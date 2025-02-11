# user_view.py
import streamlit as st
import os
import json
import requests
import re
import pycountry

import chromadb
from chromadb.config import Settings

DEBUG_MODE = True  # <--- Ensure we have a global switch for debug

# =======================
# 1) Minimal "OpenAI" HTTP Client 
# =======================
class OpenAI:
    def __init__(self):
        if "api_key" not in st.session_state or not st.session_state["api_key"]:
            raise ValueError("OpenAI API key not found in session_state. Provide it in the sidebar.")
        self.api_key = st.session_state["api_key"].strip()
        self.base_url = "https://api.openai.com/v1"
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self.session.headers["Content-Type"] = "application/json"

    def chat_completions_create(self, model, messages, max_tokens=1024, temperature=0.2):
        if DEBUG_MODE:
            st.write(f"DEBUG => Sending chat request to OpenAI, model='{model}', #messages={len(messages)}")  # DEBUG ADDED
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        resp = self.session.post(f"{self.base_url}/chat/completions", json=payload)
        if resp.status_code != 200:
            raise Exception(f"OpenAI API error: {resp.status_code}\nResponse: {resp.text}")
        return resp.json()

# =======================
# 2) LLMCountryDetector
# =======================
class LLMCountryDetector:
    SYSTEM_PROMPT = """
        ...
    """.strip()

    def __init__(self, model: str = "gpt-3.5-turbo"):
        if "api_key" not in st.session_state or not st.session_state["api_key"]:
            raise ValueError("No API key in session for LLMCountryDetector.")
        self.client = OpenAI()
        self.model = model

    def detect_countries_in_text(self, text: str):
        if DEBUG_MODE:
            st.write(f"DEBUG => detect_countries_in_text called with text='{text[:100]}'...")  # DEBUG ADDED
        if not text.strip():
            return []
        try:
            response = self.client.chat_completions_create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=512,
                temperature=0.0
            )
            raw_content = response["choices"][0]["message"]["content"].strip()
            if DEBUG_MODE:
                st.write(f"DEBUG => raw LLM country detection content: {raw_content[:200]}...")  # DEBUG ADDED

            data = []
            # Attempt JSON parse
            try:
                data = json.loads(raw_content)
            except:
                # fallback naive parse
                data = []
                matches = re.findall(
                    r'{"detected_phrase":\s*"([^"]+)",\s*"code":\s*"([A-Z]{2})"}',
                    raw_content
                )
                for m in matches:
                    phrase, code = m
                    data.append({"detected_phrase": phrase, "code": code})

            results = []
            used = set()
            for d in data:
                code = d.get("code", "").upper()
                phrase = d.get("detected_phrase", "")
                if len(code) == 2 and code.isalpha() and phrase:
                    if code not in used:
                        used.add(code)
                        results.append({"detected_phrase": phrase, "code": code})
            if DEBUG_MODE:
                st.write(f"DEBUG => final countries detected: {results}")  # DEBUG ADDED
            return results

        except Exception as e:
            if DEBUG_MODE:
                st.write(f"DEBUG => error in detect_countries_in_text: {str(e)}")
            return []

# =======================
# 3) Global CASE Instructions
# =======================
BASE_DEFAULT_PROMPT = (
    "  <ROLE>\n"
    "    ... instructions ...\n"
    "  </ROLE>"
)

# =======================
# 4) get_chroma_client
# =======================
def get_chroma_client():
    if "chroma_folder" not in st.session_state or not st.session_state["chroma_folder"].strip():
        raise ValueError("No Chroma DB folder specified in session_state.")
    folder = st.session_state["chroma_folder"].strip()

    # DEBUG prints
    if DEBUG_MODE:
        st.write(f"DEBUG => get_chroma_client using folder='{folder}'")

    # Check if folder actually exists
    if not os.path.exists(folder):
        st.write(f"DEBUG => Folder does NOT exist yet, creating it: {folder}")
        os.makedirs(folder, exist_ok=True)
    else:
        st.write(f"DEBUG => Folder exists: {folder}")

    # Print the files inside
    import glob
    files_in_folder = glob.glob(os.path.join(folder, "*"))
    st.write(f"DEBUG => files in '{folder}': {files_in_folder}")  # DEBUG ADDED

    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=folder
        )
    )
    # List existing collections
    all_colls = client.list_collections()
    st.write(f"DEBUG => after instantiating client, list_collections() => {[c.name for c in all_colls]}")  # DEBUG ADDED
    return client

# =======================
# 5) query_chroma_for_countries
# =======================
def query_chroma_for_countries(query: str, collection_name="rag_collection", n_results=10):
    if DEBUG_MODE:
        st.write(f"DEBUG => query_chroma_for_countries -> query='{query}' collection='{collection_name}'")
    detector = LLMCountryDetector()
    detected = detector.detect_countries_in_text(query)
    codes = [d["code"] for d in detected]

    client = get_chroma_client()
    try:
        collection = client.get_collection(name=collection_name)
        st.write(f"DEBUG => got collection '{collection_name}'")  # DEBUG ADDED
    except:
        st.warning(f"No collection named '{collection_name}' found in folder: {st.session_state['chroma_folder']}")
        return [], []

    all_docs = collection.get()
    all_metadata = all_docs.get("metadatas", [])
    available_countries = set()
    for m in all_metadata:
        if m and "country_code" in m:
            available_countries.add(m["country_code"])

    if DEBUG_MODE:
        st.write(f"DEBUG => in collection '{collection_name}' => total docs={len(all_docs.get('ids', []))}, countries={available_countries}")

    combined_passages = []
    combined_metadata = []
    seen_passages = set()

    if codes:
        for code in codes:
            st.write(f"DEBUG => searching for code='{code}' in where={{country_code: code}}")  # DEBUG ADDED
            results = collection.query(query_texts=[query], where={"country_code": code}, n_results=n_results)
            pass_list = results.get("documents", [[]])[0]
            meta_list = results.get("metadatas", [[]])[0]
            for p, md in zip(pass_list, meta_list):
                key = code + p
                if key not in seen_passages:
                    combined_passages.append(p)
                    combined_metadata.append(md)
                    seen_passages.add(key)
    else:
        # fallback => broad search
        st.write("DEBUG => no countries detected => broad search")  # DEBUG ADDED
        results = collection.query(query_texts=[query], n_results=n_results)
        combined_passages = results.get("documents", [[]])[0]
        combined_metadata = results.get("metadatas", [[]])[0]

    return combined_passages, combined_metadata

# =======================
# 6) generate_answer
# =======================
def generate_answer(query: str, passages, metadata):
    messages = []
    messages.append({"role": "system", "content": BASE_DEFAULT_PROMPT})

    if passages:
        docs_text = "\n\n".join([f"Doc snippet {i+1}:\n{p}" for i,p in enumerate(passages)])
    else:
        docs_text = "No relevant documents found."
    messages.append({"role": "system", "content": f"RAG DOCUMENTS:\n{docs_text}"})

    messages.append({"role": "user", "content": query})

    try:
        openai_client = OpenAI()
        resp = openai_client.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            temperature=0.2
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating final answer: {e}"

# =======================
# 7) Directory Browser 
# =======================
def list_subfolders(path: str):
    try:
        entries = os.scandir(path)
        subfolders = []
        for e in entries:
            if e.is_dir():
                subfolders.append(e.name)
        subfolders.sort()
        return subfolders
    except Exception:
        return []

def build_directory_browser():
    if "browse_path" not in st.session_state:
        st.session_state.browse_path = os.getcwd()

    st.sidebar.markdown("**Directory Browser**")
    st.sidebar.write(f"Current: `{st.session_state.browse_path}`")

    subs = list_subfolders(st.session_state.browse_path)
    if subs:
        choice = st.sidebar.selectbox("Subfolders", options=["(Select a folder)"] + subs)
        if choice != "(Select a folder)":
            if st.sidebar.button("Go to Subfolder"):
                st.session_state.browse_path = os.path.join(st.session_state.browse_path, choice)
    else:
        st.sidebar.write("No subfolders here.")

    if st.sidebar.button("Go Up One Level"):
        parent = os.path.dirname(st.session_state.browse_path)
        if parent and os.path.isdir(parent):
            st.session_state.browse_path = parent

    if st.sidebar.button("Set as Chroma Folder"):
        st.session_state["chroma_folder"] = st.session_state.browse_path
        st.sidebar.success(f"Chroma folder set to: {st.session_state.browse_path}")

# =======================
# 8) MAIN UI (No login)
# =======================
def main():
    st.set_page_config(page_title="Simple RAG UI + Dir Browser (No Login)", layout="wide")

    st.sidebar.header("Settings")

    # API Key
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    st.session_state["api_key"] = st.sidebar.text_input("OpenAI API Key", value=st.session_state["api_key"])

    # Directory Browser
    build_directory_browser()

    # Check Collections
    if st.sidebar.button("Check Collections"):
        if not st.session_state.get("chroma_folder", "").strip():
            st.sidebar.warning("No Chroma DB folder is set. Use the directory browser above.")
        else:
            st.write("DEBUG => Checking collections in user-chosen folder...")  # DEBUG ADDED
            try:
                client = get_chroma_client()
                coll_list = client.list_collections()
                if coll_list:
                    st.sidebar.success(f"Found {len(coll_list)} collection(s): {[c.name for c in coll_list]}")
                else:
                    st.sidebar.info("No collections found in that folder.")
            except Exception as e:
                st.sidebar.error(f"Error listing collections: {str(e)}")

    st.image("https://placehold.co/150x60?text=Your+Logo", use_container_width=True)
    st.title("Simple RAG UI (No Login)")
    st.write("Enter your API key, pick a folder with your Chroma DB, then ask a question.")

    user_query = st.text_input("Your question:")
    if st.button("Get Answer"):
        if not st.session_state["api_key"].strip():
            st.error("Please provide an OpenAI API key in the sidebar.")
            return
        if not st.session_state.get("chroma_folder", "").strip():
            st.error("Please pick or set your Chroma DB folder in the sidebar.")
            return
        if not user_query.strip():
            st.warning("Please enter a question first.")
            return

        passages, meta = query_chroma_for_countries(user_query, "rag_collection", n_results=10)
        answer = generate_answer(user_query, passages, meta)
        st.markdown("### RAG-based Answer")
        st.write(answer)

if __name__ == "__main__":
    main()