import streamlit as st
import os
import json
import requests
import re
import uuid
import pycountry

CHROMA_COLLECTION_NAME = "rag_collection"
USER_DB_FILE = "users.json"  # same user DB as main app

def load_users():
    if not os.path.exists(USER_DB_FILE):
        return {}
    with open(USER_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def verify_login(username, password):
    users = load_users()
    return (username in users) and (users[username] == password)

class OpenAI:
    def __init__(self):
        if "api_key" not in st.session_state or not st.session_state["api_key"]:
            raise ValueError("OpenAI API key not found in session_state.")
        self.api_key = st.session_state["api_key"].strip()
        self.base_url = "https://api.openai.com/v1"
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self.session.headers["Content-Type"] = "application/json"

    def chat_completions_create(self, model, messages, max_tokens=1024, temperature=0.2):
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

import chromadb
from chromadb.config import Settings

def get_user_chroma_client(user_id: str):
    # If your main app uses a subfolder like "chroma_db", replicate that exactly.
    base_dir = f"chromadb_storage_user_{user_id}"
    chroma_dir = os.path.join(base_dir, "chroma_db")
    os.makedirs(chroma_dir, exist_ok=True)
    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_dir
        )
    )

class LLMCountryDetector:
    SYSTEM_PROMPT = """
        You are a specialized assistant for extracting ALL country references 
        in ANY user text. Output EXACT JSON...
    """.strip()

    def __init__(self, model: str = "gpt-3.5-turbo"):
        if "api_key" not in st.session_state or not st.session_state["api_key"]:
            raise ValueError("No API key found in session_state for LLMCountryDetector.")
        self.client = OpenAI()
        self.model = model

    def detect_countries_in_text(self, text: str):
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
            data = []
            try:
                data = json.loads(raw_content)
            except:
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
            return results
        except Exception:
            return []

BASE_DEFAULT_PROMPT = (
    "  <ROLE>\n"
    "    ... (same multi-step instructions) ...\n"
    "  </ROLE>\n\n"
    "  <INSTRUCTIONS>\n"
    "    ... see main app's instructions ...\n"
    "  </INSTRUCTIONS>\n\n"
    "  ... etc.\n"
)

def query_chroma_for_countries(user_id: str, query: str, n_results=10):
    detector = LLMCountryDetector()
    detected = detector.detect_countries_in_text(query)
    codes = [d["code"] for d in detected]

    client = get_user_chroma_client(user_id)
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except:
        st.warning("No RAG collection found. Please upload docs in main app first.")
        return [], []

    all_docs = collection.get()
    all_metadata = all_docs.get("metadatas", [])
    available_countries = set()
    for m in all_metadata:
        if m and "country_code" in m:
            available_countries.add(m["country_code"])

    combined_passages = []
    combined_metadata = []
    seen_passages = set()

    if codes:
        for code in codes:
            if code not in available_countries:
                continue
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
        results = collection.query(query_texts=[query], n_results=n_results)
        combined_passages = results.get("documents", [[]])[0]
        combined_metadata = results.get("metadatas", [[]])[0]

    return combined_passages, combined_metadata

def generate_answer_with_case_logic(user_id: str, query: str, passages, metadata):
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
        answer = resp["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        return f"Error generating final answer: {e}"

def main():
    st.set_page_config(page_title="Corporate Minimal UI (With Full RAG)")

    # 1) Build a sidebar to set the API key
    st.sidebar.header("API Key / Config")
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = ""
    st.session_state["api_key"] = st.sidebar.text_input(
        "OpenAI API Key",
        value=st.session_state["api_key"]
    )

    # 2) Logo at top, using the new parameter use_container_width
    st.image(
        "https://placehold.co/150x60?text=Your+Logo", 
        use_container_width=True
    )

    st.title("Welcome to Our Minimal RAG UI with Full Logic")

    # 3) Basic login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None

    if not st.session_state.logged_in:
        st.subheader("Login to proceed")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_login(username, password):
                st.session_state.logged_in = True
                st.session_state.user_id = username
                st.success(f"Logged in as {username}")
            else:
                st.error("Invalid credentials")
        return
    else:
        st.write(f"**Hello, {st.session_state.user_id}!**")

    # 4) Ensure we have an API key
    if not st.session_state["api_key"].strip():
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    # 5) Let user ask a question
    user_query = st.text_input("Ask a question using our RAG approach:")
    if st.button("Answer"):
        if not user_query.strip():
            st.warning("Please enter a question first.")
        else:
            passages, meta = query_chroma_for_countries(st.session_state.user_id, user_query, n_results=10)
            answer = generate_answer_with_case_logic(st.session_state.user_id, user_query, passages, meta)
            st.markdown("### Answer")
            st.write(answer)

if __name__ == "__main__":
    main()