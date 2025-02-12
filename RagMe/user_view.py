# user_view.py
import streamlit as st
import os
import json
import requests
import re
import pycountry
from openai import OpenAI
import openai
import chromadb
from chromadb.config import Settings
import hashlib
import glob

DEBUG_MODE = True  # <--- Ensure we have a global switch for debug


from pathlib import Path
import json

# Must be the first Streamlit command
st.set_page_config(page_title="RAG User View", layout="wide")

# Authentication functions
def load_users():
    user_file = Path("users.json")
    if not user_file.exists():
        return {"admin": "admin"}  # Default admin account
    return json.loads(user_file.read_text())

def verify_login(username, password):
    users = load_users()
    return username in users and users[username] == password

def create_user_session():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
        st.session_state.is_authenticated = False

# Initialize session state
create_user_session()

def get_user_specific_directory(user_id: str) -> dict:
    """
    Creates a directory structure:
      <cwd>/chromadb_storage_user_{user_id}/
         chroma_db/
         images/
         xml/
    """
    base_dir = os.path.join(os.getcwd(), f"chromadb_storage_user_{user_id}")
    chroma_dir = os.path.join(base_dir, "chroma_db")
    if DEBUG_MODE:
        st.write(f"DEBUG: Using base directory: {base_dir}")
        st.write(f"DEBUG: Chroma subfolder: {chroma_dir}")
    dirs = {
        "base": base_dir,
        "chroma": chroma_dir,
        "images": os.path.join(base_dir, "images"),
        "xml": os.path.join(base_dir, "xml")
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        if DEBUG_MODE:
            st.write(f"DEBUG: Ensured directory exists: {dir_path}")
    files = glob.glob(os.path.join(chroma_dir, "*"))
    if DEBUG_MODE:
        st.write(f"DEBUG: Files in persist directory: {files}")
    return dirs

# Authentication UI
if not st.session_state.is_authenticated:
    st.header("Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_login(username, password):
            st.session_state.user_id = username
            st.session_state.is_authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()


def init_user_view_chroma_client(user_id: str):
    if not user_id:
        return None
        
    dirs = get_user_specific_directory(user_id)
    try:
        client = chromadb.PersistentClient(
            path=dirs["chroma"],
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        return client
    except Exception as e:
        st.error(f"Error initializing ChromaDB client: {str(e)}")
        return None



def set_openai_api_key(api_key: str):
    """Initialize OpenAI client."""
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

if 'api_key' not in st.session_state:
    st.session_state.api_key = None

with st.sidebar:
    # **Only one API key textbox is now shown.**
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        try:
            # Initialize the OpenAI client to verify the key.
            client = OpenAI(api_key=api_key)
            st.session_state.api_key = api_key
            st.success("API key set successfully!")
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {e}")

# Update the OpenAIEmbeddingFunction class
class OpenAIEmbeddings:
    """
    **Definition:** Helper class that wraps the embeddings API call using our custom OpenAI client.
    """
    def __init__(self, client):
        self.client = client

    def create(self, input, model):
        payload = {
            "input": input,
            "model": model
        }
        response = self.client.session.post(f"{self.client.base_url}/embeddings", json=payload)
        if response.status_code != 200:
            raise Exception(f"Error generating embeddings: {response.status_code}\nResponse: {response.text}")
        return response.json()

class OpenAI:
    """
    **Definition:** Minimal HTTP client for OpenAI API calls. Accepts an optional `api_key`.
    If not provided, it reads the key from `st.session_state`.
    """
    def __init__(self, api_key=None):
        if api_key:
            self.api_key = api_key.strip()
        elif "api_key" in st.session_state and st.session_state["api_key"]:
            self.api_key = st.session_state["api_key"].strip()
        else:
            raise ValueError("OpenAI API key not provided. Please set it in the sidebar.")
        self.base_url = "https://api.openai.com/v1"
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self.session.headers["Content-Type"] = "application/json"

    @property
    def embeddings(self):
        return OpenAIEmbeddings(self)

    def chat_completions_create(self, model, messages, max_tokens=1024, temperature=0.2):
        if DEBUG_MODE:
            st.write(f"DEBUG => Sending chat request to OpenAI, model='{model}', #messages={len(messages)}")
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = self.session.post(f"{self.base_url}/chat/completions", json=payload)
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code}\nResponse: {response.text}")
        return response.json()

class OpenAIEmbeddingFunction:
    """
    **Definition:** Wrapper that uses our OpenAI client to generate embeddings.
    """
    def __init__(self, api_key):
        self.api_key = api_key.strip()
        self.client = OpenAI(api_key=self.api_key)
    
    def __call__(self, input):
        # Ensure the input is a list
        if not isinstance(input, list):
            input = [input]
        # (Optional) You could sanitize texts here if needed
        sanitized_texts = [text for text in input]
        if DEBUG_MODE:
            st.write(f"DEBUG => Generating embeddings for {len(sanitized_texts)} texts")
            st.write(f"DEBUG => First text preview: {sanitized_texts[0][:100] if sanitized_texts else 'None'}")
        try:
            response = self.client.embeddings.create(
                input=sanitized_texts,
                model="text-embedding-3-large"
            )
            embeddings = [item["embedding"] for item in response["data"]]
            if DEBUG_MODE:
                st.write(f"DEBUG => Generated {len(embeddings)} embeddings")
                if embeddings:
                    st.write(f"DEBUG => Embedding dimension: {len(embeddings[0])}")
            return embeddings
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return None

    def __hash__(self):
        return int(hashlib.md5(self.api_key.encode('utf-8')).hexdigest(), 16)
    
    def __eq__(self, other):
        return isinstance(other, OpenAIEmbeddingFunction) and self.api_key == other.api_key

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
    if not st.session_state.get("user_id"):
        st.error("Please log in first")
        return None
    if not st.session_state.get("api_key"):
        st.error("Please set your OpenAI API key")
        return None
    try:
        dirs = get_user_specific_directory(st.session_state["user_id"])
        client = chromadb.PersistentClient(
            path=dirs["chroma"],
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        return client
    except Exception as e:
        st.error(f"Error initializing ChromaDB client: {str(e)}")
        return None

def get_collection(collection_name: str):
    client = get_chroma_client()
    if not client:
        return None
    try:
        embedding_function = OpenAIEmbeddingFunction(st.session_state["api_key"])
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        return collection
    except Exception as e:
        st.error(f"Error accessing collection: {str(e)}")
        return None
    
def query_collection(query: str, collection_name: str):
    client = get_chroma_client()
    if not client:
        return None
        
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=OpenAIEmbeddingFunction(st.session_state.api_key)
        )
        
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        return results
        
    except Exception as e:
        st.error(f"Error querying collection: {e}")
        return None
    
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
    if client:
        try:
            collection = client.get_collection(
                name=collection_name,
                embedding_function=OpenAIEmbeddingFunction(st.session_state["api_key"])
            )
            if DEBUG_MODE:
                st.write(f"DEBUG => got collection '{collection_name}'")
                
            # Add validation
            test_results = collection.peek()
            if DEBUG_MODE and test_results:
                st.write(f"DEBUG => Collection has {len(test_results['ids'])} documents")
                
        except Exception as e:
            if DEBUG_MODE:
                st.write(f"DEBUG => Error accessing collection: {str(e)}")
            collection = None

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
def query_and_get_answer():
    if not st.session_state.get("api_key"):
        st.error("Please set your OpenAI API key")
        return
    query = st.session_state.get("question", "")
    if not query:
        st.warning("Please enter a question")
        return
    if DEBUG_MODE:
        st.write(f"DEBUG => Processing query: '{query}'")
    collection = get_collection("rag_collection")
    if not collection:
        st.error("Could not access collection")
        return
    try:
        results = collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas"]
        )
        if DEBUG_MODE:
            st.write(f"DEBUG => Query results: {results}")
        if not results or not results.get("documents"):
            st.warning("No relevant documents found")
            return
        passages = results["documents"][0]
        metadata = results["metadatas"][0] if "metadatas" in results else []
        messages = []
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant. Answer the question using only the provided context. If you cannot answer from the context, say so."
        })
        context = "\n\n".join(passages)
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        })
        client = OpenAI(api_key=st.session_state["api_key"])
        completion = client.chat_completions_create(
            model="gpt-4",
            messages=messages,
            temperature=0.0
        )
        answer = completion["choices"][0]["message"]["content"]
        st.markdown("### Answer")
        st.write(answer)
        if DEBUG_MODE:
            st.markdown("### Debug: Retrieved Passages")
            for i, (passage, meta) in enumerate(zip(passages, metadata)):
                st.markdown(f"**Passage {i+1}:**")
                st.write(passage)
                st.write(f"Metadata: {meta}")
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        if DEBUG_MODE:
            st.write(f"DEBUG => Full error: {type(e).__name__}: {str(e)}")

# In your UI code:
if st.button("Get Answer"):
    query_and_get_answer()

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
    # st.set_page_config(page_title="Simple RAG UI + Dir Browser (No Login)", layout="wide")

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
                if client:
                    try:
                        coll_list = client.list_collections()
                        if coll_list:
                            names = [c.name for c in coll_list]
                            st.sidebar.success(f"Found {len(names)} collection(s): {names}")
                            
                            # Optionally verify each collection
                            if DEBUG_MODE:
                                for name in names:
                                    try:
                                        coll = client.get_collection(name=name)
                                        count = len(coll.peek()['ids'])
                                        st.write(f"DEBUG => Collection '{name}' has {count} documents")
                                    except Exception as e:
                                        st.write(f"DEBUG => Error checking collection '{name}': {str(e)}")
                        else:
                            st.sidebar.info("No collections found")
                    except Exception as e:
                        st.sidebar.error(f"Error listing collections: {str(e)}")
                else:
                    st.sidebar.warning("ChromaDB client not initialized")
            except Exception as e:
                st.sidebar.error(f"Error accessing ChromaDB: {str(e)}")

    st.image("https://placehold.co/150x60?text=Your+Logo", use_container_width=True)
    st.title("Simple RAG UI (No Login)")
    st.write("Enter your API key, pick a folder with your Chroma DB, then ask a question.")

    st.text_input("Your question:", key="question")
    if st.button("Get Answer", key="get_answer_button"):
        query_and_get_answer()
        if not st.session_state.get("api_key"):
            st.error("Please set your OpenAI API key in the sidebar first")
        else:
            query = st.session_state.get("question", "")
            if query:
                results = query_collection(query, "rag_collection")
                if results:
                    # Process and display results
                    passages = results["documents"][0]
                    st.write("Found relevant passages:")
                    for passage in passages:
                        st.write(passage)

if __name__ == "__main__":
    main()