# user_view.py
import hnswlib
if not hasattr(hnswlib.Index, "file_handle_count"):
    hnswlib.Index.file_handle_count = 0
import streamlit as st
import sys
try:
    import pysqlite3 as sqlite3
    sys.modules["sqlite3"] = sqlite3
except ImportError:
    pass
import os



# **Disable multi-tenancy for Chroma** (must be set before importing chromadb)
os.environ["CHROMADB_DISABLE_TENANCY"] = "true"

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
from typing import List, Dict, Any, Optional, Union, Set, Tuple

DEBUG_MODE = True  # <--- Ensure we have a global switch for debug

if DEBUG_MODE:
    print("Working directory:", os.getcwd())

from pathlib import Path
import json

##############################################################################
# UNIFIED PROMPT DEFINITIONS
##############################################################################
BASE_DEFAULT_PROMPT = (
    "  <ROLE>\n"
    "    You are an extremely knowledgeable and helpful assistant. Respond to the user’s query **ONLY** by using\n"
    "    the information available in the RAG documents. Always reason step-by-step. The user appreciates\n"
    "    thorough, accurate results.\n"
    "  </ROLE>\n\n"
    "  <INSTRUCTIONS>\n"
    "    1. YOU MUST NEVER correct the RAG documents; what is written in them is to be considered the truth. Always rely exclusively on the RAG documents for any information.\n\n"
    "    2. EXTREMELY IMPORTANT:\n"
    "       - If the user’s query relates to **only one** country and your RAG does **not** have matching information\n"
    "         for that country, you must use the **CASEB** structure (but do NEVER mention 'CASEB' as a term to the user, as this is only for your internal referencing.) .\n"
    "       - If the user’s query references **multiple** countries, you must still present a **CASEA** structure (but do NEVER mention 'CASEA' as a term to the user, as this is only for your internal referencing.) for\n"
    "         each country you do have data on. For any country **not** found in the RAG documents, strictly state\n"
    "         \"No information in my documents.\" instead of presenting partial data.\n\n"
    "    3. Your response must start with a **TL;DR Summary** in bullet points (again, only using doc-based content). Emphasize crucial\n"
    "       numerical thresholds or legal references in **bold**, and any important nuance in *italics*.\n\n"
    "    4. Next, provide a **Detailed Explanation** that remains strictly grounded in the RAG documents. If helpful,\n"
    "       include a *brief scenario* illustrating how these doc-based rules might apply.\n\n"
    "    5. Conclude with an **'Other References'** section, where you may optionally add clarifications or knowledge\n"
    "       beyond the doc but **label it** as external info. Any statutory references should appear in square brackets,\n"
    "       e.g., [Section 1, Paragraph 2].\n\n"
    "    6. If the user’s query **cannot** be answered with information from the RAG documents (meaning you have\n"
    "       **zero** coverage for that country or topic), you must switch to **CASEB**, which requires:\n"
    "       - A large \"Sorry!\" header: \"The uploaded document states nothing relevant...\"\n"
    "       - A large \"Best guess\" header: attempt an interpretation, clearly flagged as conjecture.\n"
    "       - A final large header in **red**, titled \"The fun part :-)\".\n"
    "         Provide a sarcastic or lighthearted\n"
    "         reflection (with emojis) about the query.\n\n"
    "    7. In all doc-based sections, stick strictly to the RAG documents (no external knowledge), keep your\n"
    "       professional or academically rigorous style, and preserve **bold** for pivotal references and *italics*\n"
    "       for nuance.\n\n"
    "    8. Always respond in the user’s initial query language, unless otherwise instructed.\n\n"
    "    9. Present your final output in normal text (headings in large text as described), **never** in raw XML.\n"
    "  </INSTRUCTIONS>\n\n"
    "  <STRUCTURE>\n"
    "    <REMARKS_TO_STRUCTURE>\n"
    "      Please ensure the structural elements below appear in the user’s query language.\n"
    "    </REMARKS_TO_STRUCTURE>\n\n"
    "    <!-- Two possible final output scenarios -->\n\n"
    "    <!-- Case A: Document-based answer (available info) -->\n"
    "    <CASEA>\n"
    "      <HEADER_LEVEL1>TL;DR Summary  ->  </HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>Detailed Explanation  ->  </HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>Other References  ->  </HEADER_LEVEL1>\n"
    "    </CASEA>\n\n"
    "    <!-- Case B: No relevant doc coverage for the query -->\n"
    "    <CASEB>\n"
    "      <HEADER_LEVEL1>Sorry!  ->  </HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>Best guess  ->  </HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>The fun part :-)  ->  </HEADER_LEVEL1>\n"
    "    </CASEB>\n"
    "  </STRUCTURE>\n\n"
    "  <FINAL_REMARKS>\n"
    "    - Do **not** guess if you lack data for a specific country. Instead, say \"No information in my documents.\"\n"
    "      or use **CASEB** if no data is found at all.\n"
    "    - Always apply step-by-step reasoning and keep the user’s question fully in mind.\n"
    "    - Present the final response in normal prose, using headings as indicated.\n"
    "    - If you are an ADVANCED VOICE MODE assistant, any <DELTA_FROM_MAIN_PROMPT> overrides contradictory\n"
    "      instructions above.\n"
    "  </FINAL_REMARKS>"
)

# Must be the first Streamlit command
st.set_page_config(page_title="RAG User View", layout="wide")

def get_streamlit_root_path() -> str:
    """
    Returns the root path exactly as in the main app.
    """
    if os.path.exists('/mount/src'):
        return '/mount/src/ai_pocs'
    else:
        return os.getcwd()

def get_user_specific_directory(user_id: str) -> dict:
    """
    Constructs directories:
      <root>/chromadb_storage_user_{user_id}/
         chroma_db/
         images/
         xml/
    matching the main app’s logic.
    """
    root_path = get_streamlit_root_path()
    base_dir = os.path.join(root_path, f"chromadb_storage_user_{user_id}")
    chroma_dir = os.path.join(base_dir, "chroma_db")
    
    if DEBUG_MODE:
        print(f"DEBUG: Root path: {root_path}")
        print(f"DEBUG: Using base directory: {base_dir}")
        print(f"DEBUG: Chroma subfolder: {chroma_dir}")
    
    dirs = {
        "base": base_dir,
        "chroma": chroma_dir,
        "images": os.path.join(base_dir, "images"),
        "xml": os.path.join(base_dir, "xml")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        if DEBUG_MODE:
            print(f"DEBUG: Ensured directory exists: {d}")
    
    return dirs

def init_user_view_chroma_client(user_id: str):
    """
    Initializes the ChromaDB client using the user-specific directory.
    """
    if not user_id:
        return None
    dirs = get_user_specific_directory(user_id)
    try:
        from chromadb.config import Settings
        import chromadb
        client = chromadb.PersistentClient(
            path=dirs["chroma"],
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        if DEBUG_MODE:
            print(f"DEBUG: ChromaDB client initialized with path: {dirs['chroma']}")
        return client
    except Exception as e:
        import streamlit as st
        st.error(f"Error initializing ChromaDB client: {e}")
        return None

def after_login_setup(user_id: str):
    """
    Immediately after login, force the user view to use the main app’s folder.
    """
    dirs = get_user_specific_directory(user_id)
    st.session_state["chroma_folder"] = dirs["chroma"]
    st.sidebar.success(f"ChromaDB path set to: {dirs['chroma']}")

# Shared user functions
def load_users():
    user_file = Path("users.json")
    if not user_file.exists():
        return {"admin": "admin"}
    return json.loads(user_file.read_text())

def save_users(users_dict):
    Path("users.json").write_text(json.dumps(users_dict))

def verify_login(username, password):
    users = load_users()
    return username in users and users[username] == password

def create_account(username, password):
    users = load_users()
    if username in users:
        return False, "User already exists."
    users[username] = password
    save_users(users)
    return True, "Account created successfully."

def create_user_session():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
        st.session_state.is_authenticated = False

create_user_session()

# --- Login Interface ---
if not st.session_state.is_authenticated:
    st.header("Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if verify_login(username, password):
            st.session_state.user_id = username
            st.session_state.is_authenticated = True
            st.rerun()  # Refreshes the page automatically
        else:
            st.error("Invalid credentials")
    with st.expander("Create an Account"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            success, msg = create_account(new_username, new_password)
            if success:
                st.success(msg)
            else:
                st.error(msg)
    st.stop()  # Stop here if not logged in

# --- Logout Button in Sidebar ---
if st.sidebar.button("Logout", key="logout_button"):
    st.session_state.user_id = None
    st.session_state.is_authenticated = False
    st.rerun()  # Refresh page

# --- Ensure user is logged in (login block above stops if not) ---
st.sidebar.success(f"Logged in as {st.session_state.user_id}")

# --- Force the correct Chroma folder using the main app's logic ---
after_login_setup(st.session_state.user_id)

# --- Initialize the Chroma client using the forced folder ---
try:
    chroma_client = chromadb.PersistentClient(
        path=st.session_state["chroma_folder"],
        settings=Settings(anonymized_telemetry=False)
    )
except ValueError as ve:
    st.error("Error initializing PersistentClient: " + str(ve) + "\nCheck the logs for details.")
    st.stop()

if chroma_client is None:
    st.error("Could not init Chroma for user: " + str(st.session_state.user_id))
    st.stop()

# --- List Collections ---
try:
    all_collections = chroma_client.list_collections()
    st.sidebar.write("Collections:", [c.name for c in all_collections])
except Exception as e:
    st.error(f"Error listing collections: {e}")

##############################################################################
# 2) DIRECTORY PERSISTENCE
##############################################################################
def load_selected_directory() -> str:
    """Load the previously selected directory or return the root path."""
    root_path = get_streamlit_root_path()
    file_path = os.path.join(root_path, "selected_directory.txt")
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            dir_selected = f.read().strip()
            if os.path.isdir(dir_selected):
                return dir_selected
    return root_path

def save_selected_directory(directory: str):
    """Save the selected directory path."""
    root_path = get_streamlit_root_path()
    file_path = os.path.join(root_path, "selected_directory.txt")
    
    with open(file_path, "w") as f:
        f.write(directory)

def list_subfolders(path: str) -> List[str]:
    """List all visible subfolders in the given path."""
    try:
        entries = os.scandir(path)
        subfolders = []
        for e in entries:
            # Skip hidden folders and common system directories
            if e.is_dir() and not e.name.startswith('.'):
                if e.name not in {'.git', '.streamlit', '__pycache__'}:
                    subfolders.append(e.name)
        return sorted(subfolders)
    except Exception as e:
        if DEBUG_MODE:
            st.write(f"DEBUG: Error listing subfolders: {str(e)}")
        return []

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



def set_openai_api_key(api_key: str):
    """Initialize OpenAI client."""
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return None

# Update the OpenAIEmbeddingFunction class
class OpenAI:
    """
    Minimal HTTP client for OpenAI. Must define a .embeddings property so that
    calls to `self.client.embeddings.create(...)` succeed.
    """
    def __init__(self, api_key=None):
        if api_key:
            self.api_key = api_key.strip()
        elif "api_key" in st.session_state and st.session_state["api_key"]:
            self.api_key = st.session_state["api_key"].strip()
        else:
            raise ValueError("No API key provided.")
        self.base_url = "https://api.openai.com/v1"
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self.session.headers["Content-Type"] = "application/json"

    @property
    def embeddings(self):
        """Expose an embeddings property that calls the embeddings endpoint via a helper class."""
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
        resp = self.session.post(f"{self.base_url}/chat/completions", json=payload)
        if resp.status_code != 200:
            raise Exception(f"OpenAI API error: {resp.status_code}\nResponse: {resp.text}")
        return resp.json()

class OpenAIEmbeddings:
    def __init__(self, client):
        self.client = client
    def create(self, input, model):
        payload = {"input": input, "model": model}
        resp = self.client.session.post(f"{self.client.base_url}/embeddings", json=payload)
        if resp.status_code != 200:
            raise Exception(f"Error generating embeddings: {resp.status_code}\n{resp.text}")
        return resp.json()

class OpenAIEmbeddingFunction:
    def __init__(self, api_key):
        self.api_key = api_key.strip()
        self.client = OpenAI(api_key=self.api_key)
    def __call__(self, input):
        if not isinstance(input, list):
            input = [input]
        try:
            response = self.client.embeddings.create(
                input=input,
                model="text-embedding-3-large"
            )
            return [item["embedding"] for item in response["data"]]
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return None
    @property
    def embeddings(self):
        return OpenAIEmbeddings(self.client)

# =======================
# 2) LLMCountryDetector
# =======================
class LLMCountryDetector:
    SYSTEM_PROMPT = """
    Extract countries from the text and return their ISO 3166-1 alpha-2 codes.
    
    STRICT RULES:
    1. For 2-letter codes:
       - ONLY match if they are uppercase and exactly 2 letters (e.g., "CH", "US")
       - ONLY if they are valid ISO 3166-1 alpha-2 codes
       - Must be standalone (separated by spaces/punctuation)
    
    2. For words ≥4 letters:
       - If it's a valid country name spelled like "china" or "China" or "CHINA", convert to ISO alpha-2 code
       - Example: "Switzerland" -> "CH", "SWITZERLAND" -> "CH", "switzerland" -> "CH", "helvetia" -> "CH" etc.
       - Example: "Germany" -> "DE", "gErMaNy" -> "DE", "germany" -> "DE", "GERMANY" -> "DE" etc.
       - Ignore the case; yield all correctly written countries and ignore small typos, ex. "swizzerland" -> "CH"
    
    Output exact JSON format:
    [
        {"detected_phrase": "exact matched text", "code": "XX"}
    ]
    """

    def __init__(self, api_key: str, model: str = "chatgpt-4o-latest"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.valid_codes = {country.alpha_2 for country in pycountry.countries}

    def detect_countries_in_text(self, text: str) -> List[Dict[str, str]]:
        if not text.strip():
            return []

        try:
            response = self.client.chat_completions_create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.0
            )
            
            llm_content = response["choices"][0]["message"]["content"].strip()
            cleaned = re.sub(r'^```json\s*|\s*```$', '', llm_content)
            
            try:
                data = json.loads(cleaned)
                if isinstance(data, list):
                    return [{
                        "detected_phrase": item.get("detected_phrase", "").strip(),
                        "code": item.get("code", "").strip().upper()
                    } for item in data if isinstance(item, dict)]
            except json.JSONDecodeError:
                if DEBUG_MODE:
                    st.write(f"DEBUG: Failed to parse LLM response: {llm_content}")
                pass

        except Exception as e:
            if DEBUG_MODE:
                st.write(f"DEBUG: LLM error: {str(e)}")

        return []

    def _call_llm(self, text: str) -> str:
        """Calls the LLM with our SYSTEM_PROMPT, returns raw content string."""
        try:
            response = self.client.chat_completions_create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=512,
                temperature=0.2
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            # If LLM fails, return empty
            return ""

    def _try_json(self, raw_content: str) -> List[Dict[str, str]]:
        """Attempt to parse raw_content as JSON. Return list of dicts or [] if parse fails."""
        # Remove ```json fences if any
        cleaned = re.sub(r'^```json\s*|\s*```$', '', raw_content)
        try:
            data = json.loads(cleaned)
            if not isinstance(data, list):
                return []
            # Convert to { "detected_phrase": ..., "code": ... } dicts
            results = []
            for item in data:
                phrase = item.get("detected_phrase", "")
                code = item.get("code", "")
                results.append({"detected_phrase": phrase, "code": code})
            return results
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _extract_via_regex(raw_content: str) -> List[Dict[str, str]]:
        """Attempt naive regex parse of lines like {"detected_phrase": "...", "code": "..."}."""
        matches = re.findall(
            r'{"detected_phrase":\s*"([^"]+)",\s*"code":\s*"([A-Z]{2})"}',
            raw_content
        )
        # Return raw pairs
        results = []
        for phrase, code in matches:
            results.append({"detected_phrase": phrase.strip(), "code": code.strip()})
        return results

    @staticmethod
    def naive_pycountry_detection(text: str) -> List[Dict[str, str]]:
        """Fallback method if LLM-based logic is empty."""
        found = []
        used = set()

        # 1) Look for uppercase 2-letter tokens
        tokens = re.findall(r'\b[A-Z]{2}\b', text)
        for t in tokens:
            cinfo = pycountry.countries.get(alpha_2=t)
            if cinfo and t not in used:
                found.append({"detected_phrase": t, "code": t})
                used.add(t)

        # 2) Common synonyms
        special_map = {"USA": "US", "UK": "GB", "BRD": "DE", "PRC": "CN"}
        for abbr, alpha2 in special_map.items():
            pattern = re.compile(rf'\b{abbr}\b', re.IGNORECASE)
            if pattern.search(text):
                if alpha2 not in used:
                    found.append({"detected_phrase": abbr, "code": alpha2})
                    used.add(alpha2)

        # 3) Attempt partial name detection
        name_tokens = re.findall(r'\b[A-Za-z]+(?:ian|ish|ese|)?\b', text)
        for token in name_tokens:
            token_up = token.capitalize()
            for c in pycountry.countries:
                if token_up in c.name:
                    iso2 = c.alpha_2
                    if iso2 not in used:
                        found.append({"detected_phrase": token, "code": iso2})
                        used.add(iso2)
                    break
        return found

    @staticmethod
    def _deduplicate_and_validate(data_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Additional rules:
        - If phrase < 4 chars, must be 2 uppercase letters -> valid alpha2 code
        - Otherwise normal logic -> code must be 2 uppercase letters -> valid alpha2
        """
        used = set()
        final = []
        for item in data_list:
            phrase = (item.get("detected_phrase") or "").strip()
            code = (item.get("code") or "").strip().upper()

            if not phrase or not code:
                continue

            if len(phrase) < 4:
                # Must be exactly 2 uppercase letters
                if len(phrase) == 2 and phrase.isalpha() and phrase == phrase.upper():
                    if code == phrase:
                        cinfo = pycountry.countries.get(alpha_2=code)
                        if cinfo and code not in used:
                            final.append({"detected_phrase": phrase, "code": code})
                            used.add(code)
                # else skip
            else:
                # phrase >= 4 => if code is 2 uppercase letters -> check alpha2
                if len(code) == 2 and code.isalpha():
                    cinfo = pycountry.countries.get(alpha_2=code)
                    if cinfo and code not in used:
                        final.append({"detected_phrase": phrase, "code": code})
                        used.add(code)

        return final

def get_strict_filtered_passages(query_text: str, iso_codes: List[str], n_results: int = 5):
    coll = get_collection("rag_collection")
    if not coll:
        st.error("No ChromaDB collection found.")
        return [], []

    final_passages = []
    final_metadatas = []
    used_passages = set()
    missing_countries = set(iso_codes)

    if not iso_codes:
        res = coll.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        if res and res.get("documents"):
            final_passages = res["documents"][0]
            final_metadatas = res["metadatas"][0]
        return final_passages, final_metadatas

    for code in iso_codes:
        try:
            all_docs = coll.get(
                where={"country_code": code},
                include=["documents", "metadatas"]
            )
            
            if all_docs and all_docs.get("documents"):
                missing_countries.discard(code)
                res = coll.query(
                    query_texts=[query_text],
                    where={"country_code": code},
                    n_results=min(n_results, len(all_docs["documents"])),
                    include=["documents", "metadatas"]
                )
                
                if res and res.get("documents"):
                    pass_list = res["documents"][0]
                    meta_list = res["metadatas"][0]
                    
                    for p, m in zip(pass_list, meta_list):
                        passage_key = f"{code}:{p}"
                        if passage_key not in used_passages:
                            final_passages.append(p)
                            final_metadatas.append(m)
                            used_passages.add(passage_key)
            
        except Exception as e:
            if DEBUG_MODE:
                st.write(f"DEBUG => Error querying {code}: {str(e)}")
            continue

    if missing_countries and len(iso_codes) == 1:
        if DEBUG_MODE:
            st.write(f"DEBUG => No data found for single country query: {list(missing_countries)[0]}")
        st.warning(f"No data found for country code: {list(missing_countries)[0]}")
        return [], []

    return final_passages, final_metadatas

# =======================
# 4) get_chroma_client
# =======================
def get_chroma_client():
    """Get ChromaDB client with proper path handling."""
    if not st.session_state.get("chroma_folder"):
        st.warning("No Chroma DB folder is selected. Use the directory browser in the sidebar.")
        return None
        
    if not st.session_state.get("api_key"):
        st.error("Please set your OpenAI API key first.")
        return None
        
    try:
        client = chromadb.PersistentClient(
            path=st.session_state["chroma_folder"],
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        )
        
        if DEBUG_MODE:
            st.write(f"DEBUG: ChromaDB client initialized with path: {st.session_state['chroma_folder']}")
            
        return client
        
    except Exception as e:
        st.error(f"Error initializing ChromaDB client: {str(e)}")
        if DEBUG_MODE:
            st.write(f"DEBUG: Full error: {type(e).__name__}: {str(e)}")
        return None

def get_collection(collection_name="rag_collection"):
    c = get_chroma_client()
    if not c:
        return None
    try:
        embedding_function = OpenAIEmbeddingFunction(st.session_state["api_key"])
        coll = c.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        return coll
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
# 6) generate_answer
def get_strict_filtered_passages(query_text: str, iso_codes: List[str], n_results: int = 5):
    """
    1) For each ISO code in iso_codes, do a strict filter query where={"country_code": code}.
    2) Merge the top passages from each code into a single list.
    3) Return (passages, metadatas).
    """
    coll = get_collection("rag_collection")
    if not coll:
        st.error("No ChromaDB collection found.")
        return [], []

    final_passages = []
    final_metadatas = []
    used_passages = set()
    missing_countries = set(iso_codes)  # Track which countries yield no results

    if not iso_codes:
        # fallback => broad search if no codes detected
        res = coll.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        if res and res.get("documents"):
            final_passages = res["documents"][0]
            final_metadatas = res["metadatas"][0]
        return final_passages, final_metadatas

    # If iso_codes exist, do separate queries for each code
    for code in iso_codes:
        if DEBUG_MODE:
            st.write(f"DEBUG => Querying for country code: {code}")
        
        try:
            # Get ALL documents for this country first
            all_docs = coll.get(
                where={"country_code": code},
                include=["documents", "metadatas"]
            )
            
            if DEBUG_MODE:
                st.write(f"DEBUG => Found {len(all_docs.get('documents', []))} documents for {code}")
            
            if all_docs and all_docs.get("documents"):
                missing_countries.discard(code)  # Found data for this country
                # Then do similarity search within these documents
                res = coll.query(
                    query_texts=[query_text],
                    where={"country_code": code},
                    n_results=min(n_results, len(all_docs["documents"])),
                    include=["documents", "metadatas"]
                )
                
                if res and res.get("documents"):
                    pass_list = res["documents"][0]
                    meta_list = res["metadatas"][0]
                    
                    if DEBUG_MODE:
                        st.write(f"DEBUG => Retrieved {len(pass_list)} relevant passages for {code}")
                    
                    for p, m in zip(pass_list, meta_list):
                        passage_key = f"{code}:{p}"
                        if passage_key not in used_passages:
                            final_passages.append(p)
                            final_metadatas.append(m)
                            used_passages.add(passage_key)
                            
                elif DEBUG_MODE:
                    st.write(f"DEBUG => No relevant passages found for {code}")
            
        except Exception as e:
            if DEBUG_MODE:
                st.write(f"DEBUG => Error querying {code}: {str(e)}")
            continue

    if DEBUG_MODE:
        st.write(f"DEBUG => Final results: {len(final_passages)} passages from {len(iso_codes)} countries")
        for i, (p, m) in enumerate(zip(final_passages, final_metadatas)):
            st.write(f"DEBUG => Passage {i+1} from {m.get('country_code', 'unknown')}")

    # After all queries, check if we have any missing countries
    if missing_countries and len(iso_codes) == 1:
        # Special case: Single country query with no data
        if DEBUG_MODE:
            st.write(f"DEBUG => No data found for single country query: {list(missing_countries)[0]}")
        st.warning(f"No data found for country code: {list(missing_countries)[0]}")
        return [], []  # Return empty lists to trigger CASEB format in query_and_get_answer

    return final_passages, final_metadatas

def extract_image_data(text: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
    """Extract image data from chunk text or metadata."""
    if metadata and metadata.get("full_image_chunk"):
        full_chunk = metadata["full_image_chunk"]
        base64_match = re.search(r"base64=['\"]([^'\"]+)['\"]", full_chunk)
        mime_match = re.search(r"mime_type=['\"]([^'\"]+)['\"]", full_chunk)
        
        if base64_match and mime_match:
            return {
                "base64": base64_match.group(1),
                "mime_type": mime_match.group(1)
            }
    return None

def query_and_get_answer():
    global DEBUG_MODE
    if not st.session_state.get("api_key"):
        st.error("Please set your OpenAI API key")
        return

    query = st.session_state.get("question", "")
    if not query:
        st.warning("Please enter a question")
        return

    if DEBUG_MODE:
        st.write(f"DEBUG => Processing query: '{query}'")

    try:
        detector = LLMCountryDetector(api_key=st.session_state["api_key"])
        detected_countries = detector.detect_countries_in_text(query)
    except Exception as e:
        st.error(f"Error during country detection: {e}")
        detected_countries = []

    if detected_countries:
        country_list = ", ".join([d["code"] for d in detected_countries])
        if DEBUG_MODE:
            st.write(f"DEBUG => Countries detected: {country_list}")
    else:
        country_list = "None"

    iso_codes = [d["code"] for d in detected_countries]
    passages, metadata = get_strict_filtered_passages(query, iso_codes, n_results=5)

    if not passages:
        st.warning("No relevant documents found")
        return

    # Organize passages by country
    country_passages = {}
    has_images = False
    for p, m in zip(passages, metadata):
        country_code = m.get("country_code", "UNKNOWN")
        if country_code not in country_passages:
            country_passages[country_code] = []
            
        if m.get("content_type") == "image":
            has_images = True
            image_data = extract_image_data(p, metadata=m)
            if image_data:
                country_passages[country_code].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data['mime_type']};base64,{image_data['base64']}",
                        "detail": "high"
                    }
                })
                if DEBUG_MODE:
                    st.write(f"DEBUG => Added image data for {country_code}")
        else:
            country_passages[country_code].append({
                "type": "text",
                "text": p
            })

    messages = []
    system_message = BASE_DEFAULT_PROMPT + f"\n\nDetected Countries in Query: {country_list}"
    messages.append({"role": "system", "content": system_message})

    content = []
    for code in iso_codes:
        if code in country_passages:
            content.append({
                "type": "text",
                "text": f"\n[Information for {code}]"
            })
            content.extend(country_passages[code])
            content.append({
                "type": "text",
                "text": "-" * 40
            })
    
    content.append({
        "type": "text",
        "text": f"\nQuestion: {query}"
    })
    
    messages.append({"role": "user", "content": content})

    try:
        client = OpenAI(api_key=st.session_state["api_key"])
        if DEBUG_MODE:
            st.write(f"DEBUG => Sending chat request to OpenAI, model='chatgpt-4o-latest', #messages={len(messages)}")
        
        completion = client.chat_completions_create(
            model="chatgpt-4o-latest",
            messages=messages,
            max_tokens=1024,
            temperature=0.2
        )
        answer_raw = completion["choices"][0]["message"]["content"]
        answer_clean = re.sub(r"<[^>]*>", "", answer_raw)

        st.markdown("### Answer")
        st.write(answer_clean)

        if DEBUG_MODE:
            st.markdown("### Debug: Retrieved Passages by Country")
            for code, passages in country_passages.items():
                st.markdown(f"**Country: {code}**")
                for i, p in enumerate(passages, 1):
                    st.markdown(f"Passage {i}:")
                    st.write(p.get("text", "(Image data)") if isinstance(p, dict) else p)
                st.write("-" * 40)

    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        if DEBUG_MODE:
            st.write(f"DEBUG => Full error: {type(e).__name__}: {str(e)}")

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
            model="chatgpt-4o-latest",
            messages=messages,
            max_tokens=1500,
            temperature=0.3
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

# 1) The function itself
def build_directory_browser():
    """Enhanced directory browser with proper root path handling."""
    root_path = get_streamlit_root_path()
    
    # Initialize browse path if needed
    if "browse_path" not in st.session_state:
        st.session_state["browse_path"] = load_selected_directory()
        st.session_state["chroma_folder"] = st.session_state["browse_path"]
    
    with st.sidebar.container():
        st.markdown("**Directory Browser**")
        st.write(f"Root path: `{root_path}`")
        st.write(f"Current: `{st.session_state.browse_path}`")
        
        # Show available Chroma DBs
        chroma_dbs = [d for d in list_subfolders(root_path) 
                     if d.startswith('chromadb_storage_user_')]
        
        if chroma_dbs:
            st.markdown("### Available Chroma DBs")
            selected_db = st.selectbox(
                "Select a Chroma DB",
                options=["(Select a DB)"] + chroma_dbs,
                key="chroma_db_select"
            )
            
            if selected_db != "(Select a DB)":
                db_path = os.path.join(root_path, selected_db, "chroma_db")
                if st.button("Use Selected DB"):
                    st.session_state["chroma_folder"] = db_path
                    save_selected_directory(db_path)
                    st.sidebar.success(f"Chroma folder set to: {db_path}")
        
        # Regular directory navigation
        subs = list_subfolders(st.session_state.browse_path)
        choice = st.selectbox(
            "Subfolders",
            options=["(Select a folder)"] + subs,
            key="subfolder_selectbox"
        )
        
        cols = st.columns(2)
        with cols[0]:
            if subs and choice != "(Select a folder)":
                if st.button("Go to Subfolder", key="go_subfolder_button"):
                    new_path = os.path.join(st.session_state.browse_path, choice)
                    if os.path.isdir(new_path):
                        st.session_state.browse_path = new_path
                        st.rerun()
        
        with cols[1]:
            if st.button("Go Up One Level", key="go_up_button"):
                parent = os.path.dirname(st.session_state.browse_path)
                if parent and os.path.isdir(parent) and parent.startswith(root_path):
                    st.session_state.browse_path = parent
                    st.rerun()

def parse_xml_for_chunks(text: str) -> List[Dict[str, Any]]:
    """Parse XML documents into chunks, properly handling images."""
    chunks = []
    xml_docs = text.split('<?xml version="1.0" encoding="UTF-8"?>')
    
    for doc in xml_docs:
        if not doc.strip():
            continue
            
        doc = '<?xml version="1.0" encoding="UTF-8"?>' + doc
        try:
            chunk_data = parse_single_xml_doc(doc)
            chunks.extend(chunk_data)
        except Exception as e:
            st.error(f"Error parsing XML document: {e}")
            
    update_stage("chunk", {
        "chunks": [f"{c['text'][:200]}..." for c in chunks[:5]],
        "full_chunks": [{
            "text": c["text"],
            "country": c["metadata"]["country_code"],
            "metadata": c["metadata"]
        } for c in chunks],
        "total_chunks": len(chunks)
    })
    return chunks

def parse_single_xml_doc(xml_text: str, max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """Processes XML doc with proper image handling."""
    import xml.etree.ElementTree as ET
    from io import StringIO

    tree = ET.parse(StringIO(xml_text))
    root = tree.getroot()
    country_code = root.tag.upper()

    meta = {}
    metadata_elem = root.find('metadata')
    if metadata_elem is not None:
        for child in metadata_elem:
            meta[child.tag] = (child.text or "").strip()

    content_elem = root.find('content')
    if content_elem is None:
        return []

    chunks = []
    text_buffer = []
    buf_size = 0

    def flush_buffer():
        nonlocal text_buffer, buf_size
        if text_buffer:
            combined_txt = "\n".join(text_buffer).strip()
            if combined_txt:
                chunks.append({
                    "text": combined_txt,
                    "metadata": {
                        **meta,
                        "country_code": country_code,
                        "content_type": "text"
                    }
                })
        text_buffer = []
        buf_size = 0

    if content_elem.text and content_elem.text.strip():
        text_buffer.append(content_elem.text.strip())
        buf_size += len(content_elem.text)

    for element in content_elem:
        if element.tag.lower() == "image":
            flush_buffer()
            
            image_data_node = element.find('image_data')
            if image_data_node is not None:
                mime_type = image_data_node.get('mime_type', 'image/jpeg')
                base64_data = image_data_node.get('base64', '')
                
                if base64_data:
                    path_node = element.find('path')
                    path = path_node.text if path_node is not None else "unknown_path"
                    
                    chunks.append({
                        "text": f"<image_data mime_type='{mime_type}' base64='{base64_data}' />",
                        "metadata": {
                            **meta,
                            "country_code": country_code,
                            "content_type": "image",
                            "mime_type": mime_type,
                            "path": path
                        }
                    })

    flush_buffer()
    return chunks

# =======================
# 8) MAIN UI (No login)
# =======================
def main():
    # st.set_page_config(page_title="Simple RAG UI + Dir Browser (No Login)", layout="wide")

   

    st.sidebar.header("Settings")

    
    # Initialize session variable for API key if not already set.
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            try:
                # Attempt to verify the key using your custom OpenAI class
                client = OpenAI(api_key=api_key)  # <-- Make sure OpenAI is defined in your code
                st.session_state.api_key = api_key
                st.success("API key set successfully!")
            except Exception as e:
                st.error(f"Error initializing OpenAI client: {e}")


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

    st.image("https://brandpulse.ch/wp-content/uploads/2024/02/0_BP_Victorinox_Case-Study-2000x800.png", use_container_width=True)
    st.title("Knife Legislation Expert")
    # st.write("Enter your API key, pick a folder with your Chroma DB, then ask a question.")

    # --- Input the user's question (bound to session state key "question") ---
    st.text_input(
    "Your question:", 
    key="question", 
    help="Write country names out in full or use valid ISO codes in capitals, ex. 'CH', 'US', ...")

    # --- Primary "Get Answer" Button ---
    if st.button("Get Answer"):
        query_and_get_answer()

if __name__ == "__main__":
    main()