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
from typing import List, Dict, Any

DEBUG_MODE = False  # <--- Ensure we have a global switch for debug


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
    "    1. Always rely exclusively on the RAG documents for any factual information.\n\n"
    "    2. EXTREMELY IMPORTANT:\n"
    "       - If the user’s query relates to **only one** country and your RAG does **not** have matching information\n"
    "         for that country, you must use the **CASEB** structure.\n"
    "       - If the user’s query references **multiple** countries, you must still present a **CASEA** structure for\n"
    "         each country you do have data on. For any country **not** found in the RAG documents, strictly state\n"
    "         \"No information in my documents.\" instead of presenting partial data.\n\n"
    "    3. When the user explicitly asks for help, your response must start with a **High-Level, Concise\n"
    "       'Instructions to Action'** section drawn directly from the doc (e.g., \"If x > y, then do z...\").\n\n"
    "    4. Follow with a **TL;DR Summary** in bullet points (again, only using doc-based content). Emphasize crucial\n"
    "       numerical thresholds or legal references in **bold**, and any important nuance in *italics*.\n\n"
    "    5. Next, provide a **Detailed Explanation** that remains strictly grounded in the RAG documents. If helpful,\n"
    "       include a *brief scenario* illustrating how these doc-based rules might apply.\n\n"
    "    6. Conclude with an **'Other References'** section, where you may optionally add clarifications or knowledge\n"
    "       beyond the doc but **label it** as external info. Any statutory references should appear in square brackets,\n"
    "       e.g., [Section 1, Paragraph 2].\n\n"
    "    7. If the user’s query **cannot** be answered with information from the RAG documents (meaning you have\n"
    "       **zero** coverage for that country or topic), you must switch to **CASEB**, which requires:\n"
    "       - A large \"Sorry!\" header: \"The uploaded document states nothing relevant...\"\n"
    "       - A large \"Best guess\" header: attempt an interpretation, clearly flagged as conjecture.\n"
    "       - A final large header in **red**, titled \"The fun part :-)\".\n"
    "         Provide a sarcastic or lighthearted\n"
    "         reflection (with emojis) about the query.\n\n"
    "    8. In all doc-based sections, stick strictly to the RAG documents (no external knowledge), keep your\n"
    "       professional or academically rigorous style, and preserve **bold** for pivotal references and *italics*\n"
    "       for nuance.\n\n"
    "    9. Always respond in the user’s initial query language, unless otherwise instructed.\n\n"
    "    10. Present your final output in normal text (headings in large text as described), **never** in raw XML.\n"
    "  </INSTRUCTIONS>\n\n"
    "  <STRUCTURE>\n"
    "    <REMARKS_TO_STRUCTURE>\n"
    "      Please ensure the structural elements below appear in the user’s query language.\n"
    "    </REMARKS_TO_STRUCTURE>\n\n"
    "    <!-- Two possible final output scenarios -->\n\n"
    "    <!-- Case A: Document-based answer (available info) -->\n"
    "    <CASEA>\n"
    "      <HEADER_LEVEL1>Instructions to Action  ->  </HEADER_LEVEL1>\n"
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
    st.stop()  # Do not display further content until logged in

# --- Logout Button in Sidebar ---
if st.sidebar.button("Logout", key="logout_button"):
    st.session_state.user_id = None
    st.session_state.is_authenticated = False
    st.rerun()  # or st.experimental_rerun() depending on your Streamlit version

st.sidebar.success(f"Logged in as {st.session_state.user_id}")


##############################################################################
# 2) DIRECTORY PERSISTENCE
##############################################################################
def load_selected_directory() -> str:
    file_path = "selected_directory.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            dir_selected = f.read().strip()
            if os.path.isdir(dir_selected):
                return dir_selected
    return os.getcwd()

def save_selected_directory(directory: str):
    with open("selected_directory.txt", "w") as f:
        f.write(directory)

def list_subfolders(path: str):
    try:
        entries = os.scandir(path)
        subfolders = [e.name for e in entries if e.is_dir()]
        subfolders.sort()
        return subfolders
    except Exception:
        return []

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
    """
    This version ensures that all results — from JSON parsing OR regex fallback — go
    through the same short-word checks in _deduplicate_and_validate. If the LLM says
    {"detected_phrase": "the", "code": "CD"}, that gets caught and skipped.
    """

    SYSTEM_PROMPT = """
        You are a specialized assistant for extracting ALL country references in ANY user text, 
        but you MUST carefully avoid partial-word triggers. Specifically:

        1. FULL COUNTRY CODES (CRITICAL - HIGHEST PRIORITY)
           - Extract only 2-letter codes in uppercase form like "CH", "US", "CN", "DE", etc.
           - Only do so when they appear as separate tokens, never as part of an unrelated word.
           - If recognized code is alpha2 in pycountry, keep it.

        2. FULL COUNTRY NAMES / ADJECTIVES:
           - "Switzerland", "Swiss", "United States", "American", "China", "Chinese", etc.
           - If synonyms or known abbreviations (e.g. "USA" -> "US") appear, transform them.

        3. COMMON ABBREVIATIONS:
           - "PRC" => "CN", "BRD" => "DE", "UK" => "GB", etc.

        4. CONTEXTUAL REFERENCES:
           - "Swiss law" => "CH", "German regulations" => "DE", "Chinese market" => "CN"

        EXTREMELY IMPORTANT:
        - If a substring is short (e.g., "the", "to") or not uppercase, skip it.
        - Only keep 2-letter uppercase codes if truly valid alpha2. 
        - If in doubt, skip it.

        Output EXACT JSON like:
        [
            {"detected_phrase": "DE", "code": "DE"}
        ]
        Do not add commentary or extra text.
    """.strip()

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        if "api_key" not in st.session_state or not st.session_state["api_key"]:
            raise ValueError("No API key in session for LLMCountryDetector.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def detect_countries_in_text(self, text: str) -> List[Dict[str, Any]]:
        """One unified function that calls the LLM, tries JSON parse, 
           then regex parse, then fallback. ALL results get validated."""
        if not text.strip():
            return []

        # 1) Attempt LLM-based detection
        raw_content = self._call_llm(text)

        # 2) Try JSON parse
        raw_results = self._try_json(raw_content)
        if raw_results:
            validated = self._deduplicate_and_validate(raw_results)
            if validated:
                return validated
        
        # 3) Try regex parse
        regex_pairs = self._extract_via_regex(raw_content)
        if regex_pairs:
            validated = self._deduplicate_and_validate(regex_pairs)
            if validated:
                return validated

        # 4) Fallback
        fallback_raw = self.naive_pycountry_detection(text)
        return self._deduplicate_and_validate(fallback_raw)

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
                temperature=0.0
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

# =======================
# 4) get_chroma_client
# =======================
def get_chroma_client():
    if not st.session_state.get("chroma_folder"):
        st.warning("No Chroma DB folder is set. Use the directory browser in the sidebar.")
        return None
    if not st.session_state.get("api_key"):
        st.error("Please set your OpenAI API key first.")
        return None
    try:
        # The user-chosen folder is in st.session_state["chroma_folder"]
        client = chromadb.PersistentClient(
            path=st.session_state["chroma_folder"],
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        return client
    except Exception as e:
        st.error(f"Error initializing ChromaDB client: {str(e)}")
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
    # Ensure the API key is set.
    if not st.session_state.get("api_key"):
        st.error("Please set your OpenAI API key")
        return

    # Retrieve the query from session state (bound via key "question").
    query = st.session_state.get("question", "")
    if not query:
        st.warning("Please enter a question")
        return

    if DEBUG_MODE:
        st.write(f"DEBUG => Processing query: '{query}'")

    # --- AUTO-LAUNCH COUNTRY DETECTION ---
    try:
        detector = LLMCountryDetector(api_key=st.session_state["api_key"])
        detected_countries = detector.detect_countries_in_text(query)
    except Exception as e:
        st.error(f"Error during country detection: {e}")
        detected_countries = []

    if detected_countries:
        # Create a comma‑separated list of detected country codes.
        country_list = ", ".join([d["code"] for d in detected_countries])
    else:
        country_list = "None"

    if DEBUG_MODE:
        st.write(f"DEBUG => Countries detected: {country_list}")

    # --- BUILD THE SYSTEM PROMPT ---
    # Incorporate the BASE_DEFAULT_PROMPT and add a note about detected countries.
    system_message = BASE_DEFAULT_PROMPT + f"\n\nDetected Countries in Query: {country_list}"

    # --- QUERY THE COLLECTION ---
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

        # --- BUILD THE MESSAGES FOR THE LLM ---
        messages = []
        messages.append({
            "role": "system",
            "content": system_message
        })

        # Concatenate all retrieved passages to form the context.
        context = "\n\n".join(passages)
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        })

        # --- CALL THE OPENAI CHAT API ---
        client = OpenAI(api_key=st.session_state["api_key"])
        completion = client.chat_completions_create(
            model="gpt-4",
            messages=messages,
            temperature=0.0
        )
        # after you get 'answer'
        answer_raw = completion["choices"][0]["message"]["content"]
        # strip any <HEADER_LEVEL1> or similar tags
        answer_clean = re.sub(r"<[^>]*>", "", answer_raw)

        st.markdown("### Answer")
        st.write(answer_clean)

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

# 1) The function itself
def build_directory_browser():
    # If the user hasn't loaded a path or set a folder yet, do so now.
    if "browse_path" not in st.session_state:
        st.session_state["browse_path"] = load_selected_directory()
        # Immediately set chroma_folder to the loaded path
        st.session_state["chroma_folder"] = st.session_state["browse_path"]

    # Create a single container in the sidebar so the UI appears once.
    with st.sidebar.container():
        st.markdown("**Directory Browser**")
        st.write(f"Current: `{st.session_state.browse_path}`")

        subs = list_subfolders(st.session_state.browse_path)

        # We'll use stable widget keys (no path-based suffix) to avoid duplicates
        choice = st.selectbox(
            "Subfolders",
            options=["(Select a folder)"] + subs,
            key="subfolder_selectbox"
        )
        if subs and choice != "(Select a folder)":
            if st.button("Go to Subfolder", key="go_subfolder_button"):
                st.session_state.browse_path = os.path.join(st.session_state.browse_path, choice)
                st.session_state["chroma_folder"] = st.session_state.browse_path

        if st.button("Go Up One Level", key="go_up_button"):
            parent = os.path.dirname(st.session_state.browse_path)
            if parent and os.path.isdir(parent):
                st.session_state.browse_path = parent
                st.session_state["chroma_folder"] = st.session_state.browse_path

        if st.button("Set as Chroma Folder", key="set_chroma_folder_button"):
            st.session_state["chroma_folder"] = st.session_state.browse_path
            save_selected_directory(st.session_state.browse_path)
            st.sidebar.success(f"Chroma folder set to: {st.session_state.browse_path}")

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
        query_text = st.session_state.get("question", "")
        if not query_text.strip():
            st.warning("Please enter a question.")
        else:
            # 1) Run country detection
            try:
                detector = LLMCountryDetector(api_key=st.session_state["api_key"])
                detected_list = detector.detect_countries_in_text(query_text)
            except Exception as e:
                st.error(f"Error during country detection: {e}")
                detected_list = []

            # 2) Show a nice globe with user term vs. identified country ISO
            if detected_list:
                st.write("🌍 **You seem to request legal information for the following countries:**")
                for item in detected_list:
                    # e.g. "'Switzerland' -> CH"
                    st.write(f"  • '{item['detected_phrase']}' → {item['code']}")
            else:
                st.write("❌ No country codes detected in query")

            # 3) Build the system prompt
            if detected_list:
                c_str = ", ".join([d["code"] for d in detected_list])
            else:
                c_str = "None"
            system_message = BASE_DEFAULT_PROMPT + f"\n\nDetected countries: {c_str}"

            # 4) Query the rag_collection
            coll = get_collection("rag_collection")
            if not coll:
                st.error("No ChromaDB collection available. Check your folder or API key.")
            else:
                results = coll.query(
                    query_texts=[query_text],
                    n_results=5,
                    include=["documents", "metadatas"]
                )
                if not results or not results.get("documents"):
                    st.warning("No relevant documents found.")
                else:
                    passages = results["documents"][0]
                    metadata = results["metadatas"][0]
                    context = "\n\n".join(passages)

                    # 5) Use GPT to formulate the final answer
                    messages = []
                    messages.append({"role": "system", "content": system_message})
                    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_text}"})

                    try:
                        client = OpenAI(api_key=st.session_state["api_key"])
                        completion = client.chat_completions_create(
                            model="gpt-4",
                            messages=messages,
                            temperature=0.0
                        )
                        # after you get 'answer'
                        answer_raw = completion["choices"][0]["message"]["content"]
                        # strip any <HEADER_LEVEL1> or similar tags
                        answer_clean = re.sub(r"<[^>]*>", "", answer_raw)

                        st.markdown("### Answer")
                        st.write(answer_clean)

                        if DEBUG_MODE:
                            st.markdown("### Debug: Retrieved Passages")
                            for i, (p, m) in enumerate(zip(passages, metadata)):
                                st.markdown(f"**Passage {i+1}**:")
                                st.write(p)
                                st.write(f"Metadata: {m}")

                    except Exception as e:
                        st.error(f"Error generating final answer: {e}")

if __name__ == "__main__":
    main()