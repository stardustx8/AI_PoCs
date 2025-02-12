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

DEBUG_MODE = True  # <--- Ensure we have a global switch for debug


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
    "       - A final large header in **red**, titled \"The fun part :-)\". Label it with *(section requested in\n"
    "         Step 0 to show how output can be steered)* in normal text. Provide a sarcastic or lighthearted\n"
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
    "      <HEADER_LEVEL1>Instructions to Action</HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>TL;DR Summary</HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>Detailed Explanation</HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>Other References</HEADER_LEVEL1>\n"
    "    </CASEA>\n\n"
    "    <!-- Case B: No relevant doc coverage for the query -->\n"
    "    <CASEB>\n"
    "      <HEADER_LEVEL1>Sorry!</HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>Best guess</HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>The fun part :-)\n"
    "        <SUBTITLE>(section requested in Step 0 to show how output can be steered)</SUBTITLE>\n"
    "      </HEADER_LEVEL1>\n"
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
        You are a specialized assistant for extracting ALL country references in ANY user text. 
        You must detect and extract EVERY single country reference, including:

        1. FULL COUNTRY CODES (CRITICAL - HIGHEST PRIORITY)
           - Extract all 2-letter codes like "CH", "US", "CN", "DE", etc.
           - These MUST be extracted even when standalone
           - Never skip any 2-letter country code
           - Example: "DE vs CH" MUST return both codes

        2. FULL NAMES:
           - "Switzerland", "United States", "China", "Germany", etc.
           - "Swiss", "American", "Chinese", "German" (adjective form)
           
        3. COMMON ABBREVIATIONS:
           - "USA" => "US"
           - "PRC" => "CN"
           - "BRD" => "DE"
           
        4. CONTEXTUAL REFERENCES:
           - "Swiss law" => "CH"
           - "German regulations" => "DE"
           - "Chinese market" => "CN"

        Output format must be EXACT JSON:
        [
            {"detected_phrase": "exact text found", "code": "XX"}
        ]

        Output ONLY valid JSON. No extra text.
    """.strip()

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        if "api_key" not in st.session_state or not st.session_state["api_key"]:
            raise ValueError("No API key in session for LLMCountryDetector.")
        # Initialize using our updated OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def detect_countries_in_text(self, text: str) -> List[Dict[str, Any]]:
        if DEBUG_MODE:
            st.write(f"DEBUG => detect_countries_in_text called with text='{text[:100]}'...")
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
                st.write(f"DEBUG => LLM raw response: {raw_content}")
            # Remove markdown formatting if present
            cleaned_content = re.sub(r'^```json\s*|\s*```$', '', raw_content)
            try:
                data = json.loads(cleaned_content)
                if not isinstance(data, list):
                    if DEBUG_MODE:
                        st.write("DEBUG => JSON response was not a list")
                    data = []
                used_codes = set()
                results = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    phrase = item.get("detected_phrase", "").strip()
                    code = item.get("code", "")
                    if len(code) == 2 and code.isupper() and phrase:
                        if code not in used_codes:
                            results.append({"detected_phrase": phrase, "code": code})
                            used_codes.add(code)
                if DEBUG_MODE:
                    st.write(f"DEBUG => Detected countries: {results}")
                if not results:
                    fallback_codes = self.naive_pycountry_detection(text)
                    if fallback_codes and DEBUG_MODE:
                        st.write(f"DEBUG => Fallback detection => {fallback_codes}")
                    results = fallback_codes
                return results
            except json.JSONDecodeError as e:
                if DEBUG_MODE:
                    st.write(f"DEBUG => JSON parse error: {str(e)}")
                # Fallback: use regex extraction
                matches = re.finditer(
                    r'{"detected_phrase":\s*"([^"]+)",\s*"code":\s*"([A-Z]{2})"}',
                    cleaned_content
                )
                results = []
                used_codes = set()
                for match in matches:
                    phrase, code = match.groups()
                    if code not in used_codes:
                        results.append({"detected_phrase": phrase, "code": code})
                        used_codes.add(code)
                if not results:
                    fallback_codes = self.naive_pycountry_detection(text)
                    if fallback_codes and DEBUG_MODE:
                        st.write(f"DEBUG => Fallback detection => {fallback_codes}")
                    results = fallback_codes
                return results
        except Exception as e:
            if DEBUG_MODE:
                st.write(f"DEBUG => LLMCountryDetector error: {str(e)}")
            return self.naive_pycountry_detection(text)

    @staticmethod
    def naive_pycountry_detection(text: str) -> List[Dict[str, str]]:
        """
        Fallback method: uses pycountry to detect 2-letter codes and country names.
        """
        found_codes = []
        used_codes = set()
        # First pass: detect 2-letter sequences
        words = re.findall(r'\b[A-Za-z]{2}\b', text)
        for w in words:
            w_up = w.upper()
            country = pycountry.countries.get(alpha_2=w_up)
            if country and w_up not in used_codes:
                found_codes.append({"detected_phrase": w, "code": w_up})
                used_codes.add(w_up)
        # Second pass: detect full country names
        words = re.findall(r'\b[A-Za-z]+\b', text)
        for w in words:
            w_up = w.upper()
            if w_up in used_codes:
                continue
            try:
                for c in pycountry.countries:
                    if w_up in c.name.upper():
                        iso2 = c.alpha_2
                        if iso2 not in used_codes:
                            found_codes.append({"detected_phrase": w, "code": iso2})
                            used_codes.add(iso2)
                        break
            except Exception:
                pass
        return found_codes

    def detect_first_country_in_text(self, text: str):
        all_codes = self.detect_countries_in_text(text)
        return all_codes[0] if all_codes else None


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
        answer = completion["choices"][0]["message"]["content"]

        # --- DISPLAY THE ANSWER ---
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

    # --- Input the user's question (bound to session state key "question") ---
    st.text_input("Your question:", key="question")

    # --- Primary "Get Answer" Button ---
    if st.button("Get Answer"):
        query_text = st.session_state.get("question", "")
        if query_text.strip():
            try:
                # --- Step 1: Auto-Launch Country Detection ---
                detector = LLMCountryDetector(api_key=st.session_state["api_key"])
                detected_countries = detector.detect_countries_in_text(query_text)
                if detected_countries:
                    # Create a comma-separated list of detected country codes
                    country_list = ", ".join([d["code"] for d in detected_countries])
                else:
                    country_list = "None"
                if DEBUG_MODE:
                    st.write(f"DEBUG => Countries detected: {country_list}")

                # --- Step 2: Build the System Prompt ---
                system_message = BASE_DEFAULT_PROMPT + f"\n\nDetected Countries in Query: {country_list}"

                # --- Step 3: Query the Collection ---
                collection = get_collection("rag_collection")
                if not collection:
                    st.error("Could not access collection")
                else:
                    results = collection.query(
                        query_texts=[query_text],
                        n_results=5,
                        include=["documents", "metadatas"]
                    )
                    if not results or not results.get("documents"):
                        st.warning("No relevant documents found")
                    else:
                        passages = results["documents"][0]
                        metadata = results["metadatas"][0] if "metadatas" in results else []

                        # --- Step 4: Build the Messages for the LLM ---
                        messages = []
                        messages.append({
                            "role": "system",
                            "content": system_message
                        })
                        context = "\n\n".join(passages)
                        messages.append({
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {query_text}"
                        })

                        # --- Step 5: Call the OpenAI Chat API ---
                        client = OpenAI(api_key=st.session_state["api_key"])
                        completion = client.chat_completions_create(
                            model="gpt-4",
                            messages=messages,
                            temperature=0.0
                        )
                        answer = completion["choices"][0]["message"]["content"]

                        # --- Step 6: Display the Answer ---
                        st.markdown("### Answer")
                        st.write(answer)
                        
                        if DEBUG_MODE:
                            st.markdown("### Debug: Retrieved Passages")
                            for i, (passage, meta) in enumerate(zip(passages, metadata)):
                                st.markdown(f"**Passage {i+1}:**")
                                st.write(passage)
                                st.write(f"Metadata: {meta}")

            except Exception as e:
                st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please enter a question above.")

if __name__ == "__main__":
    main()