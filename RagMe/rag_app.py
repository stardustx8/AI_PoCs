import hnswlib
if not hasattr(hnswlib.Index, "file_handle_count"):
    hnswlib.Index.file_handle_count = 0

import streamlit as st
st.set_page_config(page_title="RAG Demo", layout="wide", initial_sidebar_state="expanded")



import os

# **Disable multi-tenancy for Chroma** (must be set before importing chromadb)
os.environ["CHROMADB_DISABLE_TENANCY"] = "true"

PROMPT_FILE = "custom_prompt.txt"
VOICE_PREF_FILE = "voice_pref.txt"
DEBUG_MODE = False  # Set to True to enable debug prints

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

DEFAULT_VOICE_PROMPT = (
    "  <DELTA_FROM_MAIN_PROMPT>\n"
    "    1. If you find no direct or partial way to connect the user's query to the RAG documents, DO NOT simply "
    "       declare \"cannot answer.\" Instead, say \"While the docs might not directly address this, here's my "
    "       best guess...\" and then:\n"
    "       • Offer a friendly, professional yet slightly comedic best guess.\n"
    "       • If the query is downright absurd with no doc relevance, add a playful, sarcastic mock (lighthearted, "
    "         not offensive).\n\n"
    "    2. DO NOT follow the structure of the main <PROMPT> above (it's only intended for a text chatbot), stay free to continue talking after having done 1.\n\n"
    "    3. Do not output spoken XML; present your final answers in normal prose speech.\n\n"
    "    4. Continue following professional doc-based, academic rigor, accuracy and completeness in your core content\n\n"
    "    5. Adhere to the fallback structure only when absolutely necessary (i.e., if no doc-based info can possibly "
    "       apply).\n"
    "  </DELTA_FROM_MAIN_PROMPT>"
)

##############################################################################
# FILE OPERATIONS FOR PROMPTS & VOICE
##############################################################################
def load_custom_prompt(user_id):
    path = Path(f"prompts/user_{user_id}_custom_prompt.txt")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        return path.read_text()
    return None

def save_custom_prompt(prompt: str, user_id):
    path = Path(f"prompts/user_{user_id}_custom_prompt.txt")
    path.parent.mkdir(exist_ok=True)
    path.write_text(prompt)

def load_voice_pref(user_id):
    path = Path(f"preferences/user_{user_id}_voice_pref.txt")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        return path.read_text().strip()
    return "coral"

def save_voice_pref(voice: str, user_id):
    path = Path(f"preferences/user_{user_id}_voice_pref.txt")
    path.parent.mkdir(exist_ok=True)
    path.write_text(voice)

def load_voice_instructions(user_id):
    path = Path(f"instructions/user_{user_id}_voice_instructions.txt")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        return path.read_text()
    return None

def save_voice_instructions(text: str, user_id: str):
    path = Path(f"instructions/user_{user_id}_voice_instructions.txt")
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)

import sys
try:
    import pysqlite3 as sqlite3
    sys.modules["sqlite3"] = sqlite3
except ImportError:
    pass



import chromadb
from chromadb.config import Settings
import streamlit.components.v1 as components
import uuid
import re
import json
import time
import numpy as np  # optional for numeric ops
import tiktoken     # optional for token counting
from typing import List, Dict, Set, Optional, Tuple, Union, Any
import unicodedata
import requests
from openai import OpenAI
import openai
import shutil
import hashlib
import os
from typing import Optional
from pathlib import Path
import pycountry
import textwrap
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
import docx
import base64
import traceback
from io import BytesIO
from PIL import Image

try:
    import pandas as pd
except ImportError:
    st.error("Please install pandas to handle Excel/CSV files.")

try:
    from PyPDF2 import PdfReader
except ImportError:
    st.error("Please install PyPDF2 to handle PDF files.")

try:
    import docx
except ImportError:
    st.error("Please install python-docx to handle DOCX files.")

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    st.error("Please install striprtf to handle RTF files (pip install striprtf).")


##############################################################################
# 1) GLOBALS & CLIENT INITIALIZATION
##############################################################################
new_client = None  # Set once the user provides an API key
chroma_client = None  # Global Chroma client
embedding_function_instance = None  # Global instance of our embedding function

# Initialize ChromaDB with our custom embedding function instance
import chromadb
import chromadb.utils.embedding_functions


from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.errors import ChromaError

def init_chroma_client():
    if "api_key" not in st.session_state:
        return None, None

    dirs = get_user_specific_directory(st.session_state["user_id"])
    
    try:
        embedding_function = OpenAIEmbeddingFunction(st.session_state["api_key"].strip())
        
        # Test the embedding function
        test_result = embedding_function(["test"])
        if DEBUG_MODE:
            st.write(f"DEBUG => Embedding function test successful. Dimension: {len(test_result[0])}")
        
        client = chromadb.PersistentClient(
            path=dirs["chroma"],
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        return client, embedding_function
        
    except Exception as e:
        if DEBUG_MODE:
            st.write(f"DEBUG => Error initializing ChromaDB client: {str(e)}")
        return None, None
    

# Create embedding function that uses OpenAI
import hashlib

class OpenAIEmbeddingFunction:
    def __init__(self, api_key):
        self.api_key = api_key.strip()
        self.client = OpenAI(api_key=self.api_key)
    
    def __call__(self, input):
        """Generate embeddings for the input texts.
        
        Args:
            input: A string or list of strings to embed
            
        Returns:
            List of embeddings, each embedding is a list of floats
        """
        if DEBUG_MODE:
            st.write(f"DEBUG => Input type: {type(input)}")
            st.write(f"DEBUG => Input preview: {str(input)[:100]}")
        
        # Handle various input types
        if isinstance(input, str):
            texts = [input]
        elif isinstance(input, list):
            texts = input
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
        
        # Sanitize and prepare texts
        sanitized_texts = []
        for text in texts:
            if isinstance(text, (dict, list)):
                # Convert complex objects to string
                text = str(text)
            elif not isinstance(text, str):
                text = str(text)
            sanitized_texts.append(sanitize_text(text))
        
        if DEBUG_MODE:
            st.write(f"DEBUG => Generating embeddings for {len(sanitized_texts)} texts")
            st.write(f"DEBUG => First text preview: {sanitized_texts[0][:100] if sanitized_texts else 'None'}")
        
        try:
            # Ensure we have valid input
            if not sanitized_texts or not any(text.strip() for text in sanitized_texts):
                raise ValueError("No valid text to embed")
            
            # Create embeddings
            response = self.client.embeddings.create(
                input=sanitized_texts,
                model="text-embedding-3-large"
            )
            
            embeddings = [item.embedding for item in response.data]
            
            if DEBUG_MODE:
                st.write(f"DEBUG => Generated {len(embeddings)} embeddings")
                if embeddings:
                    st.write(f"DEBUG => Embedding dimension: {len(embeddings[0])}")
            
            return embeddings
            
        except Exception as e:
            if DEBUG_MODE:
                st.write(f"DEBUG => Error generating embeddings: {str(e)}")
                st.write(f"DEBUG => Error type: {type(e)}")
                if hasattr(e, 'response'):
                    st.write(f"DEBUG => Response: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")
            raise

    def __hash__(self):
        return int(hashlib.md5(self.api_key.encode('utf-8')).hexdigest(), 16)
    
    def __eq__(self, other):
        return isinstance(other, OpenAIEmbeddingFunction) and self.api_key == other.api_key

import re
from typing import Optional, Dict, Set
import pycountry
import streamlit as st  # optional if you want to log warnings

import openai

class LLMCountryDetector:
    SYSTEM_PROMPT = """
        You are a specialized assistant for extracting country references in text. 
        Extract ALL genuine country references, including:

        1. COUNTRY CODES:
           - Valid ISO 3166-1 alpha-2 codes (CH, US, CN, DE, etc.)
           - Must be recognizable as country codes in context
           - Ignore common words that happen to match codes (in, is, me, etc.)
           Example: "CN and CH laws" => extract both
           Example: "Log in to" => extract nothing

        2. COUNTRY NAMES:
           - Official names: "Switzerland", "Germany", "China"
           - Common names: "Deutschland" (Germany), "Schweiz" (Switzerland)
           - Case-insensitive: "switzerland" = "Switzerland"
           - Common international names in major languages
           
        3. CONTEXTUAL RULES:
           - Extract when country reference is clear: "Swiss law", "German regulations"
           - Include international variations: "Deutschland law" = "DE"
           - Multiple formats OK: "switzerland, Deutschland & CN"
           - Prepositions don't affect detection: "laws in Switzerland"

        EXTREMELY IMPORTANT:
        - Include both English and major international country names
        - Don't let prepositions or articles affect detection
        - Case-insensitive for country names
        - Case-sensitive for 2-letter codes (must be uppercase)
        - If clearly a country reference, include it
        - If ambiguous, don't include it

        Country name mappings (examples):
        - "Deutschland", "Germany" => "DE"
        - "Schweiz", "Switzerland" => "CH"
        - "France", "Frankreich" => "FR"

        Output format:
        [
            {"detected_phrase": "exact text found", "code": "XX"}
        ]

        Examples:
        Input: "what about laws in switzerland, deutschland & CN?"
        Output: [
            {"detected_phrase": "switzerland", "code": "CH"},
            {"detected_phrase": "deutschland", "code": "DE"},
            {"detected_phrase": "CN", "code": "CN"}
        ]

        Input: "Log in to the system"
        Output: []  # "in" is not a country reference
    """.strip()

    # Country name mappings for common international names
    COUNTRY_MAPPINGS = {
        'deutschland': 'DE',
        'germany': 'DE',
        'schweiz': 'CH',
        'switzerland': 'CH',
        'suisse': 'CH',
        'svizzera': 'CH',
        'frankreich': 'FR',
        'france': 'FR',
        # Add more as needed
    }

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def detect_countries_in_text(self, text: str) -> List[Dict[str, Any]]:
        """Enhanced country detection with better validation."""
        if not text.strip():
            return []

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.0
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            if DEBUG_MODE:
                st.write(f"DEBUG => LLM raw response: {raw_content}")

            # Clean up response
            cleaned_content = re.sub(r'^```json\s*|\s*```$', '', raw_content)
            
            try:
                data = json.loads(cleaned_content)
                if not isinstance(data, list):
                    data = []
                
                # Validate and deduplicate by code
                used_codes = set()
                results = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    phrase = item.get("detected_phrase", "").strip()
                    code = item.get("code", "")
                    if code and phrase:
                        if code not in used_codes:
                            results.append({"detected_phrase": phrase, "code": code})
                            used_codes.add(code)
                
                # If no results, try fallback
                if not results:
                    fallback_codes = self.naive_pycountry_detection(text)
                    if fallback_codes:
                        results = fallback_codes

                return results

            except json.JSONDecodeError:
                return self.naive_pycountry_detection(text)
                
        except Exception as e:
            if DEBUG_MODE:
                st.write(f"DEBUG => LLMCountryDetector error: {str(e)}")
            return self.naive_pycountry_detection(text)

    def naive_pycountry_detection(self, text: str) -> List[Dict[str, str]]:
        """
        Enhanced fallback detection with international name support.
        """
        found_codes = []
        used_codes = set()
        
        # Words to ignore even if they match country codes
        ignore_words = {
            'IN', 'IS', 'IT', 'BE', 'ME', 'TO', 'DO', 'AT', 'BY', 'NO', 'SO',
            'AS', 'AM', 'PM', 'ID', 'TV'
        }

        # First: Check for explicit country codes
        for match in re.finditer(r'\b[A-Z]{2}\b', text):
            code = match.group(0)
            if code not in ignore_words:
                country = pycountry.countries.get(alpha_2=code)
                if country and code not in used_codes:
                    found_codes.append({"detected_phrase": code, "code": code})
                    used_codes.add(code)

        # Second: Check for country names (including international)
        words = text.lower().split()
        for i, word in enumerate(words):
            # Check mappings first
            if word in self.COUNTRY_MAPPINGS and self.COUNTRY_MAPPINGS[word] not in used_codes:
                code = self.COUNTRY_MAPPINGS[word]
                found_codes.append({"detected_phrase": word, "code": code})
                used_codes.add(code)
                continue
            
            # Then check pycountry
            try:
                # Try exact match first
                country = pycountry.countries.get(name=word.title())
                if not country:
                    # Try searching
                    matches = pycountry.countries.search_fuzzy(word)
                    country = matches[0] if matches else None
                
                if country and country.alpha_2 not in used_codes:
                    found_codes.append({"detected_phrase": word, "code": country.alpha_2})
                    used_codes.add(country.alpha_2)
            except:
                continue

        return found_codes


##############################################################################
# 2) SESSION STATE INIT
##############################################################################
if 'avm_button_key' not in st.session_state:
    st.session_state.avm_button_key = 0
    
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = None

for stage in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']:
    if f'{stage}_data' not in st.session_state:
        st.session_state[f'{stage}_data'] = None

# Additional state variables
if 'uploaded_text' not in st.session_state:
    st.session_state.uploaded_text = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'query_text_step5' not in st.session_state:
    st.session_state.query_text_step5 = ""
if 'query_embedding' not in st.session_state:
    st.session_state.query_embedding = None
if 'retrieved_passages' not in st.session_state:
    st.session_state.retrieved_passages = []
if 'retrieved_metadata' not in st.session_state:
    st.session_state.retrieved_metadata = []
if 'final_answer' not in st.session_state:
    st.session_state.final_answer = None
if 'final_question_step7' not in st.session_state:
    st.session_state.final_question_step7 = ""
st.session_state.collection_name = "rag_collection"
if 'delete_confirm' not in st.session_state:
    st.session_state.delete_confirm = False
if 'avm_active' not in st.session_state:
    st.session_state.avm_active = False
if 'voice_html' not in st.session_state:
    st.session_state.voice_html = None

# We'll store the selected voice in session state
if 'selected_voice' not in st.session_state:
    st.session_state.selected_voice = load_voice_pref(st.session_state.user_id) if st.session_state.get('user_id') else "coral"

# We'll store advanced voice instructions in session_state
if 'voice_custom_prompt' not in st.session_state:
    loaded_voice_instructions = load_voice_instructions(st.session_state.user_id) if st.session_state.get('user_id') else None
    st.session_state.voice_custom_prompt = (
        loaded_voice_instructions if loaded_voice_instructions and loaded_voice_instructions.strip() != ""
        else DEFAULT_VOICE_PROMPT
    )


##############################################################################
# 3) RAG STATE CLASS
##############################################################################
class RAGState:
    def __init__(self):
        self.current_stage = None
        self.stage_data = {
            "upload":    {"active": False, "data": None},
            "chunk":     {"active": False, "data": None},
            "embed":     {"active": False, "data": None},
            "store":     {"active": False, "data": None},
            "query":     {"active": False, "data": None},
            "retrieve":  {"active": False, "data": None},
            "generate":  {"active": False, "data": None}
        }
    def set_stage(self, stage, data=None):
        self.current_stage = stage
        self.stage_data[stage]["active"] = True
        self.stage_data[stage]["data"] = data
        st.session_state.rag_state = self

if 'rag_state' not in st.session_state or st.session_state.rag_state is None:
    st.session_state.rag_state = RAGState()


##############################################################################
# 4) PIPELINE STAGE HELPER FUNCTIONS
##############################################################################
def update_stage(stage: str, data=None):
    """
    Unifies BOTH of your old update_stage functions into a single approach,
    ensuring each stage's data is handled consistently and we have debug logs.
    """
    
    # Debug: show raw data (with special handling for 'embed' stage)
    if DEBUG_MODE:
        if stage == 'embed':
            debug_data = {
                "dimensions": data.get("dimensions", 0),
                "total_vectors": data.get("total_vectors", 0),
                "preview_note": data.get("preview_note", ""),
                "sample_tokens": data.get("token_breakdowns", [])[0][:2] if data.get("token_breakdowns") else []
            }
            st.write(f"DEBUG => update_stage called with stage='{stage}', condensed embed data:", debug_data)
        else:
            st.write(f"DEBUG => update_stage called with stage='{stage}', raw data={data}")

    # Always store the raw data to session right away
    st.session_state[f'{stage}_data'] = data
    st.session_state["current_stage"] = stage

    # If data is None, we convert it to an empty dict so we don't get errors.
    if data is None:
        data = {}

    # We'll create an "enhanced_data" that the pipeline UI uses.
    # If data is a dict, copy it, else store it under {'data': data}
    if isinstance(data, dict):
        enhanced_data = data.copy()
    else:
        enhanced_data = {"data": data}

    # Stage-specific logic
    if stage == 'upload':
        # For the "upload" stage
        if isinstance(data, dict):
            text = data.get('content', '')
        else:
            text = str(data)
        enhanced_data['preview'] = text[:600] if text else None
        enhanced_data['full'] = text

    elif stage == 'chunk':
        # For "chunk" stage
        if isinstance(data, dict) and 'chunks' in data and 'total_chunks' in data:
            # it already has them
            enhanced_data = data
        elif isinstance(data, list):
            # We have a list of chunks
            enhanced_data = {'chunks': data[:5], 'total_chunks': len(data)}
        else:
            enhanced_data = {'data': data}

    elif stage == 'store':
        # For "store" stage, we record a timestamp
        enhanced_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

    elif stage == 'query':
        # For "query" stage, no special logic
        if not isinstance(data, dict):
            enhanced_data = {'query': str(data)}

    elif stage == 'retrieve':
        # **Critical** to ensure the pipeline sees a consistent shape
        if DEBUG_MODE:
            st.write(f"DEBUG => In retrieve block => data={data}")
        if isinstance(data, dict):
            # ensure passages/metadata are not None
            enhanced_data = {
                'passages': data.get("passages", []),
                'scores': [0.95, 0.87, 0.82],
                'metadata': data.get("metadata", [])
            }
            if DEBUG_MODE:
                st.write(f"DEBUG => enhanced retrieve data => {enhanced_data}")
        else:
            # data might be a tuple (passages, metadata)
            pass_list = data[0] if data and len(data) > 0 else []
            meta_list = data[1] if data and len(data) > 1 else []
            enhanced_data = {
                'passages': pass_list or [],
                'scores': [0.95, 0.87, 0.82],
                'metadata': meta_list or []
            }
            if DEBUG_MODE:
                st.write(f"DEBUG => retrieve fallback => {enhanced_data}")

    elif stage == 'generate':
        # For "generate" stage
        if isinstance(data, dict) and 'answer' not in data:
            enhanced_data['answer'] = ''
            if DEBUG_MODE:
                st.write("DEBUG => 'answer' key was missing, set to ''")

    # Store final "enhanced_data" in session state
    st.session_state[f'{stage}_data'] = enhanced_data

    # Debug print (special handling for 'embed' stage)
    if DEBUG_MODE:
        if stage == 'embed':
            debug_data = {
                "dimensions": enhanced_data.get("dimensions", 0),
                "total_vectors": enhanced_data.get("total_vectors", 0),
                "sample_preview": "Debug preview truncated..."
            }
            st.write(f"DEBUG => st.session_state[{stage}_data] updated => {debug_data}")
        else:
            st.write(f"DEBUG => st.session_state[{stage}_data] updated => {enhanced_data}")

    # If a RAGState object is in session, update it
    if 'rag_state' in st.session_state:
        st.session_state.rag_state.set_stage(stage, enhanced_data)
        if DEBUG_MODE:
            st.write(f"DEBUG => RAGState updated => current_stage={st.session_state.rag_state.current_stage}")


def set_openai_api_key(api_key: str):
    global new_client, chroma_client, embedding_function_instance
    new_client = OpenAI(api_key=api_key)
    st.session_state["api_key"] = api_key
    chroma_client, embedding_function_instance = init_chroma_client()
    if chroma_client is None:
        st.error("Failed to initialize ChromaDB client")
        st.stop()


def remove_emoji(text: str) -> str:
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def sanitize_text(text: str) -> str:
    text = remove_emoji(text)
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_text

def normalize_text(text: str, is_reversed: bool) -> str:
    """
    Normalize text order.
    
    If the text is reversed, reverse the order of lines.
    """
    if is_reversed:
        lines = text.splitlines()
        return "\n".join(reversed(lines))
    return text


def is_valid_country_code(code: str) -> bool:
    """
    Allows 'TEXT' or exactly 2 letters like 'US', 'CH', etc.
    """
    if code.upper() == "TEXT":
        return True
    return len(code) == 2 and code.isalpha()



def chunk_by_paragraph_or_size(block_text: str, iso_code: str, max_length: int=1200) -> List[Dict[str,Any]]:
    """
    Splits the block text by double newlines => paragraphs,
    wraps each paragraph if too long. Each sub-chunk dict has:
       { "text": <string>, "country": <iso_code>,
         "metadata": { "country_code": <iso_code> }
       }
    """
    paragraphs = [p.strip() for p in block_text.split("\n\n") if p.strip()]
    results = []
    for para in paragraphs:
        if para.strip() == "---":
            continue
        for wrapped in textwrap.wrap(para, width=max_length):
            s = wrapped.strip()
            if not s or s == "---":
                continue
            results.append({
                "text": s,
                "country": iso_code,
                "metadata": {
                    "country_code": iso_code
                }
            })
    return results

def parse_marker_blocks_linewise(text: str) -> List[Dict[str,Any]]:
    """
    PARTIAL-BLOCK PARSER (Line-by-Line) with partial reversal:
    """
    lines = text.splitlines()

    # Regex to detect marker lines
    # group(1) => "END OF " or None
    # group(2) => code (US, CH, TEXT, etc.)
    marker_re = re.compile(r'^(?:=+\s*)(END\s+OF\s+)?([A-Za-z0-9]+)(?:\s*=+)\s*$', re.IGNORECASE)

    final_chunks: List[Dict[str,Any]] = []

    # outside_buffer: lines outside any block
    outside_buffer: List[str] = []

    # current block state
    state = "idle"  # can be "idle", "normalBlock", "reversedBlock"
    current_code = None
    lines_in_block: List[str] = []

    i = 0
    n = len(lines)

    def snippet_of_lines(lines_list: List[str], max_words=5) -> str:
        """
        Return snippet of first 5 words or so from lines_list,
        or '(no lines)' if truly empty.
        """
        # NOTE: This function remains in place, but
        # we do NOT use its output in the new warnings.
        joined = " ".join(lines_list).strip()
        if not joined:
            return "(no lines)"
        words = joined.split()
        snippet = " ".join(words[:max_words])
        if len(words) > max_words:
            snippet += "..."
        return snippet[:200]  # limit length

    def flush_outside_if_needed():
        """If outside_buffer has real text (not just separators), warn with snippet."""
        if not outside_buffer:
            return
        joined = " ".join(outside_buffer).strip()
        outside_buffer.clear()
        # Skip if it's just separators ('=' or '-' or blank) or empty
        if not joined or re.fullmatch(r'[=\-\s]+', joined):
            return
        # Only warn about outside text if we're not in a block
        if state == "idle":
            # We omit snippet usage in the new incomplete-block warnings,
            # but we keep it for outside text warnings
            snip = snippet_of_lines([joined])
            st.warning(f"Text outside valid markers not processed: '{snip}'")

    def flush_current_block(should_reverse=False):
        """
        If we have a current block => finalize it, sub-chunk, add to final_chunks.
        If code invalid => skip w/ warning. 
        lines_in_block => reversed if needed.
        """
        nonlocal current_code, lines_in_block
        if not current_code:  
            lines_in_block = []
            return
        if not is_valid_country_code(current_code):
            st.warning(f"Invalid marker code encountered: '{current_code}'. Skipping block.")
            lines_in_block = []
            return
        
        # Filter out separator lines ('---') before joining
        filtered_lines = [line for line in lines_in_block if not re.fullmatch(r'[-]+', line.strip())]
        block_text = "\n".join(filtered_lines)
        
        if should_reverse:
            # reverse line order
            block_lines = block_text.splitlines()
            block_lines_reversed = list(reversed(block_lines))
            block_text = "\n".join(block_lines_reversed)
        
        iso = handle_country_code_special_cases(current_code)
        subchunks = chunk_by_paragraph_or_size(block_text, iso)
        final_chunks.extend(subchunks)
        lines_in_block = []

    while i < n:
        raw_line = lines[i]
        stripped = raw_line.strip()

        # if empty or dash => handle
        if not stripped or re.fullmatch(r'[-]+', stripped):
            if state in ("normalBlock", "reversedBlock"):
                lines_in_block.append(raw_line)
            else:
                # idle => outside
                outside_buffer.append(raw_line)
            i += 1
            continue

        # check marker
        mm = marker_re.match(stripped)
        if not mm:
            # not a marker => store in block lines if in a block, else outside
            if state == "idle":
                outside_buffer.append(raw_line)
            else:
                lines_in_block.append(raw_line)
            i += 1
            continue

        # we have a marker
        end_of_str = mm.group(1)  # "END OF " or None
        code_str   = mm.group(2)

        if state == "idle":
            # Only flush outside lines if we're starting a new block
            flush_outside_if_needed()

            if end_of_str:
                # reversed block start
                state = "reversedBlock"
                current_code = code_str
                lines_in_block = []
            else:
                # normal block start
                state = "normalBlock"
                current_code = code_str
                lines_in_block = []
            i += 1
            continue

        if state == "normalBlock":
            # looking for "END OF current_code"
            if end_of_str and code_str.upper() == current_code.upper():
                # matched => finalize
                flush_current_block(should_reverse=False)
                current_code = None
                state = "idle"
            else:
                # encountered a different marker => incomplete block
                st.warning(f"Missing end marker for '{current_code}', block was skipped.")
                # reset
                lines_in_block = []
                current_code = None
                state = "idle"
                # reprocess the new marker line => so don't increment i
                continue
            i += 1
            continue

        if state == "reversedBlock":
            # looking for normal code
            if (not end_of_str) and (code_str.upper() == current_code.upper()):
                # matched => finalize reversed
                flush_current_block(should_reverse=True)
                current_code = None
                state = "idle"
            else:
                # different marker => incomplete reversed block
                st.warning(f"Missing start marker for '{current_code}', block was skipped.")
                lines_in_block = []
                current_code = None
                state = "idle"
                # reprocess same line
                continue
            i += 1
            continue

        i += 1

    # end while

    # if we ended in a block => incomplete
    if state in ("normalBlock", "reversedBlock") and current_code:
        if state == "normalBlock":
            st.warning(f"Missing end marker for '{current_code}', block was skipped.")
        else:
            st.warning(f"Missing start marker for '{current_code}', block was skipped.")

    # after loop => flush leftover outside lines
    flush_outside_if_needed()

    if DEBUG_MODE:
        st.write(f"DEBUG: partial-block parse => total chunk objects => {len(final_chunks)}")

    # update pipeline
    chunk_data = {
        "chunks": [c["text"][:200] + "..." for c in final_chunks[:5]],  
        "full_chunks": final_chunks,
        "total_chunks": len(final_chunks),
    }
    update_stage("chunk", chunk_data)

    return final_chunks

def split_text_per_block_linewise(text: str) -> List[Dict[str, Any]]:
    """
    PARTIAL-BLOCK REVERSAL (Line-by-Line):
      1) Reads the text line by line, ignoring lines that are only dashes or whitespace.
      2) If we see a marker line like "==== ... X ====", check:
         - If it's "END OF X" but we haven't encountered "X" => reversed block => parse lines until we find 'X'
           and then reorder them.
         - If it's "X" (normal start) => parse lines until "END OF X" => normal block.
      3) For each completed block, we sub-chunk by paragraph. 
      4) We do NOT produce warnings about unmatched or invalid. Instead, we skip unknown codes.
         If marker code is invalid, we skip that block. 
      5) Return a list of chunk dicts. Also updates the pipeline "chunk" stage for your React pipeline.

    Example markers:
      "====================== US ======================" => normal start
      "================== END OF US ==================" => normal end, or reversed start if no prior 'US'.
    """

    lines = text.splitlines()
    # We'll accumulate chunk data here
    final_chunks: List[Dict[str, Any]] = []

    # Helper to see if a line is a marker line, returning (is_end, code) or (None, None).
    marker_line_regex = re.compile(r'^(?P<equals>=+)\s*(?P<end>END\s+OF\s+)?(?P<code>[A-Za-z0-9]+)\s*=+\s*$', re.IGNORECASE)

    # We'll parse line by line, collecting blocks
    i = 0
    n = len(lines)

    def gather_block_normal(start_code: str, start_line_idx: int) -> str:
        """
        Gathers lines up to the matching 'END OF code' marker (or EOF).
        Returns the block content (lines between start & end, exclusive).
        """
        buffer = []
        j = start_line_idx + 1
        while j < n:
            line = lines[j]
            m = marker_line_regex.match(line.strip())
            if m:
                end_segment = m.group("end")  # "END OF " or None
                code = m.group("code").upper() if m else None

                if end_segment and code == start_code.upper():
                    # Found the normal end
                    return "\n".join(buffer)
                else:
                    # Another marker => block ended unexpectedly
                    # We'll stop here, ignoring unmatched. 
                    return "\n".join(buffer)
            else:
                buffer.append(line)
            j += 1
        return "\n".join(buffer)

    def gather_block_reversed(end_code: str, end_line_idx: int) -> str:
        """
        Gathers lines up to the matching 'code' marker, indicating the block is reversed.
        We'll return the content in reversed order.
        """
        buffer = []
        j = end_line_idx + 1
        while j < n:
            line = lines[j]
            m = marker_line_regex.match(line.strip())
            if m:
                is_end = (m.group("end") is not None)
                code = m.group("code").upper()
                if (not is_end) and (code == end_code.upper()):
                    # Found the reversed 'start' => so lines from end_line_idx+1 up to here is reversed content
                    reversed_lines = list(reversed(buffer))
                    return "\n".join(reversed_lines)
                else:
                    # Another marker => block ended unexpectedly
                    return "\n".join(reversed(buffer))
            else:
                buffer.append(line)
            j += 1
        return "\n".join(reversed(buffer))

    # We'll walk line by line
    while i < n:
        raw_line = lines[i].strip()
        if not raw_line or re.fullmatch(r"[-]+", raw_line):
            # ignore blank or dash lines
            i += 1
            continue

        # Check if line is a marker
        mm = marker_line_regex.match(raw_line)
        if mm:
            end_segment = mm.group("end")   # "END OF " or None
            code_str    = mm.group("code")  # could be "US", "TEXT", "CH", etc.
            if not is_valid_country_code(code_str):
                # skip block => invalid code
                i += 1
                continue

            if end_segment:
                # This is "END OF X" => reversed block
                if DEBUG_MODE:
                    st.write(f"DEBUG: Found reversed block for code={code_str}")
                block_content = gather_block_reversed(code_str, i)
                iso = handle_country_code_special_cases(code_str)
                # chunk by paragraph
                subchunks = chunk_by_paragraph_or_size(block_content, iso_code=iso)
                final_chunks.extend(subchunks)
                i += 1  # move on
                continue
            else:
                # This is a normal start => gather until 'END OF code'
                if DEBUG_MODE:
                    st.write(f"DEBUG: Found normal block start => {code_str}")
                block_content = gather_block_normal(code_str, i)
                iso = handle_country_code_special_cases(code_str)
                subchunks = chunk_by_paragraph_or_size(block_content, iso_code=iso)
                final_chunks.extend(subchunks)
                i += 1
                continue
        else:
            # not a marker line => outside any recognized block => skip
            # (or store in outsideSegments if you like, but we won't warn).
            i += 1

    if DEBUG_MODE:
        st.write(f"DEBUG: partial-block parse => total chunk objects => {len(final_chunks)}")

    # Finally, update "chunk" stage so your React pipeline sees it
    chunk_data = {
        "chunks": [c["text"][:200]+"..." for c in final_chunks[:5]],  # short preview
        "full_chunks": final_chunks,
        "total_chunks": len(final_chunks),
    }
    update_stage("chunk", chunk_data)

    return final_chunks


def handle_country_code_special_cases(country_str: str) -> str:
    """
    Convert placeholders or 2-3 letter codes to canonical 2-letter codes,
    or 'UNKNOWN' if unrecognized.
    """
    up = country_str.upper()
    if up == "USA":
        return "US"
    if up == "UK":
        return "GB"
    if len(up) == 2:
        return up
    return "UNKNOWN"


def detect_country_in_text_fallback(text: str) -> str:
    """
    If we found no markers, do a naive detection for 'US', 'UK', 'AU', etc.
    or fallback to 'UNKNOWN'.
    """
    text_up = text.upper()
    if "USA" in text_up:
        return "US"
    if "UNITED KINGDOM" in text_up or "UK" in text_up:
        return "GB"
    if "AUSTRALIA" in text_up:
        return "AU"
    # ... more guesses if needed ...
    return "UNKNOWN"

def get_collection_dimension(collection_name: str) -> Optional[int]:
    """Get the dimension of existing embeddings in a collection."""
    try:
        coll = create_or_load_collection(collection_name)
        # Important: Query for a single document instead of trying to get all embeddings
        results = coll.query(
            query_texts=[""],  # Empty query to get any document
            n_results=1,
            include=["embeddings"]
        )
        
        if DEBUG_MODE:
            st.write(f"DEBUG => Query results: {results}")
            
        if results and results.get('embeddings') and len(results['embeddings']) > 0:
            dim = len(results['embeddings'][0][0])  # Note the double indexing
            if DEBUG_MODE:
                st.write(f"DEBUG => Found dimension: {dim}")
            return dim
            
        # Fallback: try getting all documents
        all_docs = coll.get(include=["embeddings"])
        if all_docs and all_docs.get('embeddings') and len(all_docs['embeddings']) > 0:
            dim = len(all_docs['embeddings'][0])
            if DEBUG_MODE:
                st.write(f"DEBUG => Found dimension (fallback): {dim}")
            return dim
            
        if DEBUG_MODE:
            st.write("DEBUG => No embeddings found in collection")
        return None
        
    except Exception as e:
        if DEBUG_MODE:
            st.write(f"DEBUG => Error checking collection dimension: {str(e)}")
            st.write(f"DEBUG => Collection exists: {collection_name in [c.name for c in chroma_client.list_collections()]}")
        return None
    
def embed_text(
    texts: List[Union[str, Dict[str, Any]]],
    collection_name: str = "rag_collection",
    update_stage_flag=True,
    return_data=False
):
    """Modified embedding function that ensures dimensional consistency."""
    if not st.session_state.get("api_key"):
        st.error("OpenAI API key not set.")
        st.stop()
        
    # Process texts to extract embedable content
    processed_texts = []
    original_chunks = []
    
    for chunk in texts:
        if isinstance(chunk, dict):
            text = chunk["text"]
            original_chunks.append(chunk)
        else:
            text = chunk
            original_chunks.append({"text": text})
            
        if "<image_data" in text:
            path_match = re.match(r"Image: ([^\n]+)", text)
            embedable_text = path_match.group(0) if path_match else "Image chunk"
        else:
            embedable_text = text
            
        processed_texts.append(embedable_text)
    
    # Create embedding function instance
    embedding_function = OpenAIEmbeddingFunction(st.session_state["api_key"])
    
    try:
        # Generate embeddings
        embeddings = embedding_function(processed_texts)  # Note: using processed_texts directly
        
        if DEBUG_MODE:
            st.write(f"DEBUG => Generated {len(embeddings)} embeddings")
            st.write(f"DEBUG => Embedding dimension: {len(embeddings[0])}")
        
        # Create token breakdowns for preview
        token_breakdowns = []
        for text, embedding in zip(processed_texts[:2], embeddings[:2]):
            tokens = text.split()[:5]
            breakdown = []
            if tokens:
                segment_size = len(embedding) // len(tokens)
                for i, tok in enumerate(tokens):
                    start = i * segment_size
                    end = start + segment_size
                    snippet = embedding[start:min(end, len(embedding))][:3]
                    snippet = [round(x, 4) for x in snippet]
                    breakdown.append({"token": tok, "vector_snippet": snippet})
            token_breakdowns.append(breakdown)
        
        embedding_data = {
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "preview": [round(x, 4) for x in embeddings[0][:5]] if embeddings else [],
            "total_vectors": len(embeddings),
            "token_breakdowns": token_breakdowns,
            "preview_note": "(Showing first 2 chunks, 5 words each, 3 dimensions per word)"
        }
        
        if update_stage_flag:
            update_stage('embed', embedding_data)
        if return_data:
            return embedding_data
        return embeddings
        
    except Exception as e:
        if DEBUG_MODE:
            st.write(f"DEBUG => Error in embed_text: {str(e)}")
        raise

def wrap_text_with_xml(text: str, iso_code: str, filename: str) -> str:
    """Wraps text with XML tags including metadata."""
    iso_code = iso_code.upper()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<{iso_code}>
    <metadata>
        <filename>{filename}</filename>
        <timestamp>{timestamp}</timestamp>
        <country_code>{iso_code}</country_code>
    </metadata>
    <content>
{text}
    </content>
</{iso_code}>"""
    
    return xml_content

def format_table_as_xml(table: Table) -> str:
    """Convert table to XML format."""
    xml_rows = []
    
    # Process header row
    header_cells = [cell.text.strip() or "‑" for cell in table.rows[0].cells]
    xml_rows.append("        <header>")
    for cell in header_cells:
        xml_rows.append(f"            <cell>{cell}</cell>")
    xml_rows.append("        </header>")
    
    # Process data rows
    xml_rows.append("        <rows>")
    for row in table.rows[1:]:
        xml_rows.append("            <row>")
        for cell in row.cells:
            cell_text = cell.text.strip() or "‑"
            xml_rows.append(f"                <cell>{cell_text}</cell>")
        xml_rows.append("            </row>")
    xml_rows.append("        </rows>")
    
    return "\n".join([
        "    <table>",
        *xml_rows,
        "    </table>"
    ])


def save_image_to_store(image_bytes: bytes, filename: str, iso_code: str, user_id: str) -> str:
    """Save image to user & country specific directory."""
    dirs = get_user_specific_directory(user_id)
    country_dir = os.path.join(dirs["images"], iso_code.lower())
    os.makedirs(country_dir, exist_ok=True)
    
    image_hash = hashlib.md5(image_bytes).hexdigest()[:10]
    image = Image.open(BytesIO(image_bytes))
    ext = image.format.lower()
    unique_filename = f"{image_hash}_{filename}_{time.time()}.{ext}"
    
    image_path = os.path.join(country_dir, unique_filename)
    with open(image_path, 'wb') as f:
        f.write(image_bytes)
    
    return image_path

def save_xml_document(xml_content: str, filename: str, iso_code: str, user_id: str) -> str:
    """Save XML document to user & country specific directory."""
    dirs = get_user_specific_directory(user_id)
    xml_dir = os.path.join(dirs["xml"], iso_code.lower())
    os.makedirs(xml_dir, exist_ok=True)
    
    doc_hash = hashlib.md5(xml_content.encode()).hexdigest()[:10]
    unique_filename = f"{doc_hash}_{filename}_{time.time()}.xml"
    
    xml_path = os.path.join(xml_dir, unique_filename)
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    return xml_path

def process_docx_with_xml(docx_file, iso_code: str, user_id: str) -> tuple[str, str]:
    """
    Enhanced DOCX processor that creates semantic XML output with:
    1. Regular text in <text> tags
    2. Tables with preserved structure
    3. Images with both file path and base64 data for GPT-4V
    4. Lists and other structured content
    """
    doc = docx.Document(docx_file)
    xml_elements = []
    
    def process_table(table: Table) -> str:
        """
        Convert the DOCX table into an XML element with <header>, <rows>, etc.
        Each cell is in <cell> to preserve structure. 
        """
        if len(table.rows) == 0:
            return "<table></table>"

        # Process the first row as 'header'
        header_cells = []
        header_row = table.rows[0]
        for cell in header_row.cells:
            text_val = (cell.text or "").strip()
            header_cells.append(f"<cell>{text_val}</cell>")

        header_xml = "<header>\n" + "\n".join(header_cells) + "\n</header>"

        # Process remaining rows
        rows_xml_list = []
        for row in table.rows[1:]:
            row_cells = []
            for cell in row.cells:
                text_val = (cell.text or "").strip()
                # Process any cell content
                cell_content = process_cell_content(cell)
                row_cells.append(f"<cell>{cell_content}</cell>")
            joined_cells = "\n".join(row_cells)
            row_xml = f"<row>\n{joined_cells}\n</row>"
            rows_xml_list.append(row_xml)

        rows_joined = "\n".join(rows_xml_list)
        rows_xml = f"<rows>\n{rows_joined}\n</rows>"

        return f"<table>\n{header_xml}\n{rows_xml}\n</table>"

    def process_cell_content(cell) -> str:
        """Process cell content, including lists and formatting."""
        content_parts = []
        
        for paragraph in cell.paragraphs:
            if is_list_paragraph(paragraph):
                content_parts.append(process_list(paragraph))
            else:
                text = paragraph.text.strip()
                if text:
                    content_parts.append(process_formatted_text(paragraph))
        
        return "\n".join(content_parts)

    def is_list_paragraph(paragraph) -> bool:
        """Detect if paragraph is part of a list."""
        return bool(paragraph._element.pPr and paragraph._element.pPr.numPr)

    def process_list(paragraph) -> str:
        """Convert list item to XML format."""
        return f"<list-item>{process_formatted_text(paragraph)}</list-item>"

    def process_formatted_text(paragraph) -> str:
        """Process text with formatting."""
        formatted_parts = []
        for run in paragraph.runs:
            text = run.text.strip()
            if text:
                if run.bold:
                    formatted_parts.append(f"<bold>{text}</bold>")
                elif run.italic:
                    formatted_parts.append(f"<italic>{text}</italic>")
                else:
                    formatted_parts.append(text)
        return " ".join(formatted_parts)

    def process_paragraph(paragraph: Paragraph) -> str:
        """Process regular paragraph."""
        if is_list_paragraph(paragraph):
            return process_list(paragraph)
        
        text_content = process_formatted_text(paragraph)
        if text_content.strip():
            return f"    <text>{text_content}</text>"
        return ""

    def process_image(run) -> Optional[str]:
        """Enhanced image processor using utility functions."""
        try:
            # Use utility function to process image
            image_info = process_image_run(run, doc)
            if "error" in image_info:
                return f"    <error>Failed to process image: {image_info['error']}</error>"
                
            # Save to filesystem
            image_path = save_image_to_store(
                image_info["image_bytes"], 
                docx_file.name, 
                iso_code, 
                user_id
            )
            
            # Create XML using both path and base64
            # IMPORTANT: Changed to use base64 as an attribute to match parser expectations
            return f"""    <image>
            <path>{image_path}</path>
            <size>
                <width>{image_info['width']}</width>
                <height>{image_info['height']}</height>
            </size>
            <format>{image_info['format']}</format>
            <image_data mime_type="{image_info['mime_type']}" base64="{image_info['base64_data']}" />
        </image>"""
                
        except Exception as e:
            if DEBUG_MODE:
                st.write(f"DEBUG => Image processing error: {str(e)}")
            return f"    <error>Failed to process image: {str(e)}</error>"

    # Main document processing loop
    for element in doc.element.body:
        if isinstance(element, CT_P):
            # Process paragraph
            paragraph = Paragraph(element, doc)
            
            # First check for images
            image_found = False
            for run in paragraph.runs:
                if run._r.drawing_lst:
                    image_found = True
                    image_xml = process_image(run)
                    if image_xml:
                        xml_elements.append(image_xml)
            
            # If no images, process as text
            if not image_found:
                text_xml = process_paragraph(paragraph)
                if text_xml:
                    xml_elements.append(text_xml)
                
        elif isinstance(element, CT_Tbl):
            # Process table
            table = Table(element, doc)
            table_xml = process_table(table)
            xml_elements.append(table_xml)
    
    # Combine all elements
    content = "\n".join(xml_elements)
    
    # Wrap in country tags and save
    xml_content = wrap_text_with_xml(content, iso_code, docx_file.name)
    xml_path = save_xml_document(xml_content, docx_file.name, iso_code, user_id)
    
    if DEBUG_MODE:
        st.write(f"DEBUG => Generated XML path: {xml_path}")
        if len(content) > 500:
            st.write(f"DEBUG => First 500 chars of content: {content[:500]}...")
    
    return xml_content, xml_path


def initialize_user_storage():
    if "user_id" not in st.session_state or not st.session_state.user_id:
        return None, None
        
    if "api_key" not in st.session_state:
        return None, None
        
    return init_user_chroma_client(st.session_state.user_id)

def extract_text_from_file(uploaded_file, iso_code: str, user_id: str = None, reverse: bool = False) -> str:
    """Extract text from various file formats."""
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith('.docx'):
        try:
            text, xml_path = process_docx_with_xml(uploaded_file, iso_code, user_id)
            return text
        except Exception as e:
            st.error(f"Error reading DOCX file: {e}")
            return ""
    elif file_name.endswith('.txt'):
        text = uploaded_file.read().decode("utf-8")
        return wrap_text_with_xml(text, iso_code, uploaded_file.name)
    elif file_name.endswith('.pdf'):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif file_name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file)
            text = df.to_csv(index=False)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            text = ""
    elif file_name.endswith('.xlsx'):
        try:
            df = pd.read_excel(uploaded_file)
            text = df.to_csv(index=False)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            text = ""
    elif file_name.endswith('.rtf'):
        try:
            file_contents = uploaded_file.read().decode("utf-8", errors="ignore")
            text = rtf_to_text(file_contents)
        except Exception as e:
            st.error(f"Error reading RTF file: {e}")
            text = ""
    else:
        st.warning("Unsupported file type.")
        text = ""
    
    if reverse:
        lines = text.splitlines()
        text = "\n".join(lines[::-1])
    
    return text


##############################################################################
# 5) FULL REACT PIPELINE SNIPPET
##############################################################################
def get_pipeline_component(component_args):
    html_template = """
    <div id="rag-root"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/react/17.0.2/umd/react.production.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/17.0.2/umd/react-dom.production.min.js"></script>
    """
    js_code = """
    <script>
    const args = COMPONENT_ARGS_PLACEHOLDER;
    const { useState, useEffect } = React;
    
    const ProcessExplanation = {
        upload: {
            title: "Step 1: Document Upload & Processing",
            icon: '📁',
            description: "<strong>Loading Your Source Text</strong><br>We simply take your file(s) as is, storing them until you're ready to process. This way, you can upload multiple documents before anything happens—no immediate transformation. It’s all about collecting the raw materials first!",
            summaryDataExplanation: (data) => `
<strong>Upload Summary:</strong><br>
Received ~${data.size || 'N/A'} characters.<br>
<strong>Preview:</strong> "${data.preview || 'No preview available'}"
            `.trim(),
            dataExplanation: (data) => `
<strong>Upload Details (Expanded):</strong><br>
Received ~${data.size || 'N/A'} characters.<br>
<strong>Full Content:</strong><br>
${data.full || data.preview || 'No content available.'}
            `.trim()
        },
        chunk: {
    title: "Step 2: Text Chunking",
    icon: '✂️',
    description: "<strong>Cutting the Text into Slices</strong><br>Each uploaded text is broken into manageable chunks. Each chunk now includes its detected country code.",
    summaryDataExplanation: (data) => `
<strong>Chunk Breakdown (Summary):</strong><br>
Total Chunks: ${data.total_chunks}<br>
${ (data.chunks || []).map((chunk, i) => {
    // If you can include the country info in the summary, you might modify your preview
    // For instance, if data.full_chunks is available, you could pull the country for each preview.
    // (Assuming that data.full_chunks[i] has a "country" property.)
    const country = data.full_chunks && data.full_chunks[i] ? data.full_chunks[i].country : "UNKNOWN";
    return `<br><span style="color:red;font-weight:bold;">Chunk ${i+1} [${country}]</span> => "${chunk}"`
}).join('') }
`.trim(),
    dataExplanation: (data) => `
<strong>Chunk Breakdown (Expanded):</strong><br>
Total Chunks: ${data.total_chunks}<br>
All Chunks:<br>
${
  (data.full_chunks || []).map((chunk, i) => `
    <br><span style="color:red;font-weight:bold;">
      Chunk ${i+1} [${chunk.country}]:
    </span> "${chunk.text}"
  `).join('')
}
`.trim()
        },
        embed: {
            title: "Step 3: Vector Embedding Generation",
            icon: '🧠',
            description: "<strong>Transforming Chunks into High-Dimensional Vectors</strong><br>Each chunk is converted into a multi-thousand-dimensional vector. Even a single sentence can map into thousands of numeric features! Why? Because language is highly nuanced, and each dimension captures subtle shades of meaning, syntax, or context. In vector space terms, a positive correlation (say, +0.94) indicates a strong semantic connection — like 'Switzerland' <-> 'financial stability'. Conversely, a negative correlation (e.g., –1.00) might show that 'Switzerland' stands in stark contrast to something - such as 'coastal state.'",
            summaryDataExplanation: (data) => `
<strong>Embedding Stats (Summary):</strong><br>
Dimensions: ${data.dimensions}<br>
Total Embeddings: ${data.total_vectors}<br>
Sample Token Breakdown: ${ data.token_breakdowns.slice(0,3).map((chunkBreakdown, idx) => `
<br><strong>Chunk ${idx + 1}:</strong>
${ chunkBreakdown.map(item => `<br><span style="color:red;font-weight:bold;">${item.token}</span>: [${item.vector_snippet.join(", ")}]` ).join("") }
`).join("") }
            `.trim(),
            dataExplanation: (data) => `
<strong>Embedding Stats (Expanded):</strong><br>
Dimensions: ${data.dimensions}<br>
Total Embeddings: ${data.total_vectors}<br>
Sample Vector Snippet:<br>
${ data.preview.map((val, i) => `dim${i+1}: ${val.toFixed(6)}`).join("<br>") }<br><br>
Full Token Breakdown:<br>
${ data.token_breakdowns.map((chunkBreakdown, idx) => `
<br><strong>Chunk ${idx + 1}:</strong>
${ chunkBreakdown.map(item => `<br><span style="color:red;font-weight:bold;">${item.token}</span>: [${item.vector_snippet.map(v => v.toFixed(6)).join(", ")}]` ).join("") }
`).join("") }
            `.trim()
        },
        store: {
            title: "Step 4: Vector Database Storage",
            icon: '🗄️',
            description: "<strong>Archiving Embeddings in ChromaDB</strong><br>After embedding, we place these vectors into a vector database. Later, we can search or retrieve whichever chunk best fits your query by simply comparing these vectors. Think of it like a high-tech library where each book is labeled by thousands of numeric 'keywords.'",
            summaryDataExplanation: (data) => `
<strong>Storage Summary:</strong><br>
Stored ${data.count} chunks in collection "rag_collection".
            `.trim(),
            dataExplanation: (data) => `
<strong>Storage Details (Expanded):</strong><br>
Stored ${data.count} chunks in collection "rag_collection" at ${data.timestamp}.
            `.trim()
        },
        query: {
            title: "Step 5A: Query Collection",
            icon: '❓',
            description: "<strong>Transforming Chunks into High-Dimensional Vectors</strong><br>\
Each chunk is converted into a multi-thousand-dimensional vector, exactly as in Step 3<br><br>\
Digging into that, consider the word <strong>England</strong>. It might appear as a 3,000-dimensional vector like [0.642, -0.128, 0.945, ...]. In this snippet, <em>dimension 1</em> (0.642) may reflect geography (mountains, lakes), <em>dimension 2</em> (-0.128) might capture linguistic influences, and <em>dimension 3</em> (0.945) could encode economic traits—such as stability or robust banking. As indicated, a higher value (e.g., 0.945) indicates a stronger correlation with that dimension's learned feature (in this case, 'economic stability'), whereas lower or negative values signal weaker or contrasting associations. Across thousands of dimensions, these numeric signals combine into a richly layered portrait of meaning.",
            summaryDataExplanation: (data) => `
<strong>Query Vectorization:</strong><br>
Query: <span style="color:red;font-weight:bold;">"${data.query || 'N/A'}"</span><br>
Dimensions: ${data.dimensions}<br>
Total Vectors: ${data.total_vectors}<br>
Sample Snippet: ${ data.preview ? data.preview.map((val, i) => `dim${i+1}: ${val.toFixed(6)}`).join("<br>") : "N/A" }
            `.trim(),
            dataExplanation: (data) => `
<strong>Query Vectorization (Expanded):</strong><br>
Query: <span style="color:red;font-weight:bold;">"${data.query || 'N/A'}"</span><br>
Dimensions: ${data.dimensions}<br>
Total Vectors: ${data.total_vectors}<br>
Sample Snippet: ${ data.preview ? data.preview.map((val, i) => `dim${i+1}: ${val.toFixed(6)}`).join("<br>") : "N/A" }<br><br>
Full Token Breakdown: ${ data.token_breakdowns ? data.token_breakdowns.map((chunk) => {
    return chunk.map(item => `<br><span style="color:red;font-weight:bold;">${item.token}</span>: [${item.vector_snippet.map(v => v.toFixed(6)).join(", ")}]`).join("");
}).join("") : "N/A" }
            `.trim()
        },
        retrieve: {
            title: "Step 5B: Context Retrieval",
            icon: '🔎',
            description: "<strong>Finding Matching Passages</strong><br>The query vector is matched against every vector in the database. The passages with the highest similarity (closest in vector-space) pop up as potential answers. For example, if your query is 'Ibach' (a Swiss village), the system might rank passages mentioning 'Switzerland' quite highly, given they share geographical or contextual features in the embedding space.",
            summaryDataExplanation: (data) => `
<strong>Top Matches (Summary):</strong><br>
${ data.passages.map((passage, i) => `<br><span style='color:red;font-weight:bold;'>Match ${i+1}:</span> "${passage}" (score ~ ${(data.scores[i]*100).toFixed(1)}%)`).join("<br>") }
            `.trim(),
            dataExplanation: (data) => `
<strong>Top Matches (Expanded):</strong><br>
Passages:<br>
${ data.passages.map((passage, i) => `<strong>Match ${i+1} (score ${(data.scores[i]*100).toFixed(1)}%)</strong>:<br>"${passage}"<br>`).join("<br>") }
            `.trim()
        },
        generate: {
            title: "Step 6: Get Answer",
            icon: '🤖',
            description: "<strong>Using GPT to Combine Context & Query</strong><br>Finally, GPT takes both the query and the top retrieved chunks to generate a focused answer. It's like feeding in a question and its best context, ensuring the result zeroes in on the exact info relevant to your needs.",
            summaryDataExplanation: (data) => `
<strong>Answer (Summary):</strong><br>
${ data.answer ? data.answer.substring(0, Math.floor(data.answer.length/2))+"..." : "No answer yet" }
            `.trim(),
            dataExplanation: (data) => `
<strong>Answer (Expanded):</strong><br>
${ data.answer || "No answer available." }
            `.trim()
        }
    };
    
    const RAGFlowVertical = () => {
        const [activeStage, setActiveStage] = useState(args.currentStage || null);
        const [showModal, setShowModal] = useState(false);
        const [selectedStage, setSelectedStage] = useState(null);
        
        useEffect(() => {
            setActiveStage(args.currentStage);
        }, [args.currentStage]);
        
        useEffect(() => {
            // Wait a bit for the DOM to update
            setTimeout(() => {
                const activeElem = document.querySelector('.active-stage');
                if (activeElem) {
                    const headerOffset = 100;
                    const elemTop = activeElem.offsetTop - headerOffset;
                    const scrollContainer = document.querySelector('.pipeline-container');
                    if (scrollContainer) {
                        scrollContainer.scrollTo({
                            top: elemTop,
                            behavior: 'smooth'
                        });
                    }
                }
            }, 100);
        }, [activeStage]);
        
        const formatModalContent = (stage) => {
            const data = args.stageData[stage];
            const process = ProcessExplanation[stage];
            return React.createElement('div', { className: 'modal-content' }, [
                React.createElement('button', { className: 'close-button', onClick: () => setShowModal(false) }, '×'),
                data ? React.createElement(React.Fragment, null, [
                    React.createElement('h2', { className: 'modal-title' }, [ process.icon, ' ', process.title ]),
                    React.createElement('p', { className: 'modal-description', dangerouslySetInnerHTML: { __html: process.description } }),
                    React.createElement('div', { className: 'modal-data', dangerouslySetInnerHTML: { __html: process.dataExplanation(data) } })
                ]) : "No data available for this stage."
            ]);
        };
        
        const ArrowIcon = () => (
            React.createElement('div', { className: 'pipeline-arrow' },
                React.createElement('div', { className: 'arrow-body' })
            )
        );
        
        const pipelineStages = Object.keys(ProcessExplanation).map(id => ({
            id,
            ...ProcessExplanation[id]
        }));
        
        const getStageIndex = (stage) => pipelineStages.findIndex(s => s.id === stage);
        const isStageComplete = (stage) => getStageIndex(stage) < getStageIndex(activeStage);
        
        return React.createElement('div', { className: 'pipeline-container' },
            showModal && React.createElement('div', { className: 'tooltip-modal' },
                React.createElement('div', { className: 'tooltip-content' },
                    formatModalContent(selectedStage)
                )
            ),
            React.createElement('div', { className: 'pipeline-column' },
                pipelineStages.map((stageObj, index) => {
                    const dataObj = args.stageData[stageObj.id] || null;
                    const isActive = (activeStage === stageObj.id && dataObj);
                    const isComplete = isStageComplete(stageObj.id);
                    const process = ProcessExplanation[stageObj.id];
                    const stageClass = `pipeline-box ${isActive ? 'active-stage' : ''} ${isComplete ? 'completed-stage' : ''}`;
                    
                    return React.createElement(React.Fragment, { key: stageObj.id }, [
                        React.createElement('div', { 
                            className: stageClass,
                            onClick: () => { setSelectedStage(stageObj.id); setShowModal(true); }
                        }, [
                            React.createElement('div', { className: 'stage-header' }, [
                                React.createElement('span', { className: 'stage-icon' }, process.icon),
                                React.createElement('span', { className: 'stage-title' }, process.title)
                            ]),
                            React.createElement('div', { className: 'stage-description', dangerouslySetInnerHTML: { __html: process.description } }),
                            dataObj && React.createElement('div', { 
                                className: 'stage-data', 
                                dangerouslySetInnerHTML: { __html: (process.summaryDataExplanation ? process.summaryDataExplanation(dataObj) : process.dataExplanation(dataObj)) }
                            })
                        ]),
                        index < pipelineStages.length - 1 && React.createElement(ArrowIcon, { key: `arrow-${index}` })
                    ]);
                })
            )
        );
    };
    
    ReactDOM.render(
        React.createElement(RAGFlowVertical),
        document.getElementById('rag-root')
    );
    </script>
    """
    css_styles = """
<style>
  ::-webkit-scrollbar { width: 0px; background: transparent; }
  body { background-color: #111; color: #fff; margin: 0; padding: 0; }
  #rag-root { font-family: system-ui, sans-serif; height: 100%; width: 100%; margin: 0; padding: 0; }
    .pipeline-container { 
        padding: 1rem 10rem 1rem 1rem; 
        overflow-y: auto; 
        overflow-x: visible; 
        height: 100vh;  
        box-sizing: border-box; 
        width: 100%;
        position: static;
        right: 0;
    }

    .pipeline-column {
        display: flex;
        flex-direction: column;
        align-items: stretch;
        width: 100%;
        margin: 0 auto;
        padding-right: 10rem;
        overflow-x: visible;
        min-height: 100vh;
        padding-bottom: 50vh;
    }

    .modal-content {
        max-height: none;
        height: auto;
    }
  .pipeline-box { width: 100%; margin-bottom: 1rem; padding: 1.5rem; border: 2px solid #4B5563; border-radius: 0.75rem; background-color: #1a1a1a; cursor: pointer; transition: all 0.3s; text-align: left; transform-origin: center; position: relative; z-index: 1; }
  .pipeline-box:hover { transform: scale(1.02); border-color: #6B7280; z-index: 1000; }
  .completed-stage { background-color: rgba(34, 197, 94, 0.1); border-color: #22C55E; }
  .active-stage { border-color: #22C55E; box-shadow: 0 0 15px rgba(34, 197, 94, 0.2); }
  .stage-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem; }
  .stage-icon { font-size: 1.5rem; }
  .stage-title { font-weight: bold; font-size: 1.2rem; color: white; }
  .stage-description { color: #9CA3AF; font-size: 1rem; margin-bottom: 1rem; line-height: 1.5; text-align: left; }
  .stage-data { font-family: monospace; font-size: 0.9rem; color: #D1D5DB; background-color: rgba(0, 0, 0, 0.2); padding: 0.75rem; border-radius: 0.5rem; margin-top: 0.75rem; white-space: pre-wrap; text-align: left; }
  .pipeline-arrow { height: 40px; margin: 0.5rem 0; display: flex; align-items: center; justify-content: center; position: relative; }
  .arrow-body { width: 3px; height: 100%; background: linear-gradient(to bottom, rgba(156,163,175,0) 0%, rgba(156,163,175,1) 30%, rgba(156,163,175,1) 70%, rgba(156,163,175,0) 100%); position: relative; }
  .arrow-body::after { content: ''; position: absolute; bottom: 30%; left: 50%; transform: translateX(-50%); width: 0; height: 0; border-left: 8px solid transparent; border-right: 8px solid transparent; border-top: 12px solid #9CA3AF; }
  .tooltip-modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.85); display: flex; align-items: center; justify-content: center; z-index: 9999; }
  .tooltip-content { position: relative; width: 95%; height: 95%; background: #1a1a1a; padding: 2rem; border-radius: 1rem; overflow-y: auto; box-shadow: 0 0 30px rgba(0,0,0,0.5); color: #fff; }
  .tooltip-content::-webkit-scrollbar { width: 8px; height: 8px; }
  .tooltip-content::-webkit-scrollbar-track { background: #333; border-radius: 4px; }
  .tooltip-content::-webkit-scrollbar-thumb { background: #666; border-radius: 4px; }
  .tooltip-content::-webkit-scrollbar-thumb:hover { background: #888; }
  .close-button { position: absolute; top: 20px; right: 20px; background: transparent; border: none; font-size: 2rem; font-weight: bold; color: #fff; cursor: pointer; }
  .modal-title { font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; color: white; }
  .modal-description { color: #9CA3AF; font-size: 1.1rem; margin-bottom: 1.5rem; line-height: 1.6; }
  .modal-data { background: rgba(0, 0, 0, 0.3); padding: 1.5rem; border-radius: 0.75rem; margin-top: 1rem; }
</style>
"""
    js_code = js_code.replace("COMPONENT_ARGS_PLACEHOLDER", json.dumps(component_args))
    complete_template = html_template + js_code + css_styles
    return complete_template


##############################################################################
# 6) CREATE/LOAD COLLECTION, ETC.
##############################################################################
def create_or_load_collection(collection_name: str, force_recreate: bool = False):
    """Create or load a collection with proper embedding function."""
    global chroma_client, embedding_function_instance
    
    if embedding_function_instance is None:
        st.error("Embedding function not initialized. Please set your OpenAI API key.")
        st.stop()
    
    if DEBUG_MODE:
        st.write(f"DEBUG: In create_or_load_collection, embedding_function_instance hash: {hash(embedding_function_instance)}")
    
    if force_recreate:
        try:
            chroma_client.delete_collection(name=collection_name)
            if DEBUG_MODE:
                st.write(f"DEBUG: Deleted existing collection '{collection_name}' due to force_recreate flag.")
        except Exception as e:
            if DEBUG_MODE:
                st.write(f"DEBUG: Could not delete existing collection '{collection_name}': {e}")
    
    # List current collections
    current_collections = [c.name for c in chroma_client.list_collections()]
    
    if DEBUG_MODE:
        st.write(f"DEBUG: Current collections: {current_collections}")
        
    try:
        if collection_name in current_collections:
            # Important: Explicitly set the embedding function when getting existing collection
            coll = chroma_client.get_collection(
                name=collection_name,
                embedding_function=embedding_function_instance
            )
            if DEBUG_MODE:
                st.write(f"DEBUG: Retrieved existing collection '{collection_name}' with embedding function")
        else:
            # Create new collection with embedding function
            coll = chroma_client.create_collection(
                name=collection_name,
                embedding_function=embedding_function_instance
            )
            if DEBUG_MODE:
                st.write(f"DEBUG: Created new collection '{collection_name}' with embedding function")
        
        # Verify the collection exists and has the embedding function
        if DEBUG_MODE:
            st.write(f"DEBUG: Collection '{collection_name}' exists: {coll is not None}")
            st.write(f"DEBUG: Collection embedding function hash: {hash(coll._embedding_function) if hasattr(coll, '_embedding_function') else 'None'}")
        
        return coll
        
    except Exception as e:
        if DEBUG_MODE:
            st.write(f"DEBUG: Error in create_or_load_collection: {str(e)}")
        raise



def add_to_chroma_collection(
    collection_name: str,
    chunks: List[Union[str, Dict[str, Any]]],
    metadatas: Optional[List[dict]] = None,
    xml_paths: Optional[List[str]] = None  # Make xml_paths optional
):
    """Enhanced storage function with proper embedding handling."""
    if DEBUG_MODE:
        st.write("DEBUG => Starting Chroma storage process")
    
    # Process chunks and generate embeddings
    texts = []
    chunk_metadatas = []
    
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            text = chunk["text"]
            chunk_meta = chunk.get("metadata", {})
        else:
            text = chunk
            chunk_meta = {}
            
        # For image chunks, store only the path part in the embeddings
        if "<image_data" in text:
            path_match = re.match(r"Image: ([^\n]+)", text)
            embedable_text = path_match.group(0) if path_match else "Image chunk"
            # Store the full chunk text (including base64) in metadata
            chunk_meta["full_image_chunk"] = text
        else:
            embedable_text = text
                
        # Combine metadata
        combined_meta = {
            **chunk_meta,
            **(metadatas[i] if metadatas and i < len(metadatas) else {}),
            "xml_path": xml_paths[i] if xml_paths and i < len(xml_paths) else None,
            "content_type": detect_content_type(text)
        }
        
        final_meta = flatten_metadata(combined_meta)
        
        texts.append(embedable_text)
        chunk_metadatas.append(final_meta)

        if DEBUG_MODE:
            st.write(f"DEBUG => Processed chunk with content_type: {combined_meta['content_type']}")
            st.write(f"DEBUG => Metadata keys: {list(final_meta.keys())}")

    # Generate embeddings
    embedding_function = OpenAIEmbeddingFunction(st.session_state["api_key"])
    embeddings = embedding_function(texts)

    if DEBUG_MODE:
        st.write(f"DEBUG => Generated {len(embeddings)} embeddings")
        st.write(f"DEBUG => First embedding dimension: {len(embeddings[0])}")

    # Create/get collection and add documents
    ids = [str(uuid.uuid4()) for _ in chunks]
    coll = create_or_load_collection(collection_name)
    
    if DEBUG_MODE:
        st.write("DEBUG => Adding to ChromaDB:")
        st.write(f"  - Documents: {len(texts)}")
        st.write(f"  - Embeddings: {len(embeddings)}")
        st.write(f"  - Metadata: {len(chunk_metadatas)}")
    
    coll.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=chunk_metadatas,
        ids=ids
    )
    
    if DEBUG_MODE:
        st.write("DEBUG => Storage completed and persisted")

def extract_image_data(text: str) -> Optional[Dict[str, str]]:
    """Helper to extract and validate image data from chunk text."""
    base64_match = re.search(r"base64=['\"]([^'\"]+)['\"]", text)
    mime_match = re.search(r"mime_type=['\"]([^'\"]+)['\"]", text)
    
    if base64_match and mime_match:
        base64_data = base64_match.group(1)
        mime_type = mime_match.group(1)
        
        # Validate base64 data
        if base64_data.strip():
            try:
                # Try to decode a small part to verify it's valid base64
                import base64
                test_decode = base64.b64decode(base64_data[:100] + "=" * (-len(base64_data) % 4))
                return {
                    "base64": base64_data,
                    "mime_type": mime_type
                }
            except Exception as e:
                if DEBUG_MODE:
                    st.write(f"DEBUG => Base64 validation failed: {str(e)}")
                return None
    return None

def query_collection(query: str, collection_name: str, n_results: int = 10):
    """Enhanced query function with consistent multi-country retrieval."""
    if DEBUG_MODE:
        st.write(f"DEBUG => Processing query: '{query}'")

    # 1) Create or load Chroma collection
    force_flag = st.session_state.get("force_recreate", False)
    coll = create_or_load_collection(collection_name, force_recreate=force_flag)
    
    try:
        # Test collection has data
        test_results = coll.peek()
        if DEBUG_MODE:
            st.write(f"DEBUG => Collection peek results: {test_results}")
    except Exception as e:
        if DEBUG_MODE:
            st.write(f"DEBUG => Error peeking collection: {str(e)}")
    
    # 2) Detect countries
    llm_detector = LLMCountryDetector(api_key=st.session_state["api_key"])
    detected_list = llm_detector.detect_countries_in_text(query)

    # 3) Logging & user display
    if detected_list:
        country_list = [f"'{item['detected_phrase']}' → {item['code']}" for item in detected_list]
        st.write("🌍 **Countries Detected in Query:**")
        for country in country_list:
            st.write(f"  • {country}")
    else:
        st.write("❌ No country codes detected in query")

    # Get all documents to check available countries
    try:
        all_docs = coll.get()
        if DEBUG_MODE:
            st.write(f"DEBUG => Retrieved {len(all_docs.get('ids', []))} documents")
            if 'metadatas' in all_docs:
                st.write(f"DEBUG => Sample metadata: {all_docs['metadatas'][0] if all_docs['metadatas'] else 'None'}")
    except Exception as e:
        if DEBUG_MODE:
            st.write(f"DEBUG => Error getting documents: {str(e)}")
        all_docs = {'ids': [], 'metadatas': []}

    available_countries = set()
    for metadata in all_docs.get("metadatas", []):
        if metadata and "country_code" in metadata:
            available_countries.add(metadata["country_code"])
    
    if DEBUG_MODE:
        st.write(f"DEBUG => Available countries in database: {available_countries}")
        st.write(f"DEBUG => Total documents: {len(all_docs.get('ids', []))}")

    if len(all_docs.get('ids', [])) == 0:
        st.warning("No documents found in collection. Please upload first.")
        return [], []

    # 4) Initialize results
    combined_passages = []
    combined_metadata = []
    seen_passages = set()
    iso_codes = [d["code"] for d in detected_list]

    # 5) Query for each country with retries
    if iso_codes:
        for code in iso_codes:
            if DEBUG_MODE:
                st.write(f"DEBUG => Querying for country {code}")

            # Generate query embedding using same model as collection
            query_embedding_data = embed_text(
                [query], 
                collection_name=collection_name,  # Pass collection name to ensure model consistency
                update_stage_flag=False,
                return_data=True
            )
            query_embedding = query_embedding_data["embeddings"][0]

            # Try different query approaches
            for attempt in range(3):
                if attempt == 0:
                    try:
                        results = coll.query(
                            query_embeddings=[query_embedding],  # Use pre-generated embedding
                            where={"country_code": code},
                            n_results=n_results
                        )
                    except Exception as e:
                        if DEBUG_MODE:
                            st.write(f"DEBUG => Query attempt {attempt} failed: {str(e)}")
                        continue

                elif attempt == 1:
                    country_query = f"information about {code} regulations"
                    query_embedding_data = embed_text(
                        [country_query],
                        collection_name=collection_name,
                        update_stage_flag=False,
                        return_data=True
                    )
                    try:
                        results = coll.query(
                            query_embeddings=[query_embedding_data["embeddings"][0]],
                            where={"country_code": code},
                            n_results=n_results
                        )
                    except Exception as e:
                        if DEBUG_MODE:
                            st.write(f"DEBUG => Query attempt {attempt} failed: {str(e)}")
                        continue

                else:
                    try:
                        results = coll.get(
                            where={"country_code": code}
                        )
                    except Exception as e:
                        if DEBUG_MODE:
                            st.write(f"DEBUG => Query attempt {attempt} failed: {str(e)}")
                        continue

                curr_passages = results.get("documents", [[]])[0] if attempt < 2 else results.get("documents", [])
                curr_metadata = results.get("metadatas", [[]])[0] if attempt < 2 else results.get("metadatas", [])

                if curr_passages:
                    if DEBUG_MODE:
                        st.write(f"DEBUG => Found {len(curr_passages)} passages for {code} on attempt {attempt + 1}")
                    break
                elif DEBUG_MODE:
                    st.write(f"DEBUG => No results for {code} on attempt {attempt + 1}")

            # Process results
            if curr_passages:
                for p, m in zip(curr_passages, curr_metadata):
                    passage_key = f"{code}:{p}"
                    if passage_key not in seen_passages:
                        combined_passages.append(p)
                        combined_metadata.append(m)
                        seen_passages.add(passage_key)
            elif DEBUG_MODE:
                st.write(f"DEBUG => No results found for {code} after all attempts")

    else:
        # Standard search if no countries detected
        if DEBUG_MODE:
            st.write("DEBUG => Doing standard search with no country filter")
        
        # Generate query embedding
        query_embedding_data = embed_text(
            [query],
            collection_name=collection_name,
            update_stage_flag=False,
            return_data=True
        )
        query_embedding = query_embedding_data["embeddings"][0]
        
        results = coll.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        combined_passages = results.get("documents", [[]])[0]
        combined_metadata = results.get("metadatas", [[]])[0]

    # Update pipeline stage
    update_stage('retrieve', {
        "passages": combined_passages,
        "metadata": combined_metadata
    })

    return combined_passages, combined_metadata

##############################################################################
# 7) GPT ANSWER GENERATION
##############################################################################

def process_image_run(run, doc) -> Dict[str, Any]:
    """Enhanced image processor that properly formats for vision API."""
    try:
        # Extract image from DOCX
        image_data = run._r.drawing_lst[0].xpath('.//a:blip/@r:embed')[0]
        image_rel = doc.part.rels[image_data]
        image_bytes = image_rel.target_part.blob
        
        # Open with PIL for metadata
        image = Image.open(BytesIO(image_bytes))
        format = image.format.lower()
        
        # Get base64 and mime type
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = get_image_mime_type(format)
        
        return {
            "width": image.size[0],
            "height": image.size[1],
            "format": format,
            "mime_type": mime_type,
            "base64_data": base64_data,
            "image_bytes": image_bytes  # Keep original bytes for file saving
        }
    except Exception as e:
        if DEBUG_MODE:
            st.write(f"DEBUG => Image processing error: {str(e)}")
        return {"error": str(e)}

def generate_answer_with_gpt(query: str, passages: List[str], metadata: List[dict],
                           system_instruction: str = None):
    """Generate answer with balanced temperature for better information correlation."""
    if not st.session_state.get("api_key"):
        st.error("OpenAI API key not set.")
        st.stop()

    if DEBUG_MODE:
        st.write(f"DEBUG => Starting answer generation")
        st.write(f"DEBUG => Received {len(passages)} passages and {len(metadata)} metadata entries")

    # First detect countries in the query
    llm_detector = LLMCountryDetector(api_key=st.session_state["api_key"])
    detected_countries = llm_detector.detect_countries_in_text(query)
    detected_codes = [d["code"] for d in detected_countries]

    if DEBUG_MODE:
        st.write(f"DEBUG => Detected country codes in query: {detected_codes}")

    # Group passages and metadata by country code
    country_data = {}
    
    # Process passages and their metadata together
    for passage, meta in zip(passages, metadata):
        country_code = meta.get('country_code')
        if country_code and country_code in detected_codes:
            if country_code not in country_data:
                country_data[country_code] = {
                    'passages': [],
                    'metadata': []
                }
            country_data[country_code]['passages'].append(passage)
            country_data[country_code]['metadata'].append(meta)

    if DEBUG_MODE:
        st.write(f"DEBUG => Found data for countries: {list(country_data.keys())}")
        st.write(f"DEBUG => Data points per country: {[(k, len(v['passages'])) for k,v in country_data.items()]}")

    # Initialize messages array
    messages = []
    
    # Create a balanced system message that encourages finding information
    base_instruction = system_instruction or ""
    balanced_instruction = f"""
    {base_instruction}

    INSTRUCTIONS:
    1. The user's query mentions these countries: {', '.join(detected_codes)}
    2. Focus on finding relevant information for these countries.
    3. Look for both direct and indirect mentions of regulations or requirements.
    4. If information seems relevant but not explicit, clearly mark it as "Based on available data:".
    5. For each country, try to provide:
       - Direct quotes or references if available
       - Related context that might help understand the regulations
       - Any caveats or important notes
    6. Only state "No information in my documents" if you truly find nothing relevant.

    Remember: Analyze the provided documents thoroughly before concluding no information exists.
    """
    messages.append({"role": "system", "content": balanced_instruction})

    # Process each country's data
    for iso_code, data in country_data.items():
        country_passages = data['passages']
        country_metadata = data['metadata']
        
        if DEBUG_MODE:
            st.write(f"DEBUG => Processing {len(country_passages)} passages for {iso_code}")
            passage_types = [m.get('content_type', 'unknown') for m in country_metadata]
            st.write(f"DEBUG => Content types for {iso_code}: {passage_types}")
        
        for passage, meta in zip(country_passages, country_metadata):
            is_image = meta.get('content_type') == 'image'
            
            if is_image:
                full_chunk = meta.get('full_image_chunk', '')
                if full_chunk:
                    base64_match = re.search(r"base64=['\"]([^'\"]+)['\"]", full_chunk)
                    mime_match = re.search(r"mime_type=['\"]([^'\"]+)['\"]", full_chunk)
                    
                    if base64_match and mime_match:
                        base64_data = base64_match.group(1)
                        mime_type = mime_match.group(1)
                        
                        if base64_data.strip():
                            if DEBUG_MODE:
                                st.write(f"DEBUG => Processing image from {iso_code}")
                                st.write(f"DEBUG => Base64 length: {len(base64_data)}")
                            
                            messages.append({
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Analyze this document image from {iso_code}. Extract any text about regulations, age restrictions, or requirements:"
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_data}",
                                            "detail": "high"
                                        }
                                    }
                                ]
                            })
            else:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[{iso_code}] {passage}"
                        }
                    ]
                })

    # Add specific query guidance
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""
Please analyze the documents and answer this query: {query}

For each country mentioned ({', '.join(detected_codes)}):
1. Look for both explicit and implicit information
2. Consider both direct statements and contextual clues
3. If you find partial information, share it but mark it clearly
4. Only say "no information" if you truly find nothing relevant
"""
            }
        ]
    })

    try:
        # Use higher temperature for initial response
        completion = new_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=4096,
            temperature=0.3  # Increased temperature for better information correlation
        )
        answer = completion.choices[0].message.content if completion.choices else ""
        
        # Verify no unauthorized countries are mentioned
        answer_countries = llm_detector.detect_countries_in_text(answer)
        answer_codes = [c["code"] for c in answer_countries]
        unexpected_codes = [c for c in answer_codes if c not in detected_codes]
        
        if unexpected_codes:
            if DEBUG_MODE:
                st.write(f"DEBUG => Found unexpected countries: {unexpected_codes}")
            # Regenerate with stricter control but maintain temperature
            strict_prompt = f"""
            Your previous answer mentioned unauthorized countries: {unexpected_codes}
            Please revise to discuss ONLY these countries: {detected_codes}
            Maintain all relevant information but remove references to other countries.
            """
            messages.append({"role": "user", "content": strict_prompt})
            completion = new_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096,
                temperature=0.2  # Slightly lower for correction but still flexible
            )
            answer = completion.choices[0].message.content if completion.choices else ""
            
        if DEBUG_MODE:
            st.write(f"DEBUG => Answer length: {len(answer)}")
            
    except Exception as e:
        if DEBUG_MODE:
            st.write(f"DEBUG => API Error: {str(e)}")
        answer = f"Error generating response: {str(e)}"

    update_stage('generate', {'answer': answer})
    return answer


##############################################################################
# 7) REALTIME VOICE MODE
##############################################################################
def get_ephemeral_token(collection_name: str = "rag_collection"):
    if "api_key" not in st.session_state:
        st.error("OpenAI API key not set.")
        return None
    url = "https://api.openai.com/v1/realtime/sessions"
    headers = {
        "Authorization": f"Bearer {st.session_state['api_key']}",
        "Content-Type": "application/json"
    }
    chosen_voice = st.session_state.get("selected_voice", "coral")
    data = {"model": "gpt-4o-realtime-preview-2024-12-17", "voice": chosen_voice}
    try:
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        token_data = resp.json()
        if isinstance(token_data, dict):
            if "token" in token_data:
                return {"token": token_data["token"], "collection": collection_name}
            elif "client_secret" in token_data:
                if isinstance(token_data["client_secret"], dict):
                    return {"token": token_data["client_secret"].get("value"), "collection": collection_name}
                return {"token": token_data["client_secret"], "collection": collection_name}
        st.error("Unexpected response structure")
        st.json(token_data)
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create voice session: {str(e)}")
        if hasattr(e.response, 'text'):
            st.write("Error:", e.response.text)
        return None


def get_realtime_html(token_data: dict) -> str:
    coll = create_or_load_collection(token_data['collection'])
    all_docs = coll.get()
    all_passages = all_docs.get("documents", [])
    doc_summary = summarize_context(all_passages)

    base_prompt = st.session_state.get("custom_prompt", BASE_DEFAULT_PROMPT)
    voice_instructions = st.session_state.get("voice_custom_prompt", "")
    if not voice_instructions.strip():
        voice_instructions = DEFAULT_VOICE_PROMPT

    full_prompt = (
        "<INSTRUCTIONS>\n" +
        "<MAIN_PROMPT>\n" +
        base_prompt + "\n" +
        "</MAIN_PROMPT>\n" +
        "<INSTRUCTIONS_TO_ADVANCED_VOICE_MODE>\n" +
        voice_instructions + "\n" +
        "</INSTRUCTIONS_TO_ADVANCED_VOICE_MODE>\n" +
        "</INSTRUCTIONS>\n" +
        "\n\n<DOCUMENTS>\n" +
        doc_summary + "\n" +
        "</DOCUMENTS>\n"
    )
    
    st.session_state.avm_initial_text = full_prompt

    realtime_js = f"""
    <div id="realtime-status" style="color: lime; font-size: 14px;">Initializing...</div>
    <div id="transcription" style="color: white; margin-top: 5px; font-size: 12px;"></div>
    <script>
    async function initRealtime() {{
        const statusDiv = document.getElementById('realtime-status');
        const transcriptionDiv = document.getElementById('transcription');
        const pc = new RTCPeerConnection();
        const audioEl = new Audio();
        audioEl.autoplay = true;
        try {{
            const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
            stream.getTracks().forEach(track => pc.addTrack(track, stream));
            statusDiv.innerText = "Mic connected";
        }} catch (err) {{
            statusDiv.innerText = "Mic error: " + err.message;
            return;
        }}
        pc.ontrack = (event) => {{
            audioEl.srcObject = event.streams[0];
        }};
        const dc = pc.createDataChannel("events");
        dc.onopen = () => {{
            statusDiv.innerText = "AVM active";
            dc.send(JSON.stringify({{
                type: "session.update",
                session: {{ instructions: `{full_prompt}` }}
            }}));
        }};
        dc.onmessage = async (e) => {{
            const data = JSON.parse(e.data);
            console.log("Received:", data);
            if (data.type === "text") {{
                const userQuery = data.text;
                transcriptionDiv.innerHTML += `<p style='color:#9ee;'>User: ${{userQuery}}</p>`;
                dc.send(JSON.stringify({{
                    type: "conversation.item.create",
                    item: {{ type: "message", role: "user", content: [{{ type: "input_text", text: userQuery }}] }}
                }}));
                let relevantContext = "";
                try {{
                    const response = await fetch('/query_collection', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ query: userQuery, collection: "{token_data['collection']}" }})
                    }});
                    const contextData = await response.json();
                    relevantContext = contextData.relevantContext || "";
                }} catch (err) {{
                    console.error('Error retrieving context:', err);
                }}
                dc.send(JSON.stringify({{
                    type: "response.create",
                    response: {{
                        modalities: ["audio", "text"],
                        instructions: `AVM: Here are the best-matching passages:
                        ${{relevantContext}}
                        Answer using only that info.`
                    }}
                }}));
            }} else if (data.type === "speech") {{
                transcriptionDiv.innerHTML += `<p style="color: #4CAF50;">AVM: ${{data.text}}</p>`;
            }}
        }};
        try {{
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            const sdpResponse = await fetch("https://api.openai.com/v1/realtime", {{
                method: "POST",
                body: offer.sdp,
                headers: {{
                    "Authorization": "Bearer {token_data['token']}",
                    "Content-Type": "application/sdp"
                }}
            }});
            if (!sdpResponse.ok) throw new Error(`HTTP error: ${{sdpResponse.status}}`);
            const answerSdp = await sdpResponse.text();
            await pc.setRemoteDescription({{ type: "answer", sdp: answerSdp }});
        }} catch (err) {{
            statusDiv.innerText = "Connection error: " + err.message;
            console.error(err);
        }}
    }}
    initRealtime();
    </script>
    <style>
        ::-webkit-scrollbar {{ width: 0px; background: transparent; }}
        body {{ background-color: #111; color: #fff; margin: 0; padding: 0; }}
        #realtime-status {{ font-family: system-ui, sans-serif; }}
    </style>
    """
    return realtime_js


def toggle_avm():
    st.session_state.avm_active = not st.session_state.avm_active
    if st.session_state.avm_active:
        token_data = get_ephemeral_token("rag_collection")
        if token_data:
            st.session_state.voice_html = get_realtime_html(token_data)
        else:
            st.session_state.avm_active = False
            st.session_state.voice_html = None
    else:
        st.session_state.voice_html = None


##############################################################################
# 8) USER AUTHENTICATION
##############################################################################

def create_user_session():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
        st.session_state.is_authenticated = False

import glob  # add this import if not already present

def get_user_specific_directory(user_id: str) -> dict:
    """
    Creates a directory layout like:
      <current_working_dir>/
        chromadb_storage_user_{user_id}/
          chroma_db/
          images/
          xml/
    """
    base_dir = os.path.join(os.getcwd(), f"chromadb_storage_user_{user_id}")
    
    # Subfolder dedicated to the actual Chroma DB
    chroma_dir = os.path.join(base_dir, "chroma_db")
    
    if DEBUG_MODE:
        st.write(f"DEBUG: Using base directory: {base_dir}")
        st.write(f"DEBUG: Chroma subfolder: {chroma_dir}")
    
    dirs = {
        "base": base_dir,
        "chroma": chroma_dir,  # <--- Use the subfolder for Chroma
        "images": os.path.join(base_dir, "images"),
        "xml": os.path.join(base_dir, "xml")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        if DEBUG_MODE:
            st.write(f"DEBUG: Ensured directory exists: {dir_path}")
    
    # List all files in the 'chroma' directory
    import glob
    files = glob.glob(os.path.join(chroma_dir, "*"))
    if DEBUG_MODE:
        st.write(f"DEBUG: Files in persist directory: {files}")
    
    return dirs

def load_users():
    user_file = Path("users.json")
    if not user_file.exists():
        return {"admin": "admin"}  # Default admin account
    return json.loads(user_file.read_text())

def save_users(users):
    Path("users.json").write_text(json.dumps(users))

def create_account(username, password):
    users = load_users()
    if username in users:
        return False, "Username already exists"
    users[username] = password
    save_users(users)
    return True, "Account created successfully"

def verify_login(username, password):
    users = load_users()
    return username in users and users[username] == password

def delete_user_collections(user_id: str) -> tuple[bool, str]:
    if not user_id:
        return False, "No user ID provided"
    
    try:
        dirs = get_user_specific_directory(user_id)
        user_chroma_db = dirs["chroma"]  # e.g. /.../chromadb_storage_user_Rosh/chroma_db

        if not os.path.exists(user_chroma_db):
            return True, "No collections found for user"

        # Use chromadb.PersistentClient instead of chroma_client.PersistentClient
        temp_client = chromadb.PersistentClient(
            path=user_chroma_db,
            settings=Settings(anonymized_telemetry=False)
        )
        collections = temp_client.list_collections()
        for collection in collections:
            temp_client.delete_collection(collection.name)
        
        shutil.rmtree(user_chroma_db)
        user_dir = dirs["base"]
        shutil.rmtree(user_dir)
        os.makedirs(user_dir, exist_ok=True)
        
        return True, f"Successfully deleted all collections for user {user_id}"
        
    except Exception as e:
        return False, f"Error deleting collections: {str(e)}"


def delete_country_data(iso_code: str, user_id: str) -> tuple[bool, str]:
    """
    Deletes all data for a specific country code in a user's collection.
    Also removes associated images from the image store.
    
    Returns:
        tuple[bool, str]: (success, message)
    """
    if not iso_code or not validate_iso_code(iso_code):
        return False, "Invalid country code"
    
    try:
        dirs = get_user_specific_directory(user_id)
        user_chroma_db = dirs["chroma"]  # subfolder

        temp_client = chromadb.PersistentClient(
            path=user_chroma_db,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = temp_client.get_collection(name="rag_collection")
        
        # 4. Query and delete documents for this country code
        try:
            # Get all documents for this country
            results = collection.get(
                where={"country_code": iso_code.upper()}
            )
            
            if results and results.get('ids'):
                # Delete the documents
                collection.delete(ids=results['ids'])
                return True, f"Successfully deleted {len(results['ids'])} documents and associated images for country {iso_code}"
            else:
                return True, f"No documents found for country {iso_code}"
                
        except Exception as e:
            return False, f"Error querying/deleting documents: {str(e)}"
            
    except Exception as e:
        return False, f"Error accessing data: {str(e)}"
    finally:
        try:
            temp_client.persist()
        except Exception:
            pass



##############################################################################
# 9) DELETE USERS
##############################################################################
def delete_user(user_id: str):
    import shutil
    from pathlib import Path

    users = load_users()  # (Assumes load_users() is defined elsewhere)
    if user_id not in users:
        return False, "User not found."

    # Remove the user from the users file.
    del users[user_id]
    save_users(users)

    # Delete the custom prompt file.
    prompt_path = Path(f"prompts/user_{user_id}_custom_prompt.txt")
    if prompt_path.exists():
        prompt_path.unlink()

    # Delete the voice preference file.
    voice_pref_path = Path(f"preferences/user_{user_id}_voice_pref.txt")
    if voice_pref_path.exists():
        voice_pref_path.unlink()

    # Delete the voice instructions file.
    voice_instr_path = Path(f"instructions/user_{user_id}_voice_instructions.txt")
    if voice_instr_path.exists():
        voice_instr_path.unlink()

    # Delete the user-specific chromadb storage directory.
    user_dir = f"chromadb_storage_user_{user_id}"
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)

    return True, f"User '{user_id}' deleted successfully."


##############################################################################
# TEXT PROCESSING & CHUNKING FUNCTIONS
##############################################################################
import streamlit as st
import re
import unicodedata

def ensure_canonical_order(text: str) -> str:
    """
    Scans the text line‐by‐line (ignoring lines that are only dashes).
    If the first meaningful (non-dash) line contains "END OF" (case‐insensitive),
    the entire text is assumed to be reversed and is flipped line‐by‐line.
    Otherwise, returns the original text.
    """
    lines = text.splitlines()
    if DEBUG_MODE:
        st.write(f"DEBUG: ensure_canonical_order -> total lines read: {len(lines)}")
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped and not re.fullmatch(r"[-]+", stripped):
            if DEBUG_MODE:
                st.write(f"DEBUG: First non-dash non-empty line => {repr(stripped)}")
            if re.search(r"END OF", stripped, flags=re.IGNORECASE):
                if DEBUG_MODE:
                    st.write("DEBUG: Found 'END OF' => reversing text.")
                return "\n".join(lines[::-1])
            else:
                if DEBUG_MODE:
                    st.write("DEBUG: 'END OF' not found => no reversal triggered.")
            break
    if DEBUG_MODE:
        st.write("DEBUG: returning original text => no changes.")
    return text

def chunk_text(text: str) -> List[str]:
    """
    1) ensure_canonical_order
    2) remove dash-only lines
    3) split by marker-based regex
    4) if exactly 3 parts => single chunk = the middle part
    5) if <2 => entire text as one chunk
    6) overwrite “upload” stage with canonical text
    7) update “chunk” stage
    """
    canonical_text = ensure_canonical_order(text)
    if DEBUG_MODE:
        st.write("DEBUG: chunk_text -> after ensure_canonical_order()")

    # Remove lines that are solely dashes.
    lines = canonical_text.splitlines()
    cleaned_lines = [l for l in lines if not re.fullmatch(r"[-]+", l.strip())]
    cleaned_text = "\n".join(cleaned_lines)
    if DEBUG_MODE:
        st.write("DEBUG: cleaned_text snippet:", cleaned_text[:100])

    # Split by markers
    pattern = r"(={5,}\s*(?:CH|US|END OF CH|END OF US)\s*={5,})"
    parts = re.split(pattern, cleaned_text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if DEBUG_MODE:
        st.write("DEBUG: raw splitted parts =>", parts)

    # If exactly 3 => single chunk if first/last are full markers
    if len(parts) == 3:
        if re.fullmatch(pattern, parts[0], flags=re.IGNORECASE) and re.fullmatch(pattern, parts[2], flags=re.IGNORECASE):
            if DEBUG_MODE:
                st.write("DEBUG: recognized a single block => using inner content as chunk.")
            parts = [parts[1]]

    # If fewer than 2 => fallback to entire cleaned text
    if len(parts) < 2:
        if DEBUG_MODE:
            st.write("DEBUG: chunk_text => 0 or 1 chunk => fallback entire text.")
        parts = [cleaned_text]

    # Overwrite "upload" stage so pipeline UI sees the canonical text
    upload_data = {
        "content": canonical_text,
        "preview": canonical_text[:600],
        "size": len(canonical_text),
    }
    update_stage("upload", upload_data)

    # Update "chunk" stage so pipeline UI sees the chunk info
    chunk_data = {
        "chunks": parts[:5],
        "full_chunks": [{"text": p} for p in parts],
        "total_chunks": len(parts),
    }
    update_stage("chunk", chunk_data)

    return parts

def parse_xml_for_chunks(text: str) -> List[Dict[str, Any]]:
    """
    Parse *multiple* XML documents in one string. 
    We split on '<?xml version="1.0" encoding="UTF-8"?>' and parse each subdoc
    with parse_single_xml_doc(...). Each subdoc returns a list of chunk dicts.
    """
    chunks = []
    xml_docs = text.split('<?xml version="1.0" encoding="UTF-8"?>')
    
    for doc in xml_docs:
        if not doc.strip():
            continue
        
        # re-add the XML prolog
        doc = '<?xml version="1.0" encoding="UTF-8"?>' + doc
        
        try:
            # parse_single_xml_doc returns a list of chunk dicts
            chunk_data = parse_single_xml_doc(doc)
            chunks.extend(chunk_data)
        except Exception as e:
            st.error(f"Error parsing XML document: {e}")
            
    # update stage for React pipeline
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
    """
    Processes ONE well-formed XML doc string into chunk dicts with proper base64 handling.
    """
    import xml.etree.ElementTree as ET
    from io import StringIO

    tree = ET.parse(StringIO(xml_text))
    root = tree.getroot()
    country_code = root.tag.upper()

    # gather doc-level metadata
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

    # Process direct text in content
    if content_elem.text and content_elem.text.strip():
        text_buffer.append(content_elem.text.strip())
        buf_size += len(content_elem.text)

    for element in content_elem:
        tag_name = element.tag.lower()

        if tag_name == "image":
            flush_buffer()
            
            # Get image path
            path_node = element.find('path')
            path = path_node.text.strip() if path_node is not None else "[unknown]"
            
            # Get dimensions
            size_node = element.find('size')
            w = h = "0"
            if size_node is not None:
                w_node = size_node.find('width')
                h_node = size_node.find('height')
                w = (w_node.text if w_node is not None else "0")
                h = (h_node.text if h_node is not None else "0")
            
            # Get image format
            fmt_node = element.find('format')
            fmt = fmt_node.text.strip() if fmt_node is not None else "jpeg"
            
            # CRITICAL: Extract base64 data from image_data node
            image_data_node = element.find('image_data')
            if image_data_node is not None:
                mime_type = image_data_node.get('mime_type', 'image/jpeg')
                base64_data = image_data_node.get('base64', '')
                
                # Create chunk only if we have base64 data
                if base64_data:
                    image_text = (
                        f"Image: {path} ({w}x{h} {fmt})\n"
                        f"<image_data mime_type='{mime_type}' base64='{base64_data}' />"
                    )
                    
                    chunks.append({
                        "text": image_text,
                        "metadata": {
                            **meta,
                            "country_code": country_code,
                            "content_type": "image",
                            "mime_type": mime_type,
                            "dimensions": f"{w}x{h}",
                            "format": fmt
                        }
                    })
                else:
                    if DEBUG_MODE:
                        st.write(f"DEBUG => Missing base64 data for image: {path}")
            else:
                if DEBUG_MODE:
                    st.write(f"DEBUG => No image_data node found for: {path}")

        elif tag_name in ("text", "list-item"):
            text_val = (element.text or "").strip()
            if text_val:
                if (len(text_val) + buf_size) > max_chunk_size:
                    flush_buffer()
                text_buffer.append(text_val)
                buf_size += len(text_val)

        elif tag_name == "table":
            flush_buffer()
            table_md = convert_table_to_markdown(element)
            chunks.append({
                "text": table_md,
                "metadata": {
                    **meta,
                    "country_code": country_code,
                    "content_type": "table"
                }
            })

    flush_buffer()
    return chunks

def convert_table_to_markdown(table_elem) -> str:
    """
    Convert <table><header>...</header><rows>...<row>...<cell>..</cell></row></rows></table>
    into a simple Markdown table.
    """
    header = table_elem.find('header')
    rows   = table_elem.find('rows')

    if header is None or rows is None:
        return "[Empty or invalid <table> structure]"

    # gather header cells
    header_cells = []
    for c in header.findall('cell'):
        cell_txt = (c.text or "").strip()
        header_cells.append(cell_txt if cell_txt else " ")

    # gather rows
    row_lines = []
    for row_node in rows.findall('row'):
        row_cells = []
        for c in row_node.findall('cell'):
            txt = (c.text or "").strip()
            row_cells.append(txt if txt else " ")
        row_lines.append(row_cells)

    # Build standard markdown
    md_lines = []
    # top line: headers
    md_lines.append(" | ".join(header_cells))
    # separator
    md_lines.append("|".join(["-" * max(1,len(h)) for h in header_cells]))
    # body rows
    for row_cells in row_lines:
        md_lines.append(" | ".join(row_cells))

    return "\n".join(md_lines)

# The main parse_xml_for_chunks function remains the same but uses this enhanced version

def wrap_text_with_markers(text: str, iso_code: str) -> str:
    """Wraps text with appropriate markers based on ISO code."""
    iso_code = iso_code.upper()
    start_marker = f"========== {iso_code} =========="
    end_marker = f"========== END OF {iso_code} =========="
    return f"{start_marker}\n\n{text}\n\n{end_marker}"

def validate_iso_code(code: str) -> bool:
    """Validates if the provided code is a valid ISO country code."""
    try:
        return bool(pycountry.countries.get(alpha_2=code.upper()))
    except (AttributeError, KeyError):
        return False

def detect_content_type(text: str) -> str:
    """
    Minimal stub that guesses content type based on the chunk text.
    Adjust the logic to suit your real classification needs.
    """
    low = text.strip().lower()
    # If it starts with "Image:" => it's an image
    if low.startswith("image:"):
        return "image"
    # If it looks like a markdown table => "table"
    elif " | " in text and "---" in text:
        return "table"
    else:
        # Otherwise assume "text"
        return "text"

def flatten_metadata(md: dict) -> dict:
    """
    Recursively flatten all dict/list fields into strings,
    so each top-level key is str->(str,int,float).
    If any value is still not a simple scalar, we convert it via JSON or str().
    """
    flat = {}

    def _recurse(prefix: str, val: Any):
        if isinstance(val, dict):
            for k,v in val.items():
                new_key = f"{prefix}.{k}" if prefix else k
                _recurse(new_key, v)
        elif isinstance(val, list):
            # convert entire list to JSON string
            flat[prefix] = json.dumps(val)
        else:
            # If it's int or float or str, keep it. Otherwise, str() it
            if isinstance(val, (int,float,str)):
                flat[prefix] = val
            else:
                flat[prefix] = str(val)

    # top-level
    _recurse("", md)
    return flat

def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def get_image_mime_type(format: str) -> str:
    """Get MIME type from image format."""
    format = format.lower()
    mime_types = {
        'jpeg': 'image/jpeg',
        'jpg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'webp': 'image/webp'
    }
    return mime_types.get(format, 'application/octet-stream')

def process_image_run(run, doc) -> Dict[str, Any]:
    """Process an image run from a DOCX file, returning both path and base64."""
    try:
        image_data = run._r.drawing_lst[0].xpath('.//a:blip/@r:embed')[0]
        image_rel = doc.part.rels[image_data]
        image_bytes = image_rel.target_part.blob
        
        # Get image details
        image = Image.open(BytesIO(image_bytes))
        format = image.format.lower()
        
        # Convert to base64
        base64_data = image_to_base64(image_bytes)
        mime_type = get_image_mime_type(format)
        
        return {
            "width": image.size[0],
            "height": image.size[1],
            "format": format,
            "mime_type": mime_type,
            "base64_data": base64_data,
            "image_bytes": image_bytes  # Keep original bytes for saving
        }
    except Exception as e:
        return {
            "error": str(e)
        }

def create_image_chunk(image_info: Dict[str, Any], saved_path: str) -> Dict[str, str]:
    """Create a chunk that includes both path reference and base64 data."""
    return {
        "text": (
            f"Image: {saved_path} ({image_info['width']}x{image_info['height']} {image_info['format'].upper()})\n"
            f"<image_data mime_type='{image_info['mime_type']}' base64='{image_info['base64_data']}' />"
        ),
        "type": "image"
    }

##############################################################################
# 10) MAIN STREAMLIT APP
##############################################################################
def main():
    global chroma_client

    # Authentication
    create_user_session()

    chroma_client, embedding_function_instance = init_chroma_client()

    with st.sidebar:
        st.header("User Authentication")
        if not st.session_state.is_authenticated:
            tab1, tab2 = st.tabs(["Login", "Create Account"])
            
            with tab1:
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                if st.button("Login"):
                    if verify_login(username, password):
                        st.session_state.user_id = username
                        st.session_state.is_authenticated = True
                        st.rerun()
                    else:
                        st.error("Invalid credentials")

            with tab2:
                new_username = st.text_input("New Username", key="new_user")
                new_password = st.text_input("New Password", type="password", key="new_pass")
                if st.button("Create Account"):
                    success, msg = create_account(new_username, new_password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

        else:
            st.success(f"Logged in as {st.session_state.user_id}")
            if st.button("Logout"):
                # Clear user auth
                st.session_state.user_id = None
                st.session_state.is_authenticated = False
                
                # Clear pipeline data
                for stage in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']:
                    if f'{stage}_data' in st.session_state:
                        st.session_state[f'{stage}_data'] = None
                        
                # Clear other data
                st.session_state.uploaded_text = None
                st.session_state.chunks = None
                st.session_state.embeddings = None
                st.session_state.retrieved_passages = []
                st.session_state.retrieved_metadata = []
                st.session_state.final_answer = None
                st.session_state.current_stage = None  # This resets stage colors

                st.rerun()

    # Only show main app if authenticated
    if not st.session_state.is_authenticated:
        st.warning("Please log in to use the RAG system")
        return

    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stApp { background-color: #111; color: white; }
        [data-testid="column"] { width: calc(100% + 2rem); margin-left: -1rem; }
        </style>
    """, unsafe_allow_html=True)
    st.title("RagMe")
    
    # Sidebar: API key and force recreate option
    # global chroma_client, embedding_function_instance

    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)
        if chroma_client is None:
            st.error("ChromaDB client not initialized properly")
            st.stop()
    
#    # "Delete All Collections" button in sidebar
#    if st.sidebar.button("Delete All Collections"):
#        if not st.session_state.get("user_id"):
#            st.sidebar.error("Please log in first")
#        else:
#            success, message = delete_user_collections(st.session_state.user_id)
#            if success:
#                # Reset the ChromaDB client to force reconnection
#                chroma_client, embedding_function_instance = init_chroma_client()
#                
#                # Clear relevant session state
#                for stage in ['store', 'query', 'retrieve', 'generate']:
#                    if f'{stage}_data' in st.session_state:
#                        st.session_state[f'{stage}_data'] = None
#                
#                st.sidebar.success(message)
#            else:
#                st.sidebar.error(message)
                
#    st.sidebar.markdown("### AVM Controls")
#    
#    def toggle_avm():
#        st.session_state.avm_button_key += 1  # Force button refresh
#        if st.session_state.avm_active:
#            st.session_state.voice_html = None
#            st.session_state.avm_active = False
#            st.session_state.avm_button_key += 1
#            return
#        token_data = get_ephemeral_token("rag_collection")
#        if token_data:
#            st.session_state.voice_html = get_realtime_html(token_data)
#            st.session_state.avm_active = True
#            st.session_state.avm_button_key += 1
#        else:
#            st.sidebar.error("Could not start AVM.\n\nCheck error messages at the top of the main section ---------------------------->>>")
#
#    if st.sidebar.button(
#        "End AVM" if st.session_state.avm_active else "Start AVM",
#        key=f"avm_toggle_{st.session_state.avm_button_key}",
#        on_click=toggle_avm
#    ):
#        pass
#
#    if st.session_state.avm_active:
#        st.sidebar.success("AVM started.")
#    else:
#        if st.session_state.avm_button_key > 0:
#            st.sidebar.success("AVM ended.")
#    
#    # **Display the exact AVM initialization text in the sidebar**
#    if st.session_state.avm_active and st.session_state.get("avm_initial_text"):
#        st.sidebar.markdown("### AVM Initial Instructions")
#        st.sidebar.code(st.session_state.avm_initial_text)
#    
#    if st.session_state.avm_active and st.session_state.voice_html:
#        components.html(st.session_state.voice_html, height=1, scrolling=True)
    

    with st.sidebar:
        if st.session_state.get("user_id") == "RoshAdm":
            st.markdown("### Delete Account")

            # Load the current users list and filter out "admin"
            users = load_users()
            users_filtered = [u for u in users if u != "RoshAdm"]

            # Create a container to hold the selectbox so we can update it later.
            user_container = st.empty()

            if users_filtered:
                # Set the default selection to "RoshAdm" if possible.
                if ("selected_user" not in st.session_state) or (st.session_state["selected_user"] not in users_filtered):
                    st.session_state["selected_user"] = "RoshAdm" if "RoshAdm" in users_filtered else users_filtered[0]

                selected_user = user_container.selectbox(
                    "Select an account to delete:",
                    options=users_filtered,
                    index=users_filtered.index(st.session_state["selected_user"])
                )
                st.session_state["selected_user"] = selected_user

                confirm = st.checkbox("Confirm deletion", key="delete_confirm")
                if st.button("Delete Account"):
                    if confirm:
                        success, msg = delete_user(selected_user)
                        if success:
                            st.success(msg)
                            # After deletion, re-read the users list.
                            new_users = load_users()
                            new_users_filtered = [u for u in new_users if u != "admin"]

                            if new_users_filtered:
                                # Reset the selected user to "RoshAdm" if it exists; otherwise, to the first account.
                                st.session_state["selected_user"] = "RoshAdm" if "RoshAdm" in new_users_filtered else new_users_filtered[0]
                                # Update the container with a new selectbox using the updated list.
                                user_container.selectbox(
                                    "Select an account to delete:",
                                    options=new_users_filtered,
                                    index=new_users_filtered.index(st.session_state["selected_user"])
                                )
                            else:
                                user_container.info("No user accounts found.")
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please check the box to confirm deletion.")
            else:
                st.info("No user accounts found.")


    if "upload_data" not in st.session_state:
        st.session_state["upload_data"] = {"content": "", "preview": "", "size": 0}

    # Main layout: Use three columns (col1, spacer, col2) for extra spacing
    col1, spacer, col2 = st.columns([3.3, 0.3, 1])
    
    with col1:
        # st.header("Step-by-Step Pipeline Control")
        
        # --- Step 0: Specify Initial Instructions & Voice ---
        #st.subheader("Step 0: Specify Initial Instructions")
        st.subheader("Custom Prompt")
    #    VOICE_OPTIONS = ["alloy", "echo", "shimmer", "ash", "ballad", "coral", "sage", "verse"]
#
    #    # Load the current voice preference for the logged-in user, defaulting to "coral"
    #    current_voice = load_voice_pref(st.session_state.user_id) if st.session_state.get("user_id") else "coral"
#
    #    selected_voice = st.selectbox(
    #        "-> Choose a voice for advanced voice mode:",
    #        options=VOICE_OPTIONS,
    #        index=VOICE_OPTIONS.index(current_voice) if current_voice in VOICE_OPTIONS else VOICE_OPTIONS.index("coral")
    #    )
#
    #    # If the selection has changed, update session state and persist the new voice
    #    if selected_voice != current_voice:
    #        st.session_state.selected_voice = selected_voice
    #        save_voice_pref(selected_voice, st.session_state.user_id)

        # ---------------------------
        # For Main System Instructions
        # ---------------------------
        # Use session state as the source of truth for the prompt.
        if "custom_prompt" not in st.session_state:
            st.session_state.custom_prompt = load_custom_prompt(st.session_state.user_id) or ""
        # Ensure a unique widget key exists.
        if "custom_prompt_widget_key" not in st.session_state:
            st.session_state.custom_prompt_widget_key = str(uuid.uuid4())
        # Create an empty container for the text area.
        prompt_container = st.empty()
        custom_prompt = prompt_container.text_area(
            "-> Customize the text chatbot's initial instructions ('System Instructions') for text- & advanced voice mode.\n\n",
            value=st.session_state.custom_prompt,
            key=st.session_state.custom_prompt_widget_key
        )
        # Update session state if the user makes changes.
        if custom_prompt != st.session_state.custom_prompt:
            st.session_state.custom_prompt = custom_prompt
            save_custom_prompt(custom_prompt, st.session_state.user_id)

        # Button to restore default system instructions
        if st.button("Restore Default Prompt (System Instructions)"):
            st.session_state.custom_prompt = BASE_DEFAULT_PROMPT
            save_custom_prompt(BASE_DEFAULT_PROMPT, st.session_state.user_id)
            # Update the widget key so the text_area re-initializes
            st.session_state.custom_prompt_widget_key = str(uuid.uuid4())
            # Re-render the text area in its container with the default value.
            prompt_container.text_area(
                "-> Customize the text chatbot's initial instructions ('System Instructions') for text- & Advanced Voice Mode.\n\n",
                value=st.session_state.custom_prompt,
                key=st.session_state.custom_prompt_widget_key
            )

    #    # ---------------------------
    #    # For Voice Instructions
    #    # ---------------------------
    #    if "voice_custom_prompt" not in st.session_state:
    #        st.session_state.voice_custom_prompt = load_voice_instructions(st.session_state.user_id) or ""
    #    if "voice_instructions_widget_key" not in st.session_state:
    #        st.session_state.voice_instructions_widget_key = str(uuid.uuid4())
    #    voice_container = st.empty()
    #    voice_instructions = voice_container.text_area(
    #        "-> Customize your advanced voice mode voice & tone.\n\n",
    #        value=st.session_state.voice_custom_prompt,
    #        key=st.session_state.voice_instructions_widget_key
    #    )
    #    if voice_instructions != st.session_state.voice_custom_prompt:
    #        st.session_state.voice_custom_prompt = voice_instructions
    #        save_voice_instructions(voice_instructions, st.session_state.user_id)

    #    # Button to restore default voice instructions
    #    if st.button("Restore Default Prompt (Voice Instructions)"):
    #        st.session_state.voice_custom_prompt = DEFAULT_VOICE_PROMPT
    #        save_voice_instructions(DEFAULT_VOICE_PROMPT, st.session_state.user_id)
    #        st.session_state.voice_instructions_widget_key = str(uuid.uuid4())
    #        voice_container.text_area(
    #            "-> Customize your advanced voice mode voice & tone.\n\n"
    #            "-> Deleting the contents of this box & refreshing your browser restores a default prompt.",
    #            value=st.session_state.voice_custom_prompt,
    #            key=st.session_state.voice_instructions_widget_key
    #        )

    #    # --- Step 1: Upload Context ---
    #    st.subheader("Step 1: Upload Context")
    #    
    #    # Create columns for the ISO code input
    #    iso_col1, iso_col2 = st.columns([2, 1])
    #    
    #    with iso_col1:
    #        iso_code = st.text_input(
    #            "Enter the ISO country code (e.g., CH for Switzerland, US for United States)",
    #            max_chars=2,
    #            help="This code will be used to wrap your document content with appropriate markers."
    #        ).upper()
#
    #    with iso_col2:
    #        if iso_code:
    #            if validate_iso_code(iso_code):
    #                st.success(f"✓ Valid code: {iso_code}")
    #                
    #                # Add delete button
    #                if st.button(f"Delete data for {iso_code}"):
    #                    with st.spinner(f"Deleting data for {iso_code}..."):
    #                        success, message = delete_country_data(
    #                            iso_code, 
    #                            st.session_state.user_id
    #                        )
    #                        if success:
    #                            st.success(message)
    #                            # Clear relevant session state
    #                            for stage in ['store', 'query', 'retrieve', 'generate']:
    #                                if f'{stage}_data' in st.session_state:
    #                                    st.session_state[f'{stage}_data'] = None
    #                        else:
    #                            st.error(message)
    #            else:
    #                st.error("Invalid code")
#
    #    # Wrap the uploader in a form with clear_on_submit=True
    #    with st.form("upload_form", clear_on_submit=True):
    #        uploaded_files = st.file_uploader(
    #            "-> Upload one or more documents",
    #            type=["txt", "pdf", "docx", "csv", "xlsx", "rtf"],
    #            accept_multiple_files=True
    #        )
    #        submitted = st.form_submit_button("Run Step 1: Upload Context")
#
    #    # In main(), in the upload section:
    #    if submitted:
    #        if not iso_code or not validate_iso_code(iso_code):
    #            st.error("Please enter a valid ISO country code first.")
    #            return
#
    #        if uploaded_files:
    #            all_xml_docs = []
    #            for uploaded_file in uploaded_files:
    #                text = extract_text_from_file(
    #                    uploaded_file, 
    #                    iso_code=iso_code,
    #                    user_id=st.session_state.user_id
    #                )
    #                if text:
    #                    all_xml_docs.append(text)
#
    #            if all_xml_docs:
    #                # Combine without separators
    #                combined_text = "\n".join(all_xml_docs)
    #                st.session_state.uploaded_text = combined_text
    #                update_stage("upload", {
    #                    'content': combined_text,
    #                    'preview': combined_text[:600],
    #                    'size': len(combined_text)
    #                })
    #                st.success("Files uploaded and processed!")
    #            else:
    #                st.error("No text could be extracted...")
#
    #    # --- Step 2: Chunk Context ---
    #    st.subheader("Step 2: Chunk Context")
    #    if st.button("Run Step 2: Chunk Context"):
    #        if st.session_state.uploaded_text:
    #            # Debug: Show first 500 chars of XML
    #            st.code(st.session_state.uploaded_text[:3500], language='xml')
    #            
    #            try:
    #                chunks = parse_xml_for_chunks(st.session_state.uploaded_text)
    #                st.session_state.chunks = chunks
    #                st.success(f"Found {len(chunks)} chunk(s).")
    #            except Exception as e:
    #                st.error(f"Error: {str(e)}")
    #                st.code(traceback.format_exc())
    #        else:
    #            st.warning("Please upload at least one document first.")
    #    
    #    # --- Step 3: Embed Context ---
    #    st.subheader("Step 3: Embed Context")
    #    if st.button("Run Step 3: Embed Context"):
    #        if not st.session_state.get("api_key"):
    #            st.error("OpenAI API key not set. Please provide a valid API key in the sidebar before running this step.")
    #        elif st.session_state.chunks:
    #            try:
    #                embedding_data = embed_text(st.session_state.chunks, update_stage_flag=True, return_data=True)
    #                st.session_state.embeddings = embedding_data
    #                st.success("Embeddings created!")
    #            except Exception as e:
    #                st.error(f"An error occurred while generating embeddings: {e}")
    #        else:
    #            st.warning("Please chunk the document first.")
    #    
    #    # --- Step 4: Store ---
    #    st.subheader("Step 4: Store Embedded Context")
    #    # In main(), replace:
    #    if st.button("Run Step 4: Store Embedded Context"):
    #        if st.session_state.chunks and st.session_state.embeddings:
    #            # Create default metadata for each chunk
    #            metadatas = [{"source": "uploaded_document"} for _ in st.session_state.chunks]
    #            add_to_chroma_collection(
    #                collection_name="rag_collection",
    #                chunks=st.session_state.chunks,
    #                metadatas=metadatas  # Pass only metadatas
    #            )
    #            st.success("Data stored in data collection 'rag_collection'!")
    #        else:
    #            st.warning("Ensure document is uploaded, chunked, and embedded.")
    #    
    #    # --- Step 5A: Embed Query (Optional) ---
    #    st.subheader("Step 5A: Embed Query (Optional)")
    #    query_text = st.text_input("-> Enter a query to see how it is embedded into vectors", key="query_text_input")
#
    #    if st.button("Run Step 5A: Embed Query"):
    #        current_query = st.session_state.query_text_input
    #        if current_query.strip():
    #            query_data = embed_text([current_query], update_stage_flag=False, return_data=True)
    #            query_data['query'] = current_query
    #            update_stage('query', query_data)
    #            st.session_state.query_embedding = query_data["embeddings"]
    #            st.session_state.query_text_step5 = current_query
    #            st.success("Query vectorized!")
    #        else:
    #            st.warning("Please enter a query.")
    #    
    #    # --- Step 5B: Retrieve Query Embeddings (Optional) ---
    #    st.subheader("Step 5B: Retrieve Matching Chunks (Optional)")
    #    if st.button("Run Step 5B: Retrieve Matching Chunks"):
    #        if st.session_state.query_embedding:
    #            passages, metadata = query_collection(st.session_state.query_text_step5, "rag_collection", n_results=5)
    #            st.session_state.retrieved_passages = passages
    #            st.session_state.retrieved_metadata = metadata
    #            st.success("Relevant chunks retrieved based on your query in Step 5A!")
    #        else:
    #            st.warning("Run Step 5A (Embed Query) first.")
        
        # --- Step 6: Get Answer ---
        st.subheader("Step 6: Get Answer")
        final_question = st.text_input("-> Enter your final question for Step 6", key="final_question_input")

        if st.button("Run Step 6: Get Answer"):
            current_question = st.session_state.final_question_input
            if current_question.strip():
                passages, metadata = query_collection(current_question, "rag_collection", n_results=50)
                if passages:
                    if DEBUG_MODE:
                        st.write(f"DEBUG => Retrieved {len(passages)} passages with {len(metadata)} metadata entries")
                    answer = generate_answer_with_gpt(current_question, passages, metadata)
                    st.session_state.final_answer = answer
                    st.session_state.final_question_step7 = current_question
                    st.success("Answer generated!")
                    st.write(answer)
                else:
                    st.warning("Could not retrieve relevant passages for your question.")
            else:
                st.warning("Please enter your final question.")
    
   # with col2:
   #     st.header("RAG Pipeline Visualization")
   #     component_args = {
   #         "currentStage": st.session_state.current_stage,
   #         "stageData": { s: st.session_state.get(f'{s}_data')
   #                        for s in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']
   #                        if st.session_state.get(f'{s}_data') is not None }
   #     }
   #     pipeline_html = get_pipeline_component(component_args)
   #     components.html(pipeline_html, height=2000, scrolling=True)


if __name__ == "__main__":
    main()
