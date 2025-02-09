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
    "    You are an extremely smart, knowledgeable, and helpful assistant. You must answer the user’s query based "
    "    **ONLY** on the provided context from the RAG documents. Always think step-by-step. The user is very "
    "    grateful for perfect results.\n"
    "  </ROLE>\n\n"
    "  <INSTRUCTIONS>\n"
    "    1. Your response must begin with a **High-Level, Concise 'Instructions to Action'** section if the user "
    "       explicitly asks for help. Provide direct, deterministic guidance strictly from the RAG documents "
    "       (e.g., \"If w > **x** then **y** is allowed, **z** is prohibited\").\n\n"
    "    2. Then present a **TL;DR Summary** (bullet points) strictly based on the doc. Use **bold** for crucial "
    "       numeric thresholds and legal/statutory references on first mention, and *italics* for important nuances.\n\n"
    "    3. Then provide a **Detailed Explanation** (also strictly from the doc). If relevant, include a *short example "
    "       scenario* demonstrating how the doc-based rules might apply.\n\n"
    "    4. After the Detailed Explanation, include an **'Other References'** section. Here, you may add any "
    "       further clarifications or external knowledge beyond the doc, but clearly label it as such. Cite any "
    "       explicit statutory references in square brackets, e.g., [Section 1, Paragraph 2].\n\n"
    "    5. If the user’s query **cannot** be addressed with the RAG documents, then you must provide:\n"
    "       - A large \"Sorry!\" header with: \"The uploaded document states nothing relevant according to your query...\"\n"
    "       - Under another large header \"Best guess,\" try to interpret the user’s request, noting that this is a guess.\n"
    "       - Finally, **only** if no relevant doc info is found, add a last section in the same large header size, "
    "         **in red**, titled \"The fun part :-)\". Introduce it with *italics* \"(section requested in Step 0 to "
    "         show how output can be steered)\" in normal text size. Provide an amusing, sarcastic take (with emojis) "
    "         on how the query might be related.\n\n"
    "    6. Keep the doc-based sections strictly doc-based (no external info). Maintain **bold** for crucial references, "
    "       *italics* for nuance, and a professional, academically rigorous tone except in \"The fun part :-)\".\n\n"
    "    7. IMPORTANT: always answer in the language of the user's initial query unless the user requests otherwise.\n\n"
    "    8. IMPORTANT: Do **not** produce XML tags in your final output. Present your answer in normal prose with "
    "       headings in large text as described.\n"
    "  </INSTRUCTIONS>\n\n"
    "  <STRUCTURE>\n"
    "    <REMARKS_TO_STRUCTURE>"
    "      - VERY IMPORTANT: Translate all of the following elements to the user's query language."
    "    </REMARKS_TO_STRUCTURE>"
    "    <!-- Two possible cases for final output -->\n\n"
    "    <!-- Case A: Document-based answer (when RAG doc is relevant) -->\n"
    "    <CASEA>\n"
    "      <HEADER_LEVEL1>Instructions to Action</HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>TL;DR Summary</HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>Detailed Explanation</HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>Other References</HEADER_LEVEL1>\n"
    "    </CASEA>\n\n"
    "    <!-- Case B: No relevant info in the RAG doc -->\n"
    "    <CASEB>\n"
    "      <HEADER_LEVEL1>Sorry!</HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>Best guess</HEADER_LEVEL1>\n"
    "      <HEADER_LEVEL1>The fun part :-)\n"
    "        <SUBTITLE>(section requested in Step 0 to show how output can be steered)</SUBTITLE>\n"
    "      </HEADER_LEVEL1>\n"
    "    </CASEB>\n"
    "  </STRUCTURE>\n\n"
    "  <FINAL_REMARKS>\n"
    "    - Carefully follow each step and always THINK STEP-BY-STEP for an optimal, well-structured response.\n"
    "    - Always present the final answer in normal prose, not XML.\n"
    "    - ETREMELY IMPORTANT: if you are an ADVANCED VOICE MODE assistant, then the specific instructions under tag <DELTA_FROM_MAIN_PROMPT> must override all of the above that is contradictory to its explanations!!!\n"
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
import shutil
import hashlib
import os
from typing import Optional
from pathlib import Path
import pycountry
import textwrap


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

CHROMA_DIRECTORY = "chromadb_storage"

# Initialize ChromaDB with our custom embedding function instance
def init_chroma_client():
    if "api_key" not in st.session_state or not st.session_state.get("user_id"):
        return None, None
    
    user_dir = f"chromadb_storage_user_{st.session_state.user_id}"
    os.makedirs(user_dir, exist_ok=True)
    
    embedding_function_instance = OpenAIEmbeddingFunction(st.session_state["api_key"])
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=user_dir
    ))
    return client, embedding_function_instance

# Create embedding function that uses OpenAI
class OpenAIEmbeddingFunction:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def __call__(self, texts):
        if not isinstance(texts, list):
            texts = [texts]
        texts = [sanitize_text(t) for t in texts]
        response = self.client.embeddings.create(input=texts, model="text-embedding-3-large")
        return [item.embedding for item in response.data]

import re
from typing import Optional, Dict, Set
import pycountry

import streamlit as st  # optional if you want to log warnings

import re
from typing import Optional, Dict, Set
import pycountry
import streamlit as st  # optional if you want to log warnings

class CountryDetector:
    """
    Detects country references in user text by:
      1) Checking for ISO codes (alpha2, alpha3) in uppercase.
      2) Checking for official/common English names from pycountry.
      3) Checking for a built-in German translation (if available) via country.translations.
      4) Incorporating a large synonyms map (manually curated for English/German/local forms).

    Returns an alpha-2 code (e.g., 'CH' for Switzerland) if found.
    """

    def __init__(self):
        self.country_mapping = self._build_country_mapping()

    def _build_country_mapping(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        Gathers known synonyms & codes for each country:
          - alpha_2 & alpha_3 codes from pycountry.
          - Official/common name from pycountry.
          - A German translation from country.translations (if available).
          - A large manually curated synonyms map for extra coverage.
        """
        mapping = {}
        synonyms_map = self._synonyms_map()

        for country in pycountry.countries:
            alpha2 = country.alpha_2
            alpha3 = country.alpha_3

            iso_codes = {alpha2, alpha3}
            all_names = set()

            # 1) Official English name
            all_names.add(country.name.lower())

            # 2) If available, official_name and common_name
            if hasattr(country, 'official_name'):
                off = getattr(country, 'official_name')
                if off and off.lower() not in all_names:
                    all_names.add(off.lower())
            if hasattr(country, 'common_name'):
                common = getattr(country, 'common_name')
                if common and common.lower() not in all_names:
                    all_names.add(common.lower())

            # 3) Attempt to add a German translation if available
            if hasattr(country, 'translations'):
                de_name = country.translations.get('de')
                if de_name and de_name.lower() not in all_names:
                    all_names.add(de_name.lower())

            # 4) Add manually curated synonyms
            extras = synonyms_map.get(alpha2.upper(), set())
            all_names.update(extras)

            mapping[alpha2] = {
                "iso": iso_codes,
                "names": all_names
            }
            mapping[alpha3] = mapping[alpha2]

        return mapping

    def _synonyms_map(self) -> Dict[str, Set[str]]:
        """
        A large dictionary mapping ISO alpha-2 codes to a set of synonyms.
        This includes well-known English, German, and minimal local forms.
        (Extend this as needed.)
        """
        return {
            # -----------------
            # Europe
            # -----------------
            "AD": {"andorra"},
            "AL": {"albania", "albanien"},
            "AM": {"armenia", "armenien"},
            "AT": {"austria", "osterreich", "österreich"},
            "AZ": {"azerbaijan", "aserbaidschan"},
            "BA": {"bosnia", "bosnia and herzegovina", "bosnien", "herzegovina"},
            "BE": {"belgium", "belgien", "belgique"},
            "BG": {"bulgaria", "bulgarien"},
            "BY": {"belarus", "weißrussland", "weissrussland"},
            "CH": {"switzerland", "schweiz", "svizzera", "suisse", "swiss", "helvetia"},
            "CY": {"cyprus", "zypern"},
            "CZ": {"czech republic", "czechia", "tschechien", "tschchien"},
            "DE": {"germany", "deutschland"},
            "DK": {"denmark", "danmark"},
            "EE": {"estonia", "estland"},
            "ES": {"spain", "spanien", "españa"},
            "FI": {"finland", "finnland"},
            "FO": {"faroe islands", "färöer", "faroe"},
            "FR": {"france", "frankreich"},
            "GB": {"uk", "gb", "united kingdom", "england", "britain", "scotland", "wales", "großbritannien"},
            "IE": {"ireland", "irland"},
            "IS": {"iceland", "island"},
            "IT": {"italy", "italien", "italia"},
            "LI": {"liechtenstein"},
            "LT": {"lithuania", "litauen"},
            "LU": {"luxembourg", "luxemburg"},
            "LV": {"latvia", "lettland"},
            "MC": {"monaco"},
            "MD": {"moldova", "moldawien", "moldau"},
            "ME": {"montenegro", "crna gora"},
            "MK": {"macedonia", "north macedonia", "mazedonien"},
            "MT": {"malta"},
            "NL": {"netherlands", "holland", "niederlande", "holländisch"},
            "NO": {"norway", "norwegen"},
            "PL": {"poland", "polen"},
            "PT": {"portugal"},
            "RO": {"romania", "rumänien", "rumanien"},
            "RS": {"serbia", "serbien"},
            "RU": {"russia", "russland", "rossija"},
            "SE": {"sweden", "schweden"},
            "SI": {"slovenia", "slowenien"},
            "SK": {"slovakia", "slowakei"},
            "TR": {"turkey", "türkei"},
            "UA": {"ukraine", "ukraina"},
            "VA": {"vatican", "vatikan"},
            "XK": {"kosovo"},
            # -----------------
            # Asia
            # -----------------
            "AE": {"uae", "united arab emirates", "vereinigte arabische emirate"},
            "AF": {"afghanistan", "afganistan"},
            "BD": {"bangladesh", "bangladesch"},
            "BH": {"bahrain", "bahrein"},
            "BN": {"brunei"},
            "BT": {"bhutan", "butan"},
            "CN": {"china", "volksrepublik china"},
            "HK": {"hong kong"},
            "ID": {"indonesia", "indonesien"},
            "IN": {"india", "indien"},
            "IQ": {"iraq", "irak"},
            "IR": {"iran", "persia"},
            "IL": {"israel", "israël"},
            "JO": {"jordan", "jordanien"},
            "JP": {"japan"},
            "KG": {"kazakhstan", "kasachstan"},
            "KH": {"cambodia", "kambodscha", "kampuchea"},
            "KP": {"north korea", "nordkorea"},
            "KR": {"south korea", "südkorea"},
            "KW": {"kuwait", "kuweit"},
            "LA": {"laos", "lao"},
            "LB": {"lebanon", "libanon"},
            "LK": {"sri lanka", "ceylon"},
            "MM": {"myanmar", "burma"},
            "MN": {"mongolia", "mongolei"},
            "MY": {"malaysia", "malaysien"},
            "NP": {"nepal"},
            "OM": {"oman"},
            "PH": {"philippines", "philippinen"},
            "PK": {"pakistan", "pakstan"},
            "PS": {"palestine", "palästina"},
            "QA": {"qatar", "katar"},
            "SA": {"saudi arabia", "saudi-arabien"},
            "SG": {"singapore", "singapur"},
            "SY": {"syria", "syrien"},
            "TH": {"thailand", "thailändisch"},
            "TJ": {"tajikistan", "tadschikistan"},
            "TM": {"turkmenistan", "turkmenien"},
            "TW": {"taiwan", "roc"},
            "UZ": {"uzbekistan", "usbekistan"},
            "VN": {"vietnam", "viet nam"},
            "YE": {"yemen", "jemen"},
            # -----------------
            # Africa
            # -----------------
            "AO": {"angola"},
            "BF": {"burkina faso"},
            "BI": {"burundi"},
            "BJ": {"benin"},
            "BW": {"botswana"},
            "CD": {"dr congo", "democratic republic congo", "kongo-kinshasa", "zaire"},
            "CF": {"central african republic", "zentralafrikanische republik"},
            "CG": {"congo-brazzaville", "republic of congo"},
            "CI": {"cote d'ivoire", "ivory coast", "côte d’ivoire"},
            "CM": {"cameroon", "kamerun"},
            "CV": {"cape verde", "kap verde", "cabo verde"},
            "DJ": {"djibouti", "dschibuti"},
            "DZ": {"algeria", "algerien"},
            "EG": {"egypt", "ägypten"},
            "ER": {"eritrea"},
            "ET": {"ethiopia", "äthiopien"},
            "GA": {"gabon"},
            "GH": {"ghana"},
            "GM": {"gambia"},
            "GN": {"guinea", "guinea-conakry"},
            "GQ": {"equatorial guinea", "äquatorialguinea"},
            "GW": {"guinea-bissau"},
            "KE": {"kenya", "kenia"},
            "KM": {"comoros", "komoren"},
            "LR": {"liberia"},
            "LS": {"lesotho"},
            "LY": {"libya", "libyen"},
            "MA": {"morocco", "marokko"},
            "MG": {"madagascar", "madagaskar"},
            "ML": {"mali"},
            "MR": {"mauritania", "mauretanien"},
            "MU": {"mauritius"},
            "MW": {"malawi"},
            "MZ": {"mozambique", "mosambik"},
            "NA": {"namibia"},
            "NE": {"niger"},
            "NG": {"nigeria", "nijeria"},
            "RE": {"réunion", "reunion"},
            "RW": {"rwanda"},
            "SC": {"seychelles"},
            "SD": {"sudan"},
            "SL": {"sierra leone"},
            "SN": {"senegal"},
            "SO": {"somalia", "somalien"},
            "SS": {"south sudan", "südsudan"},
            "ST": {"sao tome and principe", "são tomé und príncipe"},
            "SZ": {"eswatini", "swaziland"},
            "TD": {"chad", "tsjad"},
            "TG": {"togo"},
            "TN": {"tunisia", "tunesien"},
            "TZ": {"tanzania", "tansania"},
            "UG": {"uganda"},
            "YT": {"mayotte"},
            "ZA": {"south africa", "südafrika"},
            "ZM": {"zambia", "sambia"},
            "ZW": {"zimbabwe", "simbabwe"},
            # -----------------
            # Americas
            # -----------------
            "AG": {"antigua and barbuda"},
            "AI": {"anguilla"},
            "AN": {"netherlands antilles"},  # obsolete code
            "AR": {"argentina", "argentinien"},
            "AW": {"aruba"},
            "BB": {"barbados"},
            "BO": {"bolivia", "bolivien"},
            "BR": {"brazil", "brasilien"},
            "BS": {"bahamas"},
            "BZ": {"belize"},
            "CA": {"canada", "kanada"},
            "CL": {"chile"},
            "CO": {"colombia", "kolumbien"},
            "CR": {"costa rica", "costarica"},
            "CU": {"cuba", "kuba"},
            "DM": {"dominica"},
            "DO": {"dominican republic", "dominikanische republik"},
            "EC": {"ecuador"},
            "GD": {"grenada"},
            "GF": {"french guiana", "guyane"},
            "GL": {"greenland", "grönland"},
            "GT": {"guatemala"},
            "GY": {"guyana"},
            "HN": {"honduras"},
            "HT": {"haiti"},
            "JM": {"jamaica", "jamaika"},
            "KN": {"saint kitts and nevis", "st kitts und nevis"},
            "LC": {"saint lucia"},
            "MQ": {"martinique"},
            "MX": {"mexico", "mexiko"},
            "NI": {"nicaragua"},
            "PA": {"panama", "panamá"},
            "PE": {"peru", "perú"},
            "PF": {"french polynesia", "polynésie française"},
            "PR": {"puerto rico", "puertorico"},
            "PY": {"paraguay"},
            "SR": {"suriname", "surinam"},
            "SV": {"el salvador", "elsalvador"},
            "TC": {"turks and caicos islands"},
            "TT": {"trinidad and tobago", "trinidad tobago"},
            "UY": {"uruguay"},
            "VC": {"saint vincent and the grenadines"},
            "VE": {"venezuela"},
            # -----------------
            # Oceania
            # -----------------
            "AS": {"american samoa"},
            "AU": {"australia", "australien"},
            "CK": {"cook islands"},
            "FJ": {"fiji", "fidji"},
            "FM": {"micronesia", "föderierte staaten von mikronesien", "mikronesien"},
            "GU": {"guam"},
            "KI": {"kiribati"},
            "MH": {"marshall islands"},
            "MP": {"northern mariana islands"},
            "NC": {"new caledonia", "nouvelle-calédonie"},
            "NR": {"nauru"},
            "NU": {"niue"},
            "NZ": {"new zealand", "neuseeland"},
            "PG": {"papua new guinea", "papua-neuguinea"},
            "PN": {"pitcairn islands"},
            "PW": {"palau"},
            "SB": {"solomon islands", "salomonen"},
            "TO": {"tonga"},
            "TV": {"tuvalu"},
            "VU": {"vanuatu"},
            "WF": {"wallis and futuna"},
            "WS": {"samoa"}
        }

    def get_iso_alpha2(self, country_identifier: str) -> Optional[str]:
        """
        Convert any country identifier (name or code) to ISO alpha-2 code
        by matching it in self.country_mapping.
        """
        country_identifier = country_identifier.strip().lower()
        upper_id = country_identifier.upper()
        if upper_id in self.country_mapping:
            for code in self.country_mapping[upper_id]['iso']:
                if len(code) == 2:
                    return code

        for alpha2, data in self.country_mapping.items():
            if len(alpha2) == 2:
                for known_name in data['names']:
                    if country_identifier == known_name:
                        return alpha2
        return None

    def detect_country_in_text(self, text: str) -> Optional[str]:
        """
        Return the first alpha-2 code detected in 'text':
          - Check for explicit 2/3 letter ISO codes.
          - Otherwise, search for any known synonym.
        """
        text_upper = text.upper()
        text_lower = text.lower()

        COMMON_STOPWORDS = {
            # English stopwords
            "ARE", "AND", "FOR", "THE", "WAS", "WERE", "HAS", "HAVE", "IS", "IN",
            "OF", "TO", "A", "AN", "AT", "BY", "FROM", "IT", "ON", "AS",
            # German stopwords
            "UND", "DER", "DIE", "DAS", "IST", "IN", "DES", "DEN", "EIN", "EINE",
            "MIT", "AUF", "FÜR", "NICHT", "WAR", "SIND", "HAT", "HABEN", "ICH", "DU", "AB", "ZU"
        }

        for match in re.finditer(r'\b([A-Z]{2,3})\b', text.upper()):
            token = match.group(1)
            # Skip tokens that are common stopwords in English or German
            if token in COMMON_STOPWORDS:
                continue
            iso_code = self.get_iso_alpha2(token)
            if iso_code:
                return iso_code
        for alpha2, data in self.country_mapping.items():
            if len(alpha2) == 2:
                for known_name in data['names']:
                    if known_name in text_lower:
                        return alpha2
        return None

if __name__ == '__main__':
    detector = CountryDetector()
    test_queries = [
        "I am planning a trip to Deutschland next summer.",
        "She moved to Switzerland for work.",
        "Traveling to the United Kingdom was fun.",
        "We visited Tschechien recently."  # German variant for Czech Republic
    ]
    for query in test_queries:
        code = detector.detect_country_in_text(query)
        print(f"Query: '{query}' => Detected country code: {code}")

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
    ensuring each stage’s data is handled consistently and we have debug logs.
    """

    # Debug: show raw data
    if DEBUG_MODE:
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
            st.write(f"DEBUG => retrieve fallback => {enhanced_data}")

    elif stage == 'generate':
        # For "generate" stage
        st.write(f"DEBUG => In generate block => data={data}")
        # ensure 'answer' key is present
        if isinstance(data, dict) and 'answer' not in data:
            enhanced_data['answer'] = ''
            if DEBUG_MODE:
                st.write("DEBUG => 'answer' key was missing, set to ''")

    # Store final "enhanced_data" in session state
    st.session_state[f'{stage}_data'] = enhanced_data

    # Debug print
    if DEBUG_MODE:
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


def embed_text(
    texts: List[Union[str, Dict[str, Any]]],
    openai_embedding_model: str = "text-embedding-3-large",
    update_stage_flag=True,
    return_data=False
):
    if not st.session_state.get("api_key"):
        st.error("OpenAI API key not set.")
        st.stop()
        
    processed_texts = [chunk["text"] if isinstance(chunk, dict) else chunk for chunk in texts]
    safe_texts = [sanitize_text(s) for s in processed_texts]
    
    response = new_client.embeddings.create(input=safe_texts, model=openai_embedding_model)
    embeddings = [item.embedding for item in response.data]
    
    token_breakdowns = []
    for text, embedding in zip(safe_texts, embeddings):
        # Take first 10 words for token breakdown to avoid visualization issues
        tokens = text.split()[:10]
        breakdown = []
        if tokens:
            segment_size = len(embedding) // len(tokens)
            for i, tok in enumerate(tokens):
                start = i * segment_size
                end = start + segment_size
                snippet = embedding[start:min(end, len(embedding))]
                breakdown.append({"token": tok, "vector_snippet": snippet[:10]})
        token_breakdowns.append(breakdown)
    
    embedding_data = {
        "embeddings": embeddings,
        "dimensions": len(embeddings[0]) if embeddings else 0,
        "preview": embeddings[0][:10] if embeddings else [],
        "total_vectors": len(embeddings),
        "token_breakdowns": token_breakdowns
    }
    
    if update_stage_flag:
        update_stage('embed', embedding_data)
    if return_data:
        return embedding_data
    return embeddings

def extract_text_from_file(uploaded_file, reverse=False) -> str:
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.txt'):
        text = uploaded_file.read().decode("utf-8")
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
    elif file_name.endswith('.docx'):
        try:
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error reading DOCX file: {e}")
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
    global chroma_client, embedding_function_instance
    if embedding_function_instance is None:
        st.error("Embedding function not initialized. Please set your OpenAI API key.")
        st.stop()
    
    if force_recreate:
        try:
            chroma_client.delete_collection(name=collection_name)
            st.write(f"Deleted existing collection '{collection_name}' due to force_recreate flag.")
        except Exception as e:
            st.write(f"Could not delete existing collection '{collection_name}': {e}")
    
    try:
        coll = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function_instance)
        coll_info = coll.get()
    except Exception as e:
        coll = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function_instance)
    return coll


def add_to_chroma_collection(collection_name: str, chunks: List[Union[str, Dict[str, Any]]], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None):
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Extract text and metadata from chunks if they're dictionaries
    if isinstance(chunks[0], dict):
        texts = [chunk["text"] for chunk in chunks]
        chunk_metadatas = [chunk["metadata"] for chunk in chunks]
    else:
        texts = chunks
        chunk_metadatas = metadatas if metadatas else [{"source": "uploaded_document"} for _ in chunks]
    
    force_flag = st.session_state.get("force_recreate", False)
    coll = create_or_load_collection(collection_name, force_recreate=force_flag)
    coll.add(documents=texts, metadatas=chunk_metadatas, ids=ids)
    chroma_client.persist()
    
    update_stage('store', {
        'collection': collection_name,
        'count': len(texts)
    })
    st.write(f"Stored {len(texts)} chunks in collection '{collection_name}'.")


def query_collection(query: str, collection_name: str, n_results: int = 3):
    """
    Enhanced query function that maintains existing functionality while adding country awareness
    """
    if DEBUG_MODE:
        st.write(f"DEBUG => Entering query_collection() with query='{query}', collection_name='{collection_name}'")

    # Initialize country detector
    country_detector = CountryDetector()
    
    # Get collection and check document count
    force_flag = st.session_state.get("force_recreate", False)
    coll = create_or_load_collection(collection_name, force_recreate=force_flag)
    coll_info = coll.get()
    doc_count = len(coll_info.get("ids", []))
    
    if DEBUG_MODE:
        st.write(f"DEBUG => Collection '{collection_name}' has doc_count={doc_count}")

    if doc_count == 0:
        st.warning("No documents found in collection. Please upload first.")
        return [], []
    
    # Try to detect country in query
    query_country_code = country_detector.detect_country_in_text(query)
    if DEBUG_MODE:
        st.write(f"DEBUG => Detected query_country_code={query_country_code}")

    if query_country_code:
        # If country detected, try country-specific search first
        st.write(f"Detected country code: {query_country_code}")
        results = coll.query(
            query_texts=[query],
            where={"country_code": query_country_code},
            n_results=n_results
        )
        retrieved_passages = results.get("documents", [[]])[0]
        retrieved_metadata = results.get("metadatas", [[]])[0]

        if DEBUG_MODE:
            st.write(f"DEBUG => Found {len(retrieved_passages)} passages for code={query_country_code}")

        # If we don't get enough results, do a fallback
        if len(retrieved_passages) < n_results:
            st.write(f"Found only {len(retrieved_passages)} results for {query_country_code}. Supplementing general search...")
            remaining_results = n_results - len(retrieved_passages)
            additional_results = coll.query(
                query_texts=[query],
                where={"country_code": {"$ne": query_country_code}},  # not equal to the current country
                n_results=remaining_results
            )
            add_pass = additional_results.get("documents", [[]])[0]
            add_meta = additional_results.get("metadatas", [[]])[0]
            retrieved_passages.extend(add_pass)
            retrieved_metadata.extend(add_meta)
            if DEBUG_MODE:
                st.write(f"DEBUG => After supplement => total passages={len(retrieved_passages)}")

    else:
        # No country detected, do a standard search
        if DEBUG_MODE:
            st.write("DEBUG => No country code detected, doing standard search")
        results = coll.query(
            query_texts=[query],
            n_results=n_results
        )
        retrieved_passages = results.get("documents", [[]])[0]
        retrieved_metadata = results.get("metadatas", [[]])[0]
        if DEBUG_MODE:
            st.write(f"DEBUG => Found {len(retrieved_passages)} passages (no country filter).")

    if DEBUG_MODE:
        st.write(f"DEBUG => Will call update_stage('retrieve', ...) with passages={len(retrieved_passages)}, metadata={len(retrieved_metadata)}")

    # Safely pass defaults to avoid None
    update_stage('retrieve', {
        "passages": retrieved_passages or [],
        "metadata": retrieved_metadata or []
    })

    if DEBUG_MODE:
        st.write(f"DEBUG => After update_stage('retrieve'), st.session_state['retrieve_data'] => {st.session_state.get('retrieve_data')}")

    st.write(f"Retrieved {len(retrieved_passages)} passages from collection '{collection_name}'.")

    if retrieved_passages:
        countries_found = set(meta.get('country_code', 'Unknown') for meta in retrieved_metadata)
        st.write(f"Results include information from: {', '.join(countries_found)}")

    return retrieved_passages, retrieved_metadata


##############################################################################
# 7) GPT ANSWER GENERATION
##############################################################################
def generate_answer_with_gpt(query: str, retrieved_passages: List[str], retrieved_metadata: List[dict],
                             system_instruction: str = None):
    if DEBUG_MODE:
        st.write(f"DEBUG => generate_answer_with_gpt called with query='{query}', #passages={len(retrieved_passages)}")

    if system_instruction is None:
        system_instruction = st.session_state.get("custom_prompt", BASE_DEFAULT_PROMPT)
    
    if new_client is None:
        st.error("OpenAI client not initialized. Provide an API key in the sidebar.")
        st.stop()
    
    context_text = "\n\n".join(retrieved_passages)
    final_prompt = (
        f"{system_instruction}\n\n"
        f"Context:\n{context_text}\n\n"
        f"User Query: {query}\nAnswer:"
    )

    if DEBUG_MODE:
        st.write(f"DEBUG => final_prompt length={len(final_prompt)} chars")

    completion = new_client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[{"role": "user", "content": final_prompt}],
        response_format={"type": "text"}
    )
    if completion and completion.choices:
        answer = completion.choices[0].message.content
    else:
        if DEBUG_MODE:
            st.write("DEBUG => No completion or empty choices; defaulting answer to ''")
        answer = ""

    st.write(f"DEBUG => Received answer of length={len(answer)}")

    # Mark stage=generate
    update_stage('generate', {'answer': answer})
    if DEBUG_MODE:
        st.write(f"DEBUG => Done generate => st.session_state['generate_data']={st.session_state.get('generate_data')}")

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
            st.write("Error response:", e.response.text)
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

def get_user_specific_directory(user_id: str) -> str:
    """Create unique ChromaDB directory for each user"""
    hashed_id = hashlib.sha256(user_id.encode()).hexdigest()[:10]
    return f"chromadb_storage_user_{hashed_id}"

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
    """
    Delete all ChromaDB collections for a specific user.
    
    Args:
        user_id: The ID of the user whose collections should be deleted
        
    Returns:
        tuple[bool, str]: Success status and message
    """
    if not user_id:
        return False, "No user ID provided"
    
    try:
        # Get user-specific directory
        user_dir = f"chromadb_storage_user_{user_id}"
        
        # If the directory doesn't exist, nothing to delete
        if not os.path.exists(user_dir):
            return True, "No collections found for user"
            
        # Create a new client instance for this user's directory
        temp_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=user_dir
        ))
        
        # Delete all collections for this user
        collections = temp_client.list_collections()
        for collection in collections:
            temp_client.delete_collection(name=collection.name)
            
        # Close the client connection
        temp_client.reset()
        
        # Remove the directory
        shutil.rmtree(user_dir)
        
        # Recreate empty directory
        os.makedirs(user_dir, exist_ok=True)
        
        return True, f"Successfully deleted all collections for user {user_id}"
        
    except Exception as e:
        return False, f"Error deleting collections: {str(e)}"






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
    st.write("DEBUG: chunk_text -> after ensure_canonical_order()")

    # Remove lines that are solely dashes.
    lines = canonical_text.splitlines()
    cleaned_lines = [l for l in lines if not re.fullmatch(r"[-]+", l.strip())]
    cleaned_text = "\n".join(cleaned_lines)
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






##############################################################################
# 10) MAIN STREAMLIT APP
##############################################################################
def main():
    global chroma_client

    # Authentication
    create_user_session()

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
    st.title("RAG + Advanced Voice Mode (AVM) Cockpit")
    
    # Sidebar: API key and force recreate option
    global chroma_client, embedding_function_instance
    
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)
        if chroma_client is None:
            st.error("ChromaDB client not initialized properly")
            st.stop()
    
    # "Delete All Collections" button in sidebar
    if st.sidebar.button("Delete All Collections"):
        if not st.session_state.get("user_id"):
            st.sidebar.error("Please log in first")
        else:
            success, message = delete_user_collections(st.session_state.user_id)
            if success:
                # Reset the ChromaDB client to force reconnection
                chroma_client, embedding_function_instance = init_chroma_client()
                
                # Clear relevant session state
                for stage in ['store', 'query', 'retrieve', 'generate']:
                    if f'{stage}_data' in st.session_state:
                        st.session_state[f'{stage}_data'] = None
                
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
                
    st.sidebar.markdown("### AVM Controls")
    
    def toggle_avm():
        st.session_state.avm_button_key += 1  # Force button refresh
        if st.session_state.avm_active:
            st.session_state.voice_html = None
            st.session_state.avm_active = False
            st.session_state.avm_button_key += 1
            return
        token_data = get_ephemeral_token("rag_collection")
        if token_data:
            st.session_state.voice_html = get_realtime_html(token_data)
            st.session_state.avm_active = True
            st.session_state.avm_button_key += 1
        else:
            st.sidebar.error("Could not start AVM.\n\nCheck error messages at the top of the main section ---------------------------->>>")

    if st.sidebar.button(
        "End AVM" if st.session_state.avm_active else "Start AVM",
        key=f"avm_toggle_{st.session_state.avm_button_key}",
        on_click=toggle_avm
    ):
        pass

    if st.session_state.avm_active:
        st.sidebar.success("AVM started.")
    else:
        if st.session_state.avm_button_key > 0:
            st.sidebar.success("AVM ended.")
    
    # **Display the exact AVM initialization text in the sidebar**
    if st.session_state.avm_active and st.session_state.get("avm_initial_text"):
        st.sidebar.markdown("### AVM Initial Instructions")
        st.sidebar.code(st.session_state.avm_initial_text)
    
    if st.session_state.avm_active and st.session_state.voice_html:
        components.html(st.session_state.voice_html, height=1, scrolling=True)
    

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
    col1, spacer, col2 = st.columns([1.3, 0.3, 2])
    
    with col1:
        st.header("Step-by-Step Pipeline Control")
        
        # --- Step 0: Specify Initial Instructions & Voice ---
        st.subheader("Step 0: Specify Initial Instructions & Voice")

        VOICE_OPTIONS = ["alloy", "echo", "shimmer", "ash", "ballad", "coral", "sage", "verse"]

        # Load the current voice preference for the logged-in user, defaulting to "coral"
        current_voice = load_voice_pref(st.session_state.user_id) if st.session_state.get("user_id") else "coral"

        selected_voice = st.selectbox(
            "-> Choose a voice for advanced voice mode:",
            options=VOICE_OPTIONS,
            index=VOICE_OPTIONS.index(current_voice) if current_voice in VOICE_OPTIONS else VOICE_OPTIONS.index("coral")
        )

        # If the selection has changed, update session state and persist the new voice
        if selected_voice != current_voice:
            st.session_state.selected_voice = selected_voice
            save_voice_pref(selected_voice, st.session_state.user_id)

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

        # ---------------------------
        # For Voice Instructions
        # ---------------------------
        if "voice_custom_prompt" not in st.session_state:
            st.session_state.voice_custom_prompt = load_voice_instructions(st.session_state.user_id) or ""
        if "voice_instructions_widget_key" not in st.session_state:
            st.session_state.voice_instructions_widget_key = str(uuid.uuid4())
        voice_container = st.empty()
        voice_instructions = voice_container.text_area(
            "-> Customize your advanced voice mode voice & tone.\n\n",
            value=st.session_state.voice_custom_prompt,
            key=st.session_state.voice_instructions_widget_key
        )
        if voice_instructions != st.session_state.voice_custom_prompt:
            st.session_state.voice_custom_prompt = voice_instructions
            save_voice_instructions(voice_instructions, st.session_state.user_id)

        # Button to restore default voice instructions
        if st.button("Restore Default Prompt (Voice Instructions)"):
            st.session_state.voice_custom_prompt = DEFAULT_VOICE_PROMPT
            save_voice_instructions(DEFAULT_VOICE_PROMPT, st.session_state.user_id)
            st.session_state.voice_instructions_widget_key = str(uuid.uuid4())
            voice_container.text_area(
                "-> Customize your advanced voice mode voice & tone.\n\n"
                "-> Deleting the contents of this box & refreshing your browser restores a default prompt.",
                value=st.session_state.voice_custom_prompt,
                key=st.session_state.voice_instructions_widget_key
            )

       # --- Step 1: Upload Context ---
        st.subheader("Step 1: Upload Context")

        # Define the reverse text order checkbox BEFORE the form.
        # reverse_text_order = st.sidebar.checkbox("Reverse extracted text order", value=True)

        # Wrap the uploader in a form with clear_on_submit=True.
        with st.form("upload_form", clear_on_submit=True):
            uploaded_files = st.file_uploader(
                "-> Upload one or more *.txt documents (EXPERIMENTAL: pdf, docx, csv, xlsx, rtf)",
                type=["txt", "pdf", "docx", "csv", "xlsx", "rtf"],
                accept_multiple_files=True
            )
            submitted = st.form_submit_button("Run Step 1: Upload Context")

        if submitted:
            if uploaded_files:
                combined_text = ""
                for uploaded_file in uploaded_files:
                    text = extract_text_from_file(uploaded_file)
                    if text:
                        combined_text += f"\n\n---\n\n{text}"

                if combined_text:
                    # set st.session_state.uploaded_text
                    st.session_state.uploaded_text = combined_text

                    # ALSO update "upload" stage properly:
                    update_stage("upload", {
                        'content': combined_text,
                        'preview': combined_text[:600],
                        'size': len(combined_text)
                    })

                    st.success("Files uploaded and processed!")
                else:
                    st.error("No text could be extracted...")
            else:
                st.warning("No file selected.")

        # --- Step 2: Chunk Context ---
        st.subheader("Step 2: Chunk Context")
        if st.button("Run Step 2: Chunk Context"):
            if st.session_state.uploaded_text:
                final_chunks = parse_marker_blocks_linewise(st.session_state.uploaded_text)
                st.session_state.chunks = final_chunks
                st.success(f"Found {len(final_chunks)} chunk(s).")
            else:
                st.warning("Please upload at least one document first.")
        
        # --- Step 3: Embed Context ---
        st.subheader("Step 3: Embed Context")
        if st.button("Run Step 3: Embed Context"):
            if not st.session_state.get("api_key"):
                st.error("OpenAI API key not set. Please provide a valid API key in the sidebar before running this step.")
            elif st.session_state.chunks:
                try:
                    embedding_data = embed_text(st.session_state.chunks, update_stage_flag=True, return_data=True)
                    st.session_state.embeddings = embedding_data
                    st.success("Embeddings created!")
                except Exception as e:
                    st.error(f"An error occurred while generating embeddings: {e}")
            else:
                st.warning("Please chunk the document first.")
        
        # --- Step 4: Store ---
        st.subheader("Step 4: Store Embedded Context")
        if st.button("Run Step 4: Store Embedded Context"):
            if st.session_state.chunks and st.session_state.embeddings:
                ids = [str(uuid.uuid4()) for _ in st.session_state.chunks]
                metadatas = [{"source": "uploaded_document"} for _ in st.session_state.chunks]
                add_to_chroma_collection("rag_collection", st.session_state.chunks, metadatas, ids)
                st.success("Data stored in data collection 'rag_collection'!")
            else:
                st.warning("Ensure document is uploaded, chunked, and embedded.")
        
        # --- Step 5A: Embed Query (Optional) ---
        st.subheader("Step 5A: Embed Query (Optional)")
        query_text = st.text_input("-> Enter a query to see how it is embedded into vectors", key="query_text_input")

        if st.button("Run Step 5A: Embed Query"):
            current_query = st.session_state.query_text_input
            if current_query.strip():
                query_data = embed_text([current_query], update_stage_flag=False, return_data=True)
                query_data['query'] = current_query
                update_stage('query', query_data)
                st.session_state.query_embedding = query_data["embeddings"]
                st.session_state.query_text_step5 = current_query
                st.success("Query vectorized!")
            else:
                st.warning("Please enter a query.")
        
        # --- Step 5B: Retrieve Query Embeddings (Optional) ---
        st.subheader("Step 5B: Retrieve Matching Chunks (Optional)")
        if st.button("Run Step 5B: Retrieve Matching Chunks"):
            if st.session_state.query_embedding:
                passages, metadata = query_collection(st.session_state.query_text_step5, "rag_collection", n_results=5)
                st.session_state.retrieved_passages = passages
                st.session_state.retrieved_metadata = metadata
                st.success("Relevant chunks retrieved based on your query in Step 5A!")
            else:
                st.warning("Run Step 5A (Embed Query) first.")
        
        # --- Step 6: Get Answer ---
        st.subheader("Step 6: Get Answer")
        final_question = st.text_input("-> Enter your final question for Step 6", key="final_question_input")

        if st.button("Run Step 6: Get Answer"):
            current_question = st.session_state.final_question_input
            if current_question.strip():
                passages, metadata = query_collection(current_question, "rag_collection", n_results=50)
                if passages:
                    answer = generate_answer_with_gpt(current_question, passages, metadata)
                    st.session_state.final_answer = answer
                    st.session_state.final_question_step7 = current_question
                    st.success("Answer generated!")
                    st.write(answer)
                else:
                    st.warning("Could not retrieve relevant passages for your question.")
            else:
                st.warning("Please enter your final question.")
    
    with col2:
        st.header("RAG Pipeline Visualization")
        component_args = {
            "currentStage": st.session_state.current_stage,
            "stageData": { s: st.session_state.get(f'{s}_data')
                           for s in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']
                           if st.session_state.get(f'{s}_data') is not None }
        }
        pipeline_html = get_pipeline_component(component_args)
        components.html(pipeline_html, height=2000, scrolling=True)


if __name__ == "__main__":
    main()
