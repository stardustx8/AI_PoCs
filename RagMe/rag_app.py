import streamlit as st
st.set_page_config(page_title="RAG Demo", layout="wide", initial_sidebar_state="expanded")


import os

# **Disable multi-tenancy for Chroma** (must be set before importing chromadb)
os.environ["CHROMADB_DISABLE_TENANCY"] = "true"

PROMPT_FILE = "custom_prompt.txt"
VOICE_PREF_FILE = "voice_pref.txt"

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
    "    7. Answer in English unless otherwise specified.\n\n"
    "    8. IMPORTANT: Do **not** produce XML tags in your final output. Present your answer in normal prose with "
    "       headings in large text as described.\n"
    "  </INSTRUCTIONS>\n\n"
    "  <STRUCTURE>\n"
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
from typing import List
import unicodedata
import requests
from openai import OpenAI
import shutil
import hashlib
import os
from typing import Optional
from pathlib import Path


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
    st.session_state.current_stage = stage
    if data is not None:
        if stage == 'embed' and isinstance(data, dict):
            enhanced_data = data
        else:
            enhanced_data = data.copy() if isinstance(data, dict) else {'data': data}
        if stage == 'upload':
            text = data.get('content', '') if isinstance(data, dict) else data
            enhanced_data['preview'] = text[:600] if text else None
            enhanced_data['full'] = text
        elif stage == 'chunk':
            enhanced_data = {'chunks': data[:5], 'total_chunks': len(data)}
        elif stage == 'store':
            enhanced_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        elif stage == 'query':
            enhanced_data = data.copy() if isinstance(data, dict) else {'query': data}
        elif stage == 'retrieve':
            if isinstance(data, dict):
                enhanced_data = {
                    'passages': data.get("passages"),
                    'scores': [0.95, 0.87, 0.82],
                    'metadata': data.get("metadata")
                }
            else:
                enhanced_data = {
                    'passages': data[0],
                    'scores': [0.95, 0.87, 0.82],
                    'metadata': data[1]
                }
        st.session_state[f'{stage}_data'] = enhanced_data
        if 'rag_state' in st.session_state:
            st.session_state.rag_state.set_stage(stage, enhanced_data)


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


def split_text_into_chunks(text: str) -> List[str]:
    update_stage('upload', {'content': text, 'size': len(text)})
    chunks = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    update_stage('chunk', chunks)
    return chunks


def embed_text(
    texts: List[str],
    openai_embedding_model: str = "text-embedding-3-large",
    update_stage_flag=True,
    return_data=False
):
    if not st.session_state.get("api_key"):
        st.error("OpenAI API key not set. Provide it in the sidebar.")
        st.stop()
    safe_texts = [sanitize_text(s) for s in texts]
    response = new_client.embeddings.create(input=safe_texts, model=openai_embedding_model)
    embeddings = [item.embedding for item in response.data]
    token_breakdowns = []
    for text, embedding in zip(safe_texts, embeddings):
        tokens = text.split()
        breakdown = []
        if tokens:
            dims_per_token = len(embedding) // len(tokens)
            for i, tok in enumerate(tokens):
                start = i * dims_per_token
                end = start + dims_per_token if i < len(tokens) - 1 else len(embedding)
                snippet = embedding[start:end]
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
            description: "<strong>Cutting the Text into Slices</strong><br>Once you're ready, each uploaded text is broken into manageable chunks. This segmentation helps the system handle longer documents more effectively while preserving meaning within each slice.",
            summaryDataExplanation: (data) => `
<strong>Chunk Breakdown (Summary):</strong><br>
Total Chunks: ${data.total_chunks}<br>
Sample Chunks: ${ (data.chunks || []).map((chunk, i) => `<br><span style="color:red;font-weight:bold;">Chunk ${i + 1}:</span> "${chunk}"`).join('') }
            `.trim(),
            dataExplanation: (data) => `
<strong>Chunk Breakdown (Expanded):</strong><br>
Total Chunks: ${data.total_chunks}<br>
All Chunks:<br>
${ (data.full_chunks || data.chunks || []).map((chunk, i) => `<br><span style="color:red;font-weight:bold;">Chunk ${i + 1}:</span> "${chunk}"`).join('') }
            `.trim()
        },
        embed: {
            title: "Step 3: Vector Embedding Generation",
            icon: '🧠',
            description: "<strong>Transforming Chunks into High-Dimensional Vectors</strong><br>Each chunk is converted into a multi-thousand-dimensional vector. Even a single sentence can map into thousands of numeric features! Why? Because language is highly nuanced, and each dimension captures subtle shades of meaning, syntax, or context. We can visualize these embeddings (imagine a giant 3D cloud of points) where similar tokens or phrases cluster together—like red 'Hello, AI!' tokens standing out among greyer neighbors.",
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
Each chunk is converted into a multi-thousand-dimensional vector. Even a single sentence can map into thousands of numeric features! Why? Because language is highly nuanced, and each dimension captures subtle shades of meaning, syntax, or context.<br><br>\
For example, consider the word <strong>Switzerland</strong>. It might appear as a 3,000-dimensional vector like [0.642, -0.128, 0.945, ...]. In this snippet, <em>dimension 1</em> (0.642) may reflect geography (mountains, lakes), <em>dimension 2</em> (-0.128) might capture linguistic influences, and <em>dimension 3</em> (0.945) could encode economic traits—such as stability or robust banking. A higher value (e.g., 0.945) indicates a stronger correlation with that dimension's learned feature (in this case, 'economic stability'), whereas lower or negative values signal weaker or contrasting associations. Across thousands of dimensions, these numeric signals combine into a richly layered portrait of meaning.",
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


def add_to_chroma_collection(collection_name: str, chunks: List[str], metadatas: List[dict], ids: List[str]):
    force_flag = st.session_state.get("force_recreate", False)
    coll = create_or_load_collection(collection_name, force_recreate=force_flag)
    coll.add(documents=chunks, metadatas=metadatas, ids=ids)
    chroma_client.persist()
    update_stage('store', {'collection': collection_name, 'count': len(chunks)})
    st.write(f"Stored {len(chunks)} chunks in collection '{collection_name}'.")


def query_collection(query: str, collection_name: str, n_results: int = 3):
    force_flag = st.session_state.get("force_recreate", False)
    coll = create_or_load_collection(collection_name, force_recreate=force_flag)
    coll_info = coll.get()
    doc_count = len(coll_info.get("ids", []))
    
    st.write(f"Querying collection '{collection_name}' which has {doc_count} documents.")
    if doc_count == 0:
        st.warning("No documents found in collection. Please upload first.")
        return [], []
        
    results = coll.query(query_texts=[query], n_results=n_results)
    retrieved_passages = results.get("documents", [[]])[0]
    retrieved_metadata = results.get("metadatas", [[]])[0]
    
    update_stage('retrieve', {"passages": retrieved_passages, "metadata": retrieved_metadata})
    st.write(f"Retrieved {len(retrieved_passages)} passages from collection '{collection_name}'.")
    return retrieved_passages, retrieved_metadata


##############################################################################
# 7) GPT ANSWER GENERATION
##############################################################################
def generate_answer_with_gpt(query: str, retrieved_passages: List[str], retrieved_metadata: List[dict],
                             system_instruction: str = None):
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
    
    completion = new_client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[{"role": "user", "content": final_prompt}],
        response_format={"type": "text"}
    )
    answer = completion.choices[0].message.content
    update_stage('generate', {'answer': answer})
    return answer


def summarize_context(passages: list[str]) -> str:
    combined = "\n".join(passages)
    return f"Summary of your documents:\n{combined}"


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


##############################################################################
# 9) DELETE USERS
##############################################################################

def delete_user(user_id: str):
    import shutil
    from pathlib import Path

    users = load_users()
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
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)
        if chroma_client is None:
            st.error("ChromaDB client not initialized properly")
            st.stop()
    
    # "Delete All Collections" button in sidebar
    if st.sidebar.button("Delete All Collections"):
        try:
            collections = chroma_client.list_collections()
            for collection in collections:
                chroma_client.delete_collection(name=collection.name)
            if os.path.exists(CHROMA_DIRECTORY):
                import shutil
                shutil.rmtree(CHROMA_DIRECTORY)
                os.makedirs(CHROMA_DIRECTORY, exist_ok=True)
            st.sidebar.success("All Chroma collections and storage files deleted!")
        except Exception as e:
            st.sidebar.error(f"Error deleting collections: {e}")

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
            st.sidebar.error("Could not start AVM. Check error messages at the top of the main section >>> >>> >>>")

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
        reverse_text_order = st.sidebar.checkbox("Reverse extracted text order", value=True)

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
                    text = extract_text_from_file(uploaded_file, reverse=reverse_text_order)
                    if text:
                        combined_text += f"\n\n---\n\n{text}"
                if combined_text:
                    st.session_state.uploaded_text = combined_text
                    update_stage('upload', {'content': combined_text, 'size': len(combined_text)})
                    st.success("Files uploaded and processed!")
                else:
                    st.error("No text could be extracted from the uploaded files.")
            else:
                st.warning("No file selected.")

        # --- Step 2: Chunk Context ---
        st.subheader("Step 2: Chunk Context")
        if st.button("Run Step 2: Chunk Context"):
            if st.session_state.uploaded_text:
                chunks = split_text_into_chunks(st.session_state.uploaded_text)
                st.session_state.chunks = chunks
                st.success(f"Text chunked into {len(chunks)} segments.")
            else:
                st.warning("Please upload min. 1 document first.")
        
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
                passages, metadata = query_collection(st.session_state.query_text_step5, "rag_collection")
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
                passages, metadata = query_collection(current_question, "rag_collection")
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
