import os

# **Disable multi-tenancy for Chroma** (must be set before importing chromadb)
os.environ["CHROMADB_DISABLE_TENANCY"] = "true"

import chromadb
from chromadb.config import Settings
import streamlit as st
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
    
#######################################################################
# 1) GLOBALS & CLIENT INITIALIZATION
#######################################################################
new_client = None  # Set once the user provides an API key
chroma_client = None  # Global Chroma client
embedding_function_instance = None  # Global instance of our embedding function

CHROMA_DIRECTORY = "chromadb_storage"

# Initialize ChromaDB with our custom embedding function instance
def init_chroma_client():
    if "api_key" not in st.session_state:
        return None, None
    global embedding_function_instance
    embedding_function_instance = OpenAIEmbeddingFunction(st.session_state["api_key"])
    client = chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_DIRECTORY
        )
    )
    return client, embedding_function_instance

#######################################################################
# 2) SESSION STATE INIT
#######################################################################
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
# Use default collection name
st.session_state.collection_name = "rag_collection"
if 'delete_confirm' not in st.session_state:
    st.session_state.delete_confirm = False
if 'avm_active' not in st.session_state:
    st.session_state.avm_active = False
if 'voice_html' not in st.session_state:
    st.session_state.voice_html = None

#######################################################################
# 3) RAG STATE CLASS
#######################################################################
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

#######################################################################
# 4) PIPELINE STAGE HELPER FUNCTIONS
#######################################################################
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
                enhanced_data = {'passages': data.get("passages"), 'scores': [0.95, 0.87, 0.82],
                                 'metadata': data.get("metadata")}
            else:
                enhanced_data = {'passages': data[0], 'scores': [0.95, 0.87, 0.82],
                                 'metadata': data[1]}
        st.session_state[f'{stage}_data'] = enhanced_data
        if 'rag_state' in st.session_state:
            st.session_state.rag_state.set_stage(stage, enhanced_data)

def set_openai_api_key(api_key: str):
    global new_client, chroma_client, embedding_function_instance
    new_client = OpenAI(api_key=api_key)
    st.session_state["api_key"] = api_key
    
    # Initialize ChromaDB with our embedding function
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
    embedding_data = {"embeddings": embeddings,
                      "dimensions": len(embeddings[0]) if embeddings else 0,
                      "preview": embeddings[0][:10] if embeddings else [],
                      "total_vectors": len(embeddings),
                      "token_breakdowns": token_breakdowns}
    if update_stage_flag:
        update_stage('embed', embedding_data)
    if return_data:
        return embedding_data
    return embeddings

def extract_text_from_file(uploaded_file) -> str:
    """
    Detects the file type from its extension and extracts text accordingly.
    """
    file_name = uploaded_file.name.lower()
    
    if file_name.endswith('.txt'):
        # Text file: decode as UTF-8
        text = uploaded_file.read().decode("utf-8")
        
    elif file_name.endswith('.pdf'):
        # PDF file: use PyPDF2 to extract text from each page
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
    elif file_name.endswith('.csv'):
        # CSV file: use pandas to read and convert to string
        try:
            df = pd.read_csv(uploaded_file)
            text = df.to_csv(index=False)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            text = ""
    
    elif file_name.endswith('.xlsx'):
        # Excel file: use pandas to read and convert to CSV string
        try:
            df = pd.read_excel(uploaded_file)
            text = df.to_csv(index=False)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            text = ""
    
    elif file_name.endswith('.docx'):
        # DOCX file: use python-docx to extract text from paragraphs
        try:
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error reading DOCX file: {e}")
            text = ""
    
    elif file_name.endswith('.rtf'):
        # RTF file: use striprtf to convert RTF to plain text
        try:
            # Read the RTF file as a string. Adjust encoding if necessary.
            file_contents = uploaded_file.read().decode("utf-8", errors="ignore")
            text = rtf_to_text(file_contents)
        except Exception as e:
            st.error(f"Error reading RTF file: {e}")
            text = ""
    
    else:
        st.warning("Unsupported file type.")
        text = ""
    
    return text

#######################################################################
# 5) FULL REACT PIPELINE SNIPPET (WORKING VERSION)
#######################################################################
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
            title: "Document Upload & Processing",
            icon: '📁',
            description: "<strong>Step 1: Gather Your Raw Material</strong><br>We begin by taking the text exactly as you provided and processing it.",
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
            title: "Text Chunking",
            icon: '✂️',
            description: "<strong>Step 2: Slicing Content</strong><br>Your text is divided into segments.",
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
            title: "Vector Embedding Generation",
            icon: '🧠',
            description: "<strong>Step 3: Embedding Generation</strong><br>Each chunk is converted into a vector.",
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
            title: "Vector Database Storage",
            icon: '🗄️',
            description: "<strong>Step 4: Storage</strong><br>Embeddings are stored in the database.",
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
            title: "Query Collection",
            icon: '❓',
            description: "<strong>Step 5: Query Collection</strong><br>Your query is embedded into a vector.",
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
            title: "Context Retrieval",
            icon: '🔎',
            description: "<strong>Step 6: Retrieve Chunks</strong><br>We retrieve the most similar chunks.",
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
            title: "Get Answer",
            icon: '🤖',
            description: "<strong>Step 7: Get Answer</strong><br>Your final question and retrieved chunks generate an answer.",
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
                const container = document.querySelector('.pipeline-column');
                if (activeElem && container) {
                    const headerOffset = 100; // Adjust based on your header height
                    const elemTop = activeElem.offsetTop - headerOffset;
                    
                    // Find the scrollable container (pipeline-container)
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
        const stageData = args.stageData || {};
        
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
        position: static;  /* Changed from fixed */
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
        min-height: 100vh;  /* Changed */
        padding-bottom: 50vh;  /* Add this to allow scrolling past the last element */
    }

    .modal-content {
        max-height: none;  /* Add this if there are any modal height restrictions */
        height: auto;      /* Add this */
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

#######################################################################
# 6) CREATE/LOAD COLLECTION, ETC.
#######################################################################
def create_or_load_collection(collection_name: str, force_recreate: bool = False):
    """
    Retrieve the collection with the specified name using our global
    embedding_function_instance. If force_recreate is True, delete the
    existing collection and create a new one.
    """
    global chroma_client, embedding_function_instance
    if embedding_function_instance is None:
        st.error("Embedding function not initialized. Please set your OpenAI API key.")
        st.stop()
    
    # st.write(f"Attempting to load collection '{collection_name}'...")
    
    if force_recreate:
        try:
            chroma_client.delete_collection(name=collection_name)
            st.write(f"Deleted existing collection '{collection_name}' due to force_recreate flag.")
        except Exception as e:
            st.write(f"Could not delete existing collection '{collection_name}': {e}")
    
    try:
        # When a collection already exists, ChromaDB will not update its stored embedding function.
        coll = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function_instance)
        coll_info = coll.get()
        # st.write(f"Collection '{collection_name}' found with {len(coll_info.get('ids', []))} documents.")
    except Exception as e:
        # st.write(f"Collection '{collection_name}' not found or error encountered: {e}. Creating a new collection.")
        coll = chroma_client.create_collection(name=collection_name, embedding_function=embedding_function_instance)
    return coll

def add_to_chroma_collection(collection_name: str, chunks: List[str], metadatas: List[dict], ids: List[str]):
    # Read the force flag from session_state (set via a sidebar checkbox)
    force_flag = st.session_state.get("force_recreate", False)
    coll = create_or_load_collection(collection_name, force_recreate=force_flag)
    # Use the custom embedding function to add documents.
    coll.add(documents=chunks, metadatas=metadatas, ids=ids)
    chroma_client.persist()
    update_stage('store', {'collection': collection_name, 'count': len(chunks)})
    st.write(f"Stored {len(chunks)} chunks in collection '{collection_name}'.")

def query_collection(query: str, collection_name: str, n_results: int = 3):
    # Again, use the force flag when loading the collection.
    force_flag = st.session_state.get("force_recreate", False)
    coll = create_or_load_collection(collection_name, force_recreate=force_flag)
    coll_info = coll.get()
    doc_count = len(coll_info.get("ids", []))
    
    st.write(f"Querying collection '{collection_name}' which has {doc_count} documents.")
    
    if doc_count == 0:
        st.warning("No documents found in collection. Please upload first.")
        return [], []
        
    # Query using the collection's embedded data.
    results = coll.query(query_texts=[query], n_results=n_results)
    retrieved_passages = results.get("documents", [[]])[0]
    retrieved_metadata = results.get("metadatas", [[]])[0]
    
    update_stage('retrieve', {"passages": retrieved_passages, "metadata": retrieved_metadata})
    st.write(f"Retrieved {len(retrieved_passages)} passages from collection '{collection_name}'.")
    return retrieved_passages, retrieved_metadata

def generate_answer_with_gpt(query: str, retrieved_passages: List[str], retrieved_metadata: List[dict],
                             system_instruction: str = (
                                "You are a helpful legal assistant. Answer the following query based ONLY on the provided context (the RAG regulation document). "
                                "Your answer must begin with a high level concise instructions to action if the user asked for help. Then output a concise TL;DR summary in bullet points, followed by a Detailed Explanation - all 3 sections drawing strictly from the RAG regulation document! "
                                "After the Detailed Explanation, include a new section titled 'Other references' where you may add any further relevant insights or clarifications from your own prior knowledge, "
                                "but clearly label them as separate from the doc-based content; make them bulletized, starting with the paragraphs, then prose why relevant etc.."
                                "\n\n"
                                "Be sure to:\n"
                                "1. Use bold to highlight crucial numeric thresholds, legal terms, or statutory references on first mention.\n"
                                "2. Use italics for emphasis or important nuances.\n"
                                "3. Maintain a clear, layered structure: \n"
                                "   - High-level, concise instructions to action in the user's case if the user asked for help. VERY IMPORTANT: no vague instructions, no assumptions but directly executable, deterministic guidance (ex. 'if the knife is > than 5cm x is allowed, y is prohibited') based purely on the provided document!\n"
                                "   - TL;DR summary (bullet points, doc-based only); VERY IMPORTANT: the TL;DR must only contain references to resp. be only based on the provided document, don't introduce other legal frameworks here.\n"
                                "   - Detailed Explanation (doc-based only)\n"
                                "   - Other references (your additional knowledge or commentary); VERY IMPORTANT please add explicit statutory references here (and only here), you can write all pertinent references in ""[]"".\n"
                                "4. In 'Other references,' feel free to elaborate or cite external knowledge, disclaimers, or expansions, but explicitly note this section is beyond the doc.\n"
                                "5. Refrain from using any info that is not in the doc within the TL;DR or Detailed Explanation sections.\n"
                                "6. Answer succinctly and accurately, focusing on the question asked.\n"
                                "7. Where relevant, include a *short example scenario* within the Detailed Explanation to illustrate how the doc-based rules might apply practically (e.g., carrying a **10 cm** folding knife in everyday settings).\n"
                                "8. Ensure that in the TL;DR, key numeric thresholds and terms defined by the doc are **bolded**, and consider briefly referencing what constitutes a 'weapon' under the doc’s classification criteria."
                            )):
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
    # Combine all passages without slicing
    combined = "\n".join(passages)
    # Return the complete text without truncation
    return f"Summary of your documents:\n{combined}"

#######################################################################
# 7) REALTIME VOICE MODE
#######################################################################
def get_ephemeral_token(collection_name: str = "rag_collection"):
    if "api_key" not in st.session_state:
        st.error("OpenAI API key not set.")
        return None
    url = "https://api.openai.com/v1/realtime/sessions"
    headers = {
        "Authorization": f"Bearer {st.session_state['api_key']}",
        "Content-Type": "application/json"
    }
    data = {"model": "gpt-4o-realtime-preview-2024-12-17", "voice": "coral"} #  `alloy`, `ash`, `ballad`, `coral`, `echo` `sage`, `shimmer`, `verse`
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
        st.error(f"Failed to create realtime session: {str(e)}")
        if hasattr(e.response, 'text'):
            st.write("Error response:", e.response.text)
        return None

def get_realtime_html(token_data: dict) -> str:
    # Retrieve the collection and all documents.
    coll = create_or_load_collection(token_data['collection'])
    all_docs = coll.get()
    all_passages = all_docs.get("documents", [])
    
    # Compute the document summary (this now returns the full text).
    doc_summary = summarize_context(all_passages)
    
    # Define a custom prompt prefix.
    prompt_prefix = (
        "You are a helpful legal assistant. Answer the following query based ONLY on the provided context (the RAG regulation document). \n"
        "Your answer must begin with a high level concise instructions to action if the user asked for help. Then output a concise TL;DR summary in bullet points, followed by a Detailed Explanation - all 3 sections drawing strictly from the RAG regulation document! \n"
        "After the Detailed Explanation, include a new section titled 'Other references' where you may add any further relevant insights or clarifications from your own prior knowledge, \n"
        "but clearly label them as separate from the doc-based content; make them bulletized, starting with the paragraphs, then prose why relevant etc.. \n"
        "\n\n"
        "Be sure to:\n"
        "1. Use bold to highlight crucial numeric thresholds, legal terms, or statutory references on first mention.\n"
        "2. Use italics for emphasis or important nuances.\n"
        "3. Maintain a clear, layered structure: \n"
        "   - High-level, concise instructions to action in the user's case if the user asked for help. VERY IMPORTANT: no vague instructions, no assumptions but directly executable, deterministic guidance (ex. 'if the knife is > than 5cm x is allowed, y is prohibited') based purely on the provided document!\n"
        "   - TL;DR summary (bullet points, doc-based only); VERY IMPORTANT: the TL;DR must only contain references to resp. be only based on the provided document, don't introduce other legal frameworks here.\n"
        "   - Detailed Explanation (doc-based only)\n"
        "   - Other references (your additional knowledge or commentary); VERY IMPORTANT please add explicit statutory references here (and only here), you can write all pertinent references in \"[]\".\n"
        "4. In 'Other references,' feel free to elaborate or cite external knowledge, disclaimers, or expansions, but explicitly note this section is beyond the doc.\n"
        "5. Refrain from using any info that is not in the doc within the TL;DR or Detailed Explanation sections.\n"
        "6. Answer succinctly and accurately, focusing on the question asked.\n"
        "7. Where relevant, include a *short example scenario* within the Detailed Explanation to illustrate how the doc-based rules might apply practically (e.g., carrying a **10 cm** folding knife in everyday settings).\n"
        "8. Ensure that in the TL;DR, key numeric thresholds and terms defined by the doc are **bolded**, and consider briefly referencing what constitutes a 'weapon' under the doc’s classification criteria.\n"
    )
    # Concatenate the prompt prefix with the document summary.
    full_prompt = prompt_prefix + doc_summary
    
    # **Store the full prompt in session state** so you can inspect it.
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
            // **Send the complete prompt (prompt prefix + context) to AVM**
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
        /* CSS styles remain unchanged */
        ::-webkit-scrollbar {{ width: 0px; background: transparent; }}
        body {{ background-color: #111; color: #fff; margin: 0; padding: 0; }}
        #realtime-status {{ font-family: system-ui, sans-serif; }}
        /* ... additional CSS ... */
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

#######################################################################
# 8) MAIN STREAMLIT APP
#######################################################################
def main():
    global chroma_client  # declare global so we can reassign it later if needed
    st.set_page_config(page_title="RAG Demo", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stApp { background-color: #111; color: white; }
        [data-testid="column"] { width: calc(100% + 2rem); margin-left: -1rem; }
        </style>
    """, unsafe_allow_html=True)
    st.title("RAG + Realtime Voice Demo")
    
    # Sidebar: API key and force recreate option
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)
        if chroma_client is None:
            st.error("ChromaDB client not initialized properly")
            st.stop()
    
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
            st.sidebar.error("Could not start AVM. Check error messages above.")

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
    
    # Main layout: Use three columns (col1, spacer, col2) for extra spacing
    col1, spacer, col2 = st.columns([1.3, 0.3, 2])
    
    with col1:
        st.header("Step-by-Step Pipeline Control")
    
        # Step 1: Upload
        st.subheader("Step 1: Upload")
        uploaded_files = st.file_uploader(
            "Upload one or more documents (txt, pdf, docx, csv, xlsx, rtf)",
            type=["txt", "pdf", "docx", "csv", "xlsx", "rtf"],
            accept_multiple_files=True
        )

        if st.button("Run Step 1: Upload"):
            if uploaded_files:
                combined_text = ""
                for uploaded_file in uploaded_files:
                    text = extract_text_from_file(uploaded_file)
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
        
        # Step 2: Chunk
        st.subheader("Step 2: Chunk")
        if st.button("Run Step 2: Chunk"):
            if st.session_state.uploaded_text:
                chunks = split_text_into_chunks(st.session_state.uploaded_text)
                st.session_state.chunks = chunks
                st.success(f"Text chunked into {len(chunks)} segments.")
            else:
                st.warning("Please upload a document first.")
        
        # Step 3: Embed
        st.subheader("Step 3: Embed")
        if st.button("Run Step 3: Embed"):
            if st.session_state.chunks:
                embedding_data = embed_text(st.session_state.chunks, update_stage_flag=True, return_data=True)
                st.session_state.embeddings = embedding_data
                st.success("Embeddings created!")
            else:
                st.warning("Please chunk the document first.")
        
        # Step 4: Store
        st.subheader("Step 4: Store")
        if st.button("Run Step 4: Store"):
            if st.session_state.chunks and st.session_state.embeddings:
                ids = [str(uuid.uuid4()) for _ in st.session_state.chunks]
                metadatas = [{"source": "uploaded_document"} for _ in st.session_state.chunks]
                add_to_chroma_collection("rag_collection", st.session_state.chunks, metadatas, ids)
                st.success("Data stored in collection 'rag_collection'!")
            else:
                st.warning("Ensure document is uploaded, chunked, and embedded.")
        
        # Step 5: Query Collection
        st.subheader("Step 5: Query Collection")

        # Use a unique key so that the text input's value persists independently.
        query_text = st.text_input("Enter your query for Step 5", key="query_text_input")

        if st.button("Run Step 5: Query Collection"):
            # Retrieve the current query value from session_state using the unique key.
            current_query = st.session_state.query_text_input
            
            if current_query.strip():
                # Embed the query and update the stage
                query_data = embed_text([current_query], update_stage_flag=False, return_data=True)
                query_data['query'] = current_query
                update_stage('query', query_data)
                st.session_state.query_embedding = query_data["embeddings"]
                st.session_state.query_text_step5 = current_query  # Optionally persist the current query
                st.success("Query vectorized!")
            else:
                st.warning("Please enter a query.")
        
        # Step 6: Retrieve
        st.subheader("Step 6: Retrieve")
        if st.button("Run Step 6: Retrieve"):
            if st.session_state.query_embedding:
                passages, metadata = query_collection(st.session_state.query_text_step5, "rag_collection")
                st.session_state.retrieved_passages = passages
                st.session_state.retrieved_metadata = metadata
                st.success("Relevant passages retrieved!")
            else:
                st.warning("Run Step 5 (Query Collection) first.")
        
        # Step 7: Get Answer
        st.subheader("Step 7: Get Answer")

        # Use a unique key for the text input so that its value persists independently.
        final_question = st.text_input("Enter your final question for Step 7", key="final_question_input")

        if st.button("Run Step 7: Get Answer"):
            # Retrieve the current value from session state using the unique key.
            current_question = st.session_state.final_question_input
            
            if current_question.strip():
                # Perform query and generation using the current question.
                passages, metadata = query_collection(current_question, "rag_collection")
                if passages:
                    answer = generate_answer_with_gpt(current_question, passages, metadata)
                    st.session_state.final_answer = answer
                    st.session_state.final_question_step7 = current_question  # Optional: persist the last used question.
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