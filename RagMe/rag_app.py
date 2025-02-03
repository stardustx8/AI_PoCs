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

#######################################################################
# 1) GLOBALS & CLIENT INITIALIZATION
#######################################################################
new_client = None  # We'll set an OpenAI client once the user provides an API key
CHROMA_DIRECTORY = "chromadb_storage"
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_DIRECTORY
    )
)

#######################################################################
# 2) SESSION STATE INIT
#######################################################################
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
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "demo_collection"
if 'delete_confirm' not in st.session_state:
    st.session_state.delete_confirm = False

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
    global new_client
    new_client = OpenAI(api_key=api_key)
    st.session_state["api_key"] = api_key

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
            description: "<strong>Step 1: Gather Your Raw Material</strong><br>We begin by taking the text exactly as you provided, pulling it into our pipeline, and giving it a brief once-over. It’s the essential first step in transforming your uploaded content into something we can query with intelligence.",
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
            description: "<strong>Step 2: Slicing Content into Digestible Bits</strong><br>To make it easier for our system to understand your data, we slice your text into smaller, self-contained segments.",
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
            description: "<strong>Step 3: Translating Each Chunk into Numbers</strong><br>Now we transform every chunk into a numeric representation known as an embedding—this is where meaning meets math.",
            summaryDataExplanation: (data) => `
<strong>Embedding Stats (Summary):</strong><br>
Each segment is represented by a ${data.dimensions}-dimensional vector.<br>
Total Embeddings: ${data.total_vectors}<br>
Sample Token Breakdown (first 3 chunks): ${ data.token_breakdowns.slice(0,3).map((chunkBreakdown, idx) => `
<br><strong>Chunk ${idx + 1}:</strong>
${ chunkBreakdown.map(item => `<br><span style="color:red;font-weight:bold;">${item.token}</span>: [${item.vector_snippet.join(", ")}]` ).join("") }
`).join("") }
            `.trim(),
            dataExplanation: (data) => `
<strong>Embedding Stats (Expanded):</strong><br>
Each segment is represented by a ${data.dimensions}-dimensional vector.<br>
Total Embeddings: ${data.total_vectors}<br>
<strong>Sample Vector Snippet (first 10 dims of the first embedding):</strong><br>
${ data.preview.map((val, i) => `dim${i+1}: ${val.toFixed(6)}`).join("<br>") }<br><br>
<strong>Full Token Breakdown:</strong>
${ data.token_breakdowns.map((chunkBreakdown, idx) => `
<br><strong>Chunk ${idx + 1}:</strong>
${ chunkBreakdown.map(item => `<br><span style="color:red;font-weight:bold;">${item.token}</span>: [${item.vector_snippet.map(v => v.toFixed(6)).join(", ")}]` ).join("") }
`).join("") }
            `.trim()
        },
        store: {
            title: "Vector Database Storage",
            icon: '🗄️',
            description: "<strong>Step 4: Storing Embeddings</strong><br>All embeddings are stored in a specialized vector database (ChromaDB), so we can rapidly find the best matches later.",
            summaryDataExplanation: (data) => `
<strong>Storage Summary:</strong><br>
Stored ${data.count} chunks in collection "${data.collection}".
            `.trim(),
            dataExplanation: (data) => `
<strong>Storage Details (Expanded):</strong><br>
Stored ${data.count} chunks in collection "${data.collection}" at ${data.timestamp}.
            `.trim()
        },
        query: {
            title: "Query Collection",
            icon: '❓',
            description: "<strong>Step 5: Query Collection</strong><br>Your query is embedded into a vector.",
            summaryDataExplanation: (data) => `
<strong>Query Vectorization:</strong><br>
Original Query: <span style="color:red;font-weight:bold;">"${data.query || 'N/A'}"</span><br>
Each query is represented by a ${data.dimensions}-dimensional vector.<br>
Total Vectors: ${data.total_vectors}<br>
<strong>Sample Vector Snippet (first 10 dims):</strong><br>
${ data.preview ? data.preview.map((val, i) => `dim${i+1}: ${val.toFixed(6)}`).join("<br>") : "N/A" }
            `.trim(),
            dataExplanation: (data) => `
<strong>Query Vectorization (Expanded):</strong><br>
Original Query: <span style="color:red;font-weight:bold;">"${data.query || 'N/A'}"</span><br>
Each query is represented by a ${data.dimensions}-dimensional vector.<br>
Total Vectors: ${data.total_vectors}<br>
<strong>Sample Vector Snippet (first 10 dims):</strong><br>
${ data.preview ? data.preview.map((val, i) => `dim${i+1}: ${val.toFixed(6)}`).join("<br>") : "N/A" }<br><br>
<strong>Full Token Breakdown:</strong>
${ data.token_breakdowns ? data.token_breakdowns.map((chunk) => {
    return chunk.map(item => `<br><span style="color:red;font-weight:bold;">${item.token}</span>: [${item.vector_snippet.map(v => v.toFixed(6)).join(", ")}]`).join("");
}).join("") : "N/A" }
            `.trim()
        },
        retrieve: {
            title: "Context Retrieval",
            icon: '🔎',
            description: "<strong>Step 6: Retrieve Relevant Chunks</strong><br>We retrieve the most similar chunks.",
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
            description: "<strong>Step 7: Get Answer</strong><br>We feed your final question and the retrieved chunks into GPT to generate an answer.",
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
            const activeElem = document.querySelector('.active-stage');
            if (activeElem) {
                activeElem.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }, [activeStage]);
        
        const formatModalContent = (stage) => {
            const data = args.stageData[stage];
            if (!data) return 'No data available for this stage.';
            const process = ProcessExplanation[stage];
            return React.createElement('div', { className: 'modal-content' }, [
                React.createElement('button', { className: 'close-button', onClick: () => setShowModal(false) }, '×'),
                React.createElement('h2', { className: 'modal-title' }, [ process.icon, ' ', process.title ]),
                React.createElement('p', { className: 'modal-description', dangerouslySetInnerHTML: { __html: process.description } }),
                React.createElement('div', { className: 'modal-data', dangerouslySetInnerHTML: { __html: process.dataExplanation(data) } })
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
  body { 
      background-color: #111; 
      color: #fff; 
      margin: 0; 
      padding: 0; 
  }
  #rag-root { 
      font-family: system-ui, sans-serif; 
      height: 100%; 
      width: 100%; 
      margin: 0; 
      padding: 0; 
  }
  .pipeline-container { 
      padding: 1rem 10rem 1rem 1rem; 
      overflow-y: auto; 
      overflow-x: visible; 
      height: 100vh; 
      box-sizing: border-box; 
      width: 100%; 
  }
  .pipeline-column { 
      display: flex; 
      flex-direction: column; 
      align-items: stretch; 
      width: 100%; 
      margin: 0 auto; 
      padding-right: 10rem; 
      overflow-x: visible; 
  }
  .pipeline-box { 
      width: 100%; 
      margin-bottom: 1rem; 
      padding: 1.5rem; 
      border: 2px solid #4B5563; 
      border-radius: 0.75rem; 
      background-color: #1a1a1a; 
      cursor: pointer; 
      transition: all 0.3s; 
      text-align: left; 
      transform-origin: center; 
      position: relative; 
      z-index: 1; 
  }
  .pipeline-box:hover { 
      transform: scale(1.02); 
      border-color: #6B7280; 
      z-index: 1000; 
  }
  .completed-stage { 
      background-color: rgba(34, 197, 94, 0.1); 
      border-color: #22C55E; 
  }
  .active-stage { 
      border-color: #22C55E; 
      box-shadow: 0 0 15px rgba(34, 197, 94, 0.2); 
  }
  .stage-header { 
      display: flex; 
      align-items: center; 
      gap: 0.75rem; 
      margin-bottom: 0.75rem; 
  }
  .stage-icon { 
      font-size: 1.5rem; 
  }
  .stage-title { 
      font-weight: bold; 
      font-size: 1.2rem; 
      color: white; 
  }
  .stage-description { 
      color: #9CA3AF; 
      font-size: 1rem; 
      margin-bottom: 1rem; 
      line-height: 1.5; 
      text-align: left; 
  }
  .stage-data { 
      font-family: monospace; 
      font-size: 0.9rem; 
      color: #D1D5DB; 
      background-color: rgba(0, 0, 0, 0.2); 
      padding: 0.75rem; 
      border-radius: 0.5rem; 
      margin-top: 0.75rem; 
      white-space: pre-wrap; 
      text-align: left; 
  }
  .pipeline-arrow { 
      height: 40px; 
      margin: 0.5rem 0; 
      display: flex; 
      align-items: center; 
      justify-content: center; 
      position: relative; 
  }
  .arrow-body { 
      width: 3px; 
      height: 100%; 
      background: linear-gradient(to bottom, rgba(156,163,175,0) 0%, rgba(156,163,175,1) 30%, rgba(156,163,175,1) 70%, rgba(156,163,175,0) 100%); 
      position: relative; 
  }
  .arrow-body::after { 
      content: ''; 
      position: absolute; 
      bottom: 30%; 
      left: 50%; 
      transform: translateX(-50%); 
      width: 0; 
      height: 0; 
      border-left: 8px solid transparent; 
      border-right: 8px solid transparent; 
      border-top: 12px solid #9CA3AF; 
  }
  .tooltip-modal { 
      position: fixed; 
      top: 0; 
      left: 0; 
      width: 100%; 
      height: 100%; 
      background-color: rgba(0,0,0,0.85); 
      display: flex; 
      align-items: center; 
      justify-content: center; 
      z-index: 9999; 
  }
  .tooltip-content { 
      position: relative; 
      width: 95%; 
      height: 95%; 
      background: #1a1a1a; 
      padding: 2rem; 
      border-radius: 1rem; 
      overflow-y: auto; 
      box-shadow: 0 0 30px rgba(0,0,0,0.5); 
      color: #fff; 
  }
  .tooltip-content::-webkit-scrollbar { 
      width: 8px; 
      height: 8px; 
  }
  .tooltip-content::-webkit-scrollbar-track { 
      background: #333; 
      border-radius: 4px; 
  }
  .tooltip-content::-webkit-scrollbar-thumb { 
      background: #666; 
      border-radius: 4px; 
  }
  .tooltip-content::-webkit-scrollbar-thumb:hover { 
      background: #888; 
  }
  .close-button { 
      position: absolute; 
      top: 20px; 
      right: 20px; 
      background: transparent; 
      border: none; 
      font-size: 2rem; 
      font-weight: bold; 
      color: #fff; 
      cursor: pointer; 
  }
  .modal-title { 
      font-size: 1.5rem; 
      font-weight: bold; 
      margin-bottom: 1rem; 
      color: white; 
  }
  .modal-description { 
      color: #9CA3AF; 
      font-size: 1.1rem; 
      margin-bottom: 1.5rem; 
      line-height: 1.6; 
  }
  .modal-data { 
      background: rgba(0, 0, 0, 0.3); 
      padding: 1.5rem; 
      border-radius: 0.75rem; 
      margin-top: 1rem; 
  }
</style>
"""
    js_code = js_code.replace("COMPONENT_ARGS_PLACEHOLDER", json.dumps(component_args))
    complete_template = html_template + js_code + css_styles
    return complete_template

#######################################################################
# 6) CREATE/LOAD COLLECTION, ETC.
#######################################################################
def create_or_load_collection(collection_name: str):
    if collection_name in st.session_state:
        return st.session_state[collection_name]
    try:
        coll = chroma_client.get_collection(collection_name=collection_name)
    except Exception:
        coll = chroma_client.create_collection(name=collection_name)
    st.session_state[collection_name] = coll
    return coll

def add_to_chroma_collection(collection_name: str, chunks: List[str], metadatas: List[dict], ids: List[str]):
    embeddings = embed_text(chunks)
    coll = create_or_load_collection(collection_name)
    coll.add(documents=chunks, metadatas=metadatas, ids=ids, embeddings=embeddings)
    update_stage('store', {'collection': collection_name, 'count': len(chunks),
                           'metadata': metadatas[0] if metadatas else None})

def query_collection(query: str, collection_name: str, n_results: int = 3):
    coll = create_or_load_collection(collection_name)
    doc_count = len(coll.get().get("ids", []))
    if doc_count == 0:
        st.warning("No documents found in collection. Please upload first.")
        return [], []
    query_data = embed_text([query], update_stage_flag=False, return_data=True)
    query_data['query'] = query
    update_stage('query', query_data)
    results = coll.query(query_embeddings=query_data["embeddings"], n_results=n_results)
    retrieved_passages = results.get("documents", [[]])[0]
    retrieved_metadata = results.get("metadatas", [[]])[0]
    update_stage('retrieve', {"passages": retrieved_passages, "metadata": retrieved_metadata})
    return retrieved_passages, retrieved_metadata

def generate_answer_with_gpt(query: str, retrieved_passages: List[str], retrieved_metadata: List[dict],
                             system_instruction: str = (
                                 "You are a helpful legal assistant. Answer the following query based ONLY on the provided context. "
                                 "Your answer must begin with a TL;DR summary in bullet points. Then a detailed explanation."
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
    combined = "\n".join(passages[:3])
    short_summary = combined[:1000]
    return f"Summary of your documents:\n{short_summary}"

#######################################################################
# 7) REALTIME VOICE MODE
#######################################################################
def get_ephemeral_token(collection_name: str = "demo_collection"):
    if "api_key" not in st.session_state:
        st.error("OpenAI API key not set.")
        return None
    url = "https://api.openai.com/v1/realtime/sessions"
    headers = {
        "Authorization": f"Bearer {st.session_state['api_key']}",
        "Content-Type": "application/json"
    }
    data = {"model": "gpt-4o-realtime-preview-2024-12-17", "voice": "verse"}
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
    coll = create_or_load_collection(token_data['collection'])
    all_docs = coll.get()
    all_passages = all_docs.get("documents", [])
    doc_summary = summarize_context(all_passages)
    realtime_js = f"""
    <div id="realtime-status" style="color: lime; font-size: 20px;">Initializing...</div>
    <div id="transcription" style="color: white; margin-top: 10px;"></div>
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
            statusDiv.innerText = "Microphone connected!";
        }} catch (err) {{
            statusDiv.innerText = "Microphone error: " + err.message;
            return;
        }}
        pc.ontrack = (event) => {{
            audioEl.srcObject = event.streams[0];
        }};
        const dc = pc.createDataChannel("events");
        dc.onopen = () => {{
            statusDiv.innerText = "Connected! Sending baseline session instructions...";
            dc.send(JSON.stringify({{
                type: "session.update",
                session: {{ instructions: `{doc_summary}` }}
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
                        instructions: `You are a helpful assistant. Here are the best-matching doc passages for the last user query:
                        ${{relevantContext}}
                        Please answer carefully using *only* that info.`
                    }}
                }}));
            }} else if (data.type === "speech") {{
                transcriptionDiv.innerHTML += `<p style="color: #4CAF50;">Assistant: ${{data.text}}</p>`;
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
        body {{
          background-color: #111; 
          color: #fff; 
          font-family: system-ui, sans-serif; 
          margin: 0; 
          padding: 1rem; 
        }}
        #transcription {{
          margin-top: 20px; 
          padding: 10px; 
          background-color: #222; 
          border-radius: 5px; 
          max-height: 400px; 
          overflow-y: auto;
        }}
        #transcription p {{
          margin: 5px 0; 
          padding: 5px; 
          border-bottom: 1px solid #333;
        }}
    </style>
    """
    return realtime_js

#######################################################################
# 8) MAIN STREAMLIT APP
#######################################################################
def main():
    global chroma_client  # Declare global so that we can reassign it later if needed
    st.set_page_config(page_title="RAG Demo", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stApp { background-color: #111; color: white; }
        [data-testid="column"] { width: calc(100% + 2rem); margin-left: -1rem; }
        [data-testid="column"]:first-child { width: 33.33%; padding-right: 2rem; }
        [data-testid="column"]:last-child { width: 66.67%; }
        </style>
    """, unsafe_allow_html=True)
    st.title("RAG + Realtime Voice Demo")
    
    # Sidebar: API key, collection name, delete button, voice mode
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)
    st.sidebar.markdown("### Collection Name")
    existing_colls = [c.name for c in chroma_client.list_collections()]
    if existing_colls:
        coll_choice = st.sidebar.selectbox("Select or type a collection name:",
                                           options=["<Type Custom>"] + existing_colls,
                                           index=1 if len(existing_colls) > 0 else 0)
    else:
        coll_choice = "<Type Custom>"
    if coll_choice == "<Type Custom>":
        typed_collection = st.sidebar.text_input("Custom Collection Name", value=st.session_state.collection_name)
        st.session_state.collection_name = typed_collection
    else:
        st.session_state.collection_name = coll_choice
    if st.sidebar.button("Delete all Chroma DB Files"):
        if not st.session_state.delete_confirm:
            st.session_state.delete_confirm = True
            st.sidebar.warning("Are you sure? Click again to confirm.")
        else:
            shutil.rmtree(CHROMA_DIRECTORY, ignore_errors=True)
            st.sidebar.success("Chroma DB files deleted!")
            st.session_state.delete_confirm = False
            chroma_client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=CHROMA_DIRECTORY
                )
            )
            for ckey in list(st.session_state.keys()):
                if isinstance(st.session_state.get(ckey), chromadb.api.models.Collection.Collection):
                    del st.session_state[ckey]
    voice_mode = st.sidebar.checkbox("Enable Advanced Voice Interaction", value=False)
    if voice_mode:
        if st.sidebar.button("Start Voice Session", key="sidebar_start_voice"):
            token_data = get_ephemeral_token(st.session_state.collection_name)
            if token_data:
                st.sidebar.success("Realtime session initiated! See below in main page.")
                st.session_state["voice_html"] = get_realtime_html(token_data)
            else:
                st.sidebar.error("Could not start realtime session. Check error messages above.")
    
    # Main layout: Column 1: step controls; Column 2: pipeline visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Step-by-Step Pipeline Control")
        
        # Step 1: Upload
        st.subheader("Step 1: Upload")
        uploaded_file = st.file_uploader("Upload a document (txt, pdf, docx, csv, xlsx)", type=["txt", "pdf", "docx", "csv", "xlsx"])
        if st.button("Run Step 1: Upload"):
            if uploaded_file is not None:
                # For simplicity, we read the file as text
                text = uploaded_file.read().decode("utf-8")
                st.session_state.uploaded_text = text
                update_stage('upload', {'content': text, 'size': len(text)})
                st.success("File uploaded!")
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
                add_to_chroma_collection(st.session_state.collection_name, st.session_state.chunks, metadatas, ids)
                st.success(f"Data stored in Chroma (collection '{st.session_state.collection_name}')!")
            else:
                st.warning("Ensure document is uploaded, chunked, and embedded.")
        
        # Step 5: Query Collection
        st.subheader("Step 5: Query Collection")
        query_text = st.text_input("Enter your query for Step 5", value=st.session_state.query_text_step5)
        if st.button("Run Step 5: Query Collection"):
            if query_text.strip():
                query_data = embed_text([query_text], update_stage_flag=False, return_data=True)
                query_data['query'] = query_text
                update_stage('query', query_data)
                st.session_state.query_embedding = query_data["embeddings"]
                st.session_state.query_text_step5 = query_text
                st.success("Query vectorized!")
            else:
                st.warning("Please enter a query.")
        
        # Step 6: Retrieve
        st.subheader("Step 6: Retrieve")
        if st.button("Run Step 6: Retrieve"):
            if st.session_state.query_embedding:
                passages, metadata = query_collection(st.session_state.query_text_step5, st.session_state.collection_name)
                st.session_state.retrieved_passages = passages
                st.session_state.retrieved_metadata = metadata
                st.success("Relevant passages retrieved!")
            else:
                st.warning("Run Step 5 (Query Collection) first.")
        
        # Step 7: Get Answer
        st.subheader("Step 7: Get Answer")
        final_question = st.text_input("Enter your final question for Step 7", value=st.session_state.final_question_step7)
        if st.button("Run Step 7: Get Answer"):
            if final_question.strip():
                if st.session_state.retrieved_passages:
                    answer = generate_answer_with_gpt(final_question, st.session_state.retrieved_passages, st.session_state.retrieved_metadata)
                    st.session_state.final_answer = answer
                    st.session_state.final_question_step7 = final_question
                    st.success("Answer generated!")
                    st.write(answer)
                else:
                    st.warning("No retrieved passages. Run Step 6 first.")
            else:
                st.warning("Please enter your final question.")
    
    with col2:
        st.header("🌀 Pipeline Visualization")
        component_args = {
            "currentStage": st.session_state.current_stage,
            "stageData": { s: st.session_state.get(f'{s}_data')
                           for s in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']
                           if st.session_state.get(f'{s}_data') is not None }
        }
        pipeline_html = get_pipeline_component(component_args)
        components.html(pipeline_html, height=2000, scrolling=True)
        if "voice_html" in st.session_state:
            st.header("🎤 Realtime Voice")
            components.html(st.session_state["voice_html"], height=600, scrolling=True)

if __name__ == "__main__":
    main()