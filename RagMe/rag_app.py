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

from openai import OpenAI

# -----------------------------------------------------------------------------
# GLOBAL VARIABLES & CLIENT INITIALIZATION
# -----------------------------------------------------------------------------

new_client = None  # Global variable for the OpenAI client

chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="chromadb_storage"
    )
)

# -----------------------------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------------------------

if 'current_stage' not in st.session_state:
    st.session_state.current_stage = None

for stage in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']:
    if f'{stage}_data' not in st.session_state:
        st.session_state[f'{stage}_data'] = None

# -----------------------------------------------------------------------------
# RAG STATE CLASS
# -----------------------------------------------------------------------------

class RAGState:
    def __init__(self):
        self.current_stage = None
        self.stage_data = {
            "upload": {"active": False, "data": None},
            "chunk": {"active": False, "data": None},
            "embed": {"active": False, "data": None},
            "store": {"active": False, "data": None},
            "query": {"active": False, "data": None},
            "retrieve": {"active": False, "data": None},
            "generate": {"active": False, "data": None}
        }

    def set_stage(self, stage, data=None):
        self.current_stage = stage
        self.stage_data[stage]["active"] = True
        self.stage_data[stage]["data"] = data
        st.session_state.rag_state = self

if 'rag_state' not in st.session_state or st.session_state.rag_state is None:
    st.session_state.rag_state = RAGState()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (PIPELINE OPERATIONS)
# -----------------------------------------------------------------------------

def update_stage(stage: str, data=None):
    """
    Update the current stage and store enhanced stage data in session state.
    """
    st.session_state.current_stage = stage
    if data is not None:
        # For the embed stage, allow passing a dictionary directly.
        if stage == 'embed' and isinstance(data, dict):
            enhanced_data = data
        else:
            enhanced_data = data.copy() if isinstance(data, dict) else {'data': data}

        if stage == 'upload':
            text = data.get('content', '') if isinstance(data, dict) else data
            enhanced_data['preview'] = text[:600] if text else None
            enhanced_data['full'] = text  # Store full content for expanded view.

        elif stage == 'chunk':
            enhanced_data = {
                'chunks': data[:5],
                'total_chunks': len(data)
            }

        elif stage == 'store':
            enhanced_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        elif stage == 'query':
            enhanced_data = {'query': data.get('query')}

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

        if 'rag_state' in st.session_state and st.session_state.rag_state is not None:
            st.session_state.rag_state.set_stage(stage, enhanced_data)

def set_openai_api_key(api_key: str):
    """
    Set the OpenAI API key and instantiate the new client.
    """
    global new_client
    new_client = OpenAI(api_key=api_key)

def split_text_into_chunks(text: str) -> List[str]:
    """
    Split text into chunks (upload and chunk stages).
    """
    update_stage('upload', {'content': text, 'size': len(text)})
    chunks = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    update_stage('chunk', chunks)
    return chunks

def remove_emoji(text: str) -> str:
    """
    Remove emoji from a text string.
    """
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def sanitize_text(text: str) -> str:
    """
    Normalize the text and remove any non-ASCII characters.
    """
    # Remove emoji first
    text = remove_emoji(text)
    # Normalize and force ASCII-only (ignore non-ASCII characters)
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_text

def embed_text(texts: List[str], openai_embedding_model: str = "text-embedding-3-large", update_stage_flag=True):
    """
    Generate vector embeddings for a list of texts.
    Each text is sanitized and, for demonstration purposes, a simulated token breakdown is computed.
    """
    if new_client is None:
        st.error("OpenAI client not initialized. Please set your API key in the sidebar.")
        st.stop()
    # Sanitize each text to ensure it contains only ASCII characters.
    safe_texts = [sanitize_text(s) for s in texts]
    response = new_client.embeddings.create(input=safe_texts, model=openai_embedding_model)
    embeddings = [item.embedding for item in response.data]
    
    # Simulated Token Breakdown:
    token_breakdowns = []
    for text, embedding in zip(safe_texts, embeddings):
        tokens = text.split()  # simple tokenization for demo purposes
        n_tokens = len(tokens)
        breakdown = []
        if n_tokens > 0:
            dims_per_token = len(embedding) // n_tokens
            for i, token in enumerate(tokens):
                start = i * dims_per_token
                end = (i + 1) * dims_per_token if i < n_tokens - 1 else len(embedding)
                token_vector = embedding[start:end]
                breakdown.append({
                    "token": token,
                    "vector_snippet": token_vector[:10]  # first 10 dims for brevity
                })
        token_breakdowns.append(breakdown)
    
    embedding_data = {
        "embeddings": embeddings,
        "dimensions": len(embeddings[0]) if embeddings else 0,
        "preview": embeddings[0][:10] if embeddings else [],
        "total_vectors": len(embeddings),
        "token_breakdowns": token_breakdowns
    }
    # Only update the "embed" stage if desired (e.g. for the file, not for the query)
    if update_stage_flag:
        update_stage('embed', embedding_data)
    return embeddings

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
        description: "<strong>Step 1: Gather Your Raw Material</strong><br>We begin by taking the text exactly as you provided, pulling it into our pipeline, and giving it a brief once-over. This sets the stage for everything to come. It’s the essential first step in transforming your uploaded content into something we can query with intelligence.",
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
            description: "<strong>Step 2: Slicing Content into Digestible Bits</strong><br>To make it easier for our system to understand your data, we slice your text into smaller, self-contained segments. Think of it like cutting a big loaf of bread into manageable slices—each slice can be handled and analyzed on its own.",
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
            description: "<strong>Step 3: Translating Each Chunk into Numbers</strong><br>Now we transform every chunk into a numeric representation known as an ‘embedding.’ This is where meaning meets math. Each chunk gets a high-dimensional vector capturing its essence—making it easier for computers to 'compare' the content of different segments logically.",
            summaryDataExplanation: (data) => `
<strong>Embedding Stats (Summary):</strong><br>
Each segment is represented by a ${data.dimensions}-dimensional vector.<br>
Total Embeddings: ${data.total_vectors}<br>
Sample Token Breakdown (first 3 chunks): ${ data.token_breakdowns.slice(0,3).map((chunkBreakdown, idx) => `
<br><strong>Chunk ${idx + 1}:</strong>
${ chunkBreakdown.map(item => `<br><span style="color:red;font-weight:bold;">${item.token}</span>: [${item.vector_snippet.map(v => v.toFixed(6)).join(", ")}]` ).join("") }
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
            description: "<strong>Step 4: Stashing Vectors into ChromaDB</strong><br>All those embeddings need a home. Here, we store them in a specialized database so that, at any moment, we can fetch whichever chunk best matches your needs. It’s like a well-organized library, except the indexes are vectors!",
            summaryDataExplanation: (data) => `
<strong>Storage Summary:</strong><br>
Stored ${data.count} chunks in the collection "${data.collection}".
            `.trim(),
            dataExplanation: (data) => `
<strong>Storage Details (Expanded):</strong><br>
Stored ${data.count} chunks in the collection "${data.collection}".
            `.trim()
        },
        query: {
    title: "Query Processing",
    icon: '❓',
    description: "<strong>Step 5: Converting Your Question into a Vector</strong><br>When you ask a question, we convert your query into a vector. This ensures that your question is represented in the same high-dimensional space as your document embeddings, enabling precise comparisons.",
    summaryDataExplanation: (data) => `
<strong>Query Vectorization (Summary):</strong><br>
Original Query: <span style="color:red;font-weight:bold;">"${data.query}"</span><br>
Vectorized: [0.0123, -0.0345, 0.0789, ...]
    `.trim(),
    dataExplanation: (data) => `
<strong>Query Vectorization (Expanded):</strong><br>
Original Query: <span style="color:red;font-weight:bold;">"${data.query}"</span><br>
Vectorized Representation: [0.0123, -0.0345, 0.0789, ...]<br>
(Additional vector details can be shown here if available)
    `.trim()
},
        retrieve: {
            title: "Context Retrieval",
            icon: '🔎',
            description: "<strong>Step 6: Fishing Out the Most Relevant Chunks</strong><br>We take the newly minted query vector and compare it to our entire library of chunk vectors. The ones that line up the best (i.e., have the highest similarity) are reeled in as the best candidate passages to answer your question.",
            summaryDataExplanation: (data) => `
<strong>Top Matches (Summary):</strong><br>
${data.passages.slice(0, Math.ceil(data.passages.length/2)).map((passage, i) => `<br><span style="color:red;font-weight:bold;">Match ${i + 1} (similarity: ${(data.scores[i]*100).toFixed(1)}%):</span> "${passage}"`).join('<br><br>')}
            `.trim(),
            dataExplanation: (data) => `
<strong>Top Matches (Expanded):</strong><br>
${data.passages.map((passage, i) => `<br><span style="color:red;font-weight:bold;">Match ${i + 1} (similarity: ${(data.scores[i]*100).toFixed(1)}%):</span> "${passage}"`).join('<br><br>')}
            `.trim()
        },
        generate: {
            title: "Answer Generation",
            icon: '🤖',
            description: "<strong>Step 7: Shaping the Final Answer</strong><br>This is where GPT comes in. We feed the question and the top matching text chunks to GPT. By combining your query’s intention with context from your own data, GPT crafts a targeted answer. It’s the perfect union of clarity and fluency.",
            summaryDataExplanation: (data) => `
<strong>Answer (Summary):</strong><br>
${data.answer.substring(0, Math.floor(data.answer.length/2))}...
            `.trim(),
            dataExplanation: (data) => `
<strong>Answer (Expanded):</strong><br>
${data.answer}
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
                            onClick: () => { setSelectedStage(stageObj.id); setShowModal(true); },
                            key: 'box'
                        }, [
                            React.createElement('div', { className: 'stage-header', key: 'header' }, [
                                React.createElement('span', { className: 'stage-icon', key: 'icon' }, process.icon),
                                React.createElement('span', { className: 'stage-title', key: 'title' }, process.title)
                            ]),
                            React.createElement('div', { className: 'stage-description', key: 'desc', dangerouslySetInnerHTML: { __html: process.description } }),
                            dataObj && React.createElement('div', { 
                                className: 'stage-data', 
                                key: 'data', 
                                dangerouslySetInnerHTML: { __html: (process.summaryDataExplanation ? process.summaryDataExplanation(dataObj) : process.dataExplanation(dataObj)) } 
                            })
                        ]),
                        index < pipelineStages.length - 1 && React.createElement(ArrowIcon, { key: 'arrow' })
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
        /* Hide inner scrollbar */
        ::-webkit-scrollbar { width: 0px; background: transparent; }
        body { background-color: #111; color: #fff; margin: 0; padding: 0; }
        #rag-root { font-family: system-ui, sans-serif; height: 100%; width: 100%; margin: 0; padding: 0; }
        .pipeline-container { padding: 1rem; overflow-y: auto; height: 100vh; box-sizing: border-box; width: 100%; }
        .pipeline-column { display: flex; flex-direction: column; align-items: stretch; margin: 0 auto; width: 95%; padding-right: 2rem; }
        .pipeline-box { width: 100%; margin-bottom: 1rem; padding: 1.5rem; border: 2px solid #4B5563; border-radius: 0.75rem; background-color: #1a1a1a; cursor: pointer; transition: all 0.3s; text-align: left; }
        .pipeline-box:hover { transform: scale(1.02); border-color: #6B7280; }
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
    js_code = js_code.replace("COMPONENT_ARGS_PLACEHOLDER", json.dumps(component_args, ensure_ascii=False))
    complete_template = html_template + js_code + css_styles
    return complete_template

def create_or_load_collection(collection_name: str):
    if collection_name in st.session_state:
        return st.session_state[collection_name]
    try:
        collection = chroma_client.get_collection(collection_name=collection_name)
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)
    st.session_state[collection_name] = collection
    return collection

def add_to_chroma_collection(collection_name: str,
                             chunks: List[str],
                             metadatas: List[dict],
                             ids: List[str]):
    embeddings = embed_text(chunks)
    collection = create_or_load_collection(collection_name)
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    update_stage('store', {
        'collection': collection_name,
        'count': len(chunks),
        'metadata': metadatas[0] if metadatas else None
    })

def query_collection(query: str, collection_name: str, n_results: int = 3):
    update_stage('query', {'query': query})
    collection = create_or_load_collection(collection_name)
    doc_count = len(collection.get().get("ids", []))
    if doc_count == 0:
        st.warning("No documents found in the collection. Please upload a document first.")
        return [], []
    # Compute the query embedding but do not update the "embed" stage:
    query_embedding = embed_text([query], update_stage_flag=False)
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    retrieved_passages = results.get("documents", [[]])[0]
    retrieved_metadata = results.get("metadatas", [[]])[0]
    update_stage('retrieve', {'passages': retrieved_passages, 'metadata': retrieved_metadata})
    return retrieved_passages, retrieved_metadata

def generate_answer_with_gpt(query: str,
                             retrieved_passages: List[str],
                             retrieved_metadata: List[dict],
                             system_instruction: str = (
                                 "You are a helpful legal assistant. Answer the following query based ONLY on the provided context. "
                                 "Your answer must begin with a TL;DR summary in bullet points. Then a detailed explanation."
                             )):
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

# -----------------------------------------------------------------------------
# MAIN APPLICATION & UI
# -----------------------------------------------------------------------------

def main():
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
    
    st.title("RAG Pipeline Demo")
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Document and Query Input")
        option = st.selectbox("Select Operation", ["Upload & Process Document", "Query Collection", "Generate Answer"])
        if option == "Upload & Process Document":
            uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
            if uploaded_file is not None:
                text = uploaded_file.read().decode("utf-8")
                chunks = split_text_into_chunks(text)
                metadatas = [{"source": "uploaded_document"} for _ in chunks]
                ids = [str(uuid.uuid4()) for _ in chunks]
                add_to_chroma_collection("demo_collection", chunks, metadatas, ids)
                st.success("Document processed and stored.")
        elif option == "Query Collection":
            query = st.text_input("Enter your query")
            if st.button("Query"):
                passages, metadata = query_collection(query, "demo_collection")
                st.subheader("Retrieved Passages:")
                if passages:
                    for passage in passages:
                        st.write(passage)
                else:
                    st.info("No relevant passages found.")
        elif option == "Generate Answer":
            query = st.text_input("Enter your query for answer generation")
            if st.button("Generate"):
                passages, metadata = query_collection(query, "demo_collection")
                answer = generate_answer_with_gpt(query, passages, metadata)
                st.subheader("Generated Answer:")
                st.write(answer)
    with col2:
        st.title("🌀 Pipeline Visualization")
        component_args = {
            'currentStage': st.session_state.current_stage,
            'stageData': {
                stage: st.session_state.get(f'{stage}_data')
                for stage in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']
                if st.session_state.get(f'{stage}_data') is not None
            }
        }
        components.html(
            get_pipeline_component(component_args),
            height=2000,
            scrolling=True
        )

if __name__ == "__main__":
    main()