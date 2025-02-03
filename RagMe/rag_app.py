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

#######################################################################
# 1) GLOBALS & CLIENT INITIALIZATION
#######################################################################
new_client = None  # We'll set an OpenAI client once the user provides an API key
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="chromadb_storage"
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
        # If embed stage w/ dict, keep as is, else copy
        if stage == 'embed' and isinstance(data, dict):
            enhanced_data = data
        else:
            enhanced_data = data.copy() if isinstance(data, dict) else {'data': data}

        if stage == 'upload':
            # store a preview
            text = data.get('content', '') if isinstance(data, dict) else data
            enhanced_data['preview'] = text[:600] if text else None
            enhanced_data['full'] = text

        elif stage == 'chunk':
            enhanced_data = {
                'chunks': data[:5],
                'total_chunks': len(data)
            }

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
    global new_client
    new_client = OpenAI(api_key=api_key)
    st.session_state["api_key"] = api_key

def remove_emoji(text: str) -> str:
    # Remove various emojis
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def sanitize_text(text: str) -> str:
    # remove emoji, normalize, force ASCII
    import unicodedata
    text = remove_emoji(text)
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_text

import re
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

    # optional token breakdown
    token_breakdowns = []
    for text, embedding in zip(safe_texts, embeddings):
        tokens = text.split()
        breakdown = []
        if tokens:
            dims_per_token = len(embedding) // len(tokens)
            for i, tok in enumerate(tokens):
                start = i*dims_per_token
                end = start + dims_per_token if i < len(tokens)-1 else len(embedding)
                snippet = embedding[start:end]
                breakdown.append({
                    "token": tok,
                    "vector_snippet": snippet[:10]
                })
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

#######################################################################
# 5) FULL REACT PIPELINE SNIPPET
#######################################################################
def get_pipeline_component(component_args):
    """
    The EXACT snippet from your code, with detailed text for each stage.
    """
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
Each segment is a ${data.dimensions}-dimensional vector.<br>
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
            title: "Query Vectorization",
            icon: '❓',
            description: "<strong>Step 5: Converting Your Question into a Vector</strong><br>Your query is also turned into a vector so we can measure its similarity to the stored chunks.",
            summaryDataExplanation: (data) => `
<strong>Query Vector (Summary):</strong><br>
Original Query: "<span style='color:red;font-weight:bold;'>${data.query}</span>"<br>
Total Vectors: ${data.total_vectors}<br>
            `.trim(),
            dataExplanation: (data) => `
<strong>Query Vector (Expanded):</strong><br>
Query: "<span style='color:red;font-weight:bold;'>${data.query}</span>"<br>
Vector Dimensions: ${data.dimensions}<br>
${ data.token_breakdowns ? 
    data.token_breakdowns.map((chunk, idx) => {
       return chunk.map(item => `<br>${item.token}: [${item.vector_snippet.map(v => v.toFixed(5)).join(", ")}]`).join("");
     }).join("") 
   : "No breakdown" }
            `.trim()
        },
        retrieve: {
            title: "Context Retrieval",
            icon: '🔎',
            description: "<strong>Step 6: Retrieving Relevant Chunks</strong><br>We compare your query vector to every chunk vector and retrieve the most similar ones to help answer your question.",
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
            title: "Answer Generation",
            icon: '🤖',
            description: "<strong>Step 7: GPT’s Final Answer</strong><br>We feed your query + the retrieved passages into GPT, which composes a final answer, bridging the question and your own context.",
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
                    const dataObj = stageData[stageObj.id] || null;
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
    return html_template + js_code + css_styles

#######################################################################
# 6) CREATE/LOAD COLLECTION, ETC
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
    update_stage('store', {
        'collection': collection_name,
        'count': len(chunks),
        'metadata': metadatas[0] if metadatas else None
    })

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

#######################################################################
# 7) REALTIME VOICE MODE (FIXED)
#######################################################################
def get_ephemeral_token():
    if "api_key" not in st.session_state:
        st.error("OpenAI API key not set.")
        return None
    
    url = "https://api.openai.com/v1/realtime/sessions"
    headers = {
        "Authorization": f"Bearer {st.session_state['api_key']}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o-realtime-preview-2024-12-17",
        "voice": "verse"
    }
    
    try:
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        token_data = resp.json()
        
        # Debug the response structure
        st.write("Response structure:", token_data)
        
        # Return the token directly if it's in the response
        if isinstance(token_data, dict):
            if "token" in token_data:
                return token_data["token"]
            elif "client_secret" in token_data:
                if isinstance(token_data["client_secret"], dict):
                    return token_data["client_secret"].get("value")
                return token_data["client_secret"]
                
        st.error("Unexpected response structure")
        st.json(token_data)  # Display the full response for debugging
        return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create realtime session: {str(e)}")
        if hasattr(e.response, 'text'):
            st.write("Error response:", e.response.text)
        return None

def get_realtime_html(ephemeral_token: str) -> str:
    realtime_js = f"""
    <div id="realtime-status" style="color: lime; font-size: 20px; margin-bottom: 10px;">Initializing realtime connection...</div>
    <div id="transcription" style="color: white; margin-top: 10px;"></div>
    <script>
    async function initRealtime() {{
        const statusDiv = document.getElementById('realtime-status');
        const transcriptionDiv = document.getElementById('transcription');
        statusDiv.innerText = "Starting realtime connection...";

        const pc = new RTCPeerConnection();
        
        try {{
            // Set up audio input
            const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
            stream.getTracks().forEach(track => pc.addTrack(track, stream));
            statusDiv.innerText = "Microphone connected!";
        }} catch (err) {{
            console.error("Microphone error:", err);
            statusDiv.innerText = "Error accessing microphone: " + err.message;
            return;
        }}

        // Set up audio output
        const audioEl = new Audio();
        audioEl.autoplay = true;
        
        pc.ontrack = (event) => {{
            audioEl.srcObject = event.streams[0];
        }};

        // Create data channel
        const dc = pc.createDataChannel("events");
        
        dc.onopen = () => {{
            console.log("Data channel opened");
            statusDiv.innerText = "Connected and ready! Start speaking...";
        }};

        dc.onmessage = (e) => {{
            try {{
                const data = JSON.parse(e.data);
                console.log("Received message:", data);
                
                if (data.type === "text") {{
                    transcriptionDiv.innerHTML += `<p>You: ${{data.text}}</p>`;
                }} else if (data.type === "speech") {{
                    transcriptionDiv.innerHTML += `<p style="color: #4CAF50;">Assistant: ${{data.text}}</p>`;
                }}
            }} catch (err) {{
                console.error("Error processing message:", err);
            }}
        }};

        // Create and send offer
        try {{
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            const sdpResponse = await fetch(`https://api.openai.com/v1/realtime`, {{
                method: "POST",
                body: offer.sdp,
                headers: {{
                    "Authorization": `Bearer ${{"{ephemeral_token}"}}`,
                    "Content-Type": "application/sdp"
                }}
            }});

            if (!sdpResponse.ok) {{
                throw new Error(`HTTP error! status: ${{sdpResponse.status}}`);
            }}

            const answerSdp = await sdpResponse.text();
            await pc.setRemoteDescription({{ type: "answer", sdp: answerSdp }});
            
            console.log("WebRTC connection established");
            statusDiv.innerText = "Connected! Start speaking...";
        }} catch (err) {{
            console.error("Connection error:", err);
            statusDiv.innerText = "Connection error: " + err.message;
        }}
    }}

    // Start the realtime connection
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

    st.title("RAG Pipeline Demo + Realtime Voice")
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)

    # Voice checkbox
    voice_mode = st.sidebar.checkbox("Enable Advanced Voice Interaction", value=False)

    col1, col2 = st.columns([1,2])

    with col1:
        st.header("Document + Query Input")
        operation = st.selectbox("Select Operation", ["Upload & Process Document", "Query Collection", "Generate Answer"])
        if operation == "Upload & Process Document":
            uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
            if uploaded_file is not None:
                text = uploaded_file.read().decode("utf-8")
                chunks = split_text_into_chunks(text)
                metadatas = [{"source": "uploaded_document"} for _ in chunks]
                ids = [str(uuid.uuid4()) for _ in chunks]
                add_to_chroma_collection("demo_collection", chunks, metadatas, ids)
                st.success("Document processed + stored in Chroma!")
        elif operation == "Query Collection":
            query = st.text_input("Enter your query")
            if st.button("Query"):
                passages, metadata = query_collection(query, "demo_collection")
                st.subheader("Retrieved Passages:")
                if passages:
                    for p in passages:
                        st.write(p)
                else:
                    st.info("No relevant passages found.")
        elif operation == "Generate Answer":
            query = st.text_input("Enter your question")
            if st.button("Generate Answer"):
                passages, metadata = query_collection(query, "demo_collection")
                answer = generate_answer_with_gpt(query, passages, metadata)
                st.subheader("Answer:")
                st.write(answer)

        # If voice mode is on, show button to start session
        # Inside main() function, replace the voice mode section with:
        if voice_mode:
            st.subheader("Advanced Voice Interaction")
            st.info("This uses the OpenAI Realtime API for low-latency voice. Mic must be allowed.")
            if st.button("Start Voice Session"):
                ephemeral_token = get_ephemeral_token()
                if ephemeral_token:
                    st.success("Realtime session initiated! Opening WebRTC panel below...")
                    realtime_html = get_realtime_html(ephemeral_token)
                    components.html(realtime_html, height=600, scrolling=True)
                else:
                    st.error("Could not start realtime session. Check the error messages above.")

    with col2:
        st.title("🌀 Pipeline Visualization")
        component_args = {
            "currentStage": st.session_state.current_stage,
            "stageData": {
                s: st.session_state.get(f'{s}_data')
                for s in ['upload','chunk','embed','store','query','retrieve','generate']
                if st.session_state.get(f'{s}_data') is not None
            }
        }
        pipeline_html = get_pipeline_component(component_args)
        components.html(pipeline_html, height=2000, scrolling=True)

if __name__ == "__main__":
    main()