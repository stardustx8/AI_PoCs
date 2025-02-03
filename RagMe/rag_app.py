import os
import uuid
import re
import json
import time
import requests
import base64
import ssl
import unicodedata
from typing import List

import chromadb
from chromadb.config import Settings
import streamlit as st
import streamlit.components.v1 as components
import numpy as np  # optional for numeric operations
import tiktoken     # optional for token counting

from openai import OpenAI

# -----------------------------------------------------------------------------
# Check SSL version and warn if LibreSSL is in use
# -----------------------------------------------------------------------------
st.write(f"SSL version: {ssl.OPENSSL_VERSION}")
if "LibreSSL" in ssl.OPENSSL_VERSION:
    st.error("This application requires OpenSSL 1.1.1+ for realtime voice mode. Your environment uses LibreSSL. Please use a Python environment built with OpenSSL 1.1.1+.")

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

# RAG state class for tracking pipeline stages
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

        if 'rag_state' in st.session_state and st.session_state.rag_state is not None:
            st.session_state.rag_state.set_stage(stage, enhanced_data)

def set_openai_api_key(api_key: str):
    """
    Set the OpenAI API key and instantiate the new client.
    """
    global new_client
    new_client = OpenAI(api_key=api_key)
    st.session_state["api_key"] = api_key

def split_text_into_chunks(text: str) -> List[str]:
    """
    Split text into chunks.
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
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def sanitize_text(text: str) -> str:
    """
    Normalize text and remove non-ASCII characters.
    """
    text = remove_emoji(text)
    normalized = unicodedata.normalize('NFKD', text)
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_text

def embed_text(
    texts: List[str],
    openai_embedding_model: str = "text-embedding-3-large",
    update_stage_flag=True,
    return_data=False
):
    if new_client is None:
        st.error("OpenAI client not initialized. Please set your API key in the sidebar.")
        st.stop()
    safe_texts = [sanitize_text(s) for s in texts]
    response = new_client.embeddings.create(input=safe_texts, model=openai_embedding_model)
    embeddings = [item.embedding for item in response.data]
    
    # Compute token breakdown (using simple whitespace tokenization)
    token_breakdowns = []
    for text, embedding in zip(safe_texts, embeddings):
        tokens = text.split()
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
                    "vector_snippet": token_vector[:10]  # first 10 dims for display
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
    collection = create_or_load_collection(collection_name)
    doc_count = len(collection.get().get("ids", []))
    if doc_count == 0:
        st.warning("No documents found in the collection. Please upload a document first.")
        return [], []
    
    query_embedding_data = embed_text([query], update_stage_flag=False, return_data=True)
    query_embedding_data['query'] = query
    update_stage('query', query_embedding_data)
    
    results = collection.query(query_embeddings=query_embedding_data["embeddings"], n_results=n_results)
    retrieved_passages = results.get("documents", [[]])[0]
    retrieved_metadata = results.get("metadatas", [[]])[0]
    
    update_stage('retrieve', {'passages': retrieved_passages, 'metadata': retrieved_metadata})
    return retrieved_passages, retrieved_metadata

# -----------------------------------------------------------------------------
# REALTIME API HELPER FUNCTIONS FOR ADVANCED VOICE MODE
# -----------------------------------------------------------------------------

def get_ephemeral_token():
    """
    Create an ephemeral realtime session using the OpenAI REST API.
    Returns the JSON response (which includes an ephemeral token) or None on failure.
    """
    url = "https://api.openai.com/v1/realtime/sessions"
    headers = {
       "Authorization": f"Bearer {st.session_state.get('api_key')}",
       "Content-Type": "application/json"
    }
    data = {
       "model": "gpt-4o-realtime-preview-2024-12-17",
       "voice": "verse",  # choose realtime voice (e.g. alloy, ash, verse, etc.)
       "modalities": ["audio", "text"],
       "turn_detection": None  # leave VAD at default or disable as needed
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        st.write("Realtime session initiated successfully.")
        return response.json()
    else:
        st.error("Failed to create realtime session. Check your API key and settings.")
        st.write("Response status:", response.status_code)
        st.write("Response text:", response.text)
        return None

def get_realtime_html(ephemeral_token: str) -> str:
    """
    Returns an HTML snippet that initializes a realtime connection via WebRTC.
    Debug log statements are inserted in the JavaScript.
    """
    realtime_js = f"""
    <script>
    async function initRealtime() {{
      console.log("Initializing realtime connection.");
      const EPHEMERAL_KEY = "{ephemeral_token}";
      console.log("Ephemeral Key:", EPHEMERAL_KEY);
      const pc = new RTCPeerConnection();
      console.log("RTCPeerConnection created.");

      // Create an audio element to play remote audio.
      const audioEl = document.createElement("audio");
      audioEl.autoplay = true;
      document.body.appendChild(audioEl);
      console.log("Audio element appended to document.");
      pc.ontrack = e => {{
        console.log("Received remote track:", e);
        audioEl.srcObject = e.streams[0];
      }};

      // Add local microphone input.
      try {{
        const ms = await navigator.mediaDevices.getUserMedia({{ audio: true }});
        ms.getTracks().forEach(track => {{
          console.log("Adding local track:", track);
          pc.addTrack(track, ms);
        }});
      }} catch (err) {{
        console.error("Error accessing microphone:", err);
      }}

      // Set up a data channel for realtime events.
      const dc = pc.createDataChannel("oai-events");
      dc.onmessage = (e) => {{
        console.log("Realtime event received:", JSON.parse(e.data));
      }};
      console.log("Data channel created.");

      // Create an SDP offer.
      const offer = await pc.createOffer();
      console.log("SDP Offer created:", offer.sdp);
      await pc.setLocalDescription(offer);
      console.log("Local description set.");

      // Send the SDP offer to the realtime API.
      const baseUrl = "https://api.openai.com/v1/realtime";
      const model = "gpt-4o-realtime-preview-2024-12-17";
      console.log("Sending SDP offer to realtime endpoint.");
      const sdpResponse = await fetch(`${{baseUrl}}?model=${{model}}`, {{
        method: "POST",
        body: offer.sdp,
        headers: {{
          "Authorization": `Bearer ${{EPHEMERAL_KEY}}`,
          "Content-Type": "application/sdp",
          "OpenAI-Beta": "realtime=v1"
        }},
      }});
      console.log("SDP response received:", sdpResponse);
      const answerSdp = await sdpResponse.text();
      console.log("SDP Answer received:", answerSdp);
      const answer = {{ type: "answer", sdp: answerSdp }};
      await pc.setRemoteDescription(answer);
      console.log("Remote description set. Realtime connection established.");
    }}
    initRealtime();
    </script>
    <style>
      body {{
        background-color: #111;
        color: white;
        font-family: system-ui, sans-serif;
        padding: 1rem;
      }}
    </style>
    """
    return realtime_js

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
    
    # Sidebar configuration: API key and interaction mode.
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)
    input_mode = st.sidebar.radio("Select Interaction Mode", 
                                  options=["Standard Text Interaction", "Advanced Voice Interaction"], index=0)
    voice_output_enabled = st.sidebar.checkbox("Enable Voice Output for Answers", value=False)
    
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
            # For standard interaction, use text input.
            query = st.text_input("Enter your query")
            if query and st.button("Query"):
                passages, metadata = query_collection(query, "demo_collection")
                st.subheader("Retrieved Passages:")
                if passages:
                    for passage in passages:
                        st.write(passage)
                else:
                    st.info("No relevant passages found.")
                    
        elif option == "Generate Answer":
            query = st.text_input("Enter your query for answer generation")
            if query and st.button("Generate"):
                passages, metadata = query_collection(query, "demo_collection")
                answer = generate_answer_with_gpt(query, passages, metadata)
                st.subheader("Generated Answer:")
                st.write(answer)
                if voice_output_enabled:
                    with st.spinner("Generating voice output..."):
                        # Using TTS for answer synthesis (not realtime streaming)
                        response = new_client.audio.speech.create(
                            model="tts-1",
                            voice="alloy",
                            input=answer,
                            response_format="mp3"
                        )
                        st.audio(response.stream(), format="audio/mpeg")
                        
        # --- Advanced Voice Interaction Mode ---
        if input_mode == "Advanced Voice Interaction":
            st.subheader("Advanced Voice Interaction")
            st.info("This mode uses the OpenAI Realtime API for low-latency, continuous voice interaction. Speak into your microphone.")
            if st.button("Start Voice Session"):
                token_data = get_ephemeral_token()
                if token_data is not None:
                    ephemeral_token = token_data["client_secret"]["value"]
                    st.write("Ephemeral Token obtained. Opening realtime connection...")
                    realtime_html = get_realtime_html(ephemeral_token)
                    # Render the realtime session HTML (which initializes WebRTC)
                    components.html(realtime_html, height=600, scrolling=True)
                else:
                    st.error("Could not start realtime session. Check your API key and settings.")
                    
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

def get_pipeline_component(component_args: dict) -> str:
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
            description: "<strong>Step 1: Gather Your Raw Material</strong><br>We begin by taking the text exactly as you provided.",
            summaryDataExplanation: (data) => `
<strong>Upload Summary:</strong><br>
Received ~${data.size || 'N/A'} characters.
            `.trim(),
            dataExplanation: (data) => `
<strong>Upload Details:</strong><br>
${data.full || data.preview || 'No content available.'}
            `.trim()
        },
        chunk: {
            title: "Text Chunking",
            icon: '✂️',
            description: "<strong>Step 2: Slicing Content</strong>",
            summaryDataExplanation: (data) => `
<strong>Chunk Breakdown:</strong><br>
Total Chunks: ${data.total_chunks}
            `.trim(),
            dataExplanation: (data) => `
<strong>All Chunks:</strong><br>
${ (data.full_chunks || data.chunks || []).join("<br>") }
            `.trim()
        },
        embed: {
            title: "Vector Embedding Generation",
            icon: '🧠',
            description: "<strong>Step 3: Creating Embeddings</strong>",
            summaryDataExplanation: (data) => `
Each segment is a ${data.dimensions}-dimensional vector.
            `.trim(),
            dataExplanation: (data) => `
Total Embeddings: ${data.total_vectors}
            `.trim()
        },
        store: {
            title: "Vector Database Storage",
            icon: '🗄️',
            description: "<strong>Step 4: Storing Embeddings</strong>",
            summaryDataExplanation: (data) => `
Stored ${data.count} chunks in "${data.collection}".
            `.trim(),
            dataExplanation: (data) => `
Stored ${data.count} chunks.
            `.trim()
        },
        query: {
            title: "Query Vectorization",
            icon: '❓',
            description: "<strong>Step 5: Converting Your Query</strong>",
            summaryDataExplanation: (data) => `
Original Query: "${data.query}"
            `.trim(),
            dataExplanation: (data) => `
Query Details: ${JSON.stringify(data)}
            `.trim()
        },
        retrieve: {
            title: "Context Retrieval",
            icon: '🔎',
            description: "<strong>Step 6: Retrieving Relevant Passages</strong>",
            summaryDataExplanation: (data) => `
Top Matches: ${data.passages ? data.passages.join("<br>") : "None"}
            `.trim(),
            dataExplanation: (data) => `
Retrieved Passages: ${data.passages ? data.passages.join("<br>") : "None"}
            `.trim()
        },
        generate: {
            title: "Answer Generation",
            icon: '🤖',
            description: "<strong>Step 7: Generating the Final Answer</strong>",
            summaryDataExplanation: (data) => `
Answer (Summary): ${data.answer.substring(0, Math.floor(data.answer.length/2))}...
            `.trim(),
            dataExplanation: (data) => `
Full Answer: ${data.answer}
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
        .close-button { position: absolute; top: 20px; right: 20px; background: transparent; border: none; font-size: 2rem; font-weight: bold; color: #fff; cursor: pointer; }
        .modal-title { font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; color: white; }
        .modal-description { color: #9CA3AF; font-size: 1.1rem; margin-bottom: 1.5rem; line-height: 1.6; }
        .modal-data { background: rgba(0, 0, 0, 0.3); padding: 1.5rem; border-radius: 0.75rem; margin-top: 1rem; }
    </style>
    """
    js_code = js_code.replace("COMPONENT_ARGS_PLACEHOLDER", json.dumps(component_args, ensure_ascii=False))
    return html_template + js_code

if __name__ == "__main__":
    main()