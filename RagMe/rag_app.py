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

# Track the pipeline's current stage
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = None

# Track stage data for each pipeline step
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
    Real-time pipeline stage update using st.rerun().
    """
    st.session_state.current_stage = stage
    if data is not None:
        enhanced_data = data.copy() if isinstance(data, dict) else {'data': data}

        if stage == 'upload':
            text = data.get('content', '') if isinstance(data, dict) else data
            enhanced_data['preview'] = text[:200] if text else None

        elif stage == 'chunk':
            enhanced_data = {
                'chunks': data[:5],
                'total_chunks': len(data)
            }

        elif stage == 'embed':
            # Show embedding stats
            enhanced_data = {
                'dimensions': len(data[0]) if data else 0,
                'preview': data[0][:10] if data else [],
                'total_vectors': len(data)
            }

        elif stage == 'store':
            enhanced_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        elif stage == 'query':
            # Show user query
            enhanced_data = {'query': data.get('query')}

        elif stage == 'retrieve':
            # Show top passages
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

        # Update RAGState as well
        if 'rag_state' in st.session_state and st.session_state.rag_state is not None:
            st.session_state.rag_state.set_stage(stage, enhanced_data)

    # Short sleep & st.rerun
    # time.sleep(0.8)
    # st.rerun()

def set_openai_api_key(api_key: str):
    """
    Set the OpenAI API key and instantiate the new client.
    """
    global new_client
    new_client = OpenAI(api_key=api_key)

def split_text_into_chunks(text: str) -> List[str]:
    """
    Step 1: 'upload', Step 2: 'chunk'
    """
    update_stage('upload', {'content': text, 'size': len(text)})
    # The script reruns after 'upload'; so next lines will run again after re-run
    chunks = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    update_stage('chunk', chunks)
    # Re-run after chunk
    return chunks

def embed_text(texts: List[str], openai_embedding_model: str = "text-embedding-3-large"):
    """
    'embed' stage.
    """
    if new_client is None:
        st.error("OpenAI client not initialized. Please set your API key in the sidebar.")
        st.stop()

    # Pre-update
    update_stage('embed')
    # after re-run
    response = new_client.embeddings.create(input=texts, model=openai_embedding_model)
    embeddings = [item.embedding for item in response.data]

    # Post-update
    update_stage('embed', embeddings)
    return embeddings

def get_pipeline_component(component_args):
    # Base HTML template
    html_template = """
    <div id="rag-root"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/react/17.0.2/umd/react.production.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/17.0.2/umd/react-dom.production.min.js"></script>
    """

    # JavaScript code as a separate string
    js_code = """
    <script>
    const args = COMPONENT_ARGS_PLACEHOLDER;
    
    const { useState, useEffect } = React;
    
    const RAGFlowVertical = () => {
        const [activeStage, setActiveStage] = useState(args.currentStage || null);
        const [modalContent, setModalContent] = useState(null);
        const [showModal, setShowModal] = useState(false);
        
        useEffect(() => {
            setActiveStage(args.currentStage);
        }, [args.currentStage]);
        
        const pipelineStages = [
            {
                id: 'upload',
                title: '📁 Upload Document',
                description: 'Raw doc read from disk'
            },
            {
                id: 'chunk',
                title: '✂️ Chunk Text',
                description: 'Split doc into pieces'
            },
            {
                id: 'embed',
                title: '🧠 Vector Embedding',
                description: 'OpenAI vectorizing text'
            },
            {
                id: 'store',
                title: '🗄️ Vector Storage',
                description: 'ChromaDB storing vectors'
            },
            {
                id: 'query',
                title: '❓ Query Processing',
                description: 'Embedding + searching'
            },
            {
                id: 'retrieve',
                title: '🔎 Context Retrieval',
                description: 'Finding relevant passages'
            },
            {
                id: 'generate',
                title: '🤖 Answer Generation',
                description: 'Final GPT response'
            }
        ];
        
        const stageData = args.stageData || {};
        
        function formatData(stage, data) {
            if (!data) return 'No data yet.';
            switch(stage) {
                case 'upload':
                    return `Preview: ${data.preview || ''}\nSize: ${data.size || ''} bytes`;
                case 'chunk':
                    const firstChunks = (data.chunks || []).slice(0,2).join('\\n\\n');
                    return `First chunks: ${firstChunks}\nTotal Chunks: ${data.total_chunks}`;
                case 'embed':
                    return `Dimensions: ${data.dimensions}\nVectors: ${data.total_vectors}`;
                case 'store':
                    return `Collection: ${data.collection}\nVectors Stored: ${data.count}`;
                case 'query':
                    return `User Query: ${data.query || ''}`;
                case 'retrieve':
                    if (!data.passages) return 'No passages.';
                    const firstPass = (data.passages && data.passages.length>0) 
                                    ? data.passages[0].slice(0,120) 
                                    : '';
                    return `First Passage: ${firstPass}...\nScores: [${data.scores.join(', ')}]`;
                case 'generate':
                    if (!data.answer) return 'No answer.';
                    return `Answer preview: ${data.answer.slice(0,100)}...`;
                default:
                    return JSON.stringify(data, null, 2);
            }
        }
        
        function openModal(stage) {
            const data = stageData[stage];
            if (!data) {
                setModalContent('No data for stage.');
            } else {
                setModalContent(JSON.stringify(data, null, 2));
            }
            setShowModal(true);
        }
        
        function closeModal() {
            setShowModal(false);
            setModalContent(null);
        }
        
        return React.createElement('div', { style: { padding: '1rem' } },
            showModal && React.createElement('div', { className: 'tooltip-modal' },
                React.createElement('div', { className: 'tooltip-content' },
                    React.createElement('button', { 
                        className: 'close-button',
                        onClick: closeModal 
                    }, 'Close'),
                    React.createElement('pre', null, modalContent)
                )
            ),
            React.createElement('div', { className: 'pipeline-column' },
                pipelineStages.map((stageObj, index) => {
                    const dataObj = stageData[stageObj.id] || null;
                    const isActive = (activeStage === stageObj.id && dataObj);
                    const stageClass = 'pipeline-box ' + (isActive ? 'active-stage' : '');
                    
                    return React.createElement(React.Fragment, { key: stageObj.id }, [
                        React.createElement('div', { 
                            className: stageClass,
                            onClick: () => openModal(stageObj.id),
                            key: 'box'
                        }, [
                            React.createElement('div', { 
                                className: 'stage-title',
                                key: 'title'
                            }, stageObj.title),
                            React.createElement('div', { 
                                className: 'stage-description',
                                key: 'desc'
                            }, stageObj.description),
                            React.createElement('div', { 
                                style: { 
                                    marginTop: '0.5rem',
                                    fontSize: '0.85rem',
                                    color: '#D1D5DB'
                                },
                                key: 'data'
                            }, formatData(stageObj.id, dataObj))
                        ]),
                        index < pipelineStages.length - 1 && 
                        React.createElement('div', { 
                            className: 'pipeline-arrow-down',
                            key: 'arrow'
                        }, '⬇️')
                    ]);
                })
            )
        );
    };
    """

    # CSS styles
    css_styles = """
    const styles = `
        body {
            background-color: #111;
            color: #fff;
            margin: 0;
            padding: 0;
        }
        #rag-root {
            font-family: system-ui, sans-serif;
            color: white;
            min-height: 100vh;
        }
        .pipeline-column {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 1rem auto;
        }
        .pipeline-box {
            width: 90%;
            max-width: 400px;
            margin: 0 auto 1.5rem;
            padding: 1rem;
            border: 2px solid #4B5563;
            border-radius: 0.5rem;
            position: relative;
            transition: transform 0.3s;
            background-color: #333;
            cursor: pointer;
        }
        .pipeline-box:hover {
            transform: scale(1.02);
        }
        .pipeline-arrow-down {
            margin-bottom: 1rem;
            font-size: 2rem;
            color: #9CA3AF;
        }
        .stage-title {
            font-weight: bold;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: white;
        }
        .stage-description {
            color: #9CA3AF;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        .active-stage {
            border-color: #22C55E;
        }
        .tooltip-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }
        .tooltip-content {
            background: #111;
            padding: 2rem;
            border-radius: 0.5rem;
            max-width: 80%;
            max-height: 80%;
            overflow-y: auto;
            color: #fff;
        }
        .close-button {
            background: #f87171;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
            margin-bottom: 1rem;
            font-weight: bold;
            color: white;
        }
    `;
    
    const styleSheet = document.createElement('style');
    styleSheet.textContent = styles;
    document.head.appendChild(styleSheet);
    
    ReactDOM.render(
        React.createElement(RAGFlowVertical),
        document.getElementById('rag-root')
    );
    </script>
    """

    # Combine everything
    import json
    complete_template = (
        html_template +
        js_code.replace('COMPONENT_ARGS_PLACEHOLDER', json.dumps(component_args)) +
        css_styles
    )
    
    return complete_template


def create_or_load_collection(collection_name: str):
    """
    Creates or loads a Chroma collection, storing it in session_state
    """
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
    """
    'store' step: embed, then store
    """
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
    """
    'query' step -> 'retrieve' step
    """
    update_stage('query', {'query': query})

    collection = create_or_load_collection(collection_name)
    doc_count = len(collection.get().get("ids", []))
    if doc_count == 0:
        st.warning("No documents found in the collection. Please upload a document first.")
        return [], []

    query_embedding = embed_text([query])
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
    """
    'generate' step
    """
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
    st.set_page_config(page_title="RAG Demo", layout="wide")
    st.title("RAG Pipeline Demo (Vertical & Real-Time)")

    # Sidebar: Enter API key
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)

    col1, col2 = st.columns(2)

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

    # -------------
    # REACT VISUALIZATION
    # -------------

    with col2:
        st.title("🌀 Pipeline Visualization")
        
        # Prepare dynamic data
        component_args = {
            'currentStage': st.session_state.current_stage,
            'stageData': {
                stage: st.session_state.get(f'{stage}_data')
                for stage in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']
                if st.session_state.get(f'{stage}_data') is not None
            }
        }
        
        # Render the pipeline visualization
        components.html(
            get_pipeline_component(component_args),
            height=1200,
            scrolling=True
        )

if __name__ == "__main__":
    main()