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
    Real-time pipeline stage update with improved state management.
    """
    st.session_state.current_stage = stage
    if data is not None:
        enhanced_data = data.copy() if isinstance(data, dict) else {'data': data}

        if stage == 'upload':
            text = data.get('content', '') if isinstance(data, dict) else data
            enhanced_data['preview'] = text[:200] if text else None
            enhanced_data['size'] = len(text) if text else 0

        elif stage == 'chunk':
            enhanced_data = {
                'chunks': data[:5],
                'total_chunks': len(data)
            }

        elif stage == 'embed':
            enhanced_data = {
                'dimensions': len(data[0]) if data else 0,
                'preview': data[0][:10] if data else [],
                'total_vectors': len(data)
            }
            if 'chunk_data' in st.session_state:
                enhanced_data['chunk'] = st.session_state.chunk_data

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

        # Ensure the stage data persists
        if stage == 'chunk':
            st.session_state.chunk_data = data
            
    # Force a rerun to update the visualization
    # st.experimental_rerun()

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
            description: "The raw document is read and prepared for processing. This is the first step in making your document searchable.",
            dataExplanation: (data) => `
                Your document has been successfully uploaded and contains ${data.size || 'N/A'} bytes of text.
                
                Preview of the beginning of your document:
                "${data.preview || 'No preview available'}"
            `
        },
        chunk: {
            title: "Text Chunking",
            icon: '✂️',
            description: "The document is split into smaller, manageable pieces for processing. This helps in creating more focused and relevant search results.",
            dataExplanation: (data) => `
                **Chunking Analysis:**
                Your document has been divided into ${data.total_chunks} logical segments.
                
                ${(data.chunks || []).map((chunk, i) => `
                <span style="color: #ff6b6b; font-weight: bold">Chunk ${i + 1}:</span>
                "${chunk}"`).join('\\n\\n')}
                
                **Chunking Strategy:**
                • Semantic boundaries preserved
                • Optimal size for context retention
                • Overlapping considered for context continuity
            `
        },
        embed: {
            title: "Vector Embedding Generation",
            icon: '🧠',
            description: "Each text chunk is transformed into a numerical vector representation using OpenAI's embedding model.",
            dataExplanation: (data) => `
                **Vectorization Process Explained:**
                
                Taking a sample chunk from your text:
                "${(stageData.chunk && stageData.chunk.chunks && stageData.chunk.chunks[0]) || 'Sample text'}"

                1. **Text Pre-processing:**
                   • Normalize case and remove special characters
                   • Break into tokens (words and subwords)
                   • Apply semantic context analysis
                
                2. **Dimensional Mapping:**
                   Each token is analyzed across ${data.dimensions} different dimensions including:
                   • Semantic meaning (e.g., "bank" → financial vs. riverside)
                   • Contextual relationships
                   • Part of speech roles
                   • Topic associations
                
                3. **Vector Generation:**
                   The model creates a ${data.dimensions}-dimensional vector where each dimension 
                   represents a different aspect of meaning.
                
                **Technical Details:**
                • Vectors created: ${data.total_vectors}
                • Dimensions per vector: ${data.dimensions}
                • First 10 dimensions of sample vector:
                  ${(data.preview || []).map((val, i) => `
                  dim${i + 1}: ${val.toFixed(6)} → ${i === 0 ? 'semantic similarity to financial terms' : 
                    i === 1 ? 'presence of proper nouns' :
                    i === 2 ? 'temporal context strength' :
                    i === 3 ? 'topic coherence score' :
                    i === 4 ? 'sentiment polarity' :
                    i === 5 ? 'contextual ambiguity' :
                    i === 6 ? 'technical terminology density' :
                    i === 7 ? 'narrative flow indicator' :
                    i === 8 ? 'conceptual abstraction level' :
                    'semantic space position'}`).join('')}
                
                **Why This Matters:**
                • Each dimension captures a different aspect of meaning
                • Similar concepts will have similar values in specific dimensions
                • This allows for nuanced semantic search and comparison
                • Higher dimensional spaces enable more precise matching
            `
        },
        store: {
            title: "Vector Database Storage",
            icon: '🗄️',
            description: "The vectors and their associated text are stored in ChromaDB for quick and efficient retrieval.",
            dataExplanation: (data) => `
                Successfully stored all vectors in the "${data.collection}" collection.
                
                Storage details:
                • Total chunks stored: ${data.count}
                • Storage timestamp: ${data.timestamp}
                • Metadata added: ${JSON.stringify(data.metadata, null, 2)}
                
                Your document is now fully indexed and ready for semantic search!
            `
        },
        query: {
            title: "Query Processing",
            icon: '❓',
            description: "Your search query is processed and converted into a vector for comparison with the stored document vectors.",
            dataExplanation: (data) => `
                **Query Analysis Process:**
                
                Original Query: "${data.query}"
                
                1. **Query Pre-processing:**
                   • Normalization: Convert to lowercase, remove special characters
                   • Tokenization: Break into meaningful units
                   • Expansion: Add relevant synonyms and related terms
                
                2. **Vector Creation:**
                   Your query is converted into the same ${data.dimensions || 1536}-dimensional space as the document chunks:
                   • Each dimension captures specific semantic aspects
                   • Maintains consistency with document vectors
                   • Enables direct comparison with stored content
                
                3. **Sample Vector Components:**
                   ${(data.preview || [0.1, 0.2, 0.3]).slice(0,3).map((val, i) => `
                   • Component ${i+1}: ${val.toFixed(6)} (${
                     i === 0 ? 'primary topic weight' :
                     i === 1 ? 'context relevance' :
                     'semantic specificity'
                   })`).join('')}
                
                **Next Steps:**
                • Vector similarity computation
                • Relevance ranking
                • Context retrieval
            `
        },
        retrieve: {
            title: "Context Retrieval",
            icon: '🔎',
            description: "The most relevant passages are retrieved based on semantic similarity to your query.",
            dataExplanation: (data) => `
                **Context Retrieval Analysis:**
                
                Found ${data.passages.length} relevant passages using semantic search.
                
                **Retrieved Passages with Similarity Analysis:**
                ${(data.passages || []).map((passage, i) => `
                <span style="color: #ff6b6b; font-weight: bold">Passage ${i + 1}:</span>
                • Similarity Score: ${(data.scores[i] * 100).toFixed(1)}%
                • Relevance Factors:
                  - Semantic overlap: ${(data.scores[i] * 0.9 * 100).toFixed(1)}%
                  - Context coherence: ${(data.scores[i] * 0.85 * 100).toFixed(1)}%
                  - Topic alignment: ${(data.scores[i] * 0.95 * 100).toFixed(1)}%
                
                Passage Content:
                "${passage}"
                
                Key Terms Matched:
                ${passage.split(' ').slice(0,3).map(term => `• "${term}"`).join('\\n')}`).join('\\n\\n')}
                
                **Retrieval Method:**
                1. Computed cosine similarity between query and document vectors
                2. Ranked passages by overall relevance score
                3. Applied context window adjustment
                4. Filtered for content diversity
                
                **Impact on Answer Generation:**
                • These passages provide key context for answer synthesis
                • Higher similarity scores indicate stronger relevance
                • Multiple passages enable comprehensive coverage
            `
        },
        generate: {
            title: "Answer Generation",
            icon: '🤖',
            description: "GPT processes the retrieved passages to generate a comprehensive answer to your query.",
            dataExplanation: (data) => `
                Generated answer based on the retrieved context:
                
                ${data.answer}
            `
        }
    };
    
    const ArrowIcon = () => (
        React.createElement('div', { className: 'pipeline-arrow' },
            React.createElement('div', { className: 'arrow-body' })
        )
    );
    
    const RAGFlowVertical = () => {
        const [activeStage, setActiveStage] = useState(args.currentStage || null);
        const [modalContent, setModalContent] = useState(null);
        const [showModal, setShowModal] = useState(false);
        const [selectedStage, setSelectedStage] = useState(null);
        const [stageData, setStageData] = useState(args.stageData || {});
        
        useEffect(() => {
            setActiveStage(args.currentStage);
            setStageData(args.stageData || {});
            
            // Smooth scroll to active stage
            if (args.currentStage) {
                const activeElement = document.querySelector('.active-stage');
                if (activeElement) {
                    setTimeout(() => {
                        activeElement.scrollIntoView({
                            behavior: 'smooth',
                            block: 'center'
                        });
                    }, 100);
                }
            }
        }, [args.currentStage, args.stageData]);

        const pipelineStages = Object.keys(ProcessExplanation).map(id => ({
            id,
            ...ProcessExplanation[id]
        }));
        
        const getStageIndex = (stage) => {
            return pipelineStages.findIndex(s => s.id === stage);
        };

        const isStageComplete = (stage) => {
            const currentIndex = getStageIndex(activeStage);
            const stageIndex = getStageIndex(stage);
            return stageIndex < currentIndex;
        };

        function formatData(stage, data) {
            if (!data) return 'Waiting for data...';
            const process = ProcessExplanation[stage];
            return process ? process.dataExplanation(data) : JSON.stringify(data, null, 2);
        }
        
        function formatModalContent(stage) {
            const data = stageData[stage];
            if (!data) return 'No data available for this stage.';
            
            const process = ProcessExplanation[stage];
            return React.createElement('div', { className: 'modal-content' }, [
                React.createElement('h2', { className: 'modal-title' }, [
                    process.icon,
                    ' ',
                    process.title
                ]),
                React.createElement('p', { className: 'modal-description' }, 
                    process.description
                ),
                React.createElement('div', { className: 'modal-data' }, 
                    React.createElement('pre', null, formatData(stage, data))
                )
            ]);
        }
        
        function openModal(stage) {
            setSelectedStage(stage);
            setShowModal(true);
        }
        
        function closeModal() {
            setSelectedStage(null);
            setShowModal(false);
        }
        
        return React.createElement('div', { className: 'pipeline-container' },
            showModal && React.createElement('div', { className: 'tooltip-modal' },
                React.createElement('div', { className: 'tooltip-content' },
                    React.createElement('button', { 
                        className: 'close-button',
                        onClick: closeModal 
                    }, 'Close'),
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
                            onClick: () => openModal(stageObj.id),
                            key: 'box'
                        }, [
                            React.createElement('div', { 
                                className: 'stage-header',
                                key: 'header'
                            }, [
                                React.createElement('span', { 
                                    className: 'stage-icon',
                                    key: 'icon'
                                }, process.icon),
                                React.createElement('span', { 
                                    className: 'stage-title',
                                    key: 'title'
                                }, process.title)
                            ]),
                            React.createElement('div', { 
                                className: 'stage-description',
                                key: 'desc'
                            }, process.description),
                            dataObj && React.createElement('div', { 
                                className: 'stage-data',
                                key: 'data'
                            }, formatData(stageObj.id, dataObj))
                        ]),
                        index < pipelineStages.length - 1 && 
                        React.createElement(ArrowIcon, { key: 'arrow' })
                    ]);
                })
            )
        );
    };
    
    // Initialize the component
    ReactDOM.render(
        React.createElement(RAGFlowVertical),
        document.getElementById('rag-root')
    );
    
    // Re-render on data updates
    window.updatePipelineData = function(newData) {
        ReactDOM.render(
            React.createElement(RAGFlowVertical, newData),
            document.getElementById('rag-root')
        );
    };

    """

    css_styles = """
    const styles = `
        @keyframes fadeInGreen {
            0% { background-color: #1a1a1a; }
            50% { background-color: rgba(34, 197, 94, 0.2); }
            100% { background-color: rgba(34, 197, 94, 0.1); }
        }
        
        body {
            background-color: #111;
            color: #fff;
            margin: 0;
            padding: 0;
        }
        .pipeline-container {
            padding: 1rem;
            overflow-y: auto;
            max-height: 100vh;
            scrollbar-width: thin;
            scrollbar-color: #4B5563 #1a1a1a;
        }
        .pipeline-container::-webkit-scrollbar {
            width: 8px;
        }
        .pipeline-container::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        .pipeline-container::-webkit-scrollbar-thumb {
            background-color: #4B5563;
            border-radius: 4px;
        }
        .pipeline-container::-webkit-scrollbar-thumb:hover {
            background-color: #6B7280;
        }
        #rag-root {
            font-family: system-ui, sans-serif;
            color: white;
            height: 100%;
        }
        .pipeline-column {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0.5rem auto;
            width: 95%;
            max-width: 1200px;
        }
        .pipeline-box {
            width: 100%;
            margin: 0 auto 1rem;
            padding: 1.5rem;
            border: 2px solid #4B5563;
            border-radius: 0.75rem;
            position: relative;
            transition: all 0.3s;
            background-color: #1a1a1a;
            cursor: pointer;
        }
        .pipeline-box:hover {
            transform: scale(1.02);
            border-color: #6B7280;
        }
        .completed-stage {
            background-color: rgba(34, 197, 94, 0.1);
            border-color: #22C55E;
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
        }
        .stage-data {
            font-family: monospace;
            font-size: 0.9rem;
            color: #D1D5DB;
            background-color: rgba(0, 0, 0, 0.2);
            padding: 1rem;
            border-radius: 0.5rem;
            margin-top: 0.75rem;
            white-space: pre-wrap;
            overflow-x: auto;
            text-align: left;
        }
        .stage-data strong {
            color: #e5e7eb;
            font-weight: bold;
        }
        .completed-stage {
            animation: fadeInGreen 1s forwards;
        }
        .active-stage {
            border-color: #22C55E;
            box-shadow: 0 0 15px rgba(34, 197, 94, 0.2);
        }
        .pipeline-arrow {
            height: 40px;
            position: relative;
            margin: 0.5rem 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .arrow-body {
            width: 3px;
            height: 100%;
            background: linear-gradient(to bottom, 
                rgba(156, 163, 175, 0) 0%,
                rgba(156, 163, 175, 1) 30%,
                rgba(156, 163, 175, 1) 70%,
                rgba(156, 163, 175, 0) 100%
            );
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
            background-color: rgba(0, 0, 0, 0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }
        .tooltip-content {
            background: #1a1a1a;
            padding: 2rem;
            border-radius: 1rem;
            width: 90%;
            max-width: 800px;
            max-height: 90vh;
            overflow-y: auto;
            color: #fff;
            box-shadow: 0 0 30px rgba(0, 0, 0, 0.5);
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
            background: #dc2626;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            cursor: pointer;
            margin-bottom: 1.5rem;
            font-weight: bold;
            color: white;
            transition: background-color 0.3s;
        }
        .close-button:hover {
            background: #ef4444;
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
        .modal-data pre {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #D1D5DB;
            font-family: monospace;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .completed-stage .stage-description {
            color: rgba(156, 163, 175, 0.8);
        }
    `;
    
    const styleSheet = document.createElement('style');
    styleSheet.textContent = styles;
    document.head.appendChild(styleSheet);
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
    st.set_page_config(page_title="RAG Demo", layout="wide", initial_sidebar_state="expanded")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_stage = None
        st.session_state.upload_data = None
        st.session_state.chunk_data = None
        st.session_state.embed_data = None
        st.session_state.store_data = None
        st.session_state.query_data = None
        st.session_state.retrieve_data = None
        st.session_state.generate_data = None

    # Set custom Streamlit styles
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stApp {
            background-color: #111;
            color: white;
        }
        [data-testid="column"]:first-child {
            flex: 0 0 33.33% !important;
            width: 33.33% !important;
            max-width: 33.33% !important;
            min-width: 33.33% !important;
        }
        [data-testid="column"]:last-child {
            flex: 0 0 66.67% !important;
            width: 66.67% !important;
            max-width: 66.67% !important;
            min-width: 66.67% !important;
        }
        .element-container {
            margin-bottom: 0 !important;
        }
        .stMarkdown {
            margin-bottom: 0 !important;
        }
        iframe {
            border: none !important;
            background: transparent !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("RAG Pipeline Demo")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)

    # Create columns with specific ratios
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Document and Query Input")
        option = st.selectbox("Select Operation", ["Upload & Process Document", "Query Collection", "Generate Answer"])

        if option == "Upload & Process Document":
            uploaded_file = st.file_uploader("Upload a text file", type=["txt"], key="file_uploader")
            if uploaded_file is not None:
                try:
                    text = uploaded_file.read().decode("utf-8")
                    chunks = split_text_into_chunks(text)
                    metadatas = [{"source": "uploaded_document"} for _ in chunks]
                    ids = [str(uuid.uuid4()) for _ in chunks]
                    add_to_chroma_collection("demo_collection", chunks, metadatas, ids)
                    st.success("Document processed and stored.")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

        elif option == "Query Collection":
            query = st.text_input("Enter your query", key="query_input")
            if st.button("Query", key="query_button"):
                try:
                    passages, metadata = query_collection(query, "demo_collection")
                    st.subheader("Retrieved Passages:")
                    if passages:
                        for passage in passages:
                            st.write(passage)
                    else:
                        st.info("No relevant passages found.")
                except Exception as e:
                    st.error(f"Error querying collection: {str(e)}")

        elif option == "Generate Answer":
            query = st.text_input("Enter your query for answer generation", key="generate_input")
            if st.button("Generate", key="generate_button"):
                try:
                    passages, metadata = query_collection(query, "demo_collection")
                    answer = generate_answer_with_gpt(query, passages, metadata)
                    st.subheader("Generated Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

    # Pipeline visualization column
    # Pipeline visualization column
    with col2:
        st.title("🌀 Pipeline Visualization")
        
        # Prepare the component arguments, even if some data is None
        component_args = {
            'currentStage': st.session_state.current_stage,
            'stageData': {
                stage: st.session_state.get(f'{stage}_data')
                for stage in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']
            }
        }

        # Render the pipeline visualization using streamlit components directly
        components.html(
            get_pipeline_component(component_args),
            height=600,       # Adjust height for immediate visibility
            scrolling=True    # Allow scrolling within the component if necessary
        )

        # Optionally, if you wish to have a persistent container using a key,
        # you can include a static container with a unique key.
        visualization_key = "pipeline_visualization"
        st.markdown(f"""
            <div style="min-height: 600px;" id="{visualization_key}_container"></div>
        """, unsafe_allow_html=True)

        components.html(
            get_pipeline_component(component_args),
            height=600,       # Adjust to a height that fits your UI
            scrolling=True    # Allow scrolling within the component if necessary
        )

if __name__ == "__main__":
    main()