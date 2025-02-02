import os
# **Disable multi-tenancy for Chroma** (must be set before importing chromadb)
os.environ["CHROMADB_DISABLE_TENANCY"] = "true"

import chromadb
from chromadb.config import Settings
import streamlit as st
import streamlit.components.v1 as components
import tempfile
import uuid
import re
from typing import List
import json
import time
import numpy as np       # Optional: for numerical operations if needed
import tiktoken          # Optional: for token counting

from openai import OpenAI

# -----------------------------------------------------------------------------
# Global Variables and Client Initialization
# -----------------------------------------------------------------------------

new_client = None  # Global variable for the OpenAI client

chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="chromadb_storage"
    )
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

if 'current_stage' not in st.session_state:
    st.session_state.current_stage = None

for stage in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']:
    if f'{stage}_data' not in st.session_state:
        st.session_state[f'{stage}_data'] = None

# -----------------------------------------------------------------------------
# RAG State Class for Structured Pipeline Status
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
# Helper Functions (Pipeline Operations)
# -----------------------------------------------------------------------------

def update_stage(stage: str, data=None):
    """
    **Update the current pipeline stage** and record enhanced data previews.
    
    This function sets both a flat session key (e.g., 'upload_data') and, if available,
    updates the RAGState instance.
    """
    st.session_state.current_stage = stage
    if data is not None:
        enhanced_data = data.copy() if isinstance(data, dict) else {'data': data}
        if stage == 'upload':
            text = data.get('content', '') if isinstance(data, dict) else data
            enhanced_data['preview'] = text[:200] if text else None
        elif stage == 'chunk':
            enhanced_data = {'chunks': data[:5], 'total_chunks': len(data)}
        elif stage == 'embed':
            enhanced_data = {'dimensions': len(data[0]) if data else 0,
                             'preview': data[0][:10] if data else [],
                             'total_vectors': len(data)}
        elif stage == 'store':
            enhanced_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
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
    **Set the OpenAI API key** and instantiate the new client.
    """
    global new_client
    new_client = OpenAI(api_key=api_key)

def split_text_into_chunks(text: str) -> List[str]:
    """
    **Split text into chunks** using two or more newlines as delimiters,
    with updates to the pipeline state.
    """
    update_stage('upload', {'content': text, 'size': len(text)})
    time.sleep(0.5)  # Slow down briefly for visualization clarity.
    chunks = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    update_stage('chunk', chunks)
    return chunks

def embed_text(texts: List[str], openai_embedding_model: str = "text-embedding-3-large"):
    """
    **Generate vector embeddings** for a list of text passages and update the pipeline state.
    """
    if new_client is None:
        st.error("OpenAI client not initialized. Please set your API key in the sidebar.")
        return []
    time.sleep(0.5)  # Brief delay for visibility.
    update_stage('embed')
    response = new_client.embeddings.create(input=texts, model=openai_embedding_model)
    embeddings = [item.embedding for item in response.data]
    update_stage('embed', embeddings)
    return embeddings

def create_or_load_collection(collection_name: str):
    """
    **Create or load a Chroma collection** by name.
    
    To avoid re‑creating a new (empty) collection each time, we store the collection
    in st.session_state once it is created.
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
    **Store text passages in the vector database.**
    
    This function generates embeddings for the provided chunks using your custom
    embedding function and then adds them (with metadata and IDs) to the specified collection.
    """
    time.sleep(0.5)
    update_stage('store')
    collection = create_or_load_collection(collection_name)
    embeddings = embed_text(chunks)
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    update_stage('store', {'collection': collection_name,
                           'count': len(chunks),
                           'metadata': metadatas[0] if metadatas else None})
    st.write("Stored", len(chunks), "vectors in", collection_name)

def query_collection(query: str, collection_name: str, n_results: int = 3):
    """
    **Retrieve relevant passages** from the collection using your custom OpenAI embedding.
    (Ensure that the collection's stored embeddings come from the same model as used here.)
    """
    update_stage('retrieve')
    collection = create_or_load_collection(collection_name)
    
    # Debug: retrieve all stored documents.
    stored_docs = collection.get()
    st.write("Stored documents (debug):", stored_docs)
    
    # Compute document count from stored IDs.
    doc_count = len(stored_docs.get("ids", []))
    st.write("Documents in collection:", doc_count)
    if doc_count == 0:
        st.warning("No documents found in the collection. Please upload a document first.")
        return [], []
    
    # Pre-compute the query embedding.
    query_embedding = embed_text([query])
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    
    # Query using the pre-computed embedding.
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    st.write("Raw query results:", results)  # Debug output
    
    # Extract retrieved passages and metadata.
    retrieved_passages = results.get("documents", [[]])[0]
    retrieved_metadata = results.get("metadatas", [[]])[0]
    update_stage('retrieve', {'passages': retrieved_passages, 'metadata': retrieved_metadata})
    return retrieved_passages, retrieved_metadata

def generate_answer_with_gpt(query: str,
                             retrieved_passages: List[str],
                             retrieved_metadata: List[dict],
                             system_instruction: str = (
                                 "You are a helpful legal assistant. Answer the following query based ONLY on the provided context. "
                                 "Your answer must begin with a TL;DR summary in bullet points. Each bullet point must be concrete, measurable, and deterministic, "
                                 "so that a sales employee can quickly check if his use case complies with Swiss law. "
                                 "After the TL;DR, provide a detailed explanation with explicit legal citations if possible."
                             )):
    """
    **Generate a final answer** using GPT by combining the query and the retrieved context.
    """
    update_stage('generate')
    context_text = "\n\n".join(retrieved_passages)
    final_prompt = (
        f"{system_instruction}\n\n"
        f"Context:\n{context_text}\n\n"
        f"User Query: {query}\nAnswer:"
    )
    completion = new_client.chat.completions.create(
        model="chatgpt-4o-latest",  # Or your chosen model
        messages=[{"role": "user", "content": final_prompt}],
        response_format={"type": "text"}
    )
    answer = completion.choices[0].message.content
    update_stage('generate', {'answer': answer})
    return answer

# -----------------------------------------------------------------------------
# Main Application and UI
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="RAG Demo", layout="wide")
    st.title("RAG Pipeline Demo")
    
    # Sidebar: API key and settings.
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        set_openai_api_key(api_key)
    
    # Two-column layout: left for input/controls; right for React visualization.
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Document and Query Input")
        option = st.selectbox("Select Operation", ["Upload & Process Document", "Query Collection", "Generate Answer"])
        
        if option == "Upload & Process Document":
            uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
            if uploaded_file is not None:
                progress_bar = st.progress(0)
                text = uploaded_file.read().decode("utf-8")
                progress_bar.progress(20)
                chunks = split_text_into_chunks(text)
                progress_bar.progress(50)
                metadatas = [{"source": "uploaded_document"} for _ in chunks]
                ids = [str(uuid.uuid4()) for _ in chunks]
                add_to_chroma_collection("demo_collection", chunks, metadatas, ids)
                progress_bar.progress(100)
                st.success("Document processed and stored.")
        
        elif option == "Query Collection":
            query = st.text_input("Enter your query")
            if st.button("Query"):
                passages, metadata = query_collection(query, "demo_collection")
                st.subheader("Retrieved Passages:")
                if passages and any(passage.strip() for passage in passages):
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
        st.title("🔄 Pipeline Visualization")
        # Prepare component arguments from session state.
        component_args = {
            'currentStage': st.session_state.current_stage,
            'stageData': {
                stage: st.session_state.get(f'{stage}_data')
                for stage in ['upload', 'chunk', 'embed', 'store', 'query', 'retrieve', 'generate']
                if st.session_state.get(f'{stage}_data') is not None
            }
        }
        
        # Embed the React-based visualization.
        components.html(
            f"""
            <div id="rag-root"></div>
            
            <!-- Required CDN scripts -->
            <script src="https://cdnjs.cloudflare.com/ajax/libs/react/17.0.2/umd/react.production.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/17.0.2/umd/react-dom.production.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/lucide@latest"></script>

            <style>
                /* Tailwind-like utility classes */
                .bg-blue-500 {{ background-color: #3B82F6; }}
                .bg-green-500 {{ background-color: #22C55E; }}
                .bg-yellow-500 {{ background-color: #EAB308; }}
                .bg-purple-500 {{ background-color: #A855F7; }}
                .bg-pink-500 {{ background-color: #EC4899; }}
                .bg-orange-500 {{ background-color: #F97316; }}
                .bg-red-500 {{ background-color: #EF4444; }}
                .text-white {{ color: white; }}
                .text-gray-600 {{ color: #4B5563; }}
                .text-gray-300 {{ color: #D1D5DB; }}
                .text-gray-700 {{ color: #374151; }}
                .text-gray-800 {{ color: #1F2937; }}
                .bg-gray-50 {{ background-color: #F9FAFB; }}
                .bg-gray-800 {{ background-color: #1F2937; }}
                .bg-white {{ background-color: white; }}
                .border-gray-200 {{ border-color: #E5E7EB; }}
                .rounded-lg {{ border-radius: 0.5rem; }}
                .rounded-md {{ border-radius: 0.375rem; }}
                .shadow-lg {{ box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }}
                .shadow-md {{ box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
                .shadow-sm {{ box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05); }}
                .p-6 {{ padding: 1.5rem; }}
                .p-4 {{ padding: 1rem; }}
                .p-3 {{ padding: 0.75rem; }}
                .p-2 {{ padding: 0.5rem; }}
                .px-4 {{ padding-left: 1rem; padding-right: 1rem; }}
                .py-2 {{ padding-top: 0.5rem; padding-bottom: 0.5rem; }}
                .mb-8 {{ margin-bottom: 2rem; }}
                .mb-4 {{ margin-bottom: 1rem; }}
                .mb-2 {{ margin-bottom: 0.5rem; }}
                .mt-2 {{ margin-top: 0.5rem; }}
                .mt-4 {{ margin-top: 1rem; }}
                .space-x-4 > * + * {{ margin-left: 1rem; }}
                .space-x-3 > * + * {{ margin-left: 0.75rem; }}
                .flex {{ display: flex; }}
                .items-center {{ align-items: center; }}
                .font-medium {{ font-weight: 500; }}
                .font-bold {{ font-weight: 700; }}
                .text-2xl {{ font-size: 1.5rem; }}
                .text-lg {{ font-size: 1.125rem; }}
                .text-sm {{ font-size: 0.875rem; }}
                .text-xs {{ font-size: 0.75rem; }}
                .w-6 {{ width: 1.5rem; }}
                .h-6 {{ height: 1.5rem; }}
                .w-64 {{ width: 16rem; }}
                .relative {{ position: relative; }}
                .absolute {{ position: absolute; }}
                .top-full {{ top: 105%; }}  /* Adjusted to avoid overlap */
                .left-0 {{ left: 0; }}
                .z-50 {{ z-index: 50; }}
                .cursor-pointer {{ cursor: pointer; }}
                .transition-all {{ transition-property: all; }}
                .duration-300 {{ transition-duration: 300ms; }}
                .scale-105 {{ transform: scale(1.05); }}
                .hover\:shadow-md:hover {{ box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
                .max-w-6xl {{ max-width: 72rem; }}
                .mx-auto {{ margin-left: auto; margin-right: auto; }}
                .border-2 {{ border-width: 2px; }}
                .flex-shrink-0 {{ flex-shrink: 0; }}
                .ring-2 {{ box-shadow: 0 0 0 2px #22C55E; }}
                .ring-green-500 {{ --tw-ring-color: #22C55E; }}
                /* Tooltip style adjustment */
                .tooltip {{
                    pointer-events: none;
                    white-space: normal;
                }}
            </style>

            <script>
                const RAGFlowVisualization = () => {{
                    const args = {json.dumps(component_args)};
                    const [activeStage, setActiveStage] = React.useState(args.currentStage || null);
                    const [showPreview, setShowPreview] = React.useState(false);
                    
                    React.useEffect(() => {{
                        setActiveStage(args.currentStage);
                        if (args.currentStage) {{
                            setShowPreview(true);
                            setTimeout(() => setShowPreview(false), 5000);
                        }}
                    }}, [args.currentStage]);

                    const stages = {{
                        upload: {{ 
                            title: 'Document Upload', 
                            color: 'bg-blue-500',
                            icon: 'FileText',
                            description: 'Raw document processing',
                            detailText: 'Document is read and validated'
                        }},
                        chunk: {{ 
                            title: 'Text Chunking', 
                            color: 'bg-green-500',
                            icon: 'Code',
                            description: 'Text split into semantic chunks',
                            detailText: 'Document divided into manageable pieces'
                        }},
                        embed: {{ 
                            title: 'Vector Embedding', 
                            color: 'bg-yellow-500',
                            icon: 'Brain',
                            description: 'Converting to vectors',
                            detailText: 'Text converted to numerical vectors'
                        }},
                        store: {{ 
                            title: 'Vector Storage', 
                            color: 'bg-purple-500',
                            icon: 'Database',
                            description: 'Storing in database',
                            detailText: 'Vectors stored in ChromaDB'
                        }},
                        query: {{ 
                            title: 'Query Processing', 
                            color: 'bg-pink-500',
                            icon: 'MessageSquare',
                            description: 'Processing question',
                            detailText: 'User question vectorized'
                        }},
                        retrieve: {{ 
                            title: 'Context Retrieval', 
                            color: 'bg-orange-500',
                            icon: 'Search',
                            description: 'Finding relevant passages',
                            detailText: 'Similar vectors retrieved'
                        }},
                        generate: {{ 
                            title: 'Answer Generation', 
                            color: 'bg-red-500',
                            icon: 'Bot',
                            description: 'Generating answer',
                            detailText: 'GPT generates final response'
                        }}
                    }};

                    const formatData = (stage, data) => {{
                        switch(stage) {{
                            case 'upload':
                                return `File: ${{data.filename || ''}}\\nSize: ${{data.size || ''}} bytes\\nPreview:\\n${{data.preview || ''}}`;
                            case 'chunk':
                                return data.chunks.slice(0, 3).map((chunk, i) => 
                                    `Chunk ${{i + 1}}:\\n${{chunk.slice(0, 100)}}...`
                                ).join('\\n\\n');
                            case 'embed':
                                return `Embedding dimensions: ${{data.dimensions}}\\nFirst vector preview:\\n${{
                                    JSON.stringify(data.preview.slice(0, 5))
                                }}...`;
                            case 'store':
                                return `Stored ${{data.count}} vectors in ${{data.collection}}`;
                            case 'query':
                                return `Query: "${{data.query || ''}}"`;
                            case 'retrieve':
                                return data.passages.slice(0, 2).map((p, i) => 
                                    `Match ${{i + 1}} (score: ${{data.scores[i]}}):\\n${{p.slice(0, 100)}}...`
                                ).join('\\n\\n');
                            case 'generate':
                                return `Generated answer:\\n${{data.answer.slice(0, 200)}}...`;
                            default:
                                return '';
                        }}
                    }};

                    const DataPreview = ({{ stage, data }}) => {{
                        if (!data) return null;
                        return React.createElement('div', {{
                            className: 'data-preview tooltip mt-4'
                        }}, [
                            React.createElement('pre', {{
                                className: showPreview ? 'highlight' : ''
                            }}, formatData(stage, data))
                        ]);
                    }};

                    const StageBox = ({{ stage }}) => {{
                        const stageInfo = stages[stage];
                        const isActive = activeStage === stage;
                        const hasData = args.stageData && args.stageData[stage];
                        return React.createElement('div', {{
                            className: `relative p-4 rounded-lg border-2 transition-all duration-300 
                                ${{isActive ? `${{stageInfo.color}} text-white shadow-lg animate-pulse` : 'bg-white border-gray-200'}}
                                ${{hasData ? 'ring-2 ring-green-500' : ''}}
                                hover:shadow-md cursor-pointer`,
                            onMouseEnter: () => setActiveStage(stage),
                            onMouseLeave: () => setActiveStage(args.currentStage || null)
                        }}, [
                            React.createElement('div', {{
                                className: 'flex items-center space-x-3',
                                key: 'content'
                            }}, [
                                React.createElement('span', {{
                                    className: `lucide lucide-${{stageInfo.icon.toLowerCase()}} ${{isActive ? 'text-white' : 'text-gray-600'}}`,
                                    key: 'icon'
                                }}),
                                React.createElement('span', {{
                                    className: 'font-medium',
                                    key: 'title'
                                }}, stageInfo.title)
                            ]),
                            isActive && React.createElement('div', {{
                                className: 'absolute top-full left-0 mt-2 w-64 p-3 bg-gray-800 text-white text-sm rounded-md z-50 shadow-lg',
                                key: 'tooltip'
                            }}, [
                                React.createElement('p', {{
                                    className: 'font-medium mb-2',
                                    key: 'description'
                                }}, stageInfo.description),
                                React.createElement('p', {{
                                    className: 'text-gray-300 text-xs',
                                    key: 'detail'
                                }}, stageInfo.detailText),
                                hasData && React.createElement(DataPreview, {{
                                    stage,
                                    data: args.stageData[stage],
                                    key: 'preview'
                                }})
                            ])
                        ]);
                    }};

                    const Arrow = () => React.createElement('div', {{
                        className: 'flex-shrink-0'
                    }}, React.createElement('span', {{
                        className: 'lucide lucide-arrow-right text-gray-400'
                    }}));

                    return React.createElement('div', {{
                        className: 'max-w-6xl mx-auto p-6 bg-gray-50 rounded-xl shadow-sm'
                    }}, [
                        React.createElement('h2', {{
                            className: 'text-2xl font-bold mb-8 text-center text-gray-800',
                            key: 'title'
                        }}, 'RAG Pipeline Visualization'),
                        React.createElement('div', {{
                            className: 'mb-12',
                            key: 'document-processing'
                        }}, [
                            React.createElement('h3', {{
                                className: 'text-lg font-semibold mb-4 text-gray-700',
                                key: 'doc-title'
                            }}, 'Document Processing Pipeline'),
                            React.createElement('div', {{
                                className: 'flex items-center space-x-4',
                                key: 'doc-stages'
                            }}, [
                                React.createElement(StageBox, {{ stage: 'upload', key: 'upload' }}),
                                React.createElement(Arrow, {{ key: 'arrow1' }}),
                                React.createElement(StageBox, {{ stage: 'chunk', key: 'chunk' }}),
                                React.createElement(Arrow, {{ key: 'arrow2' }}),
                                React.createElement(StageBox, {{ stage: 'embed', key: 'embed' }}),
                                React.createElement(Arrow, {{ key: 'arrow3' }}),
                                React.createElement(StageBox, {{ stage: 'store', key: 'store' }})
                            ])
                        ]),
                        React.createElement('div', {{
                            key: 'query-processing'
                        }}, [
                            React.createElement('h3', {{
                                className: 'text-lg font-semibold mb-4 text-gray-700',
                                key: 'query-title'
                            }}, 'Query Processing Pipeline'),
                            React.createElement('div', {{
                                className: 'flex items-center space-x-4',
                                key: 'query-stages'
                            }}, [
                                React.createElement(StageBox, {{ stage: 'query', key: 'query' }}),
                                React.createElement(Arrow, {{ key: 'arrow4' }}),
                                React.createElement(StageBox, {{ stage: 'retrieve', key: 'retrieve' }}),
                                React.createElement(Arrow, {{ key: 'arrow5' }}),
                                React.createElement(StageBox, {{ stage: 'generate', key: 'generate' }})
                            ])
                        ])
                    ]);
                }};
                ReactDOM.render(
                    React.createElement(RAGFlowVisualization),
                    document.getElementById('rag-root')
                );
            </script>
            """,
            height=800
        )
    
if __name__ == "__main__":
    main()