import os
# Disable multi-tenancy for Chroma (must be set before importing chromadb)
os.environ["CHROMADB_DISABLE_TENANCY"] = "true"

import chromadb
from chromadb.config import Settings

# Instantiate the Chroma client using the Settings object.
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="chromadb_storage"
    )
)

import tempfile
import uuid
import re
from typing import List

import streamlit as st

# For token counting if needed
import tiktoken

# Import the new OpenAI client interface.
from openai import OpenAI

# Global variable for the new OpenAI client.
new_client = None

# ---------------------------------------------------------------------------------------
# 1. Helper Functions
# ---------------------------------------------------------------------------------------

def set_openai_api_key(api_key: str):
    """
    Sets the OpenAI API key and instantiates the new client.
    """
    global new_client
    new_client = OpenAI(api_key=api_key)

def split_text_into_chunks(text: str) -> List[str]:
    """
    Splits the text into chunks using two or more newlines as the delimiter.
    """
    chunks = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return chunks

def embed_text(texts: List[str], openai_embedding_model: str = "text-embedding-3-large"):
    """
    Uses the new OpenAI client interface to generate embeddings for a list of text passages.
    Returns a list of embeddings.
    """
    response = new_client.embeddings.create(input=texts, model=openai_embedding_model)
    embeddings = [item.embedding for item in response.data]
    return embeddings

def create_or_load_collection(collection_name: str):
    """Creates or loads a Chroma collection by name."""
    try:
        collection = chroma_client.get_collection(collection_name=collection_name)
    except Exception:
        collection = chroma_client.create_collection(name=collection_name)
    return collection

def add_to_chroma_collection(collection_name: str,
                               chunks: List[str],
                               metadatas: List[dict],
                               ids: List[str]):
    """Adds text passages (with metadata and IDs) to a Chroma collection."""
    collection = create_or_load_collection(collection_name)
    embeddings = embed_text(chunks)
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )

def query_collection(query: str, collection_name: str, n_results: int = 3):
    """
    Retrieves the most relevant text passages from the specified collection using semantic search.
    """
    collection = create_or_load_collection(collection_name)
    results = collection.query(query_texts=[query], n_results=n_results)
    retrieved_passages = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]
    return retrieved_passages, retrieved_metadata

def generate_answer_with_gpt(query: str,
                             retrieved_passages: List[str],
                             retrieved_metadata: List[dict],
                             system_instruction: str = (
                                 "You are a helpful legal assistant. Answer the following query based ONLY on the provided context. "
                                 "Your answer must begin with a TL;DR summary in bullet points. Each bullet point must be concrete, measurable, and deterministic, "
                                 "so that a sales employee can quickly check if his use case complies with Swiss law. "
                                 "For example, your bullet points might be: "
                                 "- Automatic opening mechanism: Must not be present. "
                                 "- Total length (fully open): Must be 12 cm or less. "
                                 "- Blade length: Must be 5 cm or less. "
                                 "- Verify absence of valid exemption permits. "
                                 "- ..."
                                 "After the TL;DR, provide a detailed explanation."
                                 "Add explicit legal citations but ONLY from the provided context whenever possible. Explicitly cite all relevant criteria (ex. a knife must have total length > x cm and blade length > y cm)"
                                 "Ensure perfect numeric accuracy in relation to the provided context (no estimations or deviations, strictly adhere to context)."
                                 "Maintain a great balance of conciseness vs completeness & relevance."
                                 "Plz stay as close as possible to this main legal document in your main answer; but after your main answer cite relevant legal sources or considerations that might be relevant too in the context."
                             )):
    """
    Combines the user query and retrieved text passages into a single prompt,
    then uses the new OpenAI client interface (o1-mini) to generate an answer.
    """
    # Build the context block by joining the retrieved passages.
    context_text = "\n\n".join(retrieved_passages)
    
    # Build the final prompt.
    final_prompt = (
        f"{system_instruction}\n\n"
        f"Context:\n{context_text}\n\n"
        f"User Query: {query}\nAnswer:"
    )
    
    # Use o1-mini with a single message (role "user").
    completion = new_client.chat.completions.create(
        model="chatgpt-4o-latest", # Alternative: o1-mini-2024-09-12  o1-preview-2024-09-12  chatgpt-4o-latest
        messages=[{"role": "user", "content": final_prompt}],
        response_format={"type": "text"}
    )
    
    return completion.choices[0].message.content

# ---------------------------------------------------------------------------------------
# 2. Streamlit UI
# ---------------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="RAG App", layout="wide")
    st.title("📝 Simple RAG (Retrieval-Augmented Generation) Demo")

    # Sidebar: API Key Input
    st.sidebar.subheader("OpenAI API Key")
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to enable embeddings and GPT queries."
    )
    if api_key:
        set_openai_api_key(api_key)

    st.sidebar.write("---")
    # Sidebar: Collection Name
    st.sidebar.subheader("Chroma Collection")
    collection_name = st.sidebar.text_input(
        "Collection Name:",
        value="my_legal_collection",
        help="Name your collection (or use an existing one)."
    )

    # Main Section: File Uploader
    st.header("1. Add Your Text Document")
    uploaded_file = st.file_uploader("Upload a text-only file", type=["txt"])
    if uploaded_file and api_key:
        file_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        if st.button("Add to RAG"):
            with st.spinner("Splitting text into chunks and embedding..."):
                # Split text into chunks (simply based on newlines)
                chunks = split_text_into_chunks(file_text)
                metadatas = [{"source": uploaded_file.name} for _ in chunks]
                ids = [str(uuid.uuid4()) for _ in chunks]
                add_to_chroma_collection(collection_name, chunks, metadatas, ids)
            st.success("Document successfully added to the collection!")

    st.write("---")
    # Query Section
    st.header("2. Ask Questions About Your Collection")
    user_query = st.text_area("Enter your question:")
    if st.button("Get Answer") and user_query.strip() != "":
        if not api_key:
            st.error("Please provide your OpenAI API key first.")
        else:
            with st.spinner("Retrieving relevant passages..."):
                retrieved_passages, metadata = query_collection(
                    user_query,
                    collection_name=collection_name,
                    n_results=3
                )
            with st.spinner("Generating final answer with GPT..."):
                final_answer = generate_answer_with_gpt(user_query, retrieved_passages, metadata)
            st.subheader("Answer:")
            st.write(final_answer)
            with st.expander("Retrieved Context Passages"):
                for passage in retrieved_passages:
                    st.markdown(f"{passage}")

if __name__ == "__main__":
    main()