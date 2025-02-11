# user_view.py
import streamlit as st
import os
import json
import requests
import re
import uuid
import pycountry

# ============== ADAPT THESE AS NEEDED ================
HARD_CODED_API_KEY = "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"   # <--- Replace with your actual key
CHROMA_COLLECTION_NAME = "rag_collection"
USER_DB_FILE = "users.json"     # same user DB as main app
# =====================================================

# ==========================
#  1) USERS & LOGIN
# ==========================
def load_users():
    """Load user data from the same JSON file used in rag_app.py."""
    if not os.path.exists(USER_DB_FILE):
        return {}
    with open(USER_DB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def verify_login(username, password):
    """Check username/password matches our user store."""
    users = load_users()
    return (username in users) and (users[username] == password)

# ==========================
#  2) "OpenAI" HTTP Client 
#  (mirrors main app's approach)
# ==========================
class OpenAI:
    """
    Minimal replication of your main code’s approach, doing raw
    POST requests to the /chat/completions endpoint.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers["Content-Type"] = "application/json"

    def chat_completions_create(self, model, messages, max_tokens=1024, temperature=0.2):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        resp = self.session.post(f"{self.base_url}/chat/completions", json=payload)
        if resp.status_code != 200:
            raise Exception(f"OpenAI API error: {resp.status_code}\nResponse: {resp.text}")
        return resp.json()

# ==========================
#  3) CHROMA + Embeddings
# ==========================
#   We'll replicate a minimal version of the "init Chroma client" logic
import chromadb
from chromadb.config import Settings

def get_user_chroma_client(user_id: str):
    """
    Returns a Chroma client that points to user-specific directory,
    so the same data as the main app is used.
    """
    user_dir = f"chromadb_storage_user_{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    return chromadb.Client(
        Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=user_dir
        )
    )

# ==========================
#  4) DETECT COUNTRIES
#  (similar to LLMCountryDetector in main app)
# ==========================
class LLMCountryDetector:
    SYSTEM_PROMPT = """
        You are a specialized assistant for extracting ALL country references in ANY user text.
        You must detect and extract EVERY single country reference, including 2-letter codes,
        full country names, etc. 
        Output EXACT JSON like:
        [
            {"detected_phrase": "CH", "code": "CH"},
            {"detected_phrase": "Switzerland", "code": "CH"}
        ]
    """.strip()

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key)
        self.model = model

    def detect_countries_in_text(self, text: str):
        if not text.strip():
            return []
        # We'll do a simple chat request with the system prompt
        try:
            response = self.client.chat_completions_create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=512,
                temperature=0.0
            )
            raw_content = response["choices"][0]["message"]["content"].strip()
            data = []
            # Attempt to parse JSON
            try:
                data = json.loads(raw_content)
            except:
                # fallback: naive parse with regex
                data = []
                matches = re.findall(r'{"detected_phrase":\s*"([^"]+)",\s*"code":\s*"([A-Z]{2})"}', raw_content)
                for m in matches:
                    phrase, code = m
                    data.append({"detected_phrase": phrase, "code": code})
            # filter out invalid
            results = []
            used = set()
            for d in data:
                code = d.get("code", "").upper()
                phrase = d.get("detected_phrase", "")
                if len(code) == 2 and code.isalpha() and phrase:
                    if code not in used:
                        used.add(code)
                        results.append({"detected_phrase": phrase, "code": code})
            return results
        except Exception as e:
            # as fallback, do naive approach
            return []

# ==========================
#  5) The EXACT BASE PROMPT
#     used for CASEA / CASEB
# ==========================
BASE_DEFAULT_PROMPT = (
    "  <ROLE>\n"
    "    You are an extremely knowledgeable and helpful assistant. Respond to the user’s query **ONLY** by using\n"
    "    the information available in the RAG documents. Always reason step-by-step. The user appreciates\n"
    "    thorough, accurate results.\n"
    "  </ROLE>\n\n"
    "  <INSTRUCTIONS>\n"
    "    1. Always rely exclusively on the RAG documents for any factual information.\n\n"
    "    2. EXTREMELY IMPORTANT:\n"
    "       - If the user’s query relates to **only one** country and your RAG does **not** have matching information\n"
    "         for that country, you must use the **CASEB** structure.\n"
    "       - If the user’s query references **multiple** countries, you must still present a **CASEA** structure for\n"
    "         each country you do have data on. For any country **not** found in the RAG documents, strictly state\n"
    "         \"No information in my documents.\" instead of presenting partial data.\n\n"
    "    3. When the user explicitly asks for help, your response must start with a **High-Level, Concise\n"
    "       'Instructions to Action'** section drawn directly from the doc (e.g., \"If x > y, then do z...\").\n\n"
    "    4. Follow with a **TL;DR Summary** in bullet points (again, only using doc-based content). Emphasize crucial\n"
    "       numerical thresholds or legal references in **bold**, and any important nuance in *italics*.\n\n"
    "    5. Next, provide a **Detailed Explanation** that remains strictly grounded in the RAG documents. If helpful,\n"
    "       include a *brief scenario* illustrating how these doc-based rules might apply.\n\n"
    "    6. Conclude with an **'Other References'** section, where you may optionally add clarifications or knowledge\n"
    "       beyond the doc but **label it** as external info. Any statutory references should appear in square brackets,\n"
    "       e.g., [Section 1, Paragraph 2].\n\n"
    "    7. If the user’s query **cannot** be answered with information from the RAG documents (meaning you have\n"
    "       **zero** coverage for that country or topic), you must switch to **CASEB**, which requires:\n"
    "       - A large \"Sorry!\" header: \"The uploaded document states nothing relevant...\"\n"
    "       - A large \"Best guess\" header: attempt an interpretation, clearly flagged as conjecture.\n"
    "       - A final large header in **red**, titled \"The fun part :-)\". Label it with *(section requested in\n"
    "         Step 0 to show how output can be steered)* in normal text. Provide a sarcastic or lighthearted\n"
    "         reflection (with emojis).\n\n"
    "    8. In all doc-based sections, stick strictly to the RAG documents (no external knowledge), keep your\n"
    "       professional or academically rigorous style, and preserve **bold** for pivotal references and *italics*\n"
    "       for nuance.\n\n"
    "    9. Always respond in the user’s initial query language, unless otherwise instructed.\n\n"
    "    10. Present your final output in normal text (headings in large text as described), **never** in raw XML.\n"
    "  </INSTRUCTIONS>\n\n"
    "  <STRUCTURE>\n"
    "    <!-- see CASEA vs CASEB, etc... identical to the main app -->\n"
    "  </STRUCTURE>\n\n"
    "  <FINAL_REMARKS>\n"
    "    - Do **not** guess if you lack data for a specific country. Instead, say \"No information in my documents.\"\n"
    "      or use **CASEB** if no data is found at all.\n"
    "    - Always apply step-by-step reasoning and keep the user’s question fully in mind.\n"
    "  </FINAL_REMARKS>"
)

# ==========================
#  6) Query Collection 
#     (like query_collection in main app)
# ==========================
def query_chroma_for_countries(user_id: str, query: str, n_results=10):
    """
    We replicate your main approach: 
      1) detect countries
      2) if none found => do broad search
      3) else do one sub-query per country
      4) gather passages & metadata
    """
    # detect
    detector = LLMCountryDetector(HARD_CODED_API_KEY)
    detected = detector.detect_countries_in_text(query)
    codes = [d["code"] for d in detected]

    client = get_user_chroma_client(user_id)
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    except:
        st.warning("No RAG collection found, or user has none. Please upload/ingest docs in the main app first.")
        return [], []

    # gather all docs to see which countries exist
    all_docs = collection.get()
    all_metadata = all_docs.get("metadatas", [])
    available_countries = set()
    for m in all_metadata:
        if m and "country_code" in m:
            available_countries.add(m["country_code"])

    combined_passages = []
    combined_metadata = []
    seen_passages = set()

    if codes:
        # do queries for each code
        for code in codes:
            # if code not in available => no results
            if code not in available_countries:
                continue
            results = collection.query(query_texts=[query], where={"country_code": code}, n_results=n_results)
            pass_list = results.get("documents", [[]])[0]
            meta_list = results.get("metadatas", [[]])[0]
            for p, m in zip(pass_list, meta_list):
                unique_key = code + p
                if unique_key not in seen_passages:
                    combined_passages.append(p)
                    combined_metadata.append(m)
                    seen_passages.add(unique_key)
    else:
        # fallback broad search
        results = collection.query(query_texts=[query], n_results=n_results)
        combined_passages = results.get("documents", [[]])[0]
        combined_metadata = results.get("metadatas", [[]])[0]

    return combined_passages, combined_metadata

# ==========================
#  7) Generate Answer 
#     with the same CASE logic
# ==========================
def generate_answer_with_case_logic(user_id: str, query: str, passages, metadata):
    """
    We replicate your GPT approach:
      1) Provide the big BASE_DEFAULT_PROMPT as system content
      2) Provide the relevant RAG passages as 'context'
      3) Provide the user question
      4) Return GPT response
    """
    # Build messages array
    messages = []

    # System: the big multi-step prompt
    messages.append({"role": "system", "content": BASE_DEFAULT_PROMPT})

    # Next, we inject the relevant passages as if a "context" message
    # Minimal approach: We can combine them into a single user 'context block'.
    # (In your main app, you might do a more sophisticated approach.)
    if passages:
        docs_text = "\n\n".join([f"Doc snippet {i+1}:\n{p}" for i, p in enumerate(passages)])
    else:
        docs_text = "No relevant documents found."

    messages.append({"role": "system", "content": f"RAG DOCUMENTS:\n{docs_text}"})

    # Finally, the user query
    messages.append({"role": "user", "content": query})

    # Send to GPT
    client = OpenAI(HARD_CODED_API_KEY)
    try:
        completion = client.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            temperature=0.2
        )
        answer = completion["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        return f"Error generating final answer: {e}"

# ==========================
#  8) MAIN MINIMAL APP
# ==========================
def main():
    st.set_page_config(page_title="Corporate Minimal UI (With Full RAG)", layout="centered")

    # Logo at top-left
    st.image("https://placehold.co/150x60?text=Your+Logo", use_column_width=False)

    st.title("Welcome to Our Minimal RAG UI with Full Logic")

    # Basic login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = None

    if not st.session_state.logged_in:
        st.subheader("Login to proceed")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_login(username, password):
                st.session_state.logged_in = True
                st.session_state.user_id = username
                st.success(f"Successfully logged in as {username}")
            else:
                st.error("Invalid credentials")
        return
    else:
        st.markdown(f"**Hello, {st.session_state.user_id}!**")

    # Single question -> do the full RAG approach
    user_query = st.text_input("Ask a question (the system will attempt the same CASEA/CASEB logic):")

    if st.button("Answer"):
        if not user_query.strip():
            st.warning("Please enter a question first.")
        else:
            # 1) Retrieve relevant passages from the user's Chroma store
            passages, meta = query_chroma_for_countries(st.session_state.user_id, user_query, n_results=10)

            # 2) Generate final answer with the same prompt structure
            answer = generate_answer_with_case_logic(st.session_state.user_id, user_query, passages, meta)

            st.markdown("### Answer")
            st.write(answer)


if __name__ == "__main__":
    main()