import streamlit as st
import os
import faiss
import numpy as np
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from openai import OpenAI
from docx import Document
from helper import *

# ------------------------------
# Load API keys
# ------------------------------
LLAMAPARSE_API_KEY = st.secrets["LLAMAPARSE_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ------------------------------
# Setup LlamaParse for PDFs/TXT
# ------------------------------
parser = LlamaParse(result_type="markdown", api_key=LLAMAPARSE_API_KEY)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Healthcare Doc Chat", layout="wide")
st.title("📑 Chat with Multiple Healthcare Documents (PDF, TXT, DOCX)")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks_meta" not in st.session_state:
    st.session_state.chunks_meta = []
if "embeddings_cache" not in st.session_state:
    st.session_state.embeddings_cache = {}  # Cache embeddings per document
if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}  # Cache answers per query

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload one or more healthcare documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    all_chunks = []
    st.session_state.chunks_meta = []

    with st.spinner("Extracting and embedding text..."):
        for uploaded_file in uploaded_files:
            temp_file_path = os.path.join("./", uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                extracted_text = ""

                if ext == ".docx":
                    extracted_text = extract_text_from_docx(temp_file_path)
                elif ext in [".pdf", ".txt"]:
                    documents = SimpleDirectoryReader(
                        input_files=[temp_file_path],
                        file_extractor={".pdf": parser, ".txt": parser}
                    ).load_data()
                    for doc in documents:
                        if hasattr(doc, "text"):
                            extracted_text += doc.text + "\n"

                if not extracted_text.strip():
                    st.warning(f"No text extracted from {uploaded_file.name}")
                    continue

                # Save to .txt file (optional)
                os.makedirs("./extracted", exist_ok=True)
                txt_file_path = os.path.join("./extracted", f"{uploaded_file.name}.txt")
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)

                # Chunk + embed (cache per document)
                if uploaded_file.name in st.session_state.embeddings_cache:
                    chunks, embeddings = st.session_state.embeddings_cache[uploaded_file.name]
                else:
                    chunks = chunk_text(extracted_text)
                    embeddings = embed_text(chunks)
                    st.session_state.embeddings_cache[uploaded_file.name] = (chunks, embeddings)

                # Store metadata
                for chunk, emb in zip(chunks, embeddings):
                    all_chunks.append(emb)
                    st.session_state.chunks_meta.append({
                        "doc_name": uploaded_file.name,
                        "text": chunk
                    })

            except Exception as e:
                st.error(f"Error parsing {uploaded_file.name}: {e}")

        if all_chunks:
            embeddings_matrix = np.vstack(all_chunks)
            dim = embeddings_matrix.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings_matrix)
            st.session_state.faiss_index = index
            st.success("✅ All documents parsed, embedded, and stored in FAISS!")

# --- Chat Interface ---
st.subheader("💬 Chat with the Documents")

query = st.chat_input("Ask something about the documents...")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})

    if query in st.session_state.qa_cache:
        answer, sources_text = st.session_state.qa_cache[query]
    elif st.session_state.faiss_index is not None:
        with st.spinner("Generating answer..."):
            # Embed query
            query_vec = embed_text([query])[0].reshape(1, -1)

            # Search FAISS
            D, I = st.session_state.faiss_index.search(query_vec, k=5)
            retrieved_chunks = [st.session_state.chunks_meta[i] for i in I[0]]

            # Generate answer using helper
            answer = generate_answer(query, retrieved_chunks, st.session_state.chat_history)

            # Generate relevant excerpts (unique)
            sources_text = ""
            seen_docs = set()
            for c in retrieved_chunks:
                if c['doc_name'] in seen_docs:
                    continue
                relevant_part = extract_relevant_text(c['text'], query)
                if relevant_part.strip():
                    sources_text += f"📖 {c['doc_name']} — {relevant_part[:500]}...\n\n"
                    seen_docs.add(c['doc_name'])

            # Cache the answer
            st.session_state.qa_cache[query] = (answer, sources_text)
    else:
        answer = "Please upload and parse at least one document first."
        sources_text = ""

        st.session_state.qa_cache[query] = (answer, sources_text)

    # Append to chat
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    if sources_text:
        st.session_state.chat_history.append({"role": "sources", "content": sources_text})

# --- Display Chat ---
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.chat_message("user").write(chat["content"])
    elif chat["role"] == "assistant":
        st.chat_message("assistant").write(chat["content"])
    elif chat["role"] == "sources":
        with st.expander("📚 Relevant Excerpts"):
            st.write(chat["content"])
