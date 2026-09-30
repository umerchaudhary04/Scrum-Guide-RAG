import os
import pickle
import streamlit as st
import numpy as np
import faiss
from groq import Groq
from rank_bm25 import BM25Okapi
import torch

# --- CRITICAL: Prevent CPU Thread Starvation / Page Freezing ---
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Scrum Guide AI Assistant",
    page_icon="📋",
    layout="wide"
)

FAISS_PATH = "data/index.faiss"
METADATA_PATH = "data/metadata.pkl"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


# --- Resource Caching with Safe Loading ---
@st.cache_resource(show_spinner=False)
def load_rag_components():
    """Load FAISS index, metadata, BM25, and neural models into memory."""
    if not os.path.exists(FAISS_PATH) or not os.path.exists(METADATA_PATH):
        return None, None, None, None, None

    # 1. Load FAISS Index
    index = faiss.read_index(FAISS_PATH)

    # 2. Load Metadata and BM25
    with open(METADATA_PATH, "rb") as f:
        metadata_payload = pickle.load(f)
    
    corpus = metadata_payload["corpus"]
    bm25 = metadata_payload["bm25"]
    embedding_model_name = metadata_payload.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")

    # 3. Load Embedding Model & Cross-Encoder in evaluation mode
    embedder = SentenceTransformer(embedding_model_name, device="cpu")
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")

    return index, corpus, bm25, embedder, reranker


@st.cache_resource
def get_groq_client():
    """Initialize Groq API client safely from st.secrets."""
    if "GROQ_API_KEY" not in st.secrets:
        return None
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# --- Initialize with UI Status ---
with st.spinner("Initializing AI Models & Database (Takes ~10s on first load)..."):
    index, corpus, bm25, embedder, reranker = load_rag_components()

groq_client = get_groq_client()

# Check for missing files or keys
if index is None or corpus is None:
    st.error("⚠️ Files missing! Please make sure `data/index.faiss` and `data/metadata.pkl` exist in the repository.")
    st.stop()

if groq_client is None:
    st.error("⚠️ `GROQ_API_KEY` is missing in Streamlit Secrets (`.streamlit/secrets.toml`).")
    st.stop()


# --- Hybrid Retrieval & Reranking ---
def reciprocal_rank_fusion(dense_ranks, sparse_ranks, rrf_k=60):
    scores = {}
    for rank, idx in enumerate(dense_ranks):
        scores[idx] = scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))
    for rank, idx in enumerate(sparse_ranks):
        scores[idx] = scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def hybrid_search(query: str, top_candidates: int = 8, final_k: int = 4):
    with torch.inference_mode():
        # 1. Dense retrieval (FAISS)
        query_emb = embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        _, dense_indices = index.search(query_emb, top_candidates)
        dense_results = dense_indices[0].tolist()

        # 2. Sparse retrieval (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        sparse_results = np.argsort(bm25_scores)[::-1][:top_candidates].tolist()

        # 3. Fuse RRF
        fused = reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=60)
        candidate_ids = [idx for idx, _ in fused[:6]]
        candidate_chunks = [corpus[idx] for idx in candidate_ids]

        # 4. Cross-Encoder Rerank
        cross_pairs = [[query, item["content"]] for item in candidate_chunks]
        rerank_scores = reranker.predict(cross_pairs)

        ranked = sorted(
            zip(candidate_chunks, rerank_scores),
            key=lambda pair: pair[1],
            reverse=True
        )
        return ranked[:final_k]


# --- Safe Stream Generator ---
def stream_groq_response(prompt_text, system_instruction):
    """Safely yields text tokens to prevent Streamlit UI freezing."""
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=True
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n\n*Error communicating with Groq API: {str(e)}*"


# --- UI & Chat ---
st.title("📋 Scrum Guide Assistant")
st.caption("Hybrid Search RAG (FAISS + BM25 + Cross-Encoder) powered by Llama 3.3 on Groq.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me any question about the official 2020 Scrum Guide."}
    ]

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("E.g., What are the responsibilities of the Scrum Master?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Searching Scrum Guide..."):
        top_chunks = hybrid_search(prompt, top_candidates=8, final_k=3)

    # Build Context
    context_text = "\n\n---\n\n".join([
        f"[Page {c['page']}]:\n{c['content']}" for c, _ in top_chunks
    ])

    system_prompt = (
        "You are an expert Scrum Guide assistant. Answer accurately based ONLY on the context below.\n"
        "Always cite the exact Scrum Guide page number.\n\n"
        f"Context:\n{context_text}"
    )

    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_groq_response(prompt, system_prompt))
        
        with st.expander("🔍 View Retrieved Sources & Reranker Confidence"):
            for chunk, score in top_chunks:
                st.markdown(f"**Page {chunk['page']}** (Score: `{score:.3f}`)")
                st.write(chunk["content"])
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response_text})
