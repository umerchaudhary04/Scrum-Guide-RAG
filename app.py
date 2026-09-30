import os
import pickle
import streamlit as st
import numpy as np
import faiss
from groq import Groq
from rank_bm25 import BM25Okapi
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


# --- Resource Caching ---
@st.cache_resource(show_spinner="Loading indices and retrieval models...")
def load_rag_components():
    """Load FAISS index, metadata, BM25, and neural models into memory."""
    if not os.path.exists(FAISS_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "Index files missing! Ensure `data/index.faiss` and `data/metadata.pkl` exist."
        )

    # 1. Load FAISS
    index = faiss.read_index(FAISS_PATH)

    # 2. Load Metadata and BM25
    with open(METADATA_PATH, "rb") as f:
        metadata_payload = pickle.load(f)
    
    corpus = metadata_payload["corpus"]
    bm25: BM25Okapi = metadata_payload["bm25"]
    embedding_model_name = metadata_payload["embedding_model_name"]

    # 3. Load Embedding Model & Cross-Encoder (CPU-optimized)
    embedder = SentenceTransformer(embedding_model_name, device="cpu")
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")

    return index, corpus, bm25, embedder, reranker


@st.cache_resource
def get_groq_client():
    """Initialize Groq API client from st.secrets."""
    if "GROQ_API_KEY" not in st.secrets:
        st.error("Missing GROQ_API_KEY in `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
        st.stop()
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


# Initialize resources
index, corpus, bm25, embedder, reranker = load_rag_components()
groq_client = get_groq_client()


# --- Hybrid Retrieval & Reranking Functions ---
def reciprocal_rank_fusion(dense_ranks: list, sparse_ranks: list, rrf_k: int = 60):
    """
    Combines dense and sparse search rankings using Reciprocal Rank Fusion (RRF).
    Score = sum(1 / (k + rank))
    """
    scores = {}
    for rank, idx in enumerate(dense_ranks):
        scores[idx] = scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))
        
    for rank, idx in enumerate(sparse_ranks):
        scores[idx] = scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

    # Sort candidates by combined RRF score descending
    sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_candidates


def hybrid_search(query: str, top_dense_sparse: int = 15, final_k: int = 4):
    """
    Executes dense FAISS + sparse BM25 retrieval, merges via RRF, 
    and refines the top candidates using a Cross-Encoder.
    """
    # 1. Dense retrieval (FAISS)
    query_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    _, dense_indices = index.search(query_emb, top_dense_sparse)
    dense_results = dense_indices[0].tolist()

    # 2. Sparse retrieval (BM25)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    sparse_results = np.argsort(bm25_scores)[::-1][:top_dense_sparse].tolist()

    # 3. Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=60)
    candidate_ids = [idx for idx, _ in fused_results[:10]]
    candidate_chunks = [corpus[idx] for idx in candidate_ids]

    # 4. Cross-Encoder Reranking
    cross_pairs = [[query, item["content"]] for item in candidate_chunks]
    rerank_scores = reranker.predict(cross_pairs)

    ranked_candidates = sorted(
        zip(candidate_chunks, rerank_scores),
        key=lambda pair: pair[1],
        reverse=True
    )

    return ranked_candidates[:final_k]


# --- UI and Chat Execution ---
st.title("Scrum Guide Assistant")
st.caption("Hybrid RAG (FAISS + BM25 + Cross-Encoder) powered by Llama 3 on Groq.")

# Initialize session state for conversation
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Ask me anything about Scrum roles, artifacts, ceremonies, or rules according to the official Scrum Guide."}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User query entry point
if prompt := st.chat_input("E.g., What are the commitments for each Scrum artifact?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval and Reranking
    with st.spinner("Searching and reranking Scrum Guide sections..."):
        top_chunks_with_scores = hybrid_search(prompt, top_dense_sparse=15, final_k=4)

    # Build prompt context
    context_blocks = []
    for item, score in top_chunks_with_scores:
        context_blocks.append(
            f"[Scrum Guide Page {item['page']}] (Relevance Score: {score:.2f}):\n{item['content']}"
        )
    formatted_context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert Scrum Guide assistant. Answer the user's question accurately "
        "and concisely using ONLY the provided Scrum Guide context below. If the answer cannot "
        "be determined from the context, state that it is not covered in the Scrum Guide.\n"
        "Always cite the relevant page numbers provided in the context.\n\n"
        f"Context:\n{formatted_context}"
    )

    # Generate streaming response
    with st.chat_message("assistant"):
        stream = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=True
        )
        
        response_text = st.write_stream(stream)
        
        # Display retrieved chunks in an expander for full auditability
        with st.expander("🔍 View Retrieved Context & Reranker Scores"):
            for item, score in top_chunks_with_scores:
                st.markdown(f"**Page {item['page']}** | *Reranker Score: {score:.4f}*")
                st.text(item["content"])
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response_text})
