# Scrum Guide AI Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built to answer questions accurately against the official **2020 Scrum Guide**.

The system features a **decoupled architecture**: heavy ingestion, embedding generation, and indexing are performed offline (via Google Colab), while real-time hybrid search, cross-encoder reranking, and low-latency answer generation run on Streamlit Cloud powered by Groq.

---

## Architecture Overview
```text
[ Offline Ingestion - Google Colab ]
Scrum Guide (PDF) ──> Chunking ──> all-MiniLM-L6-v2 ──> FAISS Index (Dense)
└──> BM25 Tokenizer  ──> metadata.pkl (Sparse)
│
(Committed to repo /data)
│
[ Inference Pipeline - Streamlit Cloud ]                      ▼
User Query ──► Dense Search (FAISS)  ──┐
──► Sparse Search (BM25)  ──┴──► Reciprocal Rank Fusion (RRF)
│
▼
Cross-Encoder Reranker
(ms-marco-MiniLM-L-6-v2)
│ Top 4 Chunks
▼
Groq API (Llama 3.3)
│
▼
Streamed Response + Citations
```

### Key Technical Decisions:
1. **Decoupled Pipeline:** Streamlit Cloud containers do not re-parse PDFs or re-calculate embeddings on startup. Pre-computed indices load into memory within 1–2 seconds.
2. **Hybrid Retrieval (Dense + Sparse):**
   - **Dense (FAISS IndexFlatIP):** Captures semantic intent and paraphrased concepts.
   - **Sparse (BM25Okapi):** Accurately retrieves exact Scrum terminology (e.g., *Product Backlog refinement*, *Increment*, *Definition of Done*).
   - **Reciprocal Rank Fusion (RRF):** Combines dense and sparse candidate lists without score scale mismatch.
3. **Cross-Encoder Reranking:** Top fused candidates are evaluated jointly with the query using `cross-encoder/ms-marco-MiniLM-L-6-v2` to prioritize high-relevance chunks.
4. **Low-Latency Generation:** Queries `llama-3.3-70b-versatile` via the Groq LPU engine for fast token streaming.

---

## Repository Structure
```
scrum-guide-rag/

├── data/
│   ├── index.faiss               # Serialized FAISS vector index (~100 KB)
│   └── metadata.pkl              # Chunks corpus, page numbers & BM25 object (~250 KB)
├── colab_code.py             # Download the scrum guide pdf and generate FAISS & Metadata (runs in Colab)
├── app.py                        # Streamlit application UI and inference flow
├── requirements.txt              # Pinned Python dependencies
└── README.md
```
### Setup & Local Execution
**Prerequisites:** Python 3.10+, A free Groq API Key

1. **Clone Repository & Setup Virtual Environment**
```Bash
git clone [https://github.com/](https://github.com/)<your-username>/scrum-guide-rag.git
cd scrum-guide-rag

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Offline Index Generation (Colab)
If you modify chunking parameters or use an updated document:

1. Open a new notebook on Google Colab.
2. Copy and run the code from ingestion/ingest.py.
3. The script will automatically:
4. Download the official Scrum Guide 2020 PDF.
5. Generate embeddings and build FAISS + BM25 indices.
6. Trigger the download of index.faiss and metadata.pkl.
7. **Place the generated files in the data/ directory and commit them to Git**

## License & Attribution
1. The Scrum Guide is offered for license under the Attribution Share-Alike license of Creative Commons (CC BY-SA 4.0) by Ken Schwaber and Jeff Sutherland.
2. Application codebase is released under the MIT License.
