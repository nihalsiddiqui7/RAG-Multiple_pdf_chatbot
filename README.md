# RAG Multiple PDF Chatbot

Chat with the contents of one or more PDFs. The app ingests PDFs, chunks the text, builds a FAISS vector index with Hugging Face embeddings, and answers questions via a Hugging Face Inference model — all inside a Streamlit UI.

## Features
- Multi-PDF ingestion and text extraction (PyPDF2)
- Overlapping chunking for better context retention (LangChain splitters)
- Fast similarity search over embeddings (FAISS + Hugging Face MiniLM)
- Retrieval-augmented QA with a configurable HF Inference model (default: `mistralai/Mistral-7B-Instruct-v0.2`)
- Streamlit UI with upload + chat flow
- Environment-based secrets (`.env` is gitignored)

## Tech Stack
- Python, Streamlit
- LangChain (splitters, chains), FAISS vector store
- Hugging Face: `all-MiniLM-L6-v2` embeddings + Inference text-generation model
- PyPDF2 for PDF text extraction
- dotenv for local config

## How It Works
1) Upload PDFs in the sidebar.
2) Extract text from each page (skipping empty pages).
3) Split text into overlapping chunks (1k chars, 200 overlap).
4) Create embeddings with `all-MiniLM-L6-v2` and store in FAISS.
5) On a question, retrieve the top chunks and send them with the question to the HF model.
6) Return a grounded answer.

## Setup
### 1) Clone and enter the project
```bash
git clone <https://github.com/nihalsiddiqui7/RAG-Multiple_pdf_chatbot.git>
cd RAG-Multiple_pdf_chatbot
```

### 2) Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate # macOS/Linux
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install streamlit python-dotenv PyPDF2 langchain langchain-community langchain-text-splitters langchain-huggingface faiss-cpu
```
If `faiss-cpu` fails on Windows, try:
```bash
pip install faiss-cpu -f https://github.com/kyamagu/faiss-wheels/releases
```

### 4) Environment variables
Create a `.env` file (already gitignored):
```
HUGGINGFACEHUB_API_TOKEN=your_hf_token
# Optional: choose a different model your token can access
HF_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
```

## Run locally
```bash
streamlit run app.py
```
Then open the URL Streamlit prints (usually http://localhost:8501).

## Usage
- Sidebar → upload one or more PDFs → click Process.
- Ask questions in the main input; answers are grounded in the retrieved PDF chunks.
- If you switch PDFs, click Process again to rebuild the index.

## Deployment
- Works well on Streamlit Community Cloud or any host where `streamlit run app.py` can run.
- Add `HUGGINGFACEHUB_API_TOKEN` (and optional `HF_MODEL_ID`) as secrets/env vars on the platform.
- The FAISS index is in-memory per session; expect it to rebuild after restarts (fine for demos).

## Notes & Limits
- Very large PDFs can be slow or exceed memory; keep uploads reasonable.
- Pages with no extractable text are skipped; scanned PDFs may need OCR first.
- Model access depends on your HF token/plan; choose a model you’re allowed to use.

## Roadmap Ideas
- Persist vector stores (e.g., disk-backed FAISS) to avoid rebuilds.
- Add chat history memory and streaming responses.
- Show source snippets with citations.
- Add OCR fallback for scanned PDFs.
- Containerize with Docker for reproducible deploys.