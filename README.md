# GenAI Knowledge Assistant 📚

Ask questions about your PDFs using Retrieval-Augmented Generation (RAG). The assistant retrieves relevant document passages and uses an LLM to answer with source/page references instead of relying only on model memory.

## Features
- Upload one or more PDF documents
- Extract text page-by-page and split it into overlapping chunks
- Create semantic embeddings and store them in a local FAISS index
- Ask natural-language questions grounded in your documents
- Display retrieved source snippets and page numbers
- Streamlit UI; simple local setup

## Architecture
```
PDF upload → PyMuPDF extraction → chunking → OpenAI embeddings → FAISS
Question → question embedding → top-k retrieval → prompt with context → LLM answer + sources
```

## Requirements
- Python 3.10+
- OpenAI API key with API access

## Run locally
```bash
git clone https://github.com/saurabhkumaryadav2001/genaiproject.git
cd genaiproject
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env  # Windows; on macOS/Linux use: cp .env.example .env
# Add OPENAI_API_KEY in .env
streamlit run app.py
```

## How it works (RAG)
1. **Ingestion:** Extract each PDF page and split its text into chunks (with overlap).
2. **Embedding:** Convert chunks to vectors using an embedding model.
3. **Index:** Save vectors and metadata in FAISS for fast similarity search.
4. **Retrieval:** Embed the user's question and retrieve the most relevant chunks.
5. **Generation:** Send only those chunks as context to the chat model. The prompt instructs it to say when the answer is not present.
6. **Sources:** Show the document name and page number for retrieved chunks.

## Project structure
```
genaiproject/
├── app.py              # Streamlit UI
├── rag.py              # PDF ingestion, retrieval and generation
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Configuration
Set `OPENAI_API_KEY` in `.env`. Optional:
- `CHAT_MODEL` (default: `gpt-4o-mini`)
- `EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `TOP_K` (default: `4`)

## Security
Never commit your real API key. The included `.gitignore` excludes `.env`. Uploaded PDFs and the FAISS index remain in memory for this app session; nothing is uploaded to a third-party storage service by this project. PDF text is sent to OpenAI for embeddings and answer generation, so upload only documents you are permitted to process.

## Limitations
- Scanned/image-only PDFs need OCR (not included).
- Answers depend on the quality of extracted PDF text and retrieval.
- This is a learning/demo app; add authentication, quotas and persistent vector storage before production use.

## License
MIT
