# Architecture and Concepts

## Retrieval-Augmented Generation (RAG)
RAG combines information retrieval with text generation. Instead of asking a language model to answer from its training data alone, the application finds relevant passages from your own documents and supplies those passages as context.

## Pipeline
1. **Load:** PyMuPDF extracts selectable text from each PDF page.
2. **Chunk:** Each page is split into approximately 900-character pieces with 150-character overlap. Overlap helps avoid cutting context at boundaries.
3. **Embed:** OpenAI's embedding model maps each chunk to a vector representing semantic meaning.
4. **Index:** FAISS stores normalized vectors. Inner-product search on normalized vectors is equivalent to cosine similarity.
5. **Retrieve:** The question is embedded and the top-K closest chunks are selected.
6. **Prompt:** Retrieved passages, with filename/page metadata, are passed to the chat model.
7. **Ground:** The system prompt tells the model to answer only from context and identify missing information.
8. **Cite:** The UI presents the passages used so readers can verify the answer.

## Components
- `app.py`: Streamlit user interface and session state.
- `rag.py`: extraction, chunking, embedding, vector search and LLM call.
- `faiss.IndexFlatIP`: exact nearest-neighbor search (simple and appropriate for a demo-sized corpus).

## Improvements for production
- Persist and version indexes; use a vector database for larger collections.
- Add OCR for scanned PDFs, robust token-aware chunking and document deduplication.
- Add authentication, rate limiting, usage budgets and tenant isolation.
- Evaluate retrieval and answer quality with a curated test set.
- Add structured logs, tracing, retries and observability.
- Consider prompt-injection defenses and treat all retrieved document content as untrusted data.
