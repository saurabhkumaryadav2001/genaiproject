import os
import streamlit as st
from dotenv import load_dotenv
from rag import build_index, answer_question

load_dotenv()
st.set_page_config(page_title="GenAI Knowledge Assistant", page_icon="📚", layout="wide")
st.title("📚 GenAI Knowledge Assistant")
st.caption("Upload PDFs, then ask questions grounded in their contents.")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Set OPENAI_API_KEY in your .env file before using the assistant.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    st.header("1 · Add knowledge")
    uploads = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs", type="primary", disabled=not uploads):
        with st.spinner("Reading PDFs and creating embeddings…"):
            try:
                store, chunks = build_index(uploads)
                st.session_state.vector_store, st.session_state.chunks = store, chunks
                st.session_state.chat = []
                st.success(f"Indexed {len(chunks)} text chunks.")
            except Exception as exc:
                st.error(f"Could not process PDFs: {exc}")
    if st.session_state.vector_store:
        st.success(f"Knowledge base ready · {len(st.session_state.chunks)} chunks")
    st.divider()
    st.caption("Your files are processed in this app session. API calls send extracted text to OpenAI.")

st.header("2 · Ask a question")
for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander("Sources"):
                for source in item["sources"]:
                    st.markdown(f"- **{source['file']} · page {source['page']}**")
                    st.caption(source["text"])

question = st.chat_input("Ask something about your uploaded documents…", disabled=st.session_state.vector_store is None)
if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                result = answer_question(question, st.session_state.vector_store, st.session_state.chunks)
                st.markdown(result["answer"])
                with st.expander("Sources used"):
                    for source in result["sources"]:
                        st.markdown(f"- **{source['file']} · page {source['page']}**")
                        st.caption(source["text"])
                st.session_state.chat.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
