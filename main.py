import os
import pickle
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Equity Analysis Assistant", layout="wide", page_icon="📈")
st.title("📈 StockIntel Assistant")
st.caption("_(Equity Analysis Assistant powered by real-time GenAI insights)_")

# Persistent session state
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "process_urls_clicked" not in st.session_state:
    st.session_state.process_urls_clicked = False
if "submit_question" not in st.session_state:
    st.session_state.submit_question = False

file_path = "faiss_store.pkl"
llm = ChatGroq(model="gemma2-9b-it", temperature=0.5, max_tokens=1000)

# Sidebar URL input
st.sidebar.header("🔗 Input News URLs")
urls = [st.sidebar.text_input(f"🔹 URL {i+1}", key=f"url_{i}") for i in range(3)]
if st.sidebar.button("🚀 Process URLs"):
    st.session_state.process_urls_clicked = True

# Main container
main_placeholder = st.container()

# Process URLs if triggered
if st.session_state.process_urls_clicked:
    st.session_state.process_urls_clicked = False
    valid_urls = [url for url in urls if url.strip()]
    if valid_urls:
        with st.status("🔄 Processing URLs..."):
            try:
                loader = UnstructuredURLLoader(urls=valid_urls)
                data = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=500)
                docs = text_splitter.split_documents(data)

                embeddings = HuggingFaceEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)

                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)
                st.success("✅ URLs processed and vector store saved!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please enter at least one valid URL.")

# Ask Question
query = st.text_input("💬 Ask your equity-related question:", key="question_input")
if st.button("📤 Ask Question"):
    st.session_state.submit_question = True

# Generate answer if question submitted
if st.session_state.submit_question and query.strip():
    st.session_state.submit_question = False
    if not os.path.exists(file_path):
        st.error("⚠️ Please process some URLs first.")
    else:
        with st.spinner("🤖 Thinking... generating answer..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

        st.markdown("## 🧾 Answer")
        st.success(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.markdown("### 🔗 Sources")
            for src in sources.strip().split("\n"):
                if src:
                    st.markdown(f"- [{src}]({src})")

        st.session_state.qa_history.append({
            "question": query,
            "answer": result["answer"],
            "sources": sources
        })

# Search filter
search_term = st.text_input("🔍 Filter Q&A History:")

# Buttons for save/load/clear
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💾 Save Q&A History"):
        with open("qa_history.pkl", "wb") as f:
            pickle.dump(st.session_state.qa_history, f)
        st.success("History saved.")
with col2:
    if st.button("📂 Load Q&A History"):
        try:
            with open("qa_history.pkl", "rb") as f:
                st.session_state.qa_history = pickle.load(f)
            st.success("History loaded.")
        except FileNotFoundError:
            st.error("No saved history found.")
with col3:
    if st.button("🗑️ Clear Q&A History"):
        st.session_state.qa_history.clear()
        st.success("History cleared.")

# Display filtered history
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("## 🕘 Previous Q&A History")
    for i, entry in enumerate(reversed(st.session_state.qa_history)):
        if search_term.lower() in entry['question'].lower() or search_term.lower() in entry['answer'].lower():
            st.markdown(f"**Q{i+1}: {entry['question']}**")
            st.markdown(f"📄 *{entry['answer']}*")
            if entry['sources']:
                st.markdown("🔗 **Sources:**")
                for src in entry['sources'].strip().split("\n"):
                    if src:
                        st.markdown(f"- [{src}]({src})")
            st.markdown("---")
