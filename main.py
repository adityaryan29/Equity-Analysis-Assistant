import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Load .env ---
load_dotenv()

# --- Config ---
st.set_page_config(page_title="StockIntel Assistant", layout="wide", page_icon="📈")

# --- Title ---
st.markdown("<h1 style='color:#00FFAA'>📈 StockIntel Assistant</h1>", unsafe_allow_html=True)
st.caption("Equity Analysis Assistant powered by real-time GenAI insights")

# --- Session State for History ---
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

file_path = "faiss_store.pkl"
history_file = "qa_history.pkl"

# --- Sidebar ---
st.sidebar.header("🔗 Input News URLs")
urls = [st.sidebar.text_input(f"🔹 URL {i + 1}") for i in range(3)]
if st.sidebar.button("🚀 Process URLs"):
    valid_urls = [u for u in urls if u.strip()]
    if not valid_urls:
        st.sidebar.warning("Enter at least one valid URL.")
    else:
        with st.spinner("🔄 Processing URLs..."):
            loader = UnstructuredURLLoader(urls=valid_urls)
            data = loader.load()
            splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=500)
            docs = splitter.split_documents(data)

            embeddings = HuggingFaceEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)

            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)

        st.success("✅ URLs processed and stored!")

# --- Load/Save/Clear History ---
with st.sidebar.expander("🧠 Manage Q&A History", expanded=True):
    if st.button("💾 Save History to File"):
        with open(history_file, "wb") as f:
            pickle.dump(st.session_state.qa_history, f)
        st.success("History saved.")

    if st.button("📂 Load History from File"):
        if os.path.exists(history_file):
            with open(history_file, "rb") as f:
                st.session_state.qa_history = pickle.load(f)
            st.success("History loaded.")
        else:
            st.warning("No saved file found.")

    if st.button("🗑️ Clear History"):
        st.session_state.qa_history = []
        st.success("History cleared.")

# --- Question Input ---
st.markdown("### 💬 Ask your equity-related question")
query = st.text_input("🗣️ Question:")

if query:
    if not os.path.exists(file_path):
        st.error("Please process URLs first.")
    else:
        with st.spinner("🤖 Thinking..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            llm = ChatGroq(model="gemma2-9b-it", temperature=0.5, max_tokens=1000)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            result = chain.invoke({"question": query})

        # Show Result
        st.markdown("### 🧾 Answer")
        st.success(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.markdown("### 🔗 Sources")
            for src in sources.strip().split("\n"):
                st.markdown(f"- [{src}]({src})")

        # Append to session history
        st.session_state.qa_history.append({
            "question": query,
            "answer": result["answer"],
            "sources": sources
        })

# --- Filter/Search ---
st.markdown("### 🔍 Search Q&A History")
search_term = st.text_input("Enter keyword to filter history:")

# --- Display History ---
if st.session_state.qa_history:
    st.markdown("## 🕘 Previous Q&A History")
    filtered = [
        q for q in reversed(st.session_state.qa_history)
        if search_term.lower() in q["question"].lower() or search_term.lower() in q["answer"].lower()
    ]
    if filtered:
        for i, entry in enumerate(filtered, 1):
            st.markdown(f"**Q{i}: {entry['question']}**")
            st.markdown(f"📄 {entry['answer']}")
            if entry["sources"]:
                st.markdown("🔗 **Sources:**")
                for src in entry["sources"].strip().split("\n"):
                    st.markdown(f"- [{src}]({src})")
            st.markdown("---")
    else:
        st.info("No matching history found.")
