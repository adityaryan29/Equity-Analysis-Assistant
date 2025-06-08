import os
import streamlit as st
import pickle
import json
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Load environment variables ---
load_dotenv()

# --- Streamlit Page Config ---
st.set_page_config(page_title="Equity Analysis Assistant", layout="wide", initial_sidebar_state="expanded")
st.title("📈 StockIntel Assistant")
st.caption("_(Equity Analysis Assistant powered by real-time GenAI insights)_")

# --- Initialize Q&A history ---
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

file_path = "faiss_store.pkl"
history_file = "qa_history.json"

# --- Sidebar: URLs & History Options ---
st.sidebar.header("🔗 Input News URLs")
urls = [st.sidebar.text_input(f"🔹 URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("🚀 Process URLs")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Q&A History"):
    st.session_state.qa_history = []

if st.sidebar.button("💾 Save History to File"):
    with open(history_file, "w") as f:
        json.dump(st.session_state.qa_history, f)
    st.sidebar.success("Saved to qa_history.json")

if st.sidebar.button("📂 Load History from File"):
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            st.session_state.qa_history = json.load(f)
        st.sidebar.success("Loaded from qa_history.json")
    else:
        st.sidebar.warning("qa_history.json not found")

# --- Process URLs ---
if process_url_clicked:
    if any(url.strip() for url in urls):
        with st.status("🔄 Processing URLs...", expanded=True):
            try:
                loader = UnstructuredURLLoader(urls=[u for u in urls if u.strip()])
                st.write("📥 Fetching articles...")
                data = loader.load()

                st.write("🧩 Splitting articles...")
                splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=500)
                docs = splitter.split_documents(data)

                st.write("🧠 Embedding & saving to FAISS...")
                embeddings = HuggingFaceEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)

                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)

                st.success("✅ Vector store saved!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please enter at least one valid URL.")

# --- Load LLM ---
llm = ChatGroq(model="gemma2-9b-it", temperature=0.5, max_tokens=1000)

# --- User Query ---
st.markdown("### 💬 Ask your equity-related question")
query = st.text_input("🗣️ Question:")
if query:
    if not os.path.exists(file_path):
        st.error("⚠️ Please process URLs first.")
    else:
        with st.spinner("🤖 Thinking..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

        # Show result
        st.markdown("### 🧾 Answer")
        st.success(result["answer"])

        st.markdown("### 🔗 Sources")
        for src in result.get("sources", "").strip().split("\n"):
            if src:
                st.markdown(f"- [{src}]({src})")

        # Save to Q&A history
        st.session_state.qa_history.append({
            "question": query,
            "answer": result["answer"],
            "sources": result.get("sources", "")
        })

# --- Filter/Search Q&A History ---
st.markdown("### 📜 Q&A History")
search = st.text_input("🔍 Filter history by keyword")
filtered_history = [qa for qa in st.session_state.qa_history if search.lower() in qa["question"].lower() or search.lower() in qa["answer"].lower()]

for i, qa in enumerate(filtered_history[::-1], 1):
    st.markdown(f"**{i}. Q:** {qa['question']}")
    st.markdown(f"**A:** {qa['answer']}")
    if qa["sources"]:
        st.markdown("_Sources:_")
        for src in qa["sources"].strip().split("\n"):
            if src:
                st.markdown(f"- [{src}]({src})")
    st.markdown("---")
