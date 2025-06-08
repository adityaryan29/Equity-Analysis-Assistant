import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# ------------------------- UI SETUP -------------------------
st.set_page_config(page_title="Equity Analysis Assistant", layout="wide", page_icon="📈")

# Logo
# st.image("logo.png", width=100)
st.title("📈 StockIntel Assistant")
st.caption("_(Equity Analysis Assistant powered by real-time GenAI insights)_")

# ------------------------- SIDEBAR INPUT -------------------------
st.sidebar.header("🔗 Add Your Data Sources")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"🔹 URL {i + 1}")
    urls.append(url.strip())

uploaded_pdf = st.sidebar.file_uploader("📄 Upload a financial report (PDF)", type=["pdf"])
process_data = st.sidebar.button("🚀 Process URLs / PDF")

file_path = "faiss_store.pkl"

# ------------------------- LLM -------------------------
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.5,
    max_tokens=1000
)

# ------------------------- PROCESS URL OR PDF -------------------------
if process_data:
    try:
        docs = []

        with st.status("🔄 Processing data...", expanded=True):
            if any(urls):
                st.write("🌐 Fetching from URLs...")
                loader = UnstructuredURLLoader(urls=[url for url in urls if url])
                docs.extend(loader.load())

            if uploaded_pdf:
                st.write("📄 Reading PDF...")
                with open("temp_uploaded.pdf", "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                pdf_loader = PyPDFLoader("temp_uploaded.pdf")
                docs.extend(pdf_loader.load())

            if not docs:
                st.warning("⚠️ No valid input found. Please enter URLs or upload a PDF.")
            else:
                st.write("🧩 Splitting text into chunks...")
                splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=500
                )
                chunks = splitter.split_documents(docs)

                st.write("🧠 Creating vector embeddings...")
                embeddings = HuggingFaceEmbeddings()
                vectorstore = FAISS.from_documents(chunks, embeddings)

                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)

                st.success("✅ Data processed and saved!")
    except Exception as e:
        st.error(f"❌ Failed to process: {e}")

# ------------------------- Q&A Section -------------------------
st.markdown("### 💬 Ask your stock/equity question below")
query = st.text_input("🗣️ Your Question:")

# Initialize session state for Q&A history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if query:
    if not os.path.exists(file_path):
        st.error("⚠️ No data available. Please process some URLs or upload a PDF.")
    else:
        with st.spinner("🤖 Generating response..."):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )

            result = chain({"question": query}, return_only_outputs=True)
            st.session_state.qa_history.append(
                {"question": query, "answer": result["answer"], "sources": result.get("sources", "")}
            )

# ------------------------- Display Results -------------------------
if st.session_state.qa_history:
    st.markdown("## 🧾 Q&A History")
    for idx, entry in enumerate(reversed(st.session_state.qa_history)):
        st.markdown(f"**Q{len(st.session_state.qa_history)-idx}:** {entry['question']}")
        st.success(entry['answer'])

        sources = entry['sources']
        if sources:
            st.markdown("**🔗 Sources:**")
            for src in sources.strip().split("\n"):
                if src:
                    st.markdown(f"- [{src}]({src})")
        st.markdown("---")
