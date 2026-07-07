import os
import streamlit as st
import pickle
import time
import langchain
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv



load_dotenv() #takes environment variable from .env.

st.title("StockIntel Assistant📈")

st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
   url= st.sidebar.text_input(f"URL {i+1}")
   if url.strip():  # ✅ ONLY ADD NON-EMPTY URLS
       urls.append(url.strip())

process_url_clicked=st.sidebar.button("Process URLs")
file_path="faiss_store.pkl"

main_placefolder=st.empty()
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.5,max_tokens=1000
)
if process_url_clicked:
    # ✅ CHECK IF ANY URLS WERE PROVIDED
    if not urls:
        st.error("❌ Please enter at least one valid URL.")
        st.stop()
    
    main_placefolder.text("Command Center: Data Inbound...✅✅✅")
    
    # ✅ USE WebBaseLoader INSTEAD (more reliable)
    try:
        loader = WebBaseLoader(urls)
        data = loader.load()
    except Exception as e:
        st.error(f"❌ Failed to load URLs: {str(e)}")
        st.write("**Tip:** Make sure URLs are publicly accessible and return valid content.")
        st.stop()
    
    # ✅ CHECK IF DATA WAS LOADED
    if not data:
        st.error("❌ No content found. Check if URLs are valid and accessible.")
        st.stop()
    
    #split data
    text_splitter=RecursiveCharacterTextSplitter(separators=['\n\n','\n','.',','],
                                                 chunk_size=500)
    docs=text_splitter.split_documents(data)
    
    # ✅ CHECK IF DOCUMENTS WERE CREATED
    if not docs:
        st.error("❌ No documents generated after splitting. URLs may not contain valid content.")
        st.stop()
    
    #create embeddings and saving it to FAISS index
    embeddings=HuggingFaceEmbeddings()
    vectorstore=FAISS.from_documents(docs,embeddings)
    #saving the FAISS index to pkl file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    
    st.success("✅ URLs processed successfully!")


query = main_placefolder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        result = qa({"query": query})

        st.header("Result")
        st.write(result["result"])

        st.subheader("Sources:")
        for doc in result["source_documents"]:
            st.write(doc.metadata.get("source", "Unknown"))




