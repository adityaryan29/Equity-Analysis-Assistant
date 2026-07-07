import os
import streamlit as st
import time
import langchain
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title("StockIntel Assistant📈")
st.sidebar.title("News Article URLs")
urls=[]
for i in range(3):
    url= st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_clicked=st.sidebar.button("Process URLs")

file_path="faiss_store" 
main_placefolder=st.empty()

llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.5,max_tokens=1000
)

if process_url_clicked:
    urls = [u for u in urls if u.strip()] 
    
    if urls: 
        loader=UnstructuredURLLoader(urls=urls)
        main_placefolder.text("Command Center: Data Inbound...✅✅✅")
        data=loader.load()
        
        text_splitter=RecursiveCharacterTextSplitter(separators=['\n\n','\n','.',','], chunk_size=500)
        docs=text_splitter.split_documents(data)
        
        embeddings=HuggingFaceEmbeddings()
        vectorstore=FAISS.from_documents(docs,embeddings)
        
        vectorstore.save_local(file_path)

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True) 
        
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
