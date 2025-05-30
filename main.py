import os
import streamlit as st
import pickle
import time
import langchain
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv



load_dotenv() #takes environment variable from .env.

st.title("News Research Tool📈")

st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
   url= st.sidebar.text_input(f"URL {i+1}")
   urls.append(url)
process_url_clicked=st.sidebar.button("Process URLs")
file_path="faiss_store.pkl"

main_placefolder=st.empty()
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.5,max_tokens=1000
)
if process_url_clicked:
    #load data
    loader=UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading...Started...✅✅✅")
    data=loader.load()
    #split data
    text_splitter=RecursiveCharacterTextSplitter(separators=['\n\n','\n','.',','],
                                                 chunk_size=500)
    docs=text_splitter.split_documents(data)
    #create embeddings and saving it to FAISS index
    embeddings=HuggingFaceEmbeddings()
    vectorstore=FAISS.from_documents(docs,embeddings)
    #saving the FAISS index to pkl file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)


query=main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore=pickle.load(f)
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
            result=chain({"question":query},return_only_outputs=True)
            #{"answer":"","sources":[]}
            st.header("Result")
            st.write(result["answer"])

        #Display sources if available
        sources=result.get("sources","")
        if sources:
            st.subheader("Sources:")
            sources_list=sources.split("\n") #Split the sources by newline
            for source in sources_list:
                st.write(source)







