import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

groq_api_key = os.environ["GROQ_API_KEY"]

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

loader = PyPDFLoader("d6f936dd421fc6592aa2eca3f9ddf3c34435.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
documents = text_splitter.split_documents(docs)

db = FAISS.from_documents(documents, OllamaEmbeddings())

prompt = ChatPromptTemplate.from_template(
    """
    You are a very helpful assistent. Please provide a response in detail.
    Think in detail and step by step before providing an answer.
    <context>
    {context}
    </context>
    Question : {input}
    """
)

st.title("PDF-Q&A")
input_text = st.text_input("Enter your query here :")

llm = ChatGroq(groq_api_key=groq_api_key, model_name = "gemma2-9b-it")

retriever = db.as_retriever()

documents_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, documents_chain)

if input_text:
    response = retrieval_chain.invoke({'input' : input_text})
    st.write(response["answer"])




