import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import streamlit as st
import os

# Vector store directory
persist_dir = "vectorstore"

# Initialize OpenAI Embeddings using API key from Streamlit secrets
embedding = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

def load_pdfs(uploaded_files):
    """Load and convert uploaded PDFs into LangChain Document objects."""
    documents = []
    for file in uploaded_files:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        metadata = {"source": file.name}
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def embed_documents(documents):
    """Split documents and embed into Chroma vector DB."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(docs, embedding, persist_directory=persist_dir)
    db.persist()
    return db

def get_vectorstore():
    """Load the existing Chroma vector store."""
    return Chroma(persist_directory=persist_dir, embedding_function=embedding)

def ask_question(query):
    """Answer a user query using RetrievalQA with OpenAI and Chroma."""
    db = get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain(query)
    return result["result"]
