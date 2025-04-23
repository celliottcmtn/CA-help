import os
from pathlib import Path
import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Create embedding function
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# In-memory FAISS DB (you could persist if needed later)
VECTORSTORE_PATH = "vectorstore/faiss_index"

# Store current vector DB in session
db = None

def load_pdfs_from_folders(union_folders):
    """
    Loads all .pdf files from specified subfolders within the 'agreements' directory.
    Returns a list of LangChain Document objects.
    """
    base_path = Path("agreements")
    documents = []

    for union in union_folders:
        folder_path = base_path / union
        if not folder_path.exists():
            st.warning(f"Folder not found: {folder_path}")
            continue

        for file in folder_path.glob("*.pdf"):
            try:
                pdf = fitz.open(file)
                text = ""
                for page in pdf:
                    text += page.get_text()
                if text.strip():
                    metadata = {"source": file.name, "union": union}
                    documents.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                st.warning(f"Error reading {file.name}: {e}")

    return documents

def embed_documents(documents):
    """
    Splits and embeds documents into an in-memory FAISS vectorstore.
    """
    global db
    if not documents:
        st.warning("No documents to embed.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, embedding)
    return db

def ask_question(query):
    """
    Answers a user query using the FAISS vector store and OpenAI.
    """
    global db
    if db is None:
        return "No documents available. Please select a union to load agreements first."

    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    try:
        result = qa_chain(query)
        return result["result"]
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"
