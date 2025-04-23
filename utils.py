import os
from pathlib import Path
import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Load API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Set up embeddings
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# In-memory FAISS store
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
    Splits and embeds documents into a FAISS vectorstore (in-memory).
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
    Uses FAISS + OpenAI + custom prompt to answer a question using the agreements.
    """
    global db
    if db is None:
        return "No documents available. Please select and embed agreements first."

    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    custom_prompt = PromptTemplate.from_template("""
You are a knowledgeable assistant that helps management understand the application of collective agreements. 
Use the following excerpts of agreement text to answer the user’s question.

- Only use the provided context.
- If the answer is not clearly supported in the agreements, say: "The agreement does not provide a clear answer to this."
- In your response, specify which union agreement(s) and article number(s) are relevant.
- If multiple sections apply, cite them clearly and explain how they relate.
- Be clear, concise, and avoid legal jargon when possible.

Context:
{context}

Question: {question}

Helpful Answer:
""")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    try:
        result = qa_chain(query)
        return result["result"]
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"
