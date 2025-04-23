import fitz  # PyMuPDF
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Path to store vector database
PERSIST_DIR = "vectorstore"

# Load API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Create embeddings using OpenAI
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def load_pdfs(uploaded_files):
    """Read uploaded PDF files and return a list of LangChain Document objects."""
    documents = []
    for file in uploaded_files:
        try:
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in pdf:
                text += page.get_text()
            if text.strip():  # Only include if content exists
                metadata = {"source": file.name}
                documents.append(Document(page_content=text, metadata=metadata))
        except Exception as e:
            st.warning(f"Failed to process {file.name}: {e}")
    return documents

def embed_documents(documents):
    """Split and embed documents into a persistent ChromaDB vector store."""
    if not documents:
        st.warning("No documents to embed.")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    db = Chroma.from_documents(chunks, embedding=embedding, persist_directory=PERSIST_DIR)
    db.persist()
    return db

def get_vectorstore():
    """Load an existing Chroma vectorstore (or raise error if not initialized)."""
    try:
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    except Exception as e:
        st.error("Could not load the vectorstore. Did you upload and embed PDFs?")
        return None

def ask_question(query):
    """Run a Retrieval QA pipeline using the embedded documents and OpenAI."""
    db = get_vectorstore()
    if db is None:
        return "No documents available. Please upload and embed PDFs first."
    
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
