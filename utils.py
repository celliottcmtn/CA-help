import os
import fitz  # PyMuPDF
import hashlib
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

persist_dir = "vectorstore"
embedding = OpenAIEmbeddings()

def load_pdfs(uploaded_files):
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    db = Chroma.from_documents(docs, embedding, persist_directory=persist_dir)
    db.persist()
    return db

def get_vectorstore():
    return Chroma(persist_directory=persist_dir, embedding_function=embedding)

def ask_question(query):
    db = get_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa_chain(query)
    return result["result"]
