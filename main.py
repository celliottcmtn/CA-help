import streamlit as st
from utils import load_pdfs, embed_documents, ask_question
import os

st.set_page_config(page_title="Collective Agreement Chatbot", layout="centered")
st.title("🤖 Collective Agreement Chatbot")

# Upload PDFs
uploaded_files = st.file_uploader("Upload Collective Agreement PDFs", type="pdf", accept_multiple_files=True)

# Embed PDFs into vector DB
if uploaded_files:
    with st.spinner("Processing and embedding documents..."):
        documents = load_pdfs(uploaded_files)
        embed_documents(documents)
    st.success("PDFs embedded successfully!")

# Ask questions
question = st.text_input("Ask a question about the agreement")
if question:
    with st.spinner("Searching the agreements..."):
        answer = ask_question(question)
        st.write(answer)
