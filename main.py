import streamlit as st
from utils import load_pdfs_from_folders, embed_documents, ask_question

st.set_page_config(page_title="Collective Agreement Advisor", layout="centered")
st.title("🤖 Collective Agreement Advisor")

# Union checkboxes
st.subheader("Which union(s) should this question apply to?")
selected_unions = []
if st.checkbox("CUPE"):
    selected_unions.append("CUPE")
if st.checkbox("BCGEU - Instructor"):
    selected_unions.append("BCGEU-instructor")
if st.checkbox("BCGEU - Support"):
    selected_unions.append("BCGEU-support")

# Embed PDFs from selected unions
if selected_unions:
    with st.spinner("Loading and embedding selected agreements..."):
        documents = load_pdfs_from_folders(selected_unions)
        if documents:
            embed_documents(documents)
            st.success("Agreements embedded successfully!")
        else:
            st.warning("No documents found in selected folders.")

# Ask a question
question = st.text_input("Ask your question here")
if question:
    with st.spinner("Analyzing agreements..."):
        answer = ask_question(question)
        st.write(answer)
