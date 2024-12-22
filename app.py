import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from firebase_admin import credentials, firestore, initialize_app
import os
from dotenv import load_dotenv
import json

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Use the provided Groq API key directly
groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Load Firebase configuration from JSON file
with open('firebase_config.json') as f:
    firebase_config = json.load(f)

# Initialize Firebase
cred = credentials.Certificate(firebase_config)

try:
    initialize_app(cred)
except ValueError:
    # This means the app is already initialized, so you can use it directly
    pass

db = firestore.client()

# Set up Streamlit
st.title("Conversational RAG With PDF uploads")
st.write("Upload PDFs and get answers to your questions based on their content")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Function to extract text from PDF files using PyPDF2
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Process uploaded PDFs
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        text = get_pdf_text([uploaded_file])
        doc_ref = db.collection("pdf_contents").document(uploaded_file.name)
        doc_ref.set({"content": text})
        documents.append({"page_content": text, "metadata": {"source": uploaded_file.name}})

    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = []
    for doc in documents:
        doc_splits = text_splitter.split_text(doc["page_content"])
        for i, split in enumerate(doc_splits):
            splits.append({"page_content": split, "metadata": {"source": doc["metadata"]["source"], "split_index": i}})
    splits_content = [split["page_content"] for split in splits]
    vectorstore = FAISS.from_texts(splits_content, embedding=embeddings)
    retriever = vectorstore.as_retriever()


    # Answer question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    user_input = st.text_input("Your question:")
    if user_input:
        response = rag_chain.invoke({"input": user_input})
        st.write("Assistant:", response['answer'])
else:
    st.warning("Please upload PDF files to get started")