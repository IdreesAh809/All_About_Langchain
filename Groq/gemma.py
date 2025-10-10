import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

## Load the GROQ API key only (HuggingFace token not needed here)
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ChatGroq With Gemma Demo")

# Initialize Llama3 model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="groq/compound-mini",   
    temperature=0.2,
    max_tokens=1024
)


# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
""")

# Function for vector embedding
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")  # ✅ specify model
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Load PDF folder
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Input from user
prompt1 = st.text_input("Enter Your Question From Documents")

# Button for embeddings
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# Process the question
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please create document embeddings first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        print("Response time:", time.process_time() - start)
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

## ## Flow

# Loader → Splitter → Embeddings → FAISS → Retriever → Groq LLM → Output
