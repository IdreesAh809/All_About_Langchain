import streamlit as st
import os, time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ObjectBox VectorstoreDB With Groq-llm Demo")

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="qwen/qwen3-32b"
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
<context>
{context}
<context>
Questions:{input}
""")

# --------------------------
# Vector embedding & ObjectBox
# --------------------------
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        docs = PyPDFDirectoryLoader("./pdfs").load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs[:20])
        st.session_state.vectors = ObjectBox.from_documents(chunks, st.session_state.embeddings, embedding_dimensions=384)
        st.success("✅ ObjectBox Database is ready!")

# --------------------------
# Auto-build vectorstore on app startup
# --------------------------
vector_embedding()  # Automatically build when the app starts

# --------------------------
# Streamlit UI
# --------------------------
query = st.text_input("Enter Your Question From Documents")

if query:
    chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, chain)

    start = time.process_time()
    resp = retrieval_chain.invoke({'input': query})
    st.write("💡 Answer:", resp['answer'])
    st.write(f"⏱ Response time: {time.process_time()-start:.2f}s")

    with st.expander("Document Similarity Search"):
        for doc in resp.get("context", []):
            st.write(doc.page_content)
            st.write("--------------------------------")
