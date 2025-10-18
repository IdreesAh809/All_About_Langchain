# LangChain — Complete Practical Implementation

This repository is my complete, from-scratch implementation of **LangChain** — covering every concept, component, and integration needed to build modern **LLM-powered applications**.  
It includes experiments with different **open-source LLMs**, **vector databases**, and tools to fully understand how LangChain connects all parts of a **generative AI pipeline**.

---

##  Objective

To deeply learn and implement:

- All core LangChain modules  
- Integration with multiple LLM models (**OpenAI**, **Groq**, **Gemini**, **Hugging Face**, **Ollama models** such as **Llama 2** and **Llama 3**)  
- Use of different embeddings and vector stores (**FAISS**, **Chroma**, **ObjectBox**, etc.)  
- Development of **retrieval-based** and **agent-based** applications  
- Step-by-step experiments to master every LangChain concept  

---

##  Topics Covered

- **Prompt Templates & Prompt Engineering**  
- **Chains & Sequential Workflows**  
- **Conversational Memory**  
- **Agents & Tools Integration**  
- **Document Loaders & Text Splitters**  
- **Embeddings & Vector Databases (FAISS, Chroma, ObjectBox)**  
- **Retrieval-Augmented Generation (RAG)**  
- **Working with different LLMs** (OpenAI, Groq, Gemini, Hugging Face, Ollama — Llama 2 & 3)  
- **Building end-to-end LangChain Applications**

---

##  Repository Structure


agents/ — Experiments with LangChain Agents that connect and reason over multiple data sources, integrating tools and retrieval mechanisms for intelligent task execution.

api/ — Backend API integration layer built using FastAPI and Ollama API, demonstrating how to expose LLM and RAG functionalities through REST endpoints (includes Groq, Gemini, and OpenAI connections).

chatbot_with_langsmith/ — Chatbot example built using LangSmith for tracing, debugging, and monitoring LangChain workflows.

huggingface/ — Experiments using open-source LLMs from Hugging Face for embeddings, text generation, and local inference.

rag/ — Retrieval-Augmented Generation (RAG) implementations using multiple vector stores such as FAISS and Chroma for document retrieval and context-aware responses.

objectbox_vectorstore/ — (Experimental) ObjectBox-based local vector database implementation for storing and querying embeddings locally.

requirements.txt — Python dependencies for LangChain, FastAPI, and related libraries.

packages.txt — List of extra packages used during experimentation.

README.md — Repository overview (this file).



##  Installation

To install all required dependencies, run:

```
pip install -r requirements.txt
```
