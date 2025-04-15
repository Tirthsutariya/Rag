from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

def create_vectorstore(documents, persist_dir="chroma_db"):
    # Ensure directory exists
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    return vectorstore
