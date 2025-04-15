# document_processing/doc_loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os

def load_documents(pdf_file=None, pdf_path=None):
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(file_path=tmp_file_path)
        docs = loader.load()
        os.unlink(tmp_file_path)
        return docs
    elif pdf_path:
        loader = PyPDFLoader(file_path=pdf_path)
        return loader.load()
    else:
        raise ValueError("Either pdf_file or pdf_path must be provided")

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,     # bigger to reduce number of chunks
        chunk_overlap=300,   # increase overlap for better context retention
    )
    return text_splitter.split_documents(documents)
