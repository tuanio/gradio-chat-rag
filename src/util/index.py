import os
import numpy as np
import pickle

from langchain_community.vectorstores import DocArrayInMemorySearch, FAISS, Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader

from util.utils import load_embedding_model, load_text_splitter


def load_default_documents(text_splitter):
    path = "data/chatbot_knowledge.txt"
    pages = TextLoader(path).load()
    documents = text_splitter.split_documents(pages)
    return documents


def create_vector_store_index(
    file_path, embedding_model_repo_id="Fsoft-AIC/videberta-base"
):
    file_path_split = file_path.split(".")
    file_type = file_path_split[-1].rstrip("/")

    embedding_model = load_embedding_model(embedding_model_repo_id)
    text_splitter = load_text_splitter(is_semantic=False)

    if file_type == "csv":
        print(file_path)
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
    elif file_type == "pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    elif file_type in ['txt', 'md']:
        loader = TextLoader(file_path)
        pages = loader.load()
    elif file_type in ['doc', 'docx']:
        loader = Docx2txtLoader(file_path)
        pages = loader.load()

    # pages = UnstructuredFileLoader(file_path).load()
    documents = text_splitter.split_documents(pages)

    # always add chatbot_knowledge inside
    documents += load_default_documents(text_splitter)

    vectordb = FAISS.from_documents(documents, embedding_model)
    file_output = "./db/faiss_index"
    vectordb.save_local(file_output)

    return "Vector store index is created."
