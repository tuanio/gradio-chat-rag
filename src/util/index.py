import os
import numpy as np
import pickle

from langchain_community.vectorstores import DocArrayInMemorySearch, FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_vector_store_index(
    file_path, embedding_model_repo_id="sentence-transformers/all-roberta-large-v1"
):
    file_path_split = file_path.split(".")
    file_type = file_path_split[-1].rstrip("/")

    if file_type == "csv":
        print(file_path)
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()

    elif file_type == "pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256, chunk_overlap=128
        )

        documents = text_splitter.split_documents(pages)

    model_kwargs = {"device": "cpu"}
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_repo_id,
        model_kwargs=model_kwargs
    )

    vectordb = FAISS.from_documents(documents, embedding_model)
    file_output = "./db/faiss_index"
    vectordb.save_local(file_output)

    return "Vector store index is created."
