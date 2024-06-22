from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

model_kwargs = {"device": "cpu"}
encode_kwargs = {'normalize_embeddings': False}

def load_embedding_model(embedding_model_repo_id):
    if embedding_model_repo_id == 'openai-model':
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_repo_id,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    return embedding_model

def load_text_splitter(is_semantic=False, embedding_model=None):
    if is_semantic:
        text_splitter = SemanticChunker(embedding_model,
                            breakpoint_threshold_type="percentile")
    else:
        chunk_size = 500
        chunk_overlap = 100

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    
    return text_splitter