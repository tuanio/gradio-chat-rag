import os, gc, shutil
from util.conversation_rag import Conversation_RAG
from util.index import *
import torch


class ModelSetup:
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
    
    def setup(self):
        conv_rag = Conversation_RAG(self.embedding_model, self.llm)
        self.model, self.tokenizer, self.vectordb = conv_rag.load_model_and_tokenizer()
        return "Model Setup Complete"
