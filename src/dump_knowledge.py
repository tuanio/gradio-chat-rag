from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS

from util.utils import load_embedding_model, load_text_splitter

load_dotenv()

embedding_model = load_embedding_model('openai-model')
text_splitter = load_text_splitter(is_semantic=False, embedding_model=embedding_model)


file_path = 'data/chatbot_knowledge.txt'
file_output = "./db/faiss_index"

pages = UnstructuredFileLoader(file_path).load()
documents = text_splitter.split_documents(pages)


vectordb = FAISS.from_documents(documents, embedding_model)
vectordb.save_local(file_output)