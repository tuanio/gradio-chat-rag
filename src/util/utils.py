from langchain_huggingface import HuggingFaceEmbeddings

model_kwargs = {"device": "cpu"}
encode_kwargs = {'normalize_embeddings': False}

def load_embedding_model(embedding_model_repo_id):
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_repo_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embedding_model