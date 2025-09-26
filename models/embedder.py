import os
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer

def get_embedding_model():
    embedder = OllamaEmbeddings(model="embeddinggemma:300m", base_url="http://localhost:11434")
    
    return embedder

def get_embedding_model_HF():
    # model_name_emb = "all-MiniLM-L6-v2"
    # model_name_emb = "Alibaba-NLP/gte-multilingual-base"
    model_name_emb = "dangvantuan/vietnamese-embedding"
    embedder = SentenceTransformer(model_name_emb,token=os.getenv("HUGGINGFACEHUB_API_KEY"), trust_remote_code=True)

    return embedder