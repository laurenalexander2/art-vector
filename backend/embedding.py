import torch
from sentence_transformers import SentenceTransformer
from functools import lru_cache

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(MODEL_NAME)

def embed_texts(texts):
    model = get_model()
    with torch.no_grad():
        emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return emb
