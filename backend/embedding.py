import os
from functools import lru_cache
from typing import List

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Respect HF_HOME if set (we configure this in Docker)
HF_HOME = os.getenv("HF_HOME")
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Lazily load and cache the SentenceTransformer model.
    """
    model = SentenceTransformer(MODEL_NAME)
    return model


def embed_texts(texts: List[str]) -> torch.Tensor:
    """
    Embed a list of texts as L2-normalized vectors.
    Returns a torch.Tensor on CPU.
    """
    model = get_model()
    with torch.no_grad():
        emb = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
    return emb
