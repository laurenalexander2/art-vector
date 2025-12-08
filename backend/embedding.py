from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str):
    text = (text or "").strip()
    if not text:
        return None
    try:
        emb = _model.encode([text], normalize_embeddings=True)[0]
        return emb.tolist()
    except Exception:
        return None
