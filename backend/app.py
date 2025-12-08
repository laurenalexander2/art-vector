import csv
import io
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch

from .embedding import embed_text

app = FastAPI(title="ArtVector – Semantic Search for Museum Data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OBJECTS: List[Dict[str, Any]] = []
EMBEDDINGS: Optional[torch.Tensor] = None
EMBEDDED_INDICES: List[int] = []
UNEMBEDDED_INDICES: List[int] = []

def _build_object(row):
    img = row.get("PrimaryImage") or row.get("PrimaryImageSmall") or ""
    if not img:
        return None

    return {
        "id": row.get("ObjectID"),
        "title": row.get("Title") or "",
        "artist_display": row.get("ArtistDisplayName") or "",
        "medium": row.get("Medium") or "",
        "department": row.get("Department") or "",
        "description": row.get("ObjectName") or "",
        "image_url": img,
        "object_url": row.get("ObjectURL"),
    }

def _build_text(obj):
    return " | ".join(
        p for p in [
            obj.get("title"),
            obj.get("artist_display"),
            obj.get("medium"),
            obj.get("department"),
            obj.get("description"),
        ] if p
    )

def _cosine_knn(query_vec, matrix, k=10):
    if matrix.size(0) == 0:
        return []
    q = query_vec / query_vec.norm()
    m = matrix / matrix.norm(dim=-1, keepdim=True)
    sims = torch.matmul(m, q)
    vals, idx = torch.topk(sims, k=min(k, matrix.size(0)))
    return idx.tolist()

@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    global OBJECTS, EMBEDDINGS, EMBEDDED_INDICES, UNEMBEDDED_INDICES

    try:
        content = await file.read()
        stream = io.StringIO(content.decode("utf-8", errors="ignore"))
        reader = csv.DictReader(stream)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "CSV parse failure"})

    objects = []
    for row in reader:
        obj = _build_object(row)
        if obj:
            objects.append(obj)

    OBJECTS = objects
    EMBEDDINGS = None
    EMBEDDED_INDICES = []
    UNEMBEDDED_INDICES = list(range(len(objects)))

    return {"status": "ok", "total_objects": len(objects)}

@app.post("/process_batch")
def process_batch(limit: int = 128):
    global EMBEDDINGS, EMBEDDED_INDICES, UNEMBEDDED_INDICES

    if not UNEMBEDDED_INDICES:
        return {"processed": 0, "remaining": 0, "total_embedded": len(EMBEDDED_INDICES)}

    selected = UNEMBEDDED_INDICES[:limit]
    UNEMBEDDED_INDICES = UNEMBEDDED_INDICES[limit:]

    vectors = []
    new = []

    for idx in selected:
        obj = OBJECTS[idx]
        text = _build_text(obj)
        vec = embed_text(text)
        if vec:
            vectors.append(torch.tensor(vec))
            new.append(idx)

    if not vectors:
        return {"processed": 0, "remaining": len(UNEMBEDDED_INDICES)}

    batch = torch.stack(vectors)

    if EMBEDDINGS is None:
        EMBEDDINGS = batch
    else:
        EMBEDDINGS = torch.cat([EMBEDDINGS, batch], dim=0)

    EMBEDDED_INDICES.extend(new)

    return {"processed": len(new), "remaining": len(UNEMBEDDED_INDICES)}

@app.get("/search_text")
def search_text(q: str, limit: int = 10):
    if EMBEDDINGS is None or EMBEDDINGS.size(0) == 0:
        return []

    vec = embed_text(q)
    if not vec:
        return []

    query_vec = torch.tensor(vec)
    idxs = _cosine_knn(query_vec, EMBEDDINGS, k=limit)

    results = []
    for pos in idxs:
        obj = OBJECTS[EMBEDDED_INDICES[pos]]
        results.append(obj)
    return results
