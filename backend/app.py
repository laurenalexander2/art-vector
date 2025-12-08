import csv
import io
import os
import json
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

# -----------------------
# Persistent storage paths
# -----------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OBJECTS_PATH = os.path.join(DATA_DIR, "objects.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.pt")
INDICES_PATH = os.path.join(DATA_DIR, "indices.json")
os.makedirs(DATA_DIR, exist_ok=True)

# In-memory "database"
OBJECTS: List[Dict[str, Any]] = []
EMBEDDINGS: Optional[torch.Tensor] = None
EMBEDDED_INDICES: List[int] = []
UNEMBEDDED_INDICES: List[int] = []


# -----------------------
# Helpers
def _build_object_from_row(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    img = row.get("PrimaryImage") or row.get("PrimaryImageSmall")
    # Do NOT skip rows missing image
    # museums often separate metadata and media
    return {
        "id": row.get("ObjectID") or "",
        "title": row.get("Title") or "",
        "artist_display": row.get("ArtistDisplayName") or "",
        "medium": row.get("Medium") or "",
        "department": row.get("Department") or "",
        "description": row.get("ObjectName") or "",
        "image_url": img,  # may be None
        "object_url": row.get("ObjectURL") or "",
    }


def _build_text_for_object(obj: Dict[str, Any]) -> str:
    parts = [
        obj.get("title") or "",
        obj.get("artist_display") or "",
        obj.get("medium") or "",
        obj.get("department") or "",
        obj.get("description") or "",
    ]
    return " | ".join(p for p in parts if p.strip())


def _cosine_knn(query_vec: torch.Tensor, matrix: torch.Tensor, k: int = 10):
    """
    Simple cosine similarity nearest neighbors.
    query_vec: [D]
    matrix: [N, D]
    """
    if matrix.size(0) == 0:
        return [], []

    q = query_vec / query_vec.norm()
    m = matrix / matrix.norm(dim=-1, keepdim=True)
    sims = torch.matmul(m, q)  # [N]
    vals, idx = torch.topk(sims, k=min(k, matrix.size(0)))
    return idx.tolist(), vals.tolist()


def save_state():
    """Persist OBJECTS, EMBEDDINGS, and indices to disk."""
    try:
        with open(OBJECTS_PATH, "w", encoding="utf-8") as f:
            json.dump(OBJECTS, f)
    except Exception as e:
        print(f"[ArtVector] Failed to save OBJECTS: {e}")

    try:
        indices_payload = {
            "embedded_indices": EMBEDDED_INDICES,
            "unembedded_indices": UNEMBEDDED_INDICES,
        }
        with open(INDICES_PATH, "w", encoding="utf-8") as f:
            json.dump(indices_payload, f)
    except Exception as e:
        print(f"[ArtVector] Failed to save indices: {e}")

    try:
        if EMBEDDINGS is not None and EMBEDDINGS.size(0) > 0:
            torch.save(EMBEDDINGS, EMBEDDINGS_PATH)
        elif os.path.exists(EMBEDDINGS_PATH):
            os.remove(EMBEDDINGS_PATH)
    except Exception as e:
        print(f"[ArtVector] Failed to save embeddings: {e}")


def load_state():
    """Load OBJECTS, EMBEDDINGS, and indices from disk, if present."""
    global OBJECTS, EMBEDDINGS, EMBEDDED_INDICES, UNEMBEDDED_INDICES

    # Objects
    if os.path.exists(OBJECTS_PATH):
        try:
            with open(OBJECTS_PATH, "r", encoding="utf-8") as f:
                OBJECTS = json.load(f)
        except Exception as e:
            print(f"[ArtVector] Failed to load OBJECTS: {e}")
            OBJECTS = []

    # Indices
    if os.path.exists(INDICES_PATH):
        try:
            with open(INDICES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            EMBEDDED_INDICES = data.get("embedded_indices", [])
            UNEMBEDDED_INDICES = data.get("unembedded_indices", [])
        except Exception as e:
            print(f"[ArtVector] Failed to load indices: {e}")
            EMBEDDED_INDICES = []
            UNEMBEDDED_INDICES = list(range(len(OBJECTS)))
    else:
        EMBEDDED_INDICES = []
        UNEMBEDDED_INDICES = list(range(len(OBJECTS)))

    # Embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            EMBEDDINGS = torch.load(EMBEDDINGS_PATH, map_location="cpu")
        except Exception as e:
            print(f"[ArtVector] Failed to load embeddings: {e}")
            EMBEDDINGS = None
    else:
        EMBEDDINGS = None


# Load any existing state when the app starts
load_state()


# -----------------------
# Routes
# -----------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "total_objects": len(OBJECTS),
        "embedded": len(EMBEDDED_INDICES),
        "remaining": len(UNEMBEDDED_INDICES),
        "has_embeddings": EMBEDDINGS is not None and EMBEDDINGS.size(0) > 0 if EMBEDDINGS is not None else False,
    }


@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accept a CSV (Met Open Access style), parse it, and build in-memory objects.
    Existing objects / embeddings are reset.
    """
    global OBJECTS, EMBEDDINGS, EMBEDDED_INDICES, UNEMBEDDED_INDICES

    try:
        content = await file.read()  # This will handle ~350MB on a decently-sized machine.
        text_stream = io.StringIO(content.decode("utf-8", errors="ignore"))
        reader = csv.DictReader(text_stream)
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"Failed to read CSV: {exc}"})

    objects: List[Dict[str, Any]] = []
    for row in reader:
        obj = _build_object_from_row(row)
        if obj:
            objects.append(obj)

    OBJECTS = objects
    EMBEDDINGS = None
    EMBEDDED_INDICES = []
    UNEMBEDDED_INDICES = list(range(len(OBJECTS)))

    # Persist the new dataset (objects + empty index)
    save_state()

    return {
        "status": "ok",
        "total_objects": len(OBJECTS),
    }


@app.get("/job_status")
def job_status():
    total = len(OBJECTS)
    processed = len(EMBEDDED_INDICES)
    remaining = len(UNEMBEDDED_INDICES)
    if total == 0:
        status = "empty"
    elif remaining == 0:
        status = "done"
    else:
        status = "indexing"
    return {
        "status": status,
        "processed_count": processed,
        "total_count": total,
        "remaining": remaining,
    }


@app.post("/process_batch")
def process_batch(limit: int = 128):
    """
    Embed a batch of UNEMBEDDED_INDICES into EMBEDDINGS.
    This is called repeatedly (e.g. from the UI while polling job status).
    """
    global EMBEDDINGS, EMBEDDED_INDICES, UNEMBEDDED_INDICES

    if not UNEMBEDDED_INDICES:
        return {"processed": 0, "remaining": 0, "total_embedded": len(EMBEDDED_INDICES)}

    take = UNEMBEDDED_INDICES[:limit]
    UNEMBEDDED_INDICES = UNEMBEDDED_INDICES[limit:]

    vecs = []
    new_indices = []

    for idx in take:
        obj = OBJECTS[idx]
        text = _build_text_for_object(obj)
        vec_list = embed_text(text)
        if vec_list is None:
            continue
        vec = torch.tensor(vec_list, dtype=torch.float32)
        vecs.append(vec)
        new_indices.append(idx)

    if not vecs:
        # Even if none embedded this round, persist indices state
        save_state()
        return {"processed": 0, "remaining": len(UNEMBEDDED_INDICES), "total_embedded": len(EMBEDDED_INDICES)}

    batch_tensor = torch.stack(vecs, dim=0)  # [B, D]

    if EMBEDDINGS is None:
        EMBEDDINGS = batch_tensor
        EMBEDDED_INDICES = new_indices
    else:
        EMBEDDINGS = torch.cat([EMBEDDINGS, batch_tensor], dim=0)
        EMBEDDED_INDICES.extend(new_indices)

    # Persist updated embeddings and indices
    save_state()

    return {
        "processed": len(new_indices),
        "remaining": len(UNEMBEDDED_INDICES),
        "total_embedded": len(EMBEDDED_INDICES),
    }


@app.get("/search_text")
def search_text(q: str = Query(..., description="Text query"), limit: int = 10):
    """
    Search over whatever subset of objects has been embedded so far.
    """
    if EMBEDDINGS is None or EMBEDDINGS.size(0) == 0:
        return []

    vec_list = embed_text(q)
    if vec_list is None:
        return []

    query_vec = torch.tensor(vec_list, dtype=torch.float32)
    idxs, _ = _cosine_knn(query_vec, EMBEDDINGS, k=limit)

    # idxs are positions in EMBEDDINGS, map back via EMBEDDED_INDICES to OBJECTS
    results = []
    for pos in idxs:
        obj_idx = EMBEDDED_INDICES[pos]
        if 0 <= obj_idx < len(OBJECTS):
            obj = OBJECTS[obj_idx]
            results.append(
                {
                    "id": obj.get("id"),
                    "title": obj.get("title"),
                    "artist_display": obj.get("artist_display"),
                    "medium": obj.get("medium"),
                    "department": obj.get("department"),
                    "image_url": obj.get("image_url"),
                    "object_url": obj.get("object_url"),
                }
            )
    return results
