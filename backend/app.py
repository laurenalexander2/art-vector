import io
import csv
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .embedding import embed_texts

# -----------------------------
# Storage paths / persistent data
# -----------------------------
# Prefer DATA_DIR (Dockerfile sets DATA_DIR=/app/data),
# but fall back to DATA_PATH or /app/data.
DATA_ROOT = Path(
    os.getenv("DATA_DIR")
    or os.getenv("DATA_PATH")
    or "/app/data"
)
DATA_ROOT.mkdir(parents=True, exist_ok=True)

DB_DIR = DATA_ROOT / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "artvector.db"

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def get_db() -> sqlite3.Connection:
    """
    Get a SQLite connection with row factory enabled.
    check_same_thread=False so FastAPI can use it across requests.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    cur = conn.cursor()

    # Datasets table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT UNIQUE,
            name TEXT,
            source_type TEXT,
            original_filename TEXT,
            created_at TEXT,
            metadata_fields TEXT,
            num_objects INTEGER DEFAULT 0
        );
        """
    )

    # Objects table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_uid TEXT UNIQUE,
            dataset_id TEXT,
            original_id TEXT,
            title TEXT,
            artist TEXT,
            image_url TEXT,
            has_image INTEGER,
            raw_metadata TEXT,
            embedding TEXT,
            FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
        );
        """
    )

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_objects_dataset ON objects(dataset_id);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_objects_embedding_null ON objects(embedding);"
    )

    conn.commit()
    conn.close()


init_db()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="ArtVector API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for internal Streamlit + demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Pydantic models
# -----------------------------
class DatasetOut(BaseModel):
    dataset_id: str
    name: Optional[str]
    source_type: Optional[str]
    original_filename: Optional[str]
    created_at: Optional[str]
    metadata_fields: List[str]
    num_objects: int


class ObjectOut(BaseModel):
    object_uid: str
    dataset_id: str
    original_id: Optional[str]
    title: Optional[str]
    artist: Optional[str]
    image_url: Optional[str]
    has_image: bool
    raw_metadata: Dict[str, Any]


class SearchResult(BaseModel):
    score: float
    obj: ObjectOut


# -----------------------------
# Helper functions
# -----------------------------
def register_dataset(
    name: str, filename: str, fields: List[str], source_type: str = "museum"
) -> str:
    dataset_id = f"{name.lower().replace(' ', '_')}_{int(time.time())}"
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO datasets (
            dataset_id, name, source_type, original_filename,
            created_at, metadata_fields
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_id,
            name,
            source_type,
            filename,
            datetime.utcnow().isoformat(),
            json.dumps(fields),
        ),
    )
    conn.commit()
    conn.close()
    return dataset_id


def build_object_uid(dataset_id: str, original_id: str) -> str:
    return f"{dataset_id}__{original_id}"


def build_object_text(row: Dict[str, Any]) -> str:
    """
    Build a text string for embedding from common museum CSV fields.
    Falls back to concatenating all non-empty values.
    """
    parts: List[str] = []
    for key in [
        "Title",
        "ObjectName",
        "Artist",
        "Maker",
        "Culture",
        "Medium",
        "Classification",
        "Department",
    ]:
        v = row.get(key)
        if v:
            parts.append(str(v))

    if not parts:
        # Fallback: concatenate all non-empty values
        parts = [" ".join(str(v) for v in row.values() if v)]

    return " | ".join(parts)


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/upload_dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    source_type: Optional[str] = Form("museum"),
):
    """
    Upload a museum/collection CSV.
    Registers a dataset and inserts rows into `objects` with embedding=NULL.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()
    decoded = content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(decoded))
    fields = reader.fieldnames or []

    dataset_name = name or os.path.splitext(file.filename)[0]
    dataset_id = register_dataset(
        dataset_name, file.filename, fields, source_type or "museum"
    )

    conn = get_db()
    cur = conn.cursor()

    num_objects = 0
    for row in reader:
        original_id = row.get("ObjectID") or row.get("id") or str(num_objects + 1)
        object_uid = build_object_uid(dataset_id, original_id)

        title = row.get("Title") or row.get("ObjectName")
        artist = row.get("Artist") or row.get("Maker")

        image_url = (
            row.get("ImageURL")
            or row.get("PrimaryImage")
            or row.get("Image")
            or None
        )
        has_image = 1 if image_url else 0

        cur.execute(
            """
            INSERT OR IGNORE INTO objects (
                object_uid, dataset_id, original_id, title, artist,
                image_url, has_image, raw_metadata, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                object_uid,
                dataset_id,
                original_id,
                title,
                artist,
                image_url,
                has_image,
                json.dumps(row),
            ),
        )
        num_objects += 1

    cur.execute(
        "UPDATE datasets SET num_objects = ? WHERE dataset_id = ?",
        (num_objects, dataset_id),
    )
    conn.commit()
    conn.close()

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "num_objects": num_objects,
        "fields": fields,
    }


@app.post("/process_batch")
def process_batch(batch_size: int = 128):
    """
    Process a batch of objects whose embedding is NULL.
    Returns:
    - embedded_this_batch
    - remaining
    - total
    - done
    - example_object (small preview of one object from this batch)
    """
    conn = get_db()
    cur = conn.cursor()

    # How many remain?
    cur.execute("SELECT COUNT(*) AS c FROM objects WHERE embedding IS NULL")
    remaining = cur.fetchone()["c"]

    if remaining == 0:
        conn.close()
        return {
            "embedded_this_batch": 0,
            "remaining": 0,
            "total": 0,
            "done": True,
            "example_object": None,
        }

    # Select batch
    cur.execute(
        """
        SELECT id, raw_metadata, dataset_id, original_id, title, artist, image_url, has_image
        FROM objects
        WHERE embedding IS NULL
        ORDER BY id ASC
        LIMIT ?
        """,
        (batch_size,),
    )
    rows = cur.fetchall()

    ids: List[int] = [r["id"] for r in rows]
    metas: List[Dict[str, Any]] = [json.loads(r["raw_metadata"]) for r in rows]

    texts = [build_object_text(m) for m in metas]
    embeddings = embed_texts(texts).cpu()

    # Update DB with embeddings
    for obj_id, vec in zip(ids, embeddings):
        vec_list = vec.tolist()
        cur.execute(
            "UPDATE objects SET embedding = ? WHERE id = ?",
            (json.dumps(vec_list), obj_id),
        )

    conn.commit()

    # Recompute totals
    cur.execute("SELECT COUNT(*) AS c FROM objects")
    total = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) AS c FROM objects WHERE embedding IS NULL")
    remaining_after = cur.fetchone()["c"]

    # Build a small example object preview from the first row in this batch
    example_object: Optional[Dict[str, Any]] = None
    if rows:
        first = rows[0]
        first_meta = metas[0] if metas else {}
        example_object = {
            "dataset_id": first["dataset_id"],
            "original_id": first["original_id"],
            "title": first["title"],
            "artist": first["artist"],
            "image_url": first["image_url"],
            "has_image": bool(first["has_image"]),
            "preview_text": build_object_text(first_meta),
            "metadata": first_meta,
        }

    conn.close()

    return {
        "embedded_this_batch": len(ids),
        "remaining": remaining_after,
        "total": total,
        "done": remaining_after == 0,
        "example_object": example_object,
    }


@app.get("/job_status")
def job_status():
    """
    Global embedding job status (across all datasets).
    """
    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS c FROM objects")
    total = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) AS c FROM objects WHERE embedding IS NULL")
    remaining = cur.fetchone()["c"]

    embedded = total - remaining
    percent = float(embedded) / float(total) * 100.0 if total > 0 else 0.0

    conn.close()
    return {
        "total": total,
        "embedded": embedded,
        "remaining": remaining,
        "percent": percent,
    }


@app.get("/all_datasets", response_model=List[DatasetOut])
def all_datasets():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM datasets ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()

    out: List[DatasetOut] = []
    for r in rows:
        out.append(
            DatasetOut(
                dataset_id=r["dataset_id"],
                name=r["name"],
                source_type=r["source_type"],
                original_filename=r["original_filename"],
                created_at=r["created_at"],
                metadata_fields=json.loads(r["metadata_fields"] or "[]"),
                num_objects=r["num_objects"] or 0,
            )
        )
    return out


@app.get("/all_objects", response_model=List[ObjectOut])
def all_objects(dataset_id: Optional[str] = None, limit: int = 500):
    conn = get_db()
    cur = conn.cursor()

    if dataset_id:
        cur.execute(
            """
            SELECT *
            FROM objects
            WHERE dataset_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (dataset_id, limit),
        )
    else:
        cur.execute(
            """
            SELECT *
            FROM objects
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

    rows = cur.fetchall()
    conn.close()

    out: List[ObjectOut] = []
    for r in rows:
        out.append(
            ObjectOut(
                object_uid=r["object_uid"],
                dataset_id=r["dataset_id"],
                original_id=r["original_id"],
                title=r["title"],
                artist=r["artist"],
                image_url=r["image_url"],
                has_image=bool(r["has_image"]),
                raw_metadata=json.loads(r["raw_metadata"] or "{}"),
            )
        )
    return out


@app.get("/search_text", response_model=List[SearchResult])
def search_text(
    q: str,
    limit: int = 20,
    dataset_id: Optional[str] = None,
    images_only: bool = False,
):
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    conn = get_db()
    cur = conn.cursor()

    base_query = "SELECT * FROM objects WHERE embedding IS NOT NULL"
    params: List[Any] = []
    if dataset_id:
        base_query += " AND dataset_id = ?"
        params.append(dataset_id)
    if images_only:
        base_query += " AND has_image = 1"

    cur.execute(base_query, params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []

    obj_vecs = []
    objs = []
    for r in rows:
        emb = json.loads(r["embedding"])
        obj_vecs.append(emb)
        objs.append(r)

    obj_tensor = torch.tensor(obj_vecs, dtype=torch.float32)
    query_emb = embed_texts([q]).cpu()[0]

    # Cosine similarity since embeddings are normalized
    scores = torch.matmul(obj_tensor, query_emb)
    scores_vals = scores.numpy()

    indices = scores.argsort(descending=True)[:limit].tolist()

    results: List[SearchResult] = []
    for idx in indices:
        r = objs[idx]
        score = float(scores_vals[idx])
        results.append(
            SearchResult(
                score=score,
                obj=ObjectOut(
                    object_uid=r["object_uid"],
                    dataset_id=r["dataset_id"],
                    original_id=r["original_id"],
                    title=r["title"],
                    artist=r["artist"],
                    image_url=r["image_url"],
                    has_image=bool(r["has_image"]),
                    raw_metadata=json.loads(r["raw_metadata"] or "{}"),
                ),
            )
        )

    return results
