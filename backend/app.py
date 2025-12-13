import io
import csv
import json
import os
import sqlite3
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .embedding import embed_texts

# -----------------------------
# Storage paths
# -----------------------------
DATA_ROOT = Path(os.getenv("DATA_DIR") or os.getenv("DATA_PATH") or "/app/data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

DB_DIR = DATA_ROOT / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "artvector.db"

EMBEDDING_DIM = 384

# -----------------------------
# Globals
# -----------------------------
EMBED_THREADS: Dict[str, threading.Thread] = {}
THREAD_LOCK = threading.Lock()

SEARCH_CACHE_LOCK = threading.Lock()
SEARCH_CACHE: Dict[tuple, Dict[str, Any]] = {}

# -----------------------------
# DB helpers
# -----------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT UNIQUE,
            name TEXT,
            source_type TEXT,
            original_filename TEXT,
            created_at TEXT,
            metadata_fields TEXT,
            num_objects INTEGER DEFAULT 0,
            embedding_active INTEGER DEFAULT 0
        );
    """)

    cur.execute("""
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
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_objects_dataset ON objects(dataset_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_objects_has_image ON objects(has_image);")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_objects_dataset_has_image ON objects(dataset_id, has_image);"
    )

    conn.commit()
    conn.close()


def configure_db():
    conn = get_db()
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=8000;")
    conn.commit()
    conn.close()


init_db()
configure_db()

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="ArtVector API", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------
class DatasetOut(BaseModel):
    dataset_id: str
    name: Optional[str]
    source_type: Optional[str]
    original_filename: Optional[str]
    created_at: Optional[str]
    metadata_fields: List[str]
    num_objects: int
    embedding_active: bool


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


class EmbeddingControlRequest(BaseModel):
    dataset_id: str
    active: bool


# -----------------------------
# Dataset helpers
# -----------------------------
def register_dataset(name, filename, fields, source_type):
    dataset_id = f"{name.lower().replace(' ', '_')}_{int(time.time())}"
    conn = get_db()
    conn.execute(
        """
        INSERT INTO datasets
        (dataset_id, name, source_type, original_filename, created_at, metadata_fields)
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


def build_object_uid(dataset_id, original_id):
    return f"{dataset_id}__{original_id}"


# -----------------------------
# Upload dataset
# -----------------------------
@app.post("/upload_dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    source_type: Optional[str] = Form("museum"),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "CSV only")

    reader = csv.DictReader(io.TextIOWrapper(file.file, encoding="utf-8", errors="ignore"))
    fields = reader.fieldnames or []

    dataset_name = name or file.filename.rsplit(".", 1)[0]
    dataset_id = register_dataset(dataset_name, file.filename, fields, source_type)

    conn = get_db()
    cur = conn.cursor()

    batch = []
    count = 0

    for row in reader:
        original_id = row.get("ObjectID") or row.get("id") or str(count + 1)
        uid = build_object_uid(dataset_id, original_id)

        title = row.get("Title") or row.get("ObjectName")
        artist = row.get("Artist") or row.get("Maker")
        image_url = row.get("ImageURL") or row.get("PrimaryImage")
        has_image = 1 if image_url else 0

        batch.append(
            (
                uid,
                dataset_id,
                original_id,
                title,
                artist,
                image_url,
                has_image,
                json.dumps(row),
            )
        )
        count += 1

        if len(batch) >= 1000:
            cur.executemany(
                """
                INSERT OR IGNORE INTO objects
                (object_uid, dataset_id, original_id, title, artist,
                 image_url, has_image, raw_metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                batch,
            )
            conn.commit()
            batch.clear()

    if batch:
        cur.executemany(
            """
            INSERT OR IGNORE INTO objects
            (object_uid, dataset_id, original_id, title, artist,
             image_url, has_image, raw_metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            batch,
        )
        conn.commit()

    conn.execute(
        "UPDATE datasets SET num_objects=? WHERE dataset_id=?",
        (count, dataset_id),
    )
    conn.commit()
    conn.close()

    return {"dataset_id": dataset_id, "num_objects": count}


# -----------------------------
# Dataset listing
# -----------------------------
@app.get("/all_datasets", response_model=List[DatasetOut])
def all_datasets():
    conn = get_db()
    rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    conn.close()

    return [
        DatasetOut(
            dataset_id=r["dataset_id"],
            name=r["name"],
            source_type=r["source_type"],
            original_filename=r["original_filename"],
            created_at=r["created_at"],
            metadata_fields=json.loads(r["metadata_fields"] or "[]"),
            num_objects=r["num_objects"] or 0,
            embedding_active=bool(r["embedding_active"]),
        )
        for r in rows
    ]


# -----------------------------
# Object index
# -----------------------------
@app.get("/all_objects", response_model=List[ObjectOut])
def all_objects(dataset_id: Optional[str] = None, limit: int = 500):
    conn = get_db()
    if dataset_id:
        rows = conn.execute(
            "SELECT * FROM objects WHERE dataset_id=? ORDER BY id DESC LIMIT ?",
            (dataset_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM objects ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()

    return [
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
        for r in rows
    ]


# -----------------------------
# Optimized semantic search
# -----------------------------
def _count_embedded(dataset_id, images_only):
    conn = get_db()
    q = "SELECT COUNT(*) FROM objects WHERE embedding IS NOT NULL"
    params = []
    if dataset_id:
        q += " AND dataset_id=?"
        params.append(dataset_id)
    if images_only:
        q += " AND has_image=1"
    c = conn.execute(q, params).fetchone()[0]
    conn.close()
    return c


def _build_search_cache(dataset_id, images_only):
    conn = get_db()
    q = """
        SELECT object_uid, dataset_id, original_id, title, artist,
               image_url, has_image, embedding
        FROM objects WHERE embedding IS NOT NULL
    """
    params = []
    if dataset_id:
        q += " AND dataset_id=?"
        params.append(dataset_id)
    if images_only:
        q += " AND has_image=1"
    rows = conn.execute(q, params).fetchall()
    conn.close()

    vecs = []
    meta = []
    for r in rows:
        vecs.append(json.loads(r["embedding"]))
        meta.append(
            {
                "object_uid": r["object_uid"],
                "dataset_id": r["dataset_id"],
                "original_id": r["original_id"],
                "title": r["title"],
                "artist": r["artist"],
                "image_url": r["image_url"],
                "has_image": bool(r["has_image"]),
            }
        )

    tensor = torch.tensor(vecs, dtype=torch.float32) if vecs else None
    return {"tensor": tensor, "rows": meta, "embedded_count": len(meta)}


@app.get("/search_text", response_model=List[SearchResult])
def search_text(
    q: str,
    limit: int = 20,
    dataset_id: Optional[str] = None,
    images_only: bool = False,
):
    if not q:
        raise HTTPException(400, "Query required")

    key = (dataset_id or "__all__", images_only)
    count = _count_embedded(dataset_id, images_only)

    with SEARCH_CACHE_LOCK:
        cache = SEARCH_CACHE.get(key)
        if cache is None or cache["embedded_count"] != count:
            cache = _build_search_cache(dataset_id, images_only)
            SEARCH_CACHE[key] = cache

    if not cache["tensor"] is not None:
        return []

    query_vec = embed_texts([q]).cpu()[0]
    scores = torch.matmul(cache["tensor"], query_vec)

    k = min(limit, scores.numel())
    vals, idxs = torch.topk(scores, k=k)

    conn = get_db()
    meta_rows = conn.execute(
        f"SELECT object_uid, raw_metadata FROM objects WHERE object_uid IN ({','.join(['?']*k)})",
        [cache["rows"][i]["object_uid"] for i in idxs.tolist()],
    ).fetchall()
    conn.close()

    meta_map = {r["object_uid"]: json.loads(r["raw_metadata"] or "{}") for r in meta_rows}

    return [
        SearchResult(
            score=float(vals[i]),
            obj=ObjectOut(
                **cache["rows"][idxs[i]],
                raw_metadata=meta_map.get(cache["rows"][idxs[i]]["object_uid"], {}),
            ),
        )
        for i in range(k)
    ]
