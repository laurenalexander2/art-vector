# ArtVector (Persistent Edition)

Semantic search for museum and cultural collections — with **SQLite-backed persistence**, dataset provenance, and a multi-page Streamlit UI.

## What This App Does

- Ingests large collection CSVs (e.g. Met Open Access export)
- Registers each upload as a **dataset** with:
  - dataset ID
  - human-readable name
  - source type (e.g. museum, archive)
  - original filename
  - metadata fields
- Stores each object in **SQLite**, linked to its dataset and original ID
- Generates **semantic embeddings** using `sentence-transformers/all-MiniLM-L6-v2`
- Persists embeddings in the database so work is **not lost on restart**
- Provides:
  - 🔍 **Semantic Search** page
  - 🗂 **Dataset Overview** page
  - 📚 **Object Index** page
  - 📂 **Upload & Index** page with embedding progress

## Tech Stack

- **Backend:** FastAPI, SQLite, Torch, SentenceTransformers
- **Frontend:** Streamlit
- **Container:** Docker + Docker Compose

## Project Layout

```bash
artvector_app/
├── backend/
│   ├── app.py           # FastAPI app + SQLite persistence
│   └── embedding.py     # SentenceTransformer wrapper
├── frontend/
│   └── ui_app.py        # Multi-page Streamlit UI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Running the App

You need:

- Docker Desktop (Mac / Windows / Linux)
- Internet on first run (to download the embedding model)

From the `artvector_app` directory:

```bash
docker compose up --build
```

Then visit:

- UI: http://localhost:8501
- API (optional): http://localhost:8000/docs

A Docker volume named `artvector_data` stores:

- `artvector.db` — SQLite database with datasets, objects, and embeddings.

## Usage Flow

### 1. Upload a Dataset

Go to **Upload & Index** in the sidebar.

- Upload a CSV (Met-style Open Access exports work well)
- Optionally provide:
  - Dataset name
  - Source type (e.g. museum, archive, gallery)
- The backend:
  - Registers a dataset row in SQLite
  - Streams the CSV
  - Inserts objects linked to that dataset
  - Marks them as **unembedded** initially

### 2. Embed Objects

On the same page:

- Click **Run one embedding batch** to embed a chunk
- Or **Run batches until done** to process the entire queue

Under the hood:

- `/process_batch`:
  - Selects N objects with `embedding IS NULL`
  - Builds text from fields like Title, Artist, Medium, Culture
  - Encodes with `all-MiniLM-L6-v2` (cosine-normalized)
  - Stores the vector as JSON in SQLite

You can check progress via **Refresh status**, which calls `/job_status`.

### 3. Explore Datasets

On the **Datasets** page:

- See all datasets with:
  - dataset ID
  - name
  - source type
  - original filename
  - created-at timestamp
  - number of objects
  - metadata field list

Each upload is a **source** that is tracked forever.

### 4. Browse Objects

On the **Object Index** page:

- Filter by dataset or view across all
- Configure how many objects to load (up to 2000)
- See:
  - object UID (`{dataset_id}__{original_id}`)
  - dataset
  - artist
  - title
  - image presence

You can also expand a section to see **raw metadata** for sample objects.

### 5. Run Semantic Search

On the **Semantic Search** page:

- Enter a meaning-based text query:
  - `surrealist female portrait`
  - `bronze ritual vessel`
  - `abstract lithograph 1950s`
- Optionally:
  - Limit to a specific dataset
  - Require objects with images only
- The backend:
  - Embeds the query
  - Loads embedded objects (optionally filtered)
  - Computes cosine similarity in PyTorch
  - Returns top-k neighbors with scores and metadata

If object rows include image URLs, the UI will show **inline thumbnails**.

## Schema

### datasets

- `id` (INTEGER, PK)
- `dataset_id` (TEXT, unique) – internal slug
- `name` (TEXT)
- `source_type` (TEXT)
- `original_filename` (TEXT)
- `created_at` (TEXT, ISO)
- `metadata_fields` (TEXT, JSON array)
- `num_objects` (INTEGER)

### objects

- `id` (INTEGER, PK)
- `object_uid` (TEXT, unique) – `{dataset_id}__{original_id}`
- `dataset_id` (TEXT, FK → datasets.dataset_id)
- `original_id` (TEXT) – museum/local object ID
- `title` (TEXT)
- `artist` (TEXT)
- `image_url` (TEXT)
- `has_image` (INTEGER, 0/1)
- `raw_metadata` (TEXT, JSON)
- `embedding` (TEXT, JSON array of floats, nullable)

Objects are always linked back to their dataset and keep a full copy of the original CSV row.

## Notes & Limitations

- This is a **prototype engine**, not a production ANN service.
- For large collections (>200k objects), you may want to:
  - Move embeddings into a dedicated vector DB (pgVector, Qdrant, Vespa)
  - Use ANN indexing instead of in-memory cosine over all rows.
- SQLite is used here to give you:
  - Persistence
  - Easy inspection
  - Path to migrate later

## Extending This

Some natural next steps:

- Add image embeddings & multimodal search
- Add curator workspaces (saved sets, comparisons)
- Add object-level linking to authority vocabularies (ULAN, AAT, VIAF)
- Export search results as CSV for cataloging workflows

---

If you want to integrate with an existing museum system, you can:

- Map `original_id` to local object IDs
- Use `dataset_id` to represent specific exports or collection segments
- Keep the SQLite DB as a cache in front of an institutional system of record
