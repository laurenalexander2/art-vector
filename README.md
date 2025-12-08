📌 README.md — ArtVector
(Semantic Retrieval Engine for Cultural Data)

1. Project Overview
ArtVector is a semantic retrieval engine for museum and cultural heritage datasets.
It ingests large metadata exports (e.g., The Metropolitan Museum of Art Open Access CSV),
converts objects into latent embedding space, and enables meaning-based search
without keyword matching.
This system is intended as:
✔ a research/prototype platform
✔ an institutional discovery layer
✔ a foundation for future vector database integrations
✔ a demonstrator for semantic search over cultural objects

2. System Function
ArtVector performs:
Dataset ingestion
→ Extracts usable objects and fields
Representation learning
→ Embeds text metadata using SentenceTransformer models
Indexing and batching
→ Builds progressive embedding matrix in memory
Semantic evaluation & retrieval
→ Converts queries into embedding space
→ Computes cosine similarity to items
→ Returns top-K meaning neighbors

3. Why This Exists
Museums store millions of objects but:
indexing is literal
subject terms are inconsistent
keywords don’t capture artistic meaning
cross-collection similarity is invisible
ArtVector produces latent search:
“floral abstract etching”
“mexican surrealist woodcut”
“female portrait lithograph 1950s”
“bronze ritual vessel”
… return objects that fit meaningfully, not literally.

4. Architecture
frontend/  (Streamlit UI)
backend/   (FastAPI embedding + retrieval engine)
docker     (Isolation + reproducibility)
Components:
4.1 Backend (FastAPI)
Dataset loader
Latent embedding engine
Cosine similarity / ANN search
Progressive batching system
4.2 Embedding engine
SentenceTransformer: all-MiniLM-L6-v2
Normalized 384-dim vector output
4.3 In-memory vector store
Stores:
OBJECTS                → list of dicts
EMBEDDINGS             → torch tensor [N, D]
EMBEDDED_INDICES       → mapping embeddings → objects
UNEMBEDDED_INDICES     → work queue
4.4 Frontend
Dataset upload
Embedding progress polling
Semantic text search UI
Image preview + metadata readout

5. Execution Flow
Upload
CSV → parse rows → build object list → reset embedding state
Indexing
Loop:
take N pending objects →
build text →
embedding →
normalize →
append to tensor →
update index →
repeat until done
Search
text query →
embed →
cosine similarity →
return best neighbors

6. Technologies Used
FastAPI — backend API framework
Torch — cosine similarity + tensor operations
SentenceTransformers — semantic encoding
Streamlit — UI layer
Docker Compose — two-service orchestration

7. Installation
Requirements
Docker Desktop (Mac / Windows / Linux)
Internet (first run downloads SentenceTransformer)
Run
docker compose up --build
Visit UI:
http://localhost:8501

8. Usage
1. Upload a Met-style Open Access CSV
→ UI reads dataset → backend extracts objects
2. Start indexing
Embeds objects in progressive batches
UI polls /job_status
Progress bar updates
3. Semantic search
Enter queries like:
surrealist female portrait
religious woodcut print
bronze ritual vessel
abstract lithograph 1950s
Returns meaningfully related objects (not literal matches).

9. API Endpoints (Backend)
/upload_dataset
POST CSV → ingest objects
/process_batch
Run N embeddings → append to tensor
/job_status
Return process state
/search_text?q=...&limit=N
Return semantic neighbors

10. Embedding Model Notes
Model:
sentence-transformers/all-MiniLM-L6-v2
384-dim dense vector
cosine-normalized output
Properties:
good CPU inference speed
robust for metadata short text
meaning separation in cultural terminology
Swappable — see section 13.

11. Performance Notes
Handles 300–500k objects on a modern MacBook / cloud VM
Embedding cost scales linearly
Fast search via vector normalization and top-k similarity
Future work:
approximate nearest neighbor index
persistent vector store

12. Limitations
This version is in-memory only, meaning:
❌ embeddings disappear on restart
❌ not multi-user persistent
❌ not optimized for ANN querying
These are intentional — the app is an engine prototype, not the enterprise artifact.

13. Model Substitution Guide
To change embeddings:
Edit:
backend/embedding.py
Swap:
"sentence-transformers/all-MiniLM-L6-v2"
for:
multi-qa-MiniLM-L6-cos-v1 (ranking optimized)
all-mpnet-base-v2 (higher semantic richness)
CLIP text encoder for multimodal future work

14. Roadmap (Turning Prototype → Product)
Phase 1 — Add persistence (pgVector, Qdrant, or Vespa)
Phase 2 — Add enrichment UI (taxonomy filling, clustering, similarity sets)
Phase 3 — Add authority vocabulary linking (ULAN, AAT, VIAF)
Phase 4 — Multimodal support (image embeddings + alignment)
Phase 5 — Access control, curator workspace, annotation layer
Phase 6 — Packaging for institutional deployment

15. Concept Summary
ArtVector is an indexing engine that transforms cultural metadata into latent space, enabling institutional search and discovery by meaning rather than keywords.
