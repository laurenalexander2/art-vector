import os
import json
import time
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st

@st.cache_data(ttl=3600)
def resolve_restricted_image(original_id: str) -> str | None:
    """
    Best-effort resolver for Met restricted images.
    Returns IIIF URL if available, else None.
    """
    if not original_id:
        return None

    url = f"https://collectionapi.metmuseum.org/api/collection/v1/iiif/{original_id}/restricted"

    try:
        r = requests.head(url, timeout=2)
        if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
            return url
    except Exception:
        pass

    return None


# -----------------------------
# Styling
# -----------------------------
def inject_css():
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');
    :root {
        --serif: "Times New Roman", serif;
        --sans: "Inter", sans-serif;
        --idle: #bfbfbf;
        --hover: #ffffff;
    }
    html, body, [class^="css"]  {
        background:#000 !important;
        color:#fff !important;
        font-family:var(--sans);
    }
    section[data-testid="stSidebar"] {
        background:#000 !important;
        border-right:1px solid #222;
        padding-top:40px !important;
    }
    h1,h2,h3 {
        font-family:var(--serif);
        font-weight:400 !important;
    }
    .stButton button {
        background:#fff !important;
        color:#000 !important;
        border-radius:4px;
        padding:10px 24px;
        font-weight:500;
    }
    .result-card {
        padding:20px;
        background:#111;
        border:1px solid #222;
        border-radius:6px;
        margin-bottom:20px;
    }
    .result-card-title {
        font-family:var(--serif);
        font-size:26px;
        margin-bottom:12px;
    }
</style>
""",
        unsafe_allow_html=True,
    )


inject_css()

# -----------------------------
# Config
# -----------------------------
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="ArtVector – Semantic Search for Collections",
    layout="wide",
)

PAGES = ["Upload & Index", "Semantic Search", "Datasets", "Object Index"]

if "embedding_started" not in st.session_state:
    st.session_state["embedding_started"] = False

if "last_uploaded_dataset" not in st.session_state:
    st.session_state["last_uploaded_dataset"] = None

st.sidebar.title("ArtVector")
page = st.sidebar.radio("Navigation", PAGES)


# -----------------------------
# API helpers
# -----------------------------
def api_get(path: str, **params):
    url = f"{API_BASE}{path}"
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, files=None, data=None, json_body=None, **params):
    url = f"{API_BASE}{path}"
    if json_body is not None:
        resp = requests.post(url, params=params, json=json_body)
    else:
        resp = requests.post(url, params=params, files=files, data=data)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=10)
def load_datasets() -> List[Dict[str, Any]]:
    return api_get("/all_datasets")


# -----------------------------
# Upload & Index page
# -----------------------------
def render_upload_page():
    st.title("📂 Upload & Index Dataset")

    st.markdown(
        """
Upload a **museum or collection CSV** (Met-style exports work great).  
The backend will ingest rows, register a dataset, and queue objects for embedding.
"""
    )

    # Only show the upload form if we haven't started embedding yet
    if not st.session_state["embedding_started"]:
        with st.form("upload_form"):
            file = st.file_uploader("Upload CSV file", type=["csv"])
            dataset_name = st.text_input("Dataset name (optional)")
            source_type = st.text_input("Source type", value="museum")
            submitted = st.form_submit_button("Upload")

        if submitted and file is not None:
            with st.spinner("Uploading and ingesting dataset…"):
                files = {"file": (file.name, file.getvalue(), "text/csv")}
                data = {
                    "name": dataset_name,
                    "source_type": source_type,
                }
                res = api_post("/upload_dataset", files=files, data=data)

            st.success(
                f"Dataset uploaded: `{res['dataset_id']}` with {res['num_objects']} objects."
            )
            st.json(res)
            st.session_state["last_uploaded_dataset"] = res["dataset_id"]
    else:
        st.info(
            "Embedding has started. Upload form hidden to keep this run focused. "
            "Refresh the app or clear cache to upload a new dataset."
        )

    st.markdown("---")
    st.markdown("### 🔄 Embedding Progress & Control")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run one embedding batch"):
            st.session_state["embedding_started"] = True
            with st.spinner("Processing batch…"):
                res = api_post("/process_batch")
            st.write(res)
            example = res.get("example_object")
            if example:
                with st.expander("Example object from this batch"):
                    st.write(f"**Title:** {example.get('title') or '[Untitled]'}")
                    st.write(f"**Artist:** {example.get('artist') or 'Unknown'}")
                    st.write(f"**Dataset:** `{example.get('dataset_id')}`")
                    st.write(f"**Original ID:** {example.get('original_id')}")
                    st.write(f"**Preview text:** {example.get('preview_text')}")
                    st.json(example.get("metadata", {}))

    with col2:
        if st.button("Run batches until done (loop)"):
            st.session_state["embedding_started"] = True

            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            example_placeholder = st.empty()

            while True:
                res = api_post("/process_batch")
                status_placeholder.write(res)

                # Progress bar from /job_status
                try:
                    status = api_get("/job_status")
                    total = status.get("total", 0)
                    percent = status.get("percent", 0.0)
                    remaining = status.get("remaining", 0)
                    embedded = status.get("embedded", 0)

                    progress_bar = progress_placeholder.progress(
                        min(100, int(percent))
                    )
                    progress_placeholder.write(
                        f"Embedded: {embedded} / {total} "
                        f"({percent:.2f}%), remaining: {remaining}"
                    )
                except Exception as e:
                    progress_placeholder.write(f"Status error: {e}")

                example = res.get("example_object")
                if example:
                    with example_placeholder.container():
                        with st.expander("Example object from current batch", expanded=False):
                            st.write(
                                f"**Title:** {example.get('title') or '[Untitled]'}"
                            )
                            st.write(
                                f"**Artist:** {example.get('artist') or 'Unknown'}"
                            )
                            st.write(
                                f"**Dataset:** `{example.get('dataset_id')}` — "
                                f"Original ID: `{example.get('original_id')}`"
                            )
                            st.write(
                                f"**Preview text:** {example.get('preview_text')}"
                            )
                            st.json(example.get("metadata", {}))

                if res.get("done") or res.get("embedded_this_batch", 0) == 0:
                    break

                time.sleep(0.4)

            st.success("Embedding complete or queue exhausted.")

    st.markdown("### 📊 Current Job Status")

    if st.button("Refresh status"):
        status = api_get("/job_status")
        st.write(status)
        total = status.get("total", 0)
        percent = status.get("percent", 0.0)
        remaining = status.get("remaining", 0)
        embedded = status.get("embedded", 0)
        if total > 0:
            st.progress(min(100, int(percent)))
            st.write(
                f"Embedded: {embedded} / {total} "
                f"({percent:.2f}%), remaining: {remaining}"
            )
        else:
            st.info("No objects in the system yet.")


# -----------------------------
# Semantic Search page
# -----------------------------
def render_search_page():
    st.title("🔍 Semantic Search")

    datasets = load_datasets()
    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected_dataset = st.selectbox("Limit to dataset", dataset_options)
    dataset_id = None if selected_dataset == "All datasets" else selected_dataset

    images_only = st.checkbox("Only show objects with images", value=False)

    query = st.text_input(
        "Enter a meaning-based query", value="surrealist female portrait"
    )
    k = st.slider("Results to show", 5, 50, 15)

    if st.button("Search") and query:
        with st.spinner("Embedding query and searching…"):
            res = api_get(
                "/search_text",
                q=query,
                limit=k,
                dataset_id=dataset_id,
                images_only=images_only,
            )

        if not res:
            st.warning("No results yet. Make sure you've embedded at least some objects.")
            return

        for r in res:
            obj = r["obj"]
            score = r["score"]
            title = obj.get("title") or "[Untitled]"

            with st.container():
                st.markdown(
                    f"<div class='result-card'>"
                    f"<div class='result-card-title'>{title}</div>"
                    f"<div>Similarity score: {score:.3f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                cols = st.columns([2, 1])
                with cols[0]:
                    st.write(f"**Dataset:** `{obj['dataset_id']}`")
                    st.write(f"**Artist:** {obj.get('artist') or 'Unknown'}")
                    st.write(f"**Original ID:** {obj.get('original_id')}")
                    with st.expander("Raw metadata"):
                        st.json(obj["raw_metadata"])

                with cols[1]:
                    image_url = obj.get("image_url")

                # 1) Public-domain image
                if image_url:
                    st.image(
                        image_url,
                        caption=title,
                        use_container_width=True,
                    )
                else:
                    # 2) Try restricted IIIF
                    restricted = resolve_restricted_image(obj.get("original_id"))

                    if restricted:
                        st.image(
                            restricted,
                            caption=f"{title} (restricted)",
                            use_container_width=True,
                        )
                    else:
                        st.write("_No image available for this object._")


# -----------------------------
# Datasets page
# -----------------------------
def render_datasets_page():
    st.title("🗂 Datasets")

    datasets = load_datasets()
    if not datasets:
        st.info("No datasets yet. Upload one from the **Upload & Index** page.")
        return

    df = pd.DataFrame(datasets)
    st.dataframe(df)

    st.markdown("#### Dataset details")
    for d in datasets:
        with st.expander(f"{d.get('name') or d['dataset_id']}"):
            st.write(f"**ID:** `{d['dataset_id']}`")
            st.write(f"**Source type:** {d.get('source_type')}")
            st.write(f"**Original file:** {d.get('original_filename')}")
            st.write(f"**Created at:** {d.get('created_at')}")
            st.write(f"**Objects:** {d.get('num_objects')}")
            st.write("**Metadata fields:**")
            st.code(", ".join(d.get("metadata_fields", [])))


# -----------------------------
# Object Index page
# -----------------------------
def render_object_index_page():
    st.title("📚 Object Index")

    datasets = load_datasets()
    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected_dataset = st.selectbox("Filter by dataset", dataset_options)
    dataset_id = None if selected_dataset == "All datasets" else selected_dataset

    limit = st.slider("Max objects to load", 100, 2000, 500, step=100)

    with st.spinner("Loading objects…"):
        objs = api_get("/all_objects", dataset_id=dataset_id, limit=limit)

    if not objs:
        st.info("No objects found. Have you uploaded a dataset?")
        return

    records = []
    for o in objs:
        records.append(
            {
                "object_uid": o["object_uid"],
                "dataset_id": o["dataset_id"],
                "original_id": o.get("original_id"),
                "title": o.get("title"),
                "artist": o.get("artist"),
                "has_image": o.get("has_image"),
                "image_url": o.get("image_url"),
            }
        )

    df = pd.DataFrame(records)
    st.dataframe(df)

    st.markdown("#### Sample raw metadata")
    with st.expander("Show raw metadata for first 5 objects"):
        for o in objs[:5]:
            st.write(f"**{o.get('title') or '[Untitled]'}** — `{o['object_uid']}`")
            st.json(o["raw_metadata"])


# -----------------------------
# Page router
# -----------------------------
if page == "Upload & Index":
    render_upload_page()
elif page == "Semantic Search":
    render_search_page()
elif page == "Datasets":
    render_datasets_page()
elif page == "Object Index":
    render_object_index_page()
