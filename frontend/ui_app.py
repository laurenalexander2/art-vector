import os
import json
import time
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st


def inject_css():
    import streamlit as st
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');
    :root { --serif: "Times New Roman", serif; --sans: "Inter", sans-serif; --idle:#bfbfbf; --hover:#fff; }
    html, body { background:#000 !important; color:#fff !important; font-family:var(--sans); }
    section[data-testid="stSidebar"] { background:#000 !important; border-right:1px solid #222; padding-top:40px !important; }
    h1,h2,h3 { font-family:var(--serif); font-weight:400 !important; }
    .stButton button { background:#fff !important; color:#000 !important; border-radius:4px; padding:10px 24px; }
    .result-card { padding:20px; background:#111; border:1px solid #222; border-radius:6px; margin-bottom:20px; }
    .result-card-title { font-family:var(--serif); font-size:26px; margin-bottom:12px; }
    </style>
    """, unsafe_allow_html=True)
inject_css()
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="ArtVector – Semantic Search for Collections",
    layout="wide",
)

PAGES = ["Upload & Index", "Semantic Search", "Datasets", "Object Index"]

st.sidebar.title("ArtVector")
page = st.sidebar.radio("Navigation", PAGES)

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

def render_upload_page():
    st.title("📂 Upload & Index Dataset")

    st.markdown(
        """Upload a **museum or collection CSV** (Met-style exports work great).  
        The backend will ingest rows, register a dataset, and queue objects for embedding.
        """
    )

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
        st.success(f"Dataset uploaded: `{res['dataset_id']}` with {res['num_objects']} objects.")
        st.json(res)

    st.markdown("### 🔄 Embedding Progress")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run one embedding batch"):
            with st.spinner("Processing batch…"):
                res = api_post("/process_batch")
            st.write(res)

    with col2:
        if st.button("Run batches until done (may take a while)"):
            progress = st.empty()
            status_box = st.empty()
            while True:
                res = api_post("/process_batch")
                status_box.write(res)
                if res.get("done") or res.get("embedded_this_batch", 0) == 0:
                    break
                progress.write(f"Embedded this batch: {res['embedded_this_batch']}, remaining: {res['remaining']}")
                time.sleep(0.5)
            st.success("Embedding complete or queue exhausted.")

    st.markdown("### 📊 Current Job Status")
    if st.button("Refresh status"):
        status = api_get("/job_status")
        st.write(status)

def render_search_page():
    st.title("🔍 Semantic Search")

    datasets = load_datasets()
    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected_dataset = st.selectbox("Limit to dataset", dataset_options)
    dataset_id = None if selected_dataset == "All datasets" else selected_dataset

    images_only = st.checkbox("Only show objects with images", value=False)

    query = st.text_input("Enter a meaning-based query", value="surrealist female portrait")
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
            with st.expander(f"{obj.get('title') or '[Untitled]'} — score {score:.3f}"):
                meta = obj["raw_metadata"]
                left, right = st.columns([2, 1])
                with left:
                    st.write(f"**Dataset:** `{obj['dataset_id']}`")
                    st.write(f"**Artist:** {obj.get('artist')}")
                    st.write(f"**Original ID:** {obj.get('original_id')}")
                    st.json(meta)
                with right:
                    if obj.get("image_url"):
                        st.image(obj["image_url"], caption=obj.get("title") or "Image")

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

if page == "Upload & Index":
    render_upload_page()
elif page == "Semantic Search":
    render_search_page()
elif page == "Datasets":
    render_datasets_page()
elif page == "Object Index":
    render_object_index_page()