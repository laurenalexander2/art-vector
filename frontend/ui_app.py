import os
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# Config
# ============================================================

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="ArtVector – Semantic Search for Collections",
    layout="wide",
)

PAGES = ["Upload & Index", "Semantic Search", "Datasets", "Object Index"]

st.sidebar.title("ArtVector")
page = st.sidebar.radio("Navigation", PAGES)

# ============================================================
# Met restricted image resolver (optimistic, correct)
# ============================================================

@st.cache_data(ttl=3600)
def resolve_restricted_image(object_id: str) -> str | None:
    if not object_id:
        return None
    return f"https://collectionapi.metmuseum.org/api/collection/v1/iiif/{object_id}/restricted"

# ============================================================
# API helpers
# ============================================================

def api_get(path: str, **params):
    url = f"{API_BASE}{path}"
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def api_post(path: str, files=None, data=None):
    url = f"{API_BASE}{path}"
    r = requests.post(url, files=files, data=data)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=10)
def load_datasets() -> List[Dict[str, Any]]:
    return api_get("/all_datasets")

# ============================================================
# Upload & Index Page
# ============================================================

def render_upload_page():
    st.title("Upload & Index")

    with st.form("upload_form"):
        file = st.file_uploader("Upload CSV file", type=["csv"])
        dataset_name = st.text_input("Dataset name (optional)")
        source_type = st.text_input("Source type", value="museum")
        submitted = st.form_submit_button("Upload")

    if submitted and file:
        with st.spinner("Uploading and ingesting dataset…"):
            files = {"file": (file.name, file.getvalue(), "text/csv")}
            data = {"name": dataset_name, "source_type": source_type}
            res = api_post("/upload_dataset", files=files, data=data)

        st.success(
            f"Dataset uploaded: `{res['dataset_id']}` · {res['num_objects']} objects."
        )

# ============================================================
# Semantic Search Page (CORRECT + PRELOADED)
# ============================================================

def render_search_page():
    st.title("Semantic Search")

    datasets = load_datasets()
    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected = st.selectbox("Limit search to dataset", dataset_options)
    dataset_id = None if selected == "All datasets" else selected

    images_only = st.checkbox("Only show objects with images", value=False)

    # Preloaded query (runs automatically)
    query = st.text_input(
        "Enter a meaning-based query",
        value="artworks that depict cosmic awe",
    )

    k = st.slider("Results to show", 6, 48, 18)

    if not query:
        return

    with st.spinner("Searching…"):
        res = api_get(
            "/search_text",
            q=query,
            limit=k,
            dataset_id=dataset_id,
        )

    if not res:
        st.warning("No results found.")
        return

    cards = []
    rendered = 0

    for r in res:
        obj = r["obj"]

        # AUTHORITATIVE SOURCE
        meta = obj.get("raw_metadata")
        if not meta:
            continue  # correctness: do not guess

        title = meta.get("Title") or "Untitled"
        artist = meta.get("Artist Display Name") or "Unknown artist"
        date = meta.get("Object Date") or ""
        medium = meta.get("Medium") or ""
        place = meta.get("Culture") or meta.get("Country") or ""

        object_id = meta.get("Object ID")
        met_link = meta.get("Link Resource")

        image_url = resolve_restricted_image(object_id)

        if images_only and not image_url:
            continue

        rendered += 1

        cards.append(
            f"""
<div class="result-card">
  <div class="result-image">
    {"<img src='" + image_url + "' />" if image_url else ""}
  </div>
  <div class="result-body">
    <div class="result-title">{title}</div>
    <div class="result-meta">
      {artist}<br>
      {date}<br>
      {medium}<br>
      {place}
    </div>
    {"<a href='" + met_link + "' target='_blank'>View on Met →</a>" if met_link else ""}
  </div>
</div>
"""
        )

    if rendered == 0:
        st.info("Results found, but none matched the image filter.")
        return

    html = f"""
<style>
.result-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 24px;
}}

.result-card {{
  border: 1px solid #ccc;
  padding: 8px;
}}

.result-image img {{
  width: 100%;
  height: auto;
}}

.result-title {{
  font-weight: bold;
}}
</style>

<div class="result-grid">
{''.join(cards)}
</div>
"""

    components.html(html, height=1200, scrolling=True)

# ============================================================
# Datasets Page
# ============================================================

def render_datasets_page():
    st.title("Datasets")
    df = pd.DataFrame(load_datasets())
    st.dataframe(df)

# ============================================================
# Object Index Page
# ============================================================

def render_object_index_page():
    st.title("Object Index")

    datasets = load_datasets()
    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected = st.selectbox("Filter by dataset", dataset_options)
    dataset_id = None if selected == "All datasets" else selected

    limit = st.slider("Max objects to load", 100, 2000, 500, step=100)

    objs = api_get("/all_objects", dataset_id=dataset_id, limit=limit)
    df = pd.DataFrame(objs)
    st.dataframe(df)

# ============================================================
# Page Router
# ============================================================

if page == "Upload & Index":
    render_upload_page()
elif page == "Semantic Search":
    render_search_page()
elif page == "Datasets":
    render_datasets_page()
elif page == "Object Index":
    render_object_index_page()
