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
    page_title="ArtVector",
    layout="wide",
)

# ============================================================
# Styling (neutral, stable)
# ============================================================

def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');

html, body {
    font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
    background: #fafafa;
    color: #111;
}

h1, h2, h3 {
    font-weight: 500;
}

.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 20px;
}

.result-card {
    background: #fff;
    border: 1px solid #e5e5e5;
    border-radius: 6px;
    overflow: hidden;
}

.result-image {
    width: 100%;
    height: 220px;
    background: #f0f0f0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    color: #777;
}

.result-image img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

.result-body {
    padding: 12px 14px;
}

.result-title {
    font-size: 15px;
    font-weight: 500;
    margin-bottom: 6px;
}

.result-meta {
    font-size: 13px;
    color: #444;
    line-height: 1.45;
}

.result-link {
    margin-top: 8px;
    font-size: 12px;
}
</style>
""",
        unsafe_allow_html=True,
    )

inject_css()

# ============================================================
# API helpers (timeout-safe)
# ============================================================

def api_get(path: str, **params):
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ReadTimeout:
        st.error(
            "Search is taking longer than expected. "
            "Try reducing results or selecting a dataset."
        )
        return None


@st.cache_data(ttl=10)
def load_datasets() -> List[Dict[str, Any]]:
    return api_get("/all_datasets") or []


# ============================================================
# Image helpers (Met-only, browser-resolved)
# ============================================================

def met_open_access_image(obj: Dict[str, Any]) -> str | None:
    return obj.get("primaryImageSmall") or obj.get("primaryImage") or None


def met_restricted_image(object_id: str | None) -> str | None:
    if not object_id:
        return None
    # IMPORTANT: do NOT fetch or verify server-side
    return f"https://collectionapi.metmuseum.org/api/collection/v1/iiif/{object_id}/restricted"


# ============================================================
# Semantic Search Page
# ============================================================

def render_search_page():
    st.title("Semantic Search")

    datasets = load_datasets()
    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected = st.selectbox("Limit search to dataset", dataset_options)
    dataset_id = None if selected == "All datasets" else selected

    images_only = st.checkbox("Only show objects with images", value=False)

    query = st.text_input(
        "Enter a meaning-based query",
        value="modern marble sculpture",
    )

    k = st.slider("Results to show", 5, 50, 20)

    run = st.button("Search", type="primary")

    if not run or not query.strip():
        return

    with st.spinner("Searching…"):
        res = api_get(
            "/search_text",
            q=query,
            limit=k,
            dataset_id=dataset_id,
        )

    if not res:
        return

    cards_html = []
    shown = 0

    for r in res:
        if shown >= k:
            break

        obj = r["obj"]

        object_id = obj.get("original_id")
        title = obj.get("title") or "Untitled"
        artist = obj.get("artistDisplayName") or obj.get("artist") or "Unknown"
        date = obj.get("objectDate") or ""
        medium = obj.get("medium") or ""
        culture = obj.get("culture") or obj.get("country") or ""

        image_url = met_open_access_image(obj)
        if not image_url:
            image_url = met_restricted_image(object_id)

        has_image_candidate = image_url is not None

        if images_only and not has_image_candidate:
            continue

        met_link = (
            obj.get("objectURL")
            or (f"https://www.metmuseum.org/art/collection/search/{object_id}" if object_id else "")
        )

        img_html = (
            f"<img src='{image_url}' onerror=\"this.style.display='none'; this.parentElement.innerHTML='No image available';\" />"
            if image_url
            else "No image available"
        )

        cards_html.append(
            f"""
<div class="result-card">
  <div class="result-image">
    {img_html}
  </div>
  <div class="result-body">
    <div class="result-title">{title}</div>
    <div class="result-meta">
      {artist}<br>
      {date}<br>
      {medium}<br>
      {culture}<br>
      Object ID: {object_id or "—"}
    </div>
    <div class="result-link">
      <a href="{met_link}" target="_blank">View on Met →</a>
    </div>
  </div>
</div>
"""
        )

        shown += 1

    if shown == 0:
        st.info("No results matched the current filters.")
        return

    components.html(
        f"<div class='result-grid'>{''.join(cards_html)}</div>",
        height=min(400 + shown * 260, 3000),
        scrolling=True,
    )


# ============================================================
# Datasets Page (with preview)
# ============================================================

def render_datasets_page():
    st.title("Datasets")

    datasets = load_datasets()
    if not datasets:
        st.info("No datasets available.")
        return

    dataset_ids = [d["dataset_id"] for d in datasets]
    selected = st.selectbox("Select a dataset", dataset_ids)

    if not selected:
        return

    st.subheader("Sample objects")

    objs = api_get("/all_objects", dataset_id=selected, limit=5)
    if not objs:
        st.info("No objects found.")
        return

    df = pd.DataFrame(
        [
            {
                "Object ID": o.get("original_id"),
                "Title": o.get("title"),
                "Artist": o.get("artistDisplayName") or o.get("artist"),
                "Date": o.get("objectDate"),
                "Medium": o.get("medium"),
            }
            for o in objs
        ]
    )
    st.dataframe(df, use_container_width=True)


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
            res = requests.post(
                f"{API_BASE}/upload_dataset", files=files, data=data
            ).json()

        st.success(
            f"Dataset uploaded: `{res['dataset_id']}` · {res['num_objects']} objects."
        )


# ============================================================
# Navigation
# ============================================================

st.sidebar.title("ArtVector")
page = st.sidebar.radio(
    "Navigation",
    ["Semantic Search", "Datasets", "Upload & Index"],
)

if page == "Semantic Search":
    render_search_page()
elif page == "Datasets":
    render_datasets_page()
elif page == "Upload & Index":
    render_upload_page()
