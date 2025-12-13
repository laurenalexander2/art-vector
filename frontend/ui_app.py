import os
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# Met restricted image resolver (optimistic IIIF)
# ============================================================

@st.cache_data(ttl=3600)
def resolve_restricted_image(object_id: str) -> str | None:
    if not object_id:
        return None
    return f"https://collectionapi.metmuseum.org/api/collection/v1/iiif/{object_id}/restricted"


# ============================================================
# Styling
# ============================================================

def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');

:root {
    --serif: "Times New Roman", serif;
    --sans: "Inter", sans-serif;
}

html, body, [class^="css"] {
    background:#000 !important;
    color:#fff !important;
    font-family:var(--sans);
}

section[data-testid="stSidebar"] {
    background:#000 !important;
    border-right:1px solid #222;
}

h1, h2, h3 {
    font-family:var(--serif);
    font-weight:400 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

inject_css()

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
# API helpers
# ============================================================

def api_get(path: str, **params):
    url = f"{API_BASE}{path}"
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def api_post(path: str, files=None, data=None, json_body=None):
    url = f"{API_BASE}{path}"
    if json_body is not None:
        r = requests.post(url, json=json_body)
    else:
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
# Semantic Search Page (CARD GRID)
# ============================================================

def render_search_page():
    def render_search_page():
    st.title("Semantic Search")
    
    datasets = load_datasets()
    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected = st.selectbox("Limit search to dataset", dataset_options)
    dataset_id = None if selected == "All datasets" else selected
    
    images_only = st.checkbox("Only show objects with images", value=False)
    
    # Pre-loaded query
    default_query = "artworks that depict cosmic awe"
    
    # Input field with the default query
    query = st.text_input("Enter a meaning-based query", value=default_query)
    
    k = st.slider("Results to show", 6, 48, 18)
    
    # Automatically trigger the search on page load
    if st.button("Search") or query == default_query:
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

            object_id = obj.get("original_id")
            image_url = obj.get("image_url") or resolve_restricted_image(object_id)

            if images_only and not image_url:
                continue

            rendered += 1

            title = obj.get("title") or "Untitled"
            artist = obj.get("artist") or "Unknown artist"
            date = obj.get("year") or obj.get("date") or ""
            medium = obj.get("medium") or ""
            place = obj.get("place") or obj.get("culture") or ""

            met_link = (
                f"https://www.metmuseum.org/art/collection/search/{object_id}"
                if object_id
                else None
            )

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
    {"<div class='result-link'><a href='" + met_link + "' target='_blank'>View on Met →</a></div>" if met_link else ""}
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
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));
  gap:24px;
}}

.result-card {{
  background:#0f0f0f;
  border:1px solid #222;
  border-radius:6px;
  overflow:hidden;
}}

.result-image img {{
  width:100%;
  height:360px;
  object-fit:contain;
  background:#000;
}}

.result-body {{
  padding:16px;
}}

.result-title {{
  font-family:var(--serif);
  font-size:20px;
  margin-bottom:6px;
}}

.result-meta {{
  font-size:14px;
  color:#ccc;
  line-height:1.45;
}}

.result-link {{
  margin-top:10px;
  font-size:13px;
}}
</style>

<div class="result-grid">
{''.join(cards)}
</div>
"""

        components.html(
            html,
            height=min(350 + rendered * 420, 3000),
            scrolling=True,
        )

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
