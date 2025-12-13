import os
import json
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Restricted image resolver (Met IIIF)
# ============================================================

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
        padding-top:40px !important;
    }
    h1,h2,h3 {
        font-family:var(--serif);
        font-weight:400 !important;
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
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, files=None, data=None, json_body=None):
    url = f"{API_BASE}{path}"
    if json_body is not None:
        resp = requests.post(url, json=json_body)
    else:
        resp = requests.post(url, files=files, data=data)
    resp.raise_for_status()
    return resp.json()


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

    if submitted and file is not None:
        with st.spinner("Uploading and ingesting dataset…"):
            files = {"file": (file.name, file.getvalue(), "text/csv")}
            data = {"name": dataset_name, "source_type": source_type}
            res = api_post("/upload_dataset", files=files, data=data)

        st.success(
            f"Dataset uploaded: `{res['dataset_id']}` · {res['num_objects']} objects."
        )
        st.json(res)


# ============================================================
# Semantic Search Page (FIXED)
# ============================================================

def render_search_page():
    st.title("Semantic Search")

    datasets = load_datasets()
    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected = st.selectbox("Limit search to dataset", dataset_options)
    dataset_id = None if selected == "All datasets" else selected

    images_only = st.checkbox("Only show objects with images", value=False)

    query = st.text_input("Enter a meaning-based query")
    k = st.slider("Results to show", 5, 50, 15)

    if st.button("Search") and query:
        with st.spinner("Embedding query and searching…"):
            res = api_get(
                "/search_text",
                q=query,
                limit=k,
                dataset_id=dataset_id,
            )

        if not res:
            st.warning("No results found.")
            return

        shown = 0

        for r in res:
            obj = r["obj"]
            score = r["score"]
            title = obj.get("title") or "[Untitled]"

            image_url = obj.get("image_url")
            restricted_url = None

            if not image_url:
                restricted_url = resolve_restricted_image(obj.get("original_id"))

            has_any_image = bool(image_url or restricted_url)

            if images_only and not has_any_image:
                continue

            shown += 1

            object_page = f"https://www.metmuseum.org/art/collection/search/{obj.get('original_id')}"

            with st.expander(f"{title} — score {score:.3f}"):
                left, right = st.columns([2, 1])

                with left:
                    st.markdown(
                        f"[View on Met website]({object_page})",
                        unsafe_allow_html=True,
                    )
                    st.write(f"**Dataset:** `{obj['dataset_id']}`")
                    st.write(f"**Artist:** {obj.get('artist') or 'Unknown'}")
                    st.write(f"**Original ID:** {obj.get('original_id')}")
                    st.json(obj["raw_metadata"])

                with right:
                    if image_url:
                        st.image(image_url, caption=title, use_container_width=True)
                    elif restricted_url:
                        st.image(
                            restricted_url,
                            caption=f"{title} (restricted)",
                            use_container_width=True,
                        )
                    else:
                        st.caption("No image available")

        if images_only and shown == 0:
            st.info("No results with images found for this query.")


# ============================================================
# Datasets Page
# ============================================================

def render_datasets_page():
    st.title("Datasets")

    datasets = load_datasets()
    if not datasets:
        st.info("No datasets uploaded yet.")
        return

    df = pd.DataFrame(datasets)
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

    with st.spinner("Loading objects…"):
        objs = api_get("/all_objects", dataset_id=dataset_id, limit=limit)

    if not objs:
        st.info("No objects found.")
        return

    records = [
        {
            "object_uid": o["object_uid"],
            "dataset_id": o["dataset_id"],
            "original_id": o.get("original_id"),
            "title": o.get("title"),
            "artist": o.get("artist"),
            "image_url": o.get("image_url"),
        }
        for o in objs
    ]

    df = pd.DataFrame(records)
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
