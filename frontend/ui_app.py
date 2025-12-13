import os
import random
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# Config
# ============================================================

API_BASE = os.getenv("API_BASE", "http://localhost:8000")  # your backend (ArtVector)
MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"

st.set_page_config(
    page_title="ArtVector – Semantic Search for Collections",
    layout="wide",
)

# Page order (as requested)
PAGES = ["Semantic Search", "Datasets", "Upload & Index"]
st.sidebar.title("ArtVector")
page = st.sidebar.radio("Navigation", PAGES)

# ============================================================
# Backend API helpers (ArtVector)
# ============================================================

def api_get(path: str, **params):
    url = f"{API_BASE}{path}"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def api_post(path: str, files=None, data=None):
    url = f"{API_BASE}{path}"
    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=10)
def load_datasets() -> List[Dict[str, Any]]:
    return api_get("/all_datasets")

# ============================================================
# Met API helpers (authoritative metadata + open access images)
# ============================================================

@st.cache_data(ttl=86400)
def met_get_object(object_id: int) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{MET_API_BASE}/objects/{int(object_id)}", timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def met_restricted_iiif_url(object_id: int) -> str:
    return f"https://collectionapi.metmuseum.org/api/collection/v1/iiif/{int(object_id)}/restricted"

@st.cache_data(ttl=86400)
def met_restricted_exists(object_id: int) -> bool:
    """
    True if Met restricted IIIF endpoint returns an image.
    Uses HEAD for speed; caches for a day.
    """
    url = met_restricted_iiif_url(object_id)
    try:
        r = requests.head(url, timeout=3, allow_redirects=True)
        ctype = (r.headers.get("Content-Type") or "").lower()
        return (r.status_code == 200) and ("image" in ctype)
    except Exception:
        return False

def extract_object_id_from_backend_obj(obj: Dict[str, Any]) -> Optional[int]:
    """
    Prefer Object ID from raw_metadata (CSV keys) if present, else obj["original_id"].
    """
    meta = obj.get("raw_metadata") or {}
    oid = meta.get("Object ID") or meta.get("ObjectID") or meta.get("objectID")
    if oid is None:
        oid = obj.get("original_id")
    if oid is None:
        return None
    try:
        return int(str(oid).strip())
    except Exception:
        return None

def met_wall_label_fields(m: Dict[str, Any]) -> Dict[str, str]:
    """
    Minimal wall-label fields from Met API response.
    """
    title = m.get("title") or "Untitled"
    artist = m.get("artistDisplayName") or "Unknown artist"
    date = m.get("objectDate") or ""
    medium = m.get("medium") or ""
    place = m.get("culture") or m.get("country") or ""
    object_id = str(m.get("objectID") or "")
    link = m.get("objectURL") or (f"https://www.metmuseum.org/art/collection/search/{object_id}" if object_id else "")
    return {
        "title": title,
        "artist": artist,
        "date": date,
        "medium": medium,
        "place": place,
        "object_id": object_id,
        "link": link,
    }

def met_best_image(m: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    """
    Returns (image_url, is_real_image)
    - Open Access: primaryImageSmall (or primaryImage)
    - Otherwise: restricted IIIF if it exists
    - Else: None
    """
    oid = m.get("objectID")
    if not oid:
        return None, False

    primary_small = (m.get("primaryImageSmall") or "").strip()
    primary = (m.get("primaryImage") or "").strip()

    if primary_small:
        return primary_small, True
    if primary:
        return primary, True

    # restricted
    if met_restricted_exists(int(oid)):
        return met_restricted_iiif_url(int(oid)), True

    return None, False

# ============================================================
# Cards (HTML)
# ============================================================

def render_cards(cards: List[Dict[str, Any]]):
    # cards: {title, artist, date, medium, place, object_id, link, image_url, has_image}
    html_cards = []
    for c in cards:
        img_block = ""
        if c["image_url"]:
            img_block = f"<img src='{c['image_url']}' alt='Image' />"
        else:
            img_block = f"""
            <div class="placeholder">
              <div class="placeholder-title">No image available</div>
              <div class="placeholder-sub">Object ID: {c['object_id']}</div>
            </div>
            """

        html_cards.append(
            f"""
<div class="card">
  <div class="media">{img_block}</div>
  <div class="body">
    <div class="title">{c['title']}</div>
    <div class="meta">
      <div>{c['artist']}</div>
      <div class="muted">{c['date']}</div>
      <div class="muted">{c['medium']}</div>
      <div class="muted">{c['place']}</div>
      <div class="muted">Object ID: {c['object_id']}</div>
    </div>
    <div class="link"><a href="{c['link']}" target="_blank" rel="noopener">View on Met →</a></div>
  </div>
</div>
"""
        )

    html = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');

.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 18px;
  padding: 6px 2px;
  font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
}}

.card {{
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 10px;
  overflow: hidden;
  background: #fff;
}}

.media {{
  width: 100%;
  aspect-ratio: 4 / 5;
  background: #f4f4f4;
  display: flex;
  align-items: center;
  justify-content: center;
}}

.media img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
}}

.placeholder {{
  width: 100%;
  height: 100%;
  padding: 16px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  color: rgba(0,0,0,0.65);
  text-align: center;
  gap: 6px;
}}

.placeholder-title {{
  font-weight: 600;
}}

.placeholder-sub {{
  font-size: 12px;
}}

.body {{
  padding: 12px 12px 14px 12px;
}}

.title {{
  font-family: "Times New Roman", serif;
  font-size: 18px;
  line-height: 1.25;
  margin-bottom: 8px;
}}

.meta {{
  font-size: 13px;
  line-height: 1.35;
  color: rgba(0,0,0,0.80);
}}

.muted {{
  color: rgba(0,0,0,0.55);
}}

.link {{
  margin-top: 10px;
  font-size: 13px;
}}

.link a {{
  text-decoration: none;
}}

.link a:hover {{
  text-decoration: underline;
}}
</style>

<div class="grid">
{''.join(html_cards)}
</div>
"""
    components.html(html, height=1150, scrolling=True)

# ============================================================
# Page: Semantic Search
# ============================================================

def render_search_page():
    st.title("Semantic Search")

    datasets = load_datasets()
    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected = st.selectbox("Limit search to dataset", dataset_options)
    dataset_id = None if selected == "All datasets" else selected

    images_only = st.checkbox("Only show objects with images", value=False)

    default_query = "artworks that depict cosmic awe"
    query = st.text_input("Enter a meaning-based query", value=default_query)

    k = st.slider("Results to show", 5, 50, 20)

    # Auto-run once per (dataset_id, query) pair on load
    key = f"auto_ran::{dataset_id}::{query}"
    if key not in st.session_state:
        st.session_state[key] = False

    run = st.button("Search", type="primary") or (not st.session_state[key] and query.strip() != "")
    if not run:
        return

    st.session_state[key] = True

    with st.spinner("Searching…"):
        res = api_get("/search_text", q=query, limit=k, dataset_id=dataset_id)

    if not res:
        st.warning("No results found.")
        return

    cards = []
    for r in res:
        obj = r.get("obj") or {}
        oid = extract_object_id_from_backend_obj(obj)
        if oid is None:
            continue

        met_obj = met_get_object(oid)
        if not met_obj:
            continue

        fields = met_wall_label_fields(met_obj)
        image_url, has_image = met_best_image(met_obj)

        if images_only and not has_image:
            continue

        cards.append(
            {
                **fields,
                "image_url": image_url,
                "has_image": has_image,
            }
        )

    if not cards:
        st.info("No results to display (try turning off the image filter).")
        return

    render_cards(cards)

# ============================================================
# Page: Datasets (combined with object preview)
# ============================================================

def render_datasets_page():
    st.title("Datasets")

    datasets = load_datasets()
    if not datasets:
        st.info("No datasets uploaded yet.")
        return

    df = pd.DataFrame(datasets)
    st.dataframe(df, use_container_width=True)

    st.subheader("Preview a dataset")

    dataset_ids = [d["dataset_id"] for d in datasets]
    chosen = st.selectbox("Select a dataset to preview", dataset_ids)

    # pull a small preview from your backend
    preview_count = 5
    with st.spinner("Loading preview…"):
        objs = api_get("/all_objects", dataset_id=chosen, limit=200)

    if not objs:
        st.info("No objects found for this dataset.")
        return

    # choose 5 sample objects (random, stable-ish)
    sample = random.sample(objs, k=min(preview_count, len(objs)))

    cards = []
    for o in sample:
        oid = extract_object_id_from_backend_obj(o)
        if oid is None:
            continue
        met_obj = met_get_object(oid)
        if not met_obj:
            continue

        fields = met_wall_label_fields(met_obj)
        image_url, has_image = met_best_image(met_obj)

        # For preview: always show placeholders as needed
        cards.append(
            {
                **fields,
                "image_url": image_url,
                "has_image": has_image,
            }
        )

    if not cards:
        st.info("Could not build a preview from Met API for these sample objects.")
        return

    render_cards(cards)

# ============================================================
# Page: Upload & Index
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

        st.success(f"Dataset uploaded: `{res['dataset_id']}` · {res['num_objects']} objects.")

# ============================================================
# Router
# ============================================================

if page == "Semantic Search":
    render_search_page()
elif page == "Datasets":
    render_datasets_page()
elif page == "Upload & Index":
    render_upload_page()
