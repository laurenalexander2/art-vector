import os
import base64
import time
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from requests.exceptions import ReadTimeout, ConnectionError, HTTPError

# ============================================================
# Config
# ============================================================

API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="ArtVector – Semantic Search for Collections",
    layout="wide",
)

PAGES = ["Semantic Search", "Browse", "Upload & Index"]
DEFAULT_PAGE = "Semantic Search"
DEFAULT_QUERY = "modern marble sculpture"

# ============================================================
# Session defaults
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = DEFAULT_PAGE
if "query" not in st.session_state:
    st.session_state.query = DEFAULT_QUERY
if "k" not in st.session_state:
    st.session_state.k = 18
if "lazy_images" not in st.session_state:
    st.session_state.lazy_images = 12  # resolve images for first N cards
if "auto_search" not in st.session_state:
    st.session_state.auto_search = False

# ============================================================
# API helpers (timeouts + retries)
# ============================================================

API_TIMEOUTS = {
    "/search_text": 180,
    "/warmup": 120,
    "/all_objects": 120,
    "/all_datasets": 30,
}

def api_get(path: str, timeout: Optional[int] = None, retries: int = 2, backoff: float = 0.8, **params):
    url = f"{API_BASE}{path}"
    t = timeout or API_TIMEOUTS.get(path, 30)

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=t)
            r.raise_for_status()
            return r.json()
        except (ReadTimeout, ConnectionError) as e:
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise
        except HTTPError:
            raise

def api_post(path: str, files=None, data=None):
    url = f"{API_BASE}{path}"
    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=10, show_spinner=False)
def api_is_up() -> bool:
    # Use /all_datasets as a simple health probe (exists in your backend)
    try:
        _ = api_get("/all_datasets", timeout=10, retries=0)
        return True
    except Exception:
        return False

# ============================================================
# Early connectivity check (prevents stack-trace crash)
# ============================================================

if not api_is_up():
    st.sidebar.title("ArtVector")
    st.error(
        "Cannot reach backend API.\n\n"
        f"API_BASE is set to: `{API_BASE}`\n\n"
        "If you're deployed (e.g., Railway), `localhost` will NOT point to your backend.\n"
        "Set `API_BASE` on the frontend service to your backend URL and redeploy."
    )
    st.stop()

@st.cache_data(ttl=30)
def load_datasets() -> List[Dict[str, Any]]:
    return api_get("/all_datasets")

@st.cache_data(ttl=3600, show_spinner=False)
def warm_backend_once():
    # Optional; harmless if backend doesn't implement /warmup
    try:
        api_get("/warmup", timeout=120, retries=0)
    except Exception:
        pass

warm_backend_once()

@st.cache_data(ttl=60, show_spinner=False)
def cached_search(q: str, limit: int, dataset_id: Optional[str]):
    return api_get("/search_text", q=q, limit=limit, dataset_id=dataset_id, timeout=180, retries=2)

# ============================================================
# Image resolution (YOUR RULE: primaryImageSmall -> primaryImage -> restricted -> none)
# ============================================================

REQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ArtVector/1.0)",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_image_bytes(url: str) -> Optional[Tuple[bytes, str]]:
    try:
        r = requests.get(
            url,
            headers=REQ_HEADERS,
            timeout=20,
            stream=True,
            allow_redirects=True,
        )
        if r.status_code != 200:
            return None
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip() or "image/jpeg"
        content = r.content
        if len(content) < 1024:
            return None
        return content, ctype
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def met_object_endpoint(object_id: str) -> Dict[str, Any]:
    try:
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        r = requests.get(url, headers=REQ_HEADERS, timeout=20)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}

def met_restricted_iiif_url(object_id: str) -> str:
    return f"https://collectionapi.metmuseum.org/api/collection/v1/iiif/{object_id}/restricted"

def resolve_met_image_data_url(meta: Dict[str, Any]) -> Optional[str]:
    """
    Enforces:
      primaryImageSmall -> primaryImage -> forced restricted -> None
    Returns base64 data URL or None.
    """
    object_id = meta.get("Object ID") or meta.get("objectID") or meta.get("object_id")
    if not object_id:
        return None

    oid = str(object_id).strip()
    if not oid:
        return None

    j = met_object_endpoint(oid)

    # 1) primaryImageSmall
    url = j.get("primaryImageSmall")
    if url:
        fetched = fetch_image_bytes(url)
        if fetched:
            bts, ctype = fetched
            return f"data:{ctype};base64,{base64.b64encode(bts).decode('utf-8')}"

    # 2) primaryImage
    url = j.get("primaryImage")
    if url:
        fetched = fetch_image_bytes(url)
        if fetched:
            bts, ctype = fetched
            return f"data:{ctype};base64,{base64.b64encode(bts).decode('utf-8')}"

    # 3) forced restricted
    url = met_restricted_iiif_url(oid)
    fetched = fetch_image_bytes(url)
    if fetched:
        bts, ctype = fetched
        return f"data:{ctype};base64,{base64.b64encode(bts).decode('utf-8')}"

    return None

# ============================================================
# UI: Card rendering
# ============================================================

CARD_CSS = """
<style>
.av-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
  gap: 18px;
}
.av-card {
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 16px;
  overflow: hidden;
  background: white;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.av-img {
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: cover;
  display: block;
  background: #f3f3f3;
}
.av-body {
  padding: 12px 12px 14px 12px;
}
.av-title {
  font-size: 14px;
  font-weight: 650;
  line-height: 1.25;
  margin-bottom: 6px;
}
.av-meta {
  font-size: 12px;
  opacity: 0.75;
  line-height: 1.35;
  margin-bottom: 10px;
}
.av-links a {
  font-size: 12px;
  text-decoration: none;
}
.av-badge {
  position: absolute;
  top: 10px;
  left: 10px;
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(0,0,0,0.55);
  color: white;
}
.av-imgwrap { position: relative; }
</style>
"""

def render_cards(records: List[Dict[str, Any]], height: int = 1100, resolve_images_for_first_n: int = 12):
    cards_html = []
    for idx, rec in enumerate(records):
        meta = rec.get("raw_metadata") or {}

        title = meta.get("Title") or meta.get("title") or "Untitled"
        artist = meta.get("Artist Display Name") or meta.get("artist") or "Unknown artist"
        date = meta.get("Object Date") or meta.get("date") or ""
        medium = meta.get("Medium") or meta.get("medium") or ""
        place = meta.get("Culture") or meta.get("Country") or meta.get("place") or ""

        object_id = meta.get("Object ID") or meta.get("objectID") or meta.get("object_id") or ""
        dataset_id = rec.get("dataset_id") or rec.get("dataset") or ""

        met_link = meta.get("Link Resource") or meta.get("objectURL") or ""
        detail_link = f"?page=Semantic%20Search&object={object_id}&dataset={dataset_id}" if object_id else ""

        img_data_url = None
        if idx < resolve_images_for_first_n:
            img_data_url = resolve_met_image_data_url(meta)

        img_tag = f'<img class="av-img" src="{img_data_url}" />' if img_data_url else '<div class="av-img"></div>'
        badge = "image" if img_data_url else ("skipped" if idx >= resolve_images_for_first_n else "no image")

        cards_html.append(f"""
<div class="av-card">
  <div class="av-imgwrap">
    <div class="av-badge">{badge}</div>
    {img_tag}
  </div>
  <div class="av-body">
    <div class="av-title">{title}</div>
    <div class="av-meta">
      {artist}<br/>
      {date}<br/>
      {medium}<br/>
      {place}
    </div>
    <div class="av-links">
      {"<a href='" + met_link + "' target='_blank'>View source →</a><br/>" if met_link else ""}
      {"<a href='" + detail_link + "'>Open object →</a>" if detail_link else ""}
    </div>
  </div>
</div>
""")

    html = CARD_CSS + f"""
<div class="av-grid">
{''.join(cards_html)}
</div>
"""
    components.html(html, height=height, scrolling=True)

# ============================================================
# Query param helpers
# ============================================================

qp = dict(st.query_params)

def qp_get(key: str) -> Optional[str]:
    v = qp.get(key)
    if v is None:
        return None
    if isinstance(v, list) and len(v) > 0:
        return v[0]
    if isinstance(v, str):
        return v
    return None

qp_page = qp_get("page")
if qp_page in PAGES:
    st.session_state.page = qp_page

# ============================================================
# Pages
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

def render_object_detail(object_id: str, dataset_id: Optional[str]):
    st.title(f"Object {object_id}")

    # Backend doesn't expose /object, so fallback via /all_objects
    objs = api_get("/all_objects", dataset_id=dataset_id, limit=5000, timeout=120, retries=1)

    obj = None
    for o in objs:
        meta = o.get("raw_metadata") or o
        oid = meta.get("Object ID") or meta.get("object_id") or meta.get("objectID")
        if str(oid) == str(object_id):
            obj = o
            break

    if not obj:
        st.warning("Object not found.")
        return

    meta = obj.get("raw_metadata") or obj

    img = resolve_met_image_data_url(meta)
    if img:
        components.html(CARD_CSS + f'<img class="av-img" src="{img}" />', height=420)
    else:
        st.info("No retrievable image for this object (restricted or missing).")

    st.subheader("Metadata")
    st.json(meta)

def render_search_page():
    st.title("Semantic Search")

    datasets = load_datasets()
    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected = st.selectbox("Limit search to dataset", dataset_options)
    dataset_id = None if selected == "All datasets" else selected

    st.session_state.query = st.text_input(
        "Enter a meaning-based query",
        value=st.session_state.query,
        key="query_input",
    )

    images_only = st.checkbox("Only show objects with retrievable images (slower)", value=False)
    st.session_state.k = st.slider("Results to show", 6, 48, st.session_state.k)

    st.session_state.auto_search = st.checkbox("Auto-search on change", value=st.session_state.auto_search)
    st.session_state.lazy_images = st.slider("Resolve images for first N results", 0, 48, st.session_state.lazy_images)

    object_id = qp_get("object")
    qp_dataset = qp_get("dataset")
    if object_id:
        render_object_detail(object_id=object_id, dataset_id=qp_dataset or dataset_id)
        st.markdown("← Use your browser back button to return to results.")
        return

    query = st.session_state.query.strip()
    if not query:
        return

    do_search = True
    if not st.session_state.auto_search:
        do_search = st.button("Search")

    if not do_search:
        return

    with st.spinner("Searching…"):
        try:
            res = cached_search(query, st.session_state.k, dataset_id)
        except ReadTimeout:
            st.error("Search timed out. Try again, reduce results, or limit to a dataset.")
            return
        except Exception as e:
            st.error(f"Search failed: {type(e).__name__}: {e}")
            return

    if not res:
        st.warning("No results found.")
        return

    records = []
    for r in res:
        obj = r.get("obj") or {}
        meta = obj.get("raw_metadata") or {}
        if not meta:
            continue

        # expensive: triggers network calls; keep it but warn above
        if images_only and not resolve_met_image_data_url(meta):
            continue

        records.append({"dataset_id": obj.get("dataset_id"), "raw_metadata": meta})

    if not records:
        st.info("Results found, but none matched your filters (or images are restricted/unavailable).")
        return

    render_cards(records, resolve_images_for_first_n=st.session_state.lazy_images)

def render_browse_page():
    st.title("Browse (Datasets + Objects)")

    datasets = load_datasets()
    if not datasets:
        st.info("No datasets loaded yet.")
        return

    df_ds = pd.DataFrame(datasets)
    st.subheader("Datasets")
    st.dataframe(df_ds, use_container_width=True)

    st.divider()

    dataset_options = ["All datasets"] + [d["dataset_id"] for d in datasets]
    selected = st.selectbox("Filter objects by dataset", dataset_options)
    dataset_id = None if selected == "All datasets" else selected

    limit = st.slider("Max objects to load", 100, 5000, 500, step=100)

    with st.spinner("Loading objects…"):
        objs = api_get("/all_objects", dataset_id=dataset_id, limit=limit, timeout=120, retries=1)

    df = pd.DataFrame(objs)
    st.subheader("Objects")
    st.dataframe(df, use_container_width=True)

# ============================================================
# Sidebar + Router
# ============================================================

st.sidebar.title("ArtVector")
page = st.sidebar.radio(
    "Navigation",
    PAGES,
    index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0,
)
st.session_state.page = page

if page == "Semantic Search":
    render_search_page()
elif page == "Browse":
    render_browse_page()
elif page == "Upload & Index":
    render_upload_page()
