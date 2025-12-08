import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="ArtVector – Semantic Museum Search", layout="wide")
st.title("ArtVector – Semantic Search over Museum Collections")

tab_upload, tab_search = st.tabs(["Upload & Index", "Search"])


def format_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # NaN or negative
        return "estimating..."
    m, s = divmod(int(seconds), 60)
    if m == 0:
        return f"~{s}s remaining"
    if m < 60:
        return f"~{m}m {s}s remaining"
    h, m = divmod(m, 60)
    return f"~{h}h {m}m remaining"


with tab_upload:
    st.subheader("Upload Met Open Access CSV (or similar)")

    # Show current backend health
    try:
        health = requests.get(f"{API_BASE}/health", timeout=10).json()
        st.caption(
            f"Backend: {health.get('status')} | "
            f"Objects: {health.get('total_objects', 0)} | "
            f"Embedded: {health.get('embedded', 0)} | "
            f"Remaining: {health.get('remaining', 0)}"
        )
    except Exception as e:
        st.error(f"Could not reach backend: {e}")
        health = None

    csv_file = st.file_uploader(
        "Drag and drop a CSV file here", type=["csv"], key="csv_uploader"
    )

    if csv_file is not None and st.button("Upload and initialize dataset", type="primary"):
        files = {"file": (csv_file.name, csv_file.getvalue(), "text/csv")}
        with st.spinner("Uploading and parsing CSV on the server..."):
            try:
                resp = requests.post(f"{API_BASE}/upload_dataset", files=files, timeout=600)
            except Exception as e:
                st.error(f"Upload failed: {e}")
                resp = None

        if resp is None:
            pass
        elif resp.status_code != 200:
            st.error("Upload failed:")
            st.code(resp.text)
        else:
            data = resp.json()
            total = data.get("total_objects", 0)
            st.success(f"Dataset loaded with {total} objects. You can now start indexing.")

    st.markdown("### Indexing status")

    status_placeholder = st.empty()
    progress_bar = st.empty()
    eta_placeholder = st.empty()

    if st.button("Start / continue indexing", type="secondary"):
        start_time = time.time()
        last_processed = 0
        last_time = start_time

        while True:
            # Ask API to process a batch
            try:
                _ = requests.post(
                    f"{API_BASE}/process_batch", params={"limit": 128}, timeout=600
                )
            except Exception as e:
                status_placeholder.error(f"Error calling process_batch: {e}")
                break

            # Get status
            try:
                status_resp = requests.get(f"{API_BASE}/job_status", timeout=60)
            except Exception as e:
                status_placeholder.error(f"Error getting job_status: {e}")
                break

            if status_resp.status_code != 200:
                status_placeholder.error("job_status failed:")
                status_placeholder.code(status_resp.text)
                break

            sdata = status_resp.json()
            status = sdata.get("status")
            processed = sdata.get("processed_count", 0)
            total = sdata.get("total_count", 0)
            remaining = sdata.get("remaining", 0)

            pct = int(processed / total * 100) if total > 0 else 0

            # ETA calculation
            now = time.time()
            elapsed = now - start_time
            if processed > 0 and total > 0:
                rate = processed / max(elapsed, 1e-6)  # objects per second
                est_total_time = total / rate
                eta_seconds = est_total_time - elapsed
            else:
                eta_seconds = -1

            status_placeholder.info(
                f"Status: {status} | Embedded: {processed} / {total} | Remaining: {remaining}"
            )
            progress_bar.progress(min(pct, 100))
            eta_placeholder.caption(f"ETA: {format_eta(eta_seconds)}")

            if status == "done" or remaining == 0:
                status_placeholder.success(
                    f"Indexing complete! Embedded {processed} / {total} objects."
                )
                eta_placeholder.caption("ETA: done")
                break

            time.sleep(1)


with tab_search:
    st.subheader("Search embedded artworks (semantic text search)")

    query = st.text_input("Query", "surrealist self-portrait")
    limit = st.slider("Max results", 5, 50, 15)

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            # Check if any embeddings exist
            try:
                health = requests.get(f"{API_BASE}/health", timeout=10).json()
            except Exception as e:
                st.error(f"Could not reach backend: {e}")
                health = None

            if not health or not health.get("has_embeddings"):
                st.warning(
                    "No embeddings found yet. "
                    "Please upload a dataset and complete (or at least start) indexing."
                )
            else:
                with st.spinner("Searching..."):
                    try:
                        resp = requests.get(
                            f"{API_BASE}/search_text",
                            params={"q": query, "limit": limit},
                            timeout=120,
                        )
                    except Exception as e:
                        st.error(f"Search request failed: {e}")
                        resp = None

                if resp is None:
                    pass
                elif resp.status_code != 200:
                    st.error("Search failed:")
                    st.code(resp.text)
                else:
                    try:
                        results = resp.json()
                    except Exception:
                        st.error("Backend returned a non-JSON response.")
                        st.code(resp.text)
                        results = []

                    st.markdown(f"### {len(results)} results")

                    if not results:
                        st.info("No results for this query (index exists, but no close matches).")
                    else:
                        for r in results:
                            cols = st.columns([1, 3])
                            with cols[0]:
                                if r.get("image_url"):
                                    st.image(r["image_url"])
                            with cols[1]:
                                title = r.get("title") or "Untitled"
                                artist = r.get("artist_display") or "Unknown artist"
                                medium = r.get("medium") or ""
                                dept = r.get("department") or ""

                                st.markdown(f"**{title}**")
                                st.markdown(artist)
                                if medium:
                                    st.markdown(medium)
                                if dept:
                                    st.caption(dept)
                                if r.get("object_url"):
                                    st.markdown(f"[View on museum site]({r['object_url']})")

                            st.markdown("---")
