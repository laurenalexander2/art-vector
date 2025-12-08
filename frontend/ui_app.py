import os
import time
import requests
import streamlit as st

API = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(layout="wide")

st.title("ArtVector – Semantic Art Search")

tab1, tab2 = st.tabs(["Upload + Index", "Search"])

with tab1:
    file = st.file_uploader("Upload Met CSV", type=["csv"])
    if file and st.button("Upload"):
        r = requests.post(f"{API}/upload_dataset",
                          files={"file": (file.name, file.getvalue(), "text/csv")},
                          timeout=600)
        st.success(f"Loaded dataset: {r.json()['total_objects']} objects")

    if st.button("Start indexing"):
        progress = st.empty()
        while True:
            requests.post(f"{API}/process_batch", params={"limit": 128})
            status = requests.get(f"{API}/upload_dataset", timeout=30)
            time.sleep(1)

with tab2:
    query = st.text_input("Query")
    if st.button("Search"):
        r = requests.get(f"{API}/search_text", params={"q": query, "limit": 10})
        st.write(r.json())
