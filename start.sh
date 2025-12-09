#!/bin/bash

echo "Starting backend (FastAPI) on port 8000..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

echo "Starting frontend (Streamlit) on port 8501..."
streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
