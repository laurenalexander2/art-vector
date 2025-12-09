#!/bin/bash

echo "Starting FastAPI backend on port 8000..."
uvicorn backend.app:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit frontend on port 8501..."
streamlit run frontend/ui_app.py --server.address=0.0.0.0 --server.port=8501
