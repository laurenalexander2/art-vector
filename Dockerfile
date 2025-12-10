########## Stage 1: Build layer ##########
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed for building wheels (e.g., pillow, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU Torch first to avoid conflicts
RUN pip install --no-cache-dir torch==2.2.0

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MiniLM model into HF cache
RUN python3 - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EOF


########## Stage 2: Runtime image ##########
FROM python:3.11-slim

WORKDIR /app

# Runtime deps for OpenCV, pillow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Persistent storage setup
RUN mkdir -p /app/data/db \
             /app/data/uploads \
             /app/data/embeddings \
             /app/data/cache \
             /app/.hf_cache

ENV DATA_DIR=/app/data
ENV HF_HOME=/app/.hf_cache
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY backend ./backend
COPY frontend ./frontend
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000 8501

CMD ["./start.sh"]
