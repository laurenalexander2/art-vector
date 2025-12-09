########## Stage 1: Build layer ##########
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU Torch first
RUN pip install --no-cache-dir torch==2.2.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install rest of deps
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MiniLM
RUN python3 - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EOF


########## Stage 2: Runtime image ##########
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY backend ./backend
COPY frontend ./frontend
COPY start.sh .
RUN chmod +x start.sh

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.hf_cache

EXPOSE 8000 8501

CMD ["./start.sh"]
