########## Stage 1: Build layer ##########
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only Torch **first**
RUN pip install --no-cache-dir torch==2.2.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# THEN install everything else
RUN pip install --no-cache-dir -r requirements.txt


# Pre-download small model only (MiniLM)
RUN python3 - <<EOF
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EOF

# Cleanup to reduce size
RUN rm -rf /root/.cache/pip && \
    rm -rf /root/.cache/huggingface && \
    apt-get purge -y build-essential git && \
    apt-get autoremove -y


########## Stage 2: Runtime image ##########
FROM python:3.11-slim

WORKDIR /app

# Copy python environment from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY backend ./backend
COPY frontend ./frontend
COPY requirements.txt .
COPY Dockerfile .
COPY README.md .

# Environment
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.hf_cache

EXPOSE 8000 8501

CMD ["bash"]
