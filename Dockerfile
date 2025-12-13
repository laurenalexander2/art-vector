FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.hf_cache
ENV DATA_DIR=/app/data

# System dependencies for torch, pillow, opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We install torch separately to avoid pip re-solving 200 deps
RUN pip install --no-cache-dir torch==2.2.0 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MiniLM model (cached in HF_HOME)
RUN python3 - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EOF

# App directories
RUN mkdir -p /app/data /app/.hf_cache

# Copy app code
COPY backend ./backend
COPY frontend ./frontend
COPY start.sh .

RUN chmod +x start.sh

EXPOSE 8501

CMD ["./start.sh"]
