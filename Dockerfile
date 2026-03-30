FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip/setuptools ────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# ── PyTorch (CUDA 12.1) ───────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    torch==2.3.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Clone Fish Speech v1.5.0 ──────────────────────────────────────────────────
RUN git clone https://github.com/fishaudio/fish-speech.git /app/fish-speech
WORKDIR /app/fish-speech
RUN git checkout tags/v1.5.0

# ── Install Fish Speech dependencies ─────────────────────────────────────────
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Extra dependencies needed by api_server.py ───────────────────────────────
RUN pip3 install --no-cache-dir \
    runpod \
    uvicorn \
    ormsgpack \
    kui

# ── Download model weights at build time ──────────────────────────────────────
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(\
    'fishaudio/fish-speech-1.5', \
    local_dir='/app/checkpoints/fish-speech-1.5', \
    ignore_patterns=['*.git*', '*.gitattributes'] \
)"

# ── Set PYTHONPATH so tools/ and fish_speech/ are importable ─────────────────
ENV PYTHONPATH="/app/fish-speech:${PYTHONPATH}"

# ── Cache buster — bump to force rebuild ──────────────────────────────────────
ARG CACHE_BUST=2026-03-30a

# ── Copy handler ──────────────────────────────────────────────────────────────
WORKDIR /app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
