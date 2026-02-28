FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app/fish-speech:${PYTHONPATH}"

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

# ── PyTorch ───────────────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    torch==2.3.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Core inference dependencies ───────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    runpod \
    numpy \
    soundfile \
    huggingface_hub \
    transformers \
    librosa \
    einops \
    natsort \
    loralib \
    vector-quantize-pytorch \
    x-transformers \
    encodec \
    silero-vad \
    pyrootutils \
    cachetools \
    omegaconf \
    hydra-core

# ── Clone Fish Speech 1.5 ─────────────────────────────────────────────────────
RUN git clone https://github.com/fishaudio/fish-speech.git /app/fish-speech

WORKDIR /app/fish-speech

# Try 1.5 tag, fall back to main
RUN git checkout tags/1.5 2>/dev/null || git checkout main

# Install from requirements.txt if it exists (skip editable install)
RUN if [ -f requirements.txt ]; then \
        pip3 install --no-cache-dir -r requirements.txt || true; \
    fi

# ── Download model weights at build time ──────────────────────────────────────
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(\
    'fishaudio/fish-speech-1.5', \
    local_dir='/app/checkpoints/fish-speech-1.5', \
    ignore_patterns=['*.git*'] \
)"

# Cache buster — bump to force rebuild
ARG CACHE_BUST=2026-02-27c

# ── Copy handler ──────────────────────────────────────────────────────────────
WORKDIR /app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
