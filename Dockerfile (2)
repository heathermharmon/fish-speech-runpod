FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

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
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir \
    torch==2.3.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    runpod \
    numpy \
    soundfile \
    huggingface_hub \
    transformers \
    librosa

# ── Clone Fish Speech 1.5 ─────────────────────────────────────────────────────
RUN git clone https://github.com/fishaudio/fish-speech.git /app/fish-speech

WORKDIR /app/fish-speech

# Pin to 1.5 release tag
RUN git checkout tags/1.5 || git checkout main

# Install Fish Speech and its dependencies
RUN pip3 install --no-cache-dir -e ".[stable]" || pip3 install --no-cache-dir -e .

# ── Download model weights at build time ──────────────────────────────────────
# Bakes weights into the image for fast cold starts (adds ~1.5GB to image size)
RUN pip3 install --no-cache-dir huggingface_hub[cli]

RUN huggingface-cli download fishaudio/fish-speech-1.5 \
    --local-dir /app/checkpoints/fish-speech-1.5 \
    --exclude "*.git*"

# Cache buster — forces rebuild of layers above when bumped
ARG CACHE_BUST=2026-02-27

# ── Copy handler ──────────────────────────────────────────────────────────────
WORKDIR /app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
