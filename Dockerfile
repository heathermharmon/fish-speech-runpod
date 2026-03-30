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
    build-essential \
    cmake \
    libsndfile1 \
    libsndfile1-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip ───────────────────────────────────────────────────────────────
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

# ── Install Fish Speech dependencies from pyproject.toml ─────────────────────
# Install the package in editable mode — this reads pyproject.toml and installs
# all declared dependencies automatically (inference + API server only, no training tools)
RUN pip3 install --no-cache-dir \
    "numpy<=1.26.4" \
    "transformers>=4.45.2" \
    "hydra-core>=1.3.2" \
    "lightning>=2.1.0" \
    "natsort>=8.4.0" \
    "einops>=0.7.0" \
    "librosa>=0.10.1" \
    "rich>=13.5.3" \
    "kui>=1.6.0" \
    "uvicorn>=0.30.0" \
    "loguru>=0.6.0" \
    "loralib>=0.1.2" \
    "pyrootutils>=1.0.4" \
    "vector_quantize_pytorch==1.14.24" \
    "resampy>=0.4.3" \
    "einx[torch]==0.2.2" \
    "zstandard>=0.22.0" \
    "ormsgpack" \
    "tiktoken>=0.8.0" \
    "pydantic==2.9.2" \
    "cachetools" \
    "soundfile" \
    "pydub" \
    "grpcio>=1.58.0" \
    "opencc-python-reimplemented==0.1.7" \
    "silero-vad" \
    "huggingface_hub" \
    "runpod"

# ── Set PYTHONPATH ────────────────────────────────────────────────────────────
ENV PYTHONPATH="/app/fish-speech:${PYTHONPATH}"

# ── Cache buster ──────────────────────────────────────────────────────────────
ARG CACHE_BUST=2026-03-30d

# ── Copy handler ──────────────────────────────────────────────────────────────
WORKDIR /app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
