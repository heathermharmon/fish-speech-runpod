1 -FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
      1 +FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
      2
      3  ENV DEBIAN_FRONTEND=noninteractive
      4  ENV PYTHONUNBUFFERED=1
      5 +ENV PYTHONPATH="/app/fish-speech:${PYTHONPATH}"
      6
      7  # ── System dependencies
         ───────────────────────────────────────────────────────
      8  RUN apt-get update && apt-get install -y \
     ...
      12      git \
      13      curl \
      14      wget \
      15 +    build-essential \
      16 +    cmake \
      17 +    libsndfile1 \
      18 +    libsndfile1-dev \
      19      && rm -rf /var/lib/apt/lists/*
      20
      21 -# ── Python dependencies
         - ───────────────────────────────────────────────────────
      22 -RUN pip3 install --no-cache-dir --upgrade pip
      21 +# ── Upgrade pip/setuptools
         + ────────────────────────────────────────────────────
      22 +RUN pip3 install --no-cache-dir --upgrade pip
         + setuptools wheel
      23
      24 +# ── PyTorch ────────────────────────────────────────────
         +───────────────────────
      25  RUN pip3 install --no-cache-dir \
      26      torch==2.3.0 \
      27      torchaudio==2.3.0 \
      28      --index-url https://download.pytorch.org/whl/cu121
      29
      30 +# ── Core inference dependencies
         +───────────────────────────────────────────────
      31  RUN pip3 install --no-cache-dir \
      32      runpod \
      33      numpy \
      34      soundfile \
      35      huggingface_hub \
      36      transformers \
      37 -    librosa
      37 +    librosa \
      38 +    einops \
      39 +    natsort \
      40 +    loralib \
      41 +    vector-quantize-pytorch \
      42 +    x-transformers \
      43 +    encodec \
      44 +    silero-vad \
      45 +    pyrootutils \
      46 +    cachetools \
      47 +    omegaconf \
      48 +    hydra-core
      49
      50  # ── Clone Fish Speech 1.5
          ─────────────────────────────────────────────────────
      51  RUN git clone
          https://github.com/fishaudio/fish-speech.git
          /app/fish-speech
      52
      53  WORKDIR /app/fish-speech
      54
      55 -# Pin to 1.5 release tag
      56 -RUN git checkout tags/1.5 || git checkout main
      55 +# Try 1.5 tag, fall back to main
      56 +RUN git checkout tags/1.5 2>/dev/null || git checkout main
      57
      58 -# Install Fish Speech and its dependencies
      59 -RUN pip3 install --no-cache-dir -e ".[stable]" || pip3
         -install --no-cache-dir -e .
      58 +# Install from requirements.txt if it exists (skip
         +editable install)
      59 +RUN if [ -f requirements.txt ]; then \
      60 +        pip3 install --no-cache-dir -r requirements.txt
         +|| true; \
      61 +    fi
      62
      63  # ── Download model weights at build time
          ──────────────────────────────────────
      64 -# Bakes weights into the image for fast cold starts (adds
         - ~1.5GB to image size)
      65 -RUN pip3 install --no-cache-dir huggingface_hub[cli]
      66 -
      64  RUN huggingface-cli download fishaudio/fish-speech-1.5 \
      65      --local-dir /app/checkpoints/fish-speech-1.5 \
      66      --exclude "*.git*"
      67
      68 -# Cache buster — forces rebuild of layers above when
         -bumped
      69 -ARG CACHE_BUST=2026-02-27
      68 +# Cache buster — bump to force rebuild
      69 +ARG CACHE_BUST=2026-02-27b
      70
      71  # ── Copy handler ───────────────────────────────────────
          ───────────────────────
      72  WORKDIR /app
