FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

  ENV DEBIAN_FRONTEND=noninteractive
  ENV PYTHONUNBUFFERED=1
  ENV PYTHONPATH="/app/fish-speech:${PYTHONPATH}"

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

  RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

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

  RUN git clone https://github.com/fishaudio/fish-speech.git
  /app/fish-speech

  WORKDIR /app/fish-speech

  RUN git checkout tags/1.5 2>/dev/null || git checkout main

  RUN if [ -f requirements.txt ]; then \
          pip3 install --no-cache-dir -r requirements.txt || true; \
      fi

  RUN huggingface-cli download fishaudio/fish-speech-1.5 \
      --local-dir /app/checkpoints/fish-speech-1.5 \
      --exclude "*.git*"

  ARG CACHE_BUST=2026-02-27b

  WORKDIR /app
  COPY handler.py .

  CMD ["python3", "-u", "handler.py"]
