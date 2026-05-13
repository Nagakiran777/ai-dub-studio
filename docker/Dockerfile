# =============================================================================
# AI Dub Studio — Dockerfile
# Base: NVIDIA CUDA 12.4 + Ubuntu 22.04
# Installs: Miniconda + all 8 pipeline conda environments
# =============================================================================

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# ── Labels ───────────────────────────────────────────────────────────────────
LABEL maintainer="Nagakiran777"
LABEL description="AI Dub Studio — Production AI Dubbing System"
LABEL version="1.0.0"

# ── Prevent interactive prompts during apt installs ──────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    # Core tools
    wget curl git vim \
    # Audio/video processing
    ffmpeg libsndfile1 \
    # PyQt6 display dependencies
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Build tools
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Install Miniconda ─────────────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Copy project code ─────────────────────────────────────────────────────────
COPY . /app/

# ── Create required directories ───────────────────────────────────────────────
RUN mkdir -p \
    /app/data/input \
    /app/data/audio \
    /app/data/outputs \
    /app/jobs \
    /app/logs \
    /app/models/cache \
    /cache/hf_cache \
    /cache/torch_cache \
    /cache/whisper_cache

# ── Environment variables for cache paths ────────────────────────────────────
ENV HF_HOME=/cache/hf_cache
ENV TRANSFORMERS_CACHE=/cache/hf_cache
ENV HF_DATASETS_CACHE=/cache/hf_cache/datasets
ENV TORCH_HOME=/cache/torch_cache
ENV WHISPER_CACHE=/cache/whisper_cache
ENV PIP_CACHE_DIR=/cache/pip_cache
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

# ── Install conda environments ────────────────────────────────────────────────
# Each stage gets its own environment from the frozen yml
# This is the slowest part of the build (~30-60 minutes)

RUN echo "Installing Stage 00 — Vocals (dub_vocals)..." && \
    conda env create -f stages/00_vocals/environment_frozen.yml && \
    conda clean -afy

RUN echo "Installing Stage 01 — ASR (dub_asr)..." && \
    conda env create -f stages/01_asr/environment_frozen.yml && \
    conda clean -afy

RUN echo "Installing Stage 01b — Diarization (dub_diar)..." && \
    conda env create -f stages/01b_diarization/environment_frozen.yml && \
    conda clean -afy

RUN echo "Installing Stage 02 — Emotion (dub_emotion)..." && \
    conda env create -f stages/02_emotion/environment_frozen.yml && \
    conda clean -afy

RUN echo "Installing Stage 03 — Translation (dub_translate)..." && \
    conda env create -f stages/03_translation/environment_frozen.yml && \
    conda clean -afy

RUN echo "Installing Stage 04 — TTS (dub_tts)..." && \
    conda env create -f stages/04_tts/environment_frozen.yml && \
    conda clean -afy

RUN echo "Installing Stage 05 — Assembly (dub_assembly)..." && \
    conda env create -f stages/05_assembly/environment_frozen.yml && \
    conda clean -afy

RUN echo "Installing Stage 06 — Frontend (dub_frontend)..." && \
    conda env create -f stages/06_frontend/environment_frozen.yml && \
    conda clean -afy

# ── Make conda available in PATH for all users ────────────────────────────────
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    echo "conda activate base" >> /etc/bash.bashrc

#  Accept Anaconda Terms of Service (required since July 2025)
RUN $CONDA_DIR/bin/conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main && \
    $CONDA_DIR/bin/conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/r

# ── Set permissions ───────────────────────────────────────────────────────────
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# ── Default command ───────────────────────────────────────────────────────────
CMD ["bash"]
