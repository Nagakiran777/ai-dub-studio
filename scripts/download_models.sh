#!/bin/bash
# =============================================================================
# AI Dub Studio — download_models.sh
# Downloads all required ML models into cache directories
# Run from project root: bash scripts/download_models.sh
# One-time operation (~11GB download)
# =============================================================================

set -e

echo ""
echo "=============================================="
echo "  AI Dub Studio — Model Downloader"
echo "  This will download ~11GB of models"
echo "  Make sure you have enough disk space"
echo "=============================================="
echo ""

# Load .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set default cache paths if not in .env
HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
TORCH_HOME=${TORCH_HOME:-"$HOME/.cache/torch"}
WHISPER_CACHE=${WHISPER_CACHE:-"$HOME/.cache/whisper"}

echo "Cache locations:"
echo "  HuggingFace: $HF_HOME"
echo "  Torch:       $TORCH_HOME"
echo "  Whisper:     $WHISPER_CACHE"
echo ""

mkdir -p "$HF_HOME" "$TORCH_HOME" "$WHISPER_CACHE"

# ── Stage 00: Demucs ─────────────────────────────────────────────────────────
echo "----------------------------------------------"
echo "Downloading Demucs mdx_extra (Stage 00)..."
echo "----------------------------------------------"
conda run -n dub_vocals python -c "
import torch
from demucs.pretrained import get_model
model = get_model('mdx_extra')
print('✅ Demucs mdx_extra downloaded')
"

# ── Stage 01: Whisper ─────────────────────────────────────────────────────────
echo ""
echo "----------------------------------------------"
echo "Downloading Whisper medium (Stage 01)..."
echo "----------------------------------------------"
conda run -n dub_asr python -c "
import whisper
import os
os.environ['WHISPER_CACHE'] = '${WHISPER_CACHE}'
model = whisper.load_model('medium')
print('✅ Whisper medium downloaded')
"

# ── Stage 01b: TitaNet ────────────────────────────────────────────────────────
echo ""
echo "----------------------------------------------"
echo "Downloading TitaNet Large (Stage 01b)..."
echo "----------------------------------------------"
conda run -n dub_diar python -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained('titanet_large')
print('✅ TitaNet Large downloaded')
"

# ── Stage 02: Emotion models ──────────────────────────────────────────────────
echo ""
echo "----------------------------------------------"
echo "Downloading Emotion models (Stage 02)..."
echo "----------------------------------------------"
conda run -n dub_emotion python -c "
from transformers import pipeline
import os
os.environ['HF_HOME'] = '${HF_HOME}'
# Acoustic emotion model
p1 = pipeline('audio-classification',
    model='audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim')
print('✅ wav2vec2 emotion model downloaded')
# Text emotion model
p2 = pipeline('text-classification',
    model='j-hartmann/emotion-english-distilroberta-base')
print('✅ distilroberta emotion model downloaded')
"

# ── Stage 04: XTTS-v2 ────────────────────────────────────────────────────────
echo ""
echo "----------------------------------------------"
echo "Downloading XTTS-v2 (Stage 04) — largest download..."
echo "----------------------------------------------"
conda run -n dub_tts python -c "
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
os.environ['HF_HOME'] = '${HF_HOME}'
from TTS.api import TTS
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
print('✅ XTTS-v2 downloaded')
"

echo ""
echo "=============================================="
echo "  All models downloaded successfully!"
echo ""
echo "  Total cache sizes:"
du -sh "$HF_HOME" 2>/dev/null && \
du -sh "$TORCH_HOME" 2>/dev/null && \
du -sh "$WHISPER_CACHE" 2>/dev/null
echo ""
echo "  You can now run: bash scripts/launch.sh"
echo "=============================================="
