#!/bin/bash
# =============================================================================
# AI Dub Studio — launch.sh
# Launches the DubStudio Pro desktop UI from WSL2
# Run from project root: bash scripts/launch.sh
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env if exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set conda path
export PATH=/home/jani/miniconda3/bin:/home/jani/miniconda3/condabin:$PATH

# Display settings for WSL2
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi
export QT_QPA_PLATFORM=xcb

# Cache paths
export HF_HOME=${HF_HOME:-"/mnt/d/\$/dub_requirements/hf_cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export TORCH_HOME=${TORCH_HOME:-"/mnt/d/\$/dub_requirements/torch_cache"}
export WHISPER_CACHE=${WHISPER_CACHE:-"/mnt/d/\$/dub_requirements/whisper_cache"}

echo ""
echo "=============================================="
echo "  AI Dub Studio — Launching..."
echo "  Project: $PROJECT_ROOT"
echo "=============================================="
echo ""

conda run -n dub_frontend python stages/06_frontend/main.py
