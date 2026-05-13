#!/bin/bash
# =============================================================================
# AI Dub Studio — setup.sh
# First-time setup: creates all 8 conda environments
# Run from project root: bash scripts/setup.sh
# =============================================================================

set -e

echo ""
echo "=============================================="
echo "  AI Dub Studio — Environment Setup"
echo "=============================================="
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Check conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Conda found: $(conda --version)"
echo ""

# Function to create environment
create_env() {
    local env_name=$1
    local yml_path=$2
    echo "----------------------------------------------"
    echo "Creating environment: $env_name"
    echo "From: $yml_path"
    echo "----------------------------------------------"
    if conda env list | grep -q "^${env_name} "; then
        echo "Environment $env_name already exists — skipping."
        echo "To recreate: conda env remove -n $env_name && bash scripts/setup.sh"
    else
        conda env create -f "$yml_path"
        echo "✅ $env_name created successfully"
    fi
    echo ""
}

# Create all 8 environments from frozen yml files
create_env "dub_vocals"    "stages/00_vocals/environment_frozen.yml"
create_env "dub_asr"       "stages/01_asr/environment_frozen.yml"
create_env "dub_diar"      "stages/01b_diarization/environment_frozen.yml"
create_env "dub_emotion"   "stages/02_emotion/environment_frozen.yml"
create_env "dub_translate" "stages/03_translation/environment_frozen.yml"
create_env "dub_tts"       "stages/04_tts/environment_frozen.yml"
create_env "dub_assembly"  "stages/05_assembly/environment_frozen.yml"
create_env "dub_frontend"  "stages/06_frontend/environment_frozen.yml"

# Create required directories
echo "Creating required directories..."
mkdir -p data/input data/audio data/outputs jobs logs models/cache
touch data/input/.gitkeep data/audio/.gitkeep \
      data/outputs/.gitkeep jobs/.gitkeep logs/.gitkeep

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo ""
echo "  Next steps:"
echo "  1. bash scripts/download_models.sh"
echo "  2. bash scripts/launch.sh"
echo "     OR double-click DubStudio.bat on Windows"
echo "=============================================="
