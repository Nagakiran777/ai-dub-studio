#!/usr/bin/env bash
# DubStudio Pro — One-time frontend environment setup
# Run once from project root: bash stages/06_frontend/setup_frontend.sh

set -e

export PATH="/home/jani/miniconda3/bin:/home/jani/miniconda3/condabin:$PATH"

echo "=== DubStudio Pro — Frontend Setup ==="

# Remove old env if it exists
if conda env list | grep -q "^dub_frontend "; then
    echo "Removing existing dub_frontend env..."
    conda env remove -n dub_frontend -y
fi

echo "Creating dub_frontend conda env..."
conda env create -f stages/06_frontend/environment.yml

echo ""
echo "=== Setup complete! ==="
echo "Run DubStudio.bat from Windows, or:"
echo "  bash stages/06_frontend/launch.sh"