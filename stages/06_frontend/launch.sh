#!/usr/bin/env bash
# DubStudio Pro — WSL2 launch script
# Usage: bash stages/06_frontend/launch.sh

export PATH="/home/jani/miniconda3/bin:/home/jani/miniconda3/condabin:$PATH"
export DISPLAY="${DISPLAY:-:0}"
export QT_QPA_PLATFORM=xcb
export QT_AUTO_SCREEN_SCALE_FACTOR=1

cd /mnt/d/\$/my/dubbing_V2

echo "Starting DubStudio Pro..."
conda run -n dub_frontend python stages/06_frontend/main.py