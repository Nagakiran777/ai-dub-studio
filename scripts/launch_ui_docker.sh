#!/bin/bash
# =============================================================================
# AI Dub Studio — launch_ui_docker.sh
# Run this AFTER docker-compose up to launch the desktop UI
# =============================================================================

# Allow X11 connections from Docker
xhost +local:docker 2>/dev/null || true

docker exec -it ai-dub-studio bash -c "
    export DISPLAY=:0
    export QT_QPA_PLATFORM=xcb
    cd /app
    conda run -n dub_frontend python stages/06_frontend/main.py
"
