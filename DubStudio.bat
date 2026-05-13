@echo off
title AI Dub Studio
echo.
echo  ==========================================
echo    AI Dub Studio - Launching...
echo  ==========================================
echo.

wsl -d Ubuntu -e bash -c "export PATH=/home/jani/miniconda3/bin:/home/jani/miniconda3/condabin:$PATH && export DISPLAY=:0 && export QT_QPA_PLATFORM=xcb && cd /mnt/d/\$/my/dubbing_V2 && conda run -n dub_frontend python stages/06_frontend/main.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ERROR: Failed to start. See error above.
    echo.
    pause
)
