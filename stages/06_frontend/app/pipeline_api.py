"""
DubStudio Pro — Pipeline API.
All subprocess calls for running pipeline stages.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional, Callable, List

PROJECT_ROOT = "/mnt/d/$/my/dubbing_V2"
CONDA_BIN = "/home/jani/miniconda3/bin"

# Stage definitions — fixed order
STAGE_DEFINITIONS = [
    {"id": "00_vocals",       "label": "Vocals Extraction",  "env": "dub_vocals",    "depends_on": []},
    {"id": "01_asr",          "label": "Speech Recognition", "env": "dub_asr",       "depends_on": ["00_vocals"]},
    {"id": "01b_diarization", "label": "Speaker Detection",  "env": "dub_diar",      "depends_on": ["01_asr"]},
    {"id": "02_emotion",      "label": "Emotion Analysis",   "env": "dub_emotion",   "depends_on": ["01b_diarization"]},
    {"id": "03_translation",  "label": "Translation",        "env": "dub_translate", "depends_on": ["02_emotion"]},
    {"id": "04_tts",          "label": "Voice Synthesis",    "env": "dub_tts",       "depends_on": ["03_translation"]},
    {"id": "05_assembly",     "label": "Final Assembly",     "env": "dub_assembly",  "depends_on": ["04_tts"]},
]

REVIEW_GATES = {
    "01b_diarization": "Review speakers and dialogue timestamps before continuing.",
    "03_translation":  "Review and edit translations before synthesizing voices.",
}

# Stage descriptions for UI
STAGE_DESCRIPTIONS = {
    "00_vocals":       "Separate vocals from background audio using Demucs",
    "01_asr":          "Transcribe speech with OpenAI Whisper (medium model)",
    "01b_diarization": "Identify and separate individual speakers",
    "02_emotion":      "Detect emotional tone and intensity per dialogue",
    "03_translation":  "Translate dialogues to target language",
    "04_tts":          "Synthesise dubbed voices with Coqui TTS",
    "05_assembly":     "Mix and render final dubbed video",
}


def _build_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{CONDA_BIN}:/home/jani/miniconda3/condabin:" + env.get("PATH", "")
    env["HF_HOME"]            = "/mnt/d/$/dub_requirements/hf_cache"
    env["TRANSFORMERS_CACHE"] = "/mnt/d/$/dub_requirements/hf_cache"
    env["HF_DATASETS_CACHE"]  = "/mnt/d/$/dub_requirements/hf_cache/datasets"
    env["TORCH_HOME"]         = "/mnt/d/$/dub_requirements/torch_cache"
    env["WHISPER_CACHE"]      = "/mnt/d/$/dub_requirements/whisper_cache"
    env["PIP_CACHE_DIR"]      = "/mnt/d/$/dub_requirements/pip_cache"
    return env


def run_stage(
    stage_id: str,
    job_id: str,
    env_name: str,
    stdout_cb: Optional[Callable[[str], None]] = None,
    stderr_cb: Optional[Callable[[str], None]] = None,
) -> subprocess.Popen:
    """
    Launch a pipeline stage as a non-blocking subprocess.
    Returns the Popen object so the caller can monitor / kill it.
    """
    cmd = [
        "conda", "run", "--no-capture-output",
        "-n", env_name,
        "python", f"stages/{stage_id}/run.py",
        "--job_id", job_id,
        "--config", f"{PROJECT_ROOT}/config.yaml",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=_build_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def get_stage_def(stage_id: str) -> Optional[dict]:
    for s in STAGE_DEFINITIONS:
        if s["id"] == stage_id:
            return s
    return None


def get_stage_description(stage_id: str) -> str:
    return STAGE_DESCRIPTIONS.get(stage_id, "")


def is_review_gate(stage_id: str) -> bool:
    return stage_id in REVIEW_GATES


def get_review_message(stage_id: str) -> str:
    return REVIEW_GATES.get(stage_id, "")