#!/usr/bin/env python3
"""
Stage 00 — Audio Extraction + Vocal Separation
===============================================
Extracts full audio from input video and separates vocals using Demucs.

Produces two files:
  1. {job_id}.wav        — full original audio (music + sfx + vocals)
                           USED BY: Stage 5 Assembly (background audio)
  2. {job_id}_vocals.wav — clean vocals only (no music, no sfx)
                           USED BY: Stage 1 ASR, Stage 1b Diarization,
                                    Stage 2 Emotion

Why separate vocals:
  - Whisper transcribes more accurately without background music
  - NeMo diarization works correctly without music polluting embeddings
  - Emotion detection is more accurate on clean speech
  - Non-lexical sounds detected more reliably

Model: Demucs htdemucs (default) or mdx_extra (higher quality, slower)
Output: data/audio/{job_id}.wav + data/audio/{job_id}_vocals.wav
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_KEY  = "00_vocals"
STAGE_NAME = "00_vocals"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments injected by pipeline.py."""
    parser = argparse.ArgumentParser(description="Stage 00 - Vocal Extraction")
    parser.add_argument("--job_id", required=True, help="Unique job identifier")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load and return the global config.yaml as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(job_id: str, config: dict[str, Any]) -> logging.Logger:
    """Configure logger writing to stdout and logs/{job_id}_00_vocals.log."""
    logs_dir = Path(config["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{job_id}_{STAGE_NAME}.log"

    fmt     = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(STAGE_NAME)
    logger.info(f"Logging initialised -> {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(job_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Load jobs/{job_id}/manifest.json."""
    manifest_path = Path(config["paths"]["jobs"]) / job_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    """Persist manifest to disk atomically."""
    job_id        = manifest["job_id"]
    manifest_path = Path(config["paths"]["jobs"]) / job_id / "manifest.json"
    tmp_path      = manifest_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp_path.replace(manifest_path)


def mark_stage_complete(
    manifest: dict[str, Any], config: dict[str, Any], logger: logging.Logger
) -> None:
    """Set stage status -> completed with UTC timestamp."""
    manifest["stages"][STAGE_KEY]["status"]       = "completed"
    manifest["stages"][STAGE_KEY]["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest, config)
    logger.info(f"Manifest updated -> {STAGE_KEY}: completed")


def mark_stage_failed(
    manifest: dict[str, Any], config: dict[str, Any], logger: logging.Logger
) -> None:
    """Set stage status -> failed with UTC timestamp."""
    manifest["stages"][STAGE_KEY]["status"]    = "failed"
    manifest["stages"][STAGE_KEY]["failed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest, config)
    logger.error(f"Manifest updated -> {STAGE_KEY}: failed")


# ---------------------------------------------------------------------------
# Audio validation
# ---------------------------------------------------------------------------

def _is_valid_wav(wav_path: Path, logger: logging.Logger) -> bool:
    """Return True if wav_path exists, is non-empty, and ffprobe can read it."""
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return False
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.warning(f"ffprobe failed for {wav_path}: {result.stderr.strip()}")
        return False
    try:
        return float(result.stdout.strip()) > 0.0
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Step 1: Extract full audio from video
# ---------------------------------------------------------------------------

def extract_full_audio(
    job_id: str,
    config: dict[str, Any],
    manifest: dict[str, Any],
    logger: logging.Logger,
) -> Path:
    """
    Extract mono 16kHz PCM WAV from input video.
    This is the ORIGINAL full audio — music + sfx + vocals.
    Used by Stage 5 Assembly as background audio.

    Skips if valid WAV already exists.
    """
    audio_dir = Path(config["paths"]["data_audio"])
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{job_id}.wav"

    if _is_valid_wav(wav_path, logger):
        logger.info(f"Full audio already exists -> {wav_path} (skipping)")
        return wav_path

    if wav_path.exists():
        logger.warning(f"Existing WAV corrupt — re-extracting: {wav_path}")
        wav_path.unlink()

    # Get input video path from job_meta.json
    job_dir       = Path(config["paths"]["jobs"]) / job_id
    job_meta_path = job_dir / "job_meta.json"

    if job_meta_path.exists():
        with job_meta_path.open("r", encoding="utf-8") as f:
            job_meta = json.load(f)
        input_video = job_meta.get("input_video") or manifest.get("input_video")
    else:
        input_video = manifest.get("input_video")

    if not input_video:
        raise ValueError(f"Cannot determine input_video path from job_meta.json")

    logger.info(f"Extracting full audio from: {input_video}")

    # Verify audio stream exists
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type",
         "-of", "default=noprint_wrappers=1:nokey=1", input_video],
        capture_output=True, text=True
    )
    if probe.returncode != 0 or "audio" not in probe.stdout:
        raise ValueError(f"Input video has no audio stream: {input_video}")

    sample_rate = config["stages"]["vocals"].get("sample_rate", 16000)
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-acodec", "pcm_s16le", str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

    if not _is_valid_wav(wav_path, logger):
        raise RuntimeError(f"Extracted WAV invalid: {wav_path}")

    logger.info(f"Full audio extracted -> {wav_path}")
    return wav_path


# ---------------------------------------------------------------------------
# Step 2: Vocal separation using Demucs
# ---------------------------------------------------------------------------

def extract_vocals(
    job_id: str,
    full_wav: Path,
    config: dict[str, Any],
    logger: logging.Logger,
) -> Path:
    """
    Separate vocals from full audio using Demucs.
    Produces clean vocals-only WAV with no music or sfx.

    Used by: Stage 1 ASR, Stage 1b Diarization, Stage 2 Emotion.

    Skips if valid vocals WAV already exists.
    """
    audio_dir   = Path(config["paths"]["data_audio"])
    vocals_path = audio_dir / f"{job_id}_vocals.wav"

    if _is_valid_wav(vocals_path, logger):
        logger.info(f"Vocals already extracted -> {vocals_path} (skipping)")
        return vocals_path

    if vocals_path.exists():
        vocals_path.unlink()

    vocals_cfg = config["stages"]["vocals"]
    model      = vocals_cfg.get("demucs_model", "htdemucs")
    tmp_dir    = Path("/tmp/demucs_pipeline")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Separating vocals using Demucs model: {model}")
    t0 = time.time()

    cmd = [
        "python3", "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "-o", str(tmp_dir),
        str(full_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr.strip()}")

    logger.info(f"Demucs completed in {time.time()-t0:.1f}s")

    # Find the vocals output file
    import glob
    pattern  = str(tmp_dir / model / f"{full_wav.stem}" / "vocals.wav")
    matches  = glob.glob(pattern)
    if not matches:
        # Try recursive search as fallback
        matches = glob.glob(str(tmp_dir / "**" / "vocals.wav"), recursive=True)

    if not matches:
        raise RuntimeError(
            f"Demucs vocals output not found. "
            f"Expected at: {pattern}"
        )

    # Copy vocals to audio dir with correct job_id name
    import shutil
    shutil.copy2(matches[-1], str(vocals_path))

    # Cleanup temp demucs output
    try:
        shutil.rmtree(str(tmp_dir / model / full_wav.stem))
    except Exception:
        pass

    if not _is_valid_wav(vocals_path, logger):
        raise RuntimeError(f"Vocals WAV appears invalid: {vocals_path}")

    # Get duration for logging
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(vocals_path)],
        capture_output=True, text=True
    )
    try:
        duration = float(probe.stdout.strip())
        logger.info(f"Vocals extracted -> {vocals_path} ({duration:.1f}s)")
    except ValueError:
        logger.info(f"Vocals extracted -> {vocals_path}")

    return vocals_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — orchestrates Stage 00 operations."""
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logging(args.job_id, config)

    logger.info(f"=== Stage 00 Vocals started | job_id={args.job_id} ===")

    manifest = load_manifest(args.job_id, config)

    # Restart-safe check
    if manifest["stages"][STAGE_KEY]["status"] == "completed":
        logger.info("Stage 00_vocals already completed. Exiting.")
        return

    try:
        # Step 1 — Extract full audio (original mix)
        logger.info("Step 1/2 -- Extracting full audio ...")
        t0       = time.time()
        full_wav = extract_full_audio(args.job_id, config, manifest, logger)
        logger.info(f"Step 1 done in {time.time()-t0:.1f}s")

        # Step 2 — Separate vocals using Demucs
        logger.info("Step 2/2 -- Separating vocals ...")
        t0         = time.time()
        vocals_wav = extract_vocals(args.job_id, full_wav, config, logger)
        logger.info(f"Step 2 done in {time.time()-t0:.1f}s")

        logger.info(f"Full audio:  {full_wav}")
        logger.info(f"Vocals only: {vocals_wav}")

        mark_stage_complete(manifest, config, logger)
        logger.info("=== Stage 00 Vocals complete ===")

    except Exception as e:
        logger.error(f"Stage 00 FAILED: {e}", exc_info=True)
        mark_stage_failed(manifest, config, logger)
        raise


if __name__ == "__main__":
    main()