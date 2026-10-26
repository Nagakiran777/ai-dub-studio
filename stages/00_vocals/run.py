#!/usr/bin/env python3
"""
Stage 00 — Audio Extraction + Vocal Separation
===============================================
Step 1: ffmpeg extracts full audio from video
Step 2: BS-Roformer separates vocals + background

Produces THREE files:
  1. {job_id}.wav            — original full audio (44100Hz stereo)
  2. {job_id}_vocals.wav     — BS-Roformer vocals (44100Hz stereo)
                               USED BY: Stage 01 ASR, Stage 01b Diarization,
                                        Stage 02 Emotion
  3. {job_id}_no_vocals.wav  — BS-Roformer BGM+SFX (44100Hz stereo)
                               USED BY: Stage 05 Assembly (background track)

CHANGE vs previous version:
  - Demucs Step 3 removed entirely
  - Back to BS-Roformer only (2 steps instead of 3)
  - Stage 01 ASR reads {job_id}_vocals.wav directly

Key features (unchanged):
  - Auto VRAM detection → scales chunk size, batch size, precision
  - Chunked processing with overlap + crossfade
  - Chunk temp files inside job folder — survives PC restarts
  - OOM recovery — auto-reduces settings and retries
  - Silent chunk detection — skips separation on silence
  - Short chunk padding — pads chunks < 15s, trims after
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_KEY  = "00_vocals"
STAGE_NAME = "00_vocals"

BS_ROFORMER_MODEL    = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
OUTPUT_SAMPLE_RATE   = 44100
OVERLAP_SECONDS      = 2.0
CROSSFADE_SECONDS    = 1.0
MIN_CHUNK_DURATION_S = 15.0


# ---------------------------------------------------------------------------
# VRAM Tier Definitions
# ---------------------------------------------------------------------------

VRAM_TIERS = [
    {
        "name":          "cpu",
        "min_vram_gb":   0.0,
        "max_vram_gb":   3.0,
        "device":        "cpu",
        "chunk_seconds": 30,
        "batch_size":    1,
        "use_fp16":      False,
        "overlap":       2.0,
        "description":   "CPU mode — no GPU or insufficient VRAM",
    },
    {
        "name":          "gpu_conservative",
        "min_vram_gb":   3.0,
        "max_vram_gb":   5.0,
        "device":        "cuda",
        "chunk_seconds": 30,
        "batch_size":    1,
        "use_fp16":      False,
        "overlap":       2.0,
        "description":   "GPU conservative — 3–5GB VRAM (RTX 2050)",
    },
    {
        "name":          "gpu_standard",
        "min_vram_gb":   5.0,
        "max_vram_gb":   8.0,
        "device":        "cuda",
        "chunk_seconds": 60,
        "batch_size":    2,
        "use_fp16":      True,
        "overlap":       3.0,
        "description":   "GPU standard — 5–8GB VRAM",
    },
    {
        "name":          "gpu_performance",
        "min_vram_gb":   8.0,
        "max_vram_gb":   12.0,
        "device":        "cuda",
        "chunk_seconds": 120,
        "batch_size":    4,
        "use_fp16":      True,
        "overlap":       4.0,
        "description":   "GPU performance — 8–12GB VRAM",
    },
    {
        "name":          "gpu_maximum",
        "min_vram_gb":   12.0,
        "max_vram_gb":   9999.0,
        "device":        "cuda",
        "chunk_seconds": 240,
        "batch_size":    8,
        "use_fp16":      True,
        "overlap":       4.0,
        "description":   "GPU maximum — 12GB+ VRAM",
    },
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 00 - Vocal Extraction")
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(job_id: str, config: dict[str, Any]) -> logging.Logger:
    logs_dir = Path(config["paths"]["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{job_id}_{STAGE_NAME}.log"
    fmt      = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
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
    manifest_path = Path(config["paths"]["jobs"]) / job_id / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    job_id        = manifest["job_id"]
    manifest_path = Path(config["paths"]["jobs"]) / job_id / "manifest.json"
    tmp_path      = manifest_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp_path.replace(manifest_path)


def mark_stage_complete(manifest, config, logger):
    manifest["stages"][STAGE_KEY]["status"]       = "completed"
    manifest["stages"][STAGE_KEY]["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest, config)
    logger.info(f"Manifest updated -> {STAGE_KEY}: completed")


def mark_stage_failed(manifest, config, logger):
    manifest["stages"][STAGE_KEY]["status"]    = "failed"
    manifest["stages"][STAGE_KEY]["failed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest, config)
    logger.error(f"Manifest updated -> {STAGE_KEY}: failed")


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def _progress_path(job_id: str, config: dict[str, Any]) -> Path:
    return Path(config["paths"]["jobs"]) / job_id / "00_vocals_progress.json"


def load_progress(job_id: str, config: dict[str, Any]) -> dict[str, Any]:
    path = _progress_path(job_id, config)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "job_id":           job_id,
        "total_chunks":     0,
        "completed_chunks": 0,
        "tier_name":        None,
        "chunk_seconds":    None,
        "overlap_seconds":  None,
        "chunks":           {},
    }


def save_progress(progress, job_id, config):
    path     = _progress_path(job_id, config)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def mark_chunk_done(chunk_id, progress, job_id, config):
    progress["chunks"][chunk_id]   = "done"
    progress["completed_chunks"]   = sum(1 for v in progress["chunks"].values() if v == "done")
    save_progress(progress, job_id, config)


# ---------------------------------------------------------------------------
# VRAM detection
# ---------------------------------------------------------------------------

def detect_vram_tier(logger: logging.Logger) -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("CUDA not available — using CPU tier")
            return VRAM_TIERS[0]
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb  = free_bytes  / 1e9
        total_gb = total_bytes / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {gpu_name} | VRAM: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
        selected = VRAM_TIERS[0]
        for tier in VRAM_TIERS:
            if tier["device"] == "cpu":
                continue
            if free_gb >= tier["min_vram_gb"]:
                selected = tier
        logger.info(f"Tier: {selected['name']} — {selected['description']}")
        return selected
    except Exception as e:
        logger.warning(f"VRAM detection failed ({e}) — using CPU tier")
        return VRAM_TIERS[0]


def downgrade_tier(current_tier: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    current_idx = next(
        (i for i, t in enumerate(VRAM_TIERS) if t["name"] == current_tier["name"]), 0
    )
    if current_idx == 0:
        return VRAM_TIERS[0]
    new_tier = VRAM_TIERS[current_idx - 1]
    logger.warning(f"OOM — downgrading: {current_tier['name']} → {new_tier['name']}")
    return new_tier


# ---------------------------------------------------------------------------
# Audio validation
# ---------------------------------------------------------------------------

def _is_valid_wav(wav_path: Path, logger: logging.Logger) -> bool:
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return False
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    try:
        return float(result.stdout.strip()) > 0.0
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Step 1: Extract full audio
# ---------------------------------------------------------------------------

def extract_full_audio(job_id, config, manifest, logger) -> Path:
    audio_dir = Path(config["paths"]["data_audio"])
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path  = audio_dir / f"{job_id}.wav"

    if _is_valid_wav(wav_path, logger):
        logger.info(f"Full audio already exists -> {wav_path} (skipping)")
        return wav_path

    if wav_path.exists():
        wav_path.unlink()

    job_dir       = Path(config["paths"]["jobs"]) / job_id
    job_meta_path = job_dir / "job_meta.json"
    input_video   = None

    if job_meta_path.exists():
        with job_meta_path.open("r", encoding="utf-8") as f:
            job_meta = json.load(f)
        input_video = job_meta.get("input_video") or job_meta.get("input_file")

    if not input_video:
        input_video = manifest.get("input_video") or manifest.get("input_file")

    if not input_video:
        raise ValueError("Cannot determine input_video from job_meta.json or manifest")
    if not Path(input_video).exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    logger.info(f"Input video: {input_video}")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type",
         "-of", "default=noprint_wrappers=1:nokey=1", input_video],
        capture_output=True, text=True,
    )
    if probe.returncode != 0 or "audio" not in probe.stdout:
        raise ValueError(f"Input video has no audio stream: {input_video}")

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-ac", "2", "-ar", str(OUTPUT_SAMPLE_RATE),
        "-acodec", "pcm_s16le", str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.strip()}")

    if not _is_valid_wav(wav_path, logger):
        raise RuntimeError(f"Extracted WAV invalid: {wav_path}")

    logger.info(f"Full audio extracted -> {wav_path}")
    return wav_path


# ---------------------------------------------------------------------------
# Step 2: BS-Roformer separation
# ---------------------------------------------------------------------------

def _build_chunks(total_samples, sample_rate, chunk_seconds, overlap_seconds):
    chunk_samples   = int(chunk_seconds   * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    chunks    = []
    chunk_idx = 0
    pos       = 0
    while pos < total_samples:
        read_start = max(0, pos - overlap_samples)
        read_end   = min(total_samples, pos + chunk_samples + overlap_samples)
        use_start  = pos
        use_end    = min(total_samples, pos + chunk_seconds * sample_rate)
        use_end    = min(use_end, total_samples)
        chunks.append({
            "chunk_id":   f"chunk_{chunk_idx:04d}",
            "read_start": read_start,
            "read_end":   read_end,
            "use_start":  use_start,
            "use_end":    use_end,
        })
        pos       += chunk_samples
        chunk_idx += 1
        if use_end >= total_samples:
            break
    return chunks


def _separate_one_chunk(
    chunk_audio, sample_rate, tier, model_file_dir,
    chunk_tmp_dir, chunk_id, logger,
) -> tuple[np.ndarray, np.ndarray]:
    from audio_separator.separator import Separator

    chunk_tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_input      = chunk_tmp_dir / f"{chunk_id}_in.wav"
    vocals_name    = f"{chunk_id}_vocals"
    no_vocals_name = f"{chunk_id}_novocals"
    vocals_out     = chunk_tmp_dir / f"{vocals_name}.wav"
    no_vocals_out  = chunk_tmp_dir / f"{no_vocals_name}.wav"

    for p in [tmp_input, vocals_out, no_vocals_out]:
        if p.exists():
            p.unlink()

    # Pad short chunks
    min_samples  = int(MIN_CHUNK_DURATION_S * sample_rate)
    original_len = chunk_audio.shape[0]
    was_padded   = False
    if original_len < min_samples:
        pad_len     = min_samples - original_len
        n_ch        = chunk_audio.shape[1] if chunk_audio.ndim > 1 else 1
        silence     = np.zeros((pad_len, n_ch), dtype=chunk_audio.dtype)
        chunk_audio = np.concatenate([chunk_audio, silence], axis=0)
        was_padded  = True
        logger.info(f"  Short chunk padded to {MIN_CHUNK_DURATION_S}s")

    # Silent chunk detection
    rms = float(np.sqrt(np.mean(chunk_audio.astype(np.float32) ** 2)))
    if rms < 1e-4:
        logger.info(f"  Silent chunk (RMS={rms:.2e}) — returning silence")
        silence_out = np.zeros_like(chunk_audio)
        if was_padded:
            silence_out = silence_out[:original_len]
        return silence_out, silence_out.copy()

    # Write input WAV via ffmpeg pipe
    raw_pcm = (chunk_audio * 32767).astype(np.int16).tobytes()
    n_ch    = chunk_audio.shape[1] if chunk_audio.ndim > 1 else 1
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "s16le",
        "-ar", str(sample_rate), "-ac", str(n_ch),
        "-i", "pipe:0", "-acodec", "pcm_s16le", str(tmp_input),
    ]
    result = subprocess.run(ffmpeg_cmd, input=raw_pcm, capture_output=True)
    if result.returncode != 0 or not tmp_input.exists() or tmp_input.stat().st_size == 0:
        raise RuntimeError(f"Failed to write chunk WAV: {result.stderr.decode()[:200]}")

    # Run BS-Roformer
    separator = Separator(
        log_level               = logging.INFO,
        model_file_dir          = model_file_dir,
        output_dir              = str(chunk_tmp_dir),
        output_format           = "WAV",
        normalization_threshold = 0.9,
        use_autocast            = tier["use_fp16"],
        mdxc_params             = {
            "segment_size":                1032,
            "override_model_segment_size": True,
            "overlap":                     16,
            "batch_size":                  tier["batch_size"],
        },
    )
    separator.load_model(model_filename=BS_ROFORMER_MODEL)
    output_files = separator.separate(
        str(tmp_input),
        custom_output_names={
            "Vocals":       vocals_name,
            "Instrumental": no_vocals_name,
        },
    )

    # Locate output files (4-strategy fallback)
    vocals_file    = None
    no_vocals_file = None

    if vocals_out.exists() and no_vocals_out.exists():
        vocals_file    = str(vocals_out)
        no_vocals_file = str(no_vocals_out)
    else:
        all_wavs = [f for f in chunk_tmp_dir.glob("*.wav") if f.name != tmp_input.name]
        for f in all_wavs:
            fl = f.name.lower()
            if "vocal" in fl and "instrumental" not in fl and "novocal" not in fl:
                vocals_file = str(f)
            elif "instrumental" in fl or "novocal" in fl:
                no_vocals_file = str(f)

        if (vocals_file is None or no_vocals_file is None) and output_files:
            for f in output_files:
                fl = Path(f).name.lower()
                if "vocal" in fl and "instrumental" not in fl:
                    vocals_file = f
                elif "instrumental" in fl or "novocal" in fl:
                    no_vocals_file = f

        if vocals_file is None or no_vocals_file is None:
            remaining = [f for f in chunk_tmp_dir.glob("*.wav") if f.name != tmp_input.name]
            if len(remaining) == 2:
                remaining.sort(key=lambda x: x.stat().st_size)
                vocals_file    = str(remaining[0])
                no_vocals_file = str(remaining[1])
            else:
                raise RuntimeError(
                    f"Cannot identify output files. "
                    f"Dir: {[f.name for f in chunk_tmp_dir.glob('*')]}"
                )

    vocals_audio,    _ = sf.read(str(vocals_file),    always_2d=True)
    no_vocals_audio, _ = sf.read(str(no_vocals_file), always_2d=True)

    if was_padded:
        vocals_audio    = vocals_audio[:original_len]
        no_vocals_audio = no_vocals_audio[:original_len]

    for p in [tmp_input, vocals_out, no_vocals_out]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    for f in chunk_tmp_dir.glob(f"{chunk_id}_*"):
        try:
            f.unlink()
        except Exception:
            pass

    return vocals_audio, no_vocals_audio


def _apply_crossfade_and_add(
    output_buf, weight_buf, chunk_audio,
    use_start, use_end, read_start,
    chunk_idx, total_chunks, sample_rate, crossfade_seconds,
) -> None:
    crossfade_samples = int(crossfade_seconds * sample_rate)
    use_length        = use_end - use_start
    offset_in_chunk   = use_start - read_start
    v_use = chunk_audio[offset_in_chunk : offset_in_chunk + use_length]
    if v_use.ndim == 1:
        v_use = v_use[:, np.newaxis]
    weight = np.ones(use_length, dtype=np.float32)
    if chunk_idx > 0 and crossfade_samples > 0:
        fade_len = min(crossfade_samples, use_length)
        weight[:fade_len] = np.linspace(0.0, 1.0, fade_len)
    if chunk_idx < total_chunks - 1 and crossfade_samples > 0:
        fade_len = min(crossfade_samples, use_length)
        weight[-fade_len:] = np.linspace(1.0, 0.0, fade_len)
    w          = weight[:, np.newaxis]
    actual_len = min(use_length, output_buf.shape[0] - use_start)
    output_buf[use_start : use_start + actual_len] += v_use[:actual_len] * w[:actual_len]
    weight_buf[use_start : use_start + actual_len] += weight[:actual_len]


def separate_vocals(job_id, full_wav, config, logger) -> tuple[Path, Path]:
    """BS-Roformer separation — produces vocals.wav + no_vocals.wav (44100Hz stereo)."""
    audio_dir      = Path(config["paths"]["data_audio"])
    vocals_path    = audio_dir / f"{job_id}_vocals.wav"
    no_vocals_path = audio_dir / f"{job_id}_no_vocals.wav"

    if _is_valid_wav(vocals_path, logger) and _is_valid_wav(no_vocals_path, logger):
        logger.info("Both output files already exist — skipping separation")
        return vocals_path, no_vocals_path

    model_file_dir = config.get("stages", {}).get("vocals", {}).get(
        "model_cache_dir",
        str(Path(config["paths"]["models_cache"]) / "audio_separator")
    )
    Path(model_file_dir).mkdir(parents=True, exist_ok=True)

    chunk_tmp_dir = Path(config["paths"]["jobs"]) / job_id / "00_vocals_chunks"
    chunk_tmp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Chunk temp dir: {chunk_tmp_dir}")

    tier     = detect_vram_tier(logger)
    progress = load_progress(job_id, config)

    audio, sample_rate = sf.read(str(full_wav), always_2d=True)
    total_samples      = audio.shape[0]
    n_channels         = audio.shape[1]
    duration_s         = total_samples / sample_rate
    logger.info(f"Audio: {duration_s:.1f}s | {sample_rate}Hz | {n_channels}ch")

    chunk_seconds   = tier["chunk_seconds"]
    overlap_seconds = tier["overlap"]

    if progress["chunk_seconds"] is not None:
        chunk_seconds   = progress["chunk_seconds"]
        overlap_seconds = progress["overlap_seconds"]
        logger.info(f"Resuming with saved chunk settings: {chunk_seconds}s")

    chunk_defs = _build_chunks(total_samples, sample_rate, chunk_seconds, overlap_seconds)
    n_chunks   = len(chunk_defs)
    logger.info(f"Total chunks: {n_chunks} ({chunk_seconds}s each, {overlap_seconds}s overlap)")

    progress.update({
        "total_chunks":    n_chunks,
        "tier_name":       tier["name"],
        "chunk_seconds":   chunk_seconds,
        "overlap_seconds": overlap_seconds,
    })
    save_progress(progress, job_id, config)

    vocals_buf    = np.zeros((total_samples, n_channels), dtype=np.float32)
    no_vocals_buf = np.zeros((total_samples, n_channels), dtype=np.float32)
    weight_buf    = np.zeros(total_samples,               dtype=np.float32)

    # Reload completed chunks from .npy
    for chunk_def in chunk_defs:
        cid = chunk_def["chunk_id"]
        if progress["chunks"].get(cid) != "done":
            continue
        v_npy  = chunk_tmp_dir / f"{cid}_vocals.npy"
        nv_npy = chunk_tmp_dir / f"{cid}_no_vocals.npy"
        if v_npy.exists() and nv_npy.exists():
            v_chunk  = np.load(str(v_npy))
            nv_chunk = np.load(str(nv_npy))
            idx = list(c["chunk_id"] for c in chunk_defs).index(cid)
            _apply_crossfade_and_add(
                vocals_buf, weight_buf, v_chunk,
                chunk_def["use_start"], chunk_def["use_end"],
                chunk_def["read_start"], idx, n_chunks, sample_rate, CROSSFADE_SECONDS,
            )
            _apply_crossfade_and_add(
                no_vocals_buf, weight_buf, nv_chunk,
                chunk_def["use_start"], chunk_def["use_end"],
                chunk_def["read_start"], idx, n_chunks, sample_rate, CROSSFADE_SECONDS,
            )
        else:
            progress["chunks"][cid] = "pending"

    # Process pending chunks
    for chunk_idx, chunk_def in enumerate(chunk_defs):
        cid = chunk_def["chunk_id"]
        if progress["chunks"].get(cid) == "done":
            continue

        logger.info(
            f"Chunk {chunk_idx+1}/{n_chunks} | "
            f"{chunk_def['read_start']/sample_rate:.1f}s → "
            f"{chunk_def['read_end']/sample_rate:.1f}s"
        )

        chunk_audio = audio[chunk_def["read_start"]:chunk_def["read_end"]]
        max_retries = len(VRAM_TIERS)

        for attempt in range(max_retries):
            try:
                t0 = time.time()
                vocals_chunk, no_vocals_chunk = _separate_one_chunk(
                    chunk_audio, sample_rate, tier, model_file_dir,
                    chunk_tmp_dir, cid, logger,
                )
                logger.info(f"  Done in {time.time()-t0:.1f}s (tier={tier['name']})")
                break
            except RuntimeError as e:
                if ("out of memory" in str(e).lower() or "cuda out" in str(e).lower()) \
                        and tier["device"] == "cuda":
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    tier = downgrade_tier(tier, logger)
                    continue
                raise
        else:
            raise RuntimeError(f"Failed chunk {cid} after {max_retries} attempts")

        if vocals_chunk.ndim    == 1: vocals_chunk    = vocals_chunk[:, np.newaxis]
        if no_vocals_chunk.ndim == 1: no_vocals_chunk = no_vocals_chunk[:, np.newaxis]

        np.save(str(chunk_tmp_dir / f"{cid}_vocals.npy"),    vocals_chunk)
        np.save(str(chunk_tmp_dir / f"{cid}_no_vocals.npy"), no_vocals_chunk)

        _apply_crossfade_and_add(
            vocals_buf, weight_buf, vocals_chunk,
            chunk_def["use_start"], chunk_def["use_end"],
            chunk_def["read_start"], chunk_idx, n_chunks, sample_rate, CROSSFADE_SECONDS,
        )
        _apply_crossfade_and_add(
            no_vocals_buf, weight_buf, no_vocals_chunk,
            chunk_def["use_start"], chunk_def["use_end"],
            chunk_def["read_start"], chunk_idx, n_chunks, sample_rate, CROSSFADE_SECONDS,
        )
        mark_chunk_done(cid, progress, job_id, config)
        logger.info(f"  Progress: {progress['completed_chunks']}/{n_chunks}")

    # Normalize and write
    logger.info("Normalizing and writing final output files...")
    weight_buf      = np.maximum(weight_buf, 1e-8)
    w2d             = weight_buf[:, np.newaxis]
    vocals_final    = np.clip(vocals_buf    / w2d, -1.0, 1.0)
    no_vocals_final = np.clip(no_vocals_buf / w2d, -1.0, 1.0)

    sf.write(str(vocals_path),    vocals_final,    sample_rate, subtype="PCM_16")
    sf.write(str(no_vocals_path), no_vocals_final, sample_rate, subtype="PCM_16")

    if not _is_valid_wav(vocals_path, logger):
        raise RuntimeError(f"vocals.wav invalid: {vocals_path}")
    if not _is_valid_wav(no_vocals_path, logger):
        raise RuntimeError(f"no_vocals.wav invalid: {no_vocals_path}")

    logger.info(f"Vocals    -> {vocals_path}")
    logger.info(f"No vocals -> {no_vocals_path}")

    try:
        shutil.rmtree(str(chunk_tmp_dir))
        logger.info("Chunk temp dir cleaned")
    except Exception as e:
        logger.warning(f"Could not clean chunk dir: {e}")

    return vocals_path, no_vocals_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logging(args.job_id, config)

    logger.info(f"=== Stage 00 Vocals started | job_id={args.job_id} ===")
    logger.info(f"Model: {BS_ROFORMER_MODEL}")

    manifest = load_manifest(args.job_id, config)

    if manifest["stages"][STAGE_KEY]["status"] == "completed":
        logger.info("Stage 00_vocals already completed — exiting")
        return

    try:
        # Step 1 — Extract full audio
        logger.info("Step 1/2 — Extracting full audio...")
        t0       = time.time()
        full_wav = extract_full_audio(args.job_id, config, manifest, logger)
        logger.info(f"Step 1 done in {time.time()-t0:.1f}s")

        # Step 2 — BS-Roformer separation
        logger.info("Step 2/2 — BS-Roformer vocal separation...")
        t0 = time.time()
        vocals_wav, no_vocals_wav = separate_vocals(
            args.job_id, full_wav, config, logger
        )
        logger.info(f"Step 2 done in {time.time()-t0:.1f}s")

        logger.info(f"Original audio:  {full_wav}")
        logger.info(f"Vocals only:     {vocals_wav}")
        logger.info(f"No vocals (BGM): {no_vocals_wav}")

        mark_stage_complete(manifest, config, logger)
        logger.info("=== Stage 00 Vocals complete ===")

    except Exception as e:
        logger.error(f"Stage 00 FAILED: {e}", exc_info=True)
        mark_stage_failed(manifest, config, logger)
        raise


if __name__ == "__main__":
    main()