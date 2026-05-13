#!/usr/bin/env python3
"""
Stage 01 — ASR (Automatic Speech Recognition)
==============================================
Transcribes vocals audio using Whisper medium with word-level timestamps.

Uses vocals WAV from Stage 00 for cleaner transcription.
Falls back to full audio if vocals WAV not found.

Speaker diarization runs separately in Stage 01b (dub_diar env).
All segments assigned speaker_00 by default here.

Output: jobs/{job_id}/01_transcription.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_KEY  = "01_asr"
STAGE_NAME = "01_asr"

# Punctuation chars for nonlexical detection
PUNCT_CHARS = set('.,!?;:\'"- \t\n\r')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments injected by pipeline.py."""
    parser = argparse.ArgumentParser(description="Stage 01 - ASR")
    parser.add_argument("--job_id",      required=True,  help="Unique job identifier")
    parser.add_argument("--config",      required=True,  help="Path to config.yaml")
    parser.add_argument("--source_lang", required=False, default=None,
                        help="Override source language (e.g. en, ja). "
                             "If not set, Whisper auto-detects.")
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
    """Configure logger writing to stdout and logs/{job_id}_01_asr.log."""
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
# GPU release
# ---------------------------------------------------------------------------

def release_gpu(logger: logging.Logger) -> None:
    """Release all GPU memory. Called in finally blocks."""
    try:
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory released.")
    except Exception as e:
        logger.warning(f"GPU release warning (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Non-lexical detection
# ---------------------------------------------------------------------------

def _load_nonlexical_tokens(config: dict[str, Any]) -> set[str]:
    """Load non-lexical token list from config as lowercase set."""
    tokens = config["stages"]["assembly"]["nonlexical_sounds"]
    return {t.lower().strip() for t in tokens}


def _strip_punct(text: str) -> str:
    """Remove punctuation, lowercase, strip whitespace."""
    cleaned = text.lower()
    cleaned = "".join(c for c in cleaned if c not in PUNCT_CHARS)
    return cleaned.strip()


def _is_nonlexical(text: str, nonlexical_tokens: set[str]) -> bool:
    """
    Return True if entire segment text matches a nonlexical token.
    "hmm" -> True. "Hmm, really?" -> False.
    """
    return _strip_punct(text) in nonlexical_tokens


def _is_punct_only(text: str) -> bool:
    """Return True if text has no actual content."""
    return len(_strip_punct(text)) == 0


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe_audio(
    wav_path: Path,
    config: dict[str, Any],
    source_lang_override: Optional[str],
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], str]:
    """
    Load Whisper medium on CUDA fp16, transcribe with word_timestamps=True,
    unload immediately.

    Returns:
        segments: list of Whisper segments with word-level timestamps
        detected_language: ISO 639-1 language code
    """
    import whisper

    asr_cfg     = config["stages"]["asr"]
    model_cache = Path(config["paths"]["models_cache"])
    model_cache.mkdir(parents=True, exist_ok=True)

    device     = asr_cfg.get("device", "cuda")
    model_name = asr_cfg.get("model", "medium")

    # Language: manual override > config > auto-detect
    language = source_lang_override
    if language and language.lower() == "auto":
        language = None
    if not language:
        cfg_lang = config.get("language", {}).get("source", "auto")
        language = None if cfg_lang == "auto" else cfg_lang

    logger.info(
        f"[ASR] Loading Whisper '{model_name}' | device={device} | fp16=True | "
        f"language={'auto-detect' if language is None else language}"
    )
    t0    = time.time()
    model = whisper.load_model(model_name, device=device, download_root=str(model_cache))
    vram  = torch.cuda.memory_allocated() / 1024 ** 2
    logger.info(f"[ASR] Loaded in {time.time()-t0:.1f}s | VRAM: {vram:.0f} MB")

    logger.info("[ASR] Transcribing with word_timestamps=True ...")
    t0     = time.time()
    result = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,
        verbose=False,
        fp16=True,
        temperature=asr_cfg.get("temperature", 0.0),
        logprob_threshold=asr_cfg.get("logprob_threshold", -1.0),
        no_speech_threshold=asr_cfg.get("no_speech_threshold", 0.6),
        condition_on_previous_text=asr_cfg.get("condition_on_previous_text", False),
        compression_ratio_threshold=asr_cfg.get("compression_ratio_threshold", 2.4),
    )

    segments          = result.get("segments", [])
    detected_language = result.get("language", language or "unknown")
    logger.info(
        f"[ASR] Done in {time.time()-t0:.1f}s | "
        f"{len(segments)} segments | detected: {detected_language}"
    )

    # Unload Whisper immediately
    del model
    torch.cuda.empty_cache()
    gc.collect()
    vram_after = torch.cuda.memory_allocated() / 1024 ** 2
    logger.info(f"[ASR] Whisper unloaded | VRAM: {vram_after:.1f} MB")

    return segments, detected_language


# ---------------------------------------------------------------------------
# Segment -> dialogue unit conversion
# ---------------------------------------------------------------------------

def _ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to human-readable HH:MM:SS.mmm format."""
    hours   = ms // 3_600_000
    minutes = (ms % 3_600_000) // 60_000
    seconds = (ms % 60_000) // 1_000
    millis  = ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _segments_to_dialogues(
    segments: list[dict[str, Any]],
    nonlexical_tokens: set[str],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Convert Whisper segments directly to dialogue units.
    Each segment = one dialogue unit.
    All assigned speaker_00 — diarization updates this in Stage 01b.
    """
    dialogues: list[dict[str, Any]] = []
    skipped = 0

    for seg in segments:
        text = seg.get("text", "").strip()

        if not text:
            continue

        # Skip punctuation-only segments (Whisper artifact)
        if _is_punct_only(text):
            skipped += 1
            continue

        start_ms = int(seg["start"] * 1000)
        end_ms   = int(seg["end"]   * 1000 + 0.999)
        if end_ms <= start_ms:
            end_ms = start_ms + 1

        nlx = _is_nonlexical(text, nonlexical_tokens)

        # Build word list
        word_list: list[dict[str, Any]] = []
        if not nlx:
            for w in seg.get("words", []):
                word_text = w.get("word", "").strip()
                if not word_text:
                    continue
                w_start = int(w.get("start", seg["start"]) * 1000)
                w_end   = int(w.get("end",   seg["end"])   * 1000 + 0.999)
                if w_end <= w_start:
                    w_end = w_start + 1
                word_list.append({
                    "word":       word_text,
                    "start_ms":   w_start,
                    "end_ms":     w_end,
                    "speaker_id": "speaker_00",  # updated by Stage 01b
                })

        idx = len(dialogues) + 1
        dialogues.append({
            "id":            f"dialogue_{idx:04d}",
            "speaker_id":    "speaker_00",  # updated by Stage 01b diarization
            "start_ms":      start_ms,
            "end_ms":        end_ms,
            "start_time":    _ms_to_timestamp(start_ms),
            "end_time":      _ms_to_timestamp(end_ms),
            "text":          text,
            "is_nonlexical": nlx,
            "words":         word_list,
        })

    if skipped:
        logger.info(f"Skipped {skipped} punctuation-only segment(s)")

    return dialogues


# ---------------------------------------------------------------------------
# Build output JSON
# ---------------------------------------------------------------------------

def build_transcription_json(
    job_id: str,
    wav_path: Path,
    dialogues: list[dict[str, Any]],
    detected_language: str,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Build the canonical 01_transcription.json envelope."""
    total_duration_ms = dialogues[-1]["end_ms"] if dialogues else 0
    dialogue_count    = sum(1 for d in dialogues if not d["is_nonlexical"])
    nonlexical_count  = sum(1 for d in dialogues if d["is_nonlexical"])

    logger.info(
        f"Transcription: {len(dialogues)} units | "
        f"{dialogue_count} dialogue | {nonlexical_count} nonlexical | "
        f"lang={detected_language} | duration={total_duration_ms}ms"
    )

    return {
        "job_id":            job_id,
        "source_language":   config.get("language", {}).get("source", "auto"),
        "detected_language": detected_language,
        "audio_path":        str(wav_path),
        "total_duration_ms": total_duration_ms,
        "dialogue_count":    len(dialogues),
        "nonlexical_count":  nonlexical_count,
        "speaker_count":     1,  # updated by Stage 01b after diarization
        "created_at":        datetime.now(timezone.utc).isoformat(),
        "dialogues":         dialogues,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — orchestrates Stage 01 ASR."""
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logging(args.job_id, config)

    logger.info(f"=== Stage 01 ASR started | job_id={args.job_id} ===")
    if args.source_lang:
        logger.info(f"Source language override: {args.source_lang}")

    manifest = load_manifest(args.job_id, config)

    if manifest["stages"][STAGE_KEY]["status"] == "completed":
        logger.info("Stage 01_asr already completed. Exiting.")
        return

    try:
        # Step 1 — Find audio to transcribe
        # Prefer vocals WAV (Stage 00 output) for cleaner transcription
        audio_dir  = Path(config["paths"]["data_audio"])
        vocals_wav = audio_dir / f"{args.job_id}_vocals.wav"
        full_wav   = audio_dir / f"{args.job_id}.wav"

        if vocals_wav.exists():
            wav_path = vocals_wav
            logger.info(f"[ASR] Step 1/3 -- Using vocals WAV: {wav_path}")
        elif full_wav.exists():
            wav_path = full_wav
            logger.warning(
                "[ASR] Step 1/3 -- Vocals WAV not found, "
                "falling back to full audio (run Stage 00 for better results)"
            )
        else:
            raise FileNotFoundError(
                f"No audio found for job {args.job_id}. "
                f"Expected: {vocals_wav} or {full_wav}. "
                f"Run Stage 00 first."
            )

        # Step 2 — Transcribe with Whisper
        logger.info("[ASR] Step 2/3 -- Transcribing ...")
        segments, detected_language = transcribe_audio(
            wav_path, config, args.source_lang, logger
        )

        if not segments:
            raise ValueError(
                "Transcription produced zero segments. "
                "Audio may be silent or non-speech."
            )

        # Step 3 — Build dialogue units
        # All assigned speaker_00 — Stage 01b updates with real speaker IDs
        logger.info("[ASR] Step 3/3 -- Building dialogue units ...")
        nonlexical_tokens = _load_nonlexical_tokens(config)
        dialogues         = _segments_to_dialogues(segments, nonlexical_tokens, logger)

        if not dialogues:
            raise ValueError("Zero dialogue units produced.")

        # Write 01_transcription.json
        transcription = build_transcription_json(
            args.job_id, wav_path, dialogues,
            detected_language, config, logger
        )

        output_dir  = Path(config["paths"]["jobs"]) / args.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "01_transcription.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)

        logger.info(f"[ASR] Written: {output_path}")

        mark_stage_complete(manifest, config, logger)
        logger.info("=== Stage 01 ASR complete ===")

    except Exception as e:
        logger.error(f"Stage 01 ASR FAILED: {e}", exc_info=True)
        mark_stage_failed(manifest, config, logger)
        raise

    finally:
        release_gpu(logger)


if __name__ == "__main__":
    main()