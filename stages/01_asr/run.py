#!/usr/bin/env python3
"""
Stage 01 — ASR (Automatic Speech Recognition)
==============================================
Changes vs previous:
  - source_lang passed from job_meta.json via --source_lang arg (from server.py)
  - in_memory=True on whisper.load_model → keeps model fully in GPU RAM, no disk paging
  - large-v3 now runs entirely on GPU without CPU fallback
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
import jobpaths as jp          # every job keeps ALL its files inside jobs/<job_id>/


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_KEY  = "01_asr"
STAGE_NAME = "01_asr"
PUNCT_CHARS = set('.,!?;:\'"- \t\n\r')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 01 - ASR")
    parser.add_argument("--job_id",      required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--source_lang", required=False, default=None,
                        help="Source language code (e.g. te, hi, en) or 'auto'")
    parser.add_argument("--asr_model", required=False, default=None,
                        help="Whisper model for the whole job: 'medium' (recommended) or 'large-v3'. "
                             "Chosen in the UI; overrides stages.asr.model in config.yaml")
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
    jp.migrate_legacy_job(".", config["paths"]["jobs"], job_id)
    logs_dir = jp.logs_dir(Path(config["paths"]["jobs"]) / job_id)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{job_id}_{STAGE_NAME}.log"
    fmt      = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt  = "%Y-%m-%d %H:%M:%S"
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


def mark_stage_complete(manifest, config, logger) -> None:
    manifest["stages"][STAGE_KEY]["status"]       = "completed"
    manifest["stages"][STAGE_KEY]["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest, config)
    logger.info(f"Manifest updated -> {STAGE_KEY}: completed")


def mark_stage_failed(manifest, config, logger) -> None:
    manifest["stages"][STAGE_KEY]["status"]    = "failed"
    manifest["stages"][STAGE_KEY]["failed_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest, config)
    logger.error(f"Manifest updated -> {STAGE_KEY}: failed")


# ---------------------------------------------------------------------------
# GPU release
# ---------------------------------------------------------------------------

def release_gpu(logger: logging.Logger) -> None:
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
    tokens = config["stages"]["assembly"]["nonlexical_sounds"]
    return {t.lower().strip() for t in tokens}


def _strip_punct(text: str) -> str:
    cleaned = text.lower()
    cleaned = "".join(c for c in cleaned if c not in PUNCT_CHARS)
    return cleaned.strip()


def _is_nonlexical(text: str, nonlexical_tokens: set[str]) -> bool:
    return _strip_punct(text) in nonlexical_tokens


def _is_punct_only(text: str) -> bool:
    return len(_strip_punct(text)) == 0


# ---------------------------------------------------------------------------
# Transcription — KEY FIX: in_memory=True keeps model fully in GPU RAM
# ---------------------------------------------------------------------------

def _use_faster_whisper(asr_cfg: dict[str, Any]) -> bool:
    return str(asr_cfg.get("backend", "openai-whisper")).strip().lower() in (
        "faster-whisper", "faster_whisper", "fasterwhisper",
    )


def transcribe_audio_faster(
    wav_path: Path,
    config: dict[str, Any],
    source_lang_override: Optional[str],
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], str]:
    """EXPERIMENT — same model (large-v3), same settings, same whole-file call and
    Whisper's own 30 s windowing as transcribe_audio(); only the engine differs
    (faster-whisper / CTranslate2 instead of openai-whisper).

    Kept identical on purpose:
      * one transcribe() call over the whole _vocals.wav (no extra chunking)
      * word_timestamps=True, temperature / thresholds / condition_on_previous_text from config
      * vad_filter=False (the built-in VAD is what merged dialogues in earlier tests)
      * beam_size=1 → greedy decoding like openai-whisper's Python API default
        (faster-whisper's own default is beam 5, which would be a logic change)
    Returns segments in the same dict shape openai-whisper produces, so everything
    downstream (dedup filter, dialogue building, JSON) is untouched.
    """
    from faster_whisper import WhisperModel

    asr_cfg     = config["stages"]["asr"]
    model_cache = Path(config["paths"]["models_cache"])
    model_cache.mkdir(parents=True, exist_ok=True)

    device     = asr_cfg.get("device", "cuda")
    model_name = asr_cfg.get("model", "medium")
    compute    = asr_cfg.get("faster_whisper_compute_type", "float16" if device == "cuda" else "int8")
    beam_size  = int(asr_cfg.get("faster_whisper_beam_size", 1))

    language = source_lang_override
    if language and language.lower() in ("auto", ""):
        language = None
    if not language:
        cfg_lang = config.get("language", {}).get("source", "auto")
        language = None if cfg_lang in ("auto", "") else cfg_lang

    logger.info(
        f"[ASR] Loading faster-whisper '{model_name}' | device={device} | compute_type={compute} | "
        f"beam_size={beam_size} | language={'auto-detect' if language is None else language}"
    )
    t0 = time.time()
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute,
        download_root=str(model_cache),
    )
    logger.info(f"[ASR] Loaded in {time.time()-t0:.1f}s")

    logger.info(f"[ASR] Transcribing (faster-whisper): {wav_path.name} ...")
    t0 = time.time()
    seg_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,
        temperature=asr_cfg.get("temperature", 0.0),
        log_prob_threshold=asr_cfg.get("logprob_threshold", -1.0),
        no_speech_threshold=asr_cfg.get("no_speech_threshold", 0.6),
        condition_on_previous_text=asr_cfg.get("condition_on_previous_text", False),
        compression_ratio_threshold=asr_cfg.get("compression_ratio_threshold", 2.4),
        beam_size=beam_size,
        vad_filter=False,
    )

    segments: list[dict[str, Any]] = []
    for s in seg_iter:   # lazy generator — the real work happens while iterating
        segments.append({
            "start": s.start,
            "end":   s.end,
            "text":  s.text,
            "avg_logprob":    s.avg_logprob,
            "no_speech_prob": s.no_speech_prob,
            "words": [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in (s.words or [])
            ],
        })

    detected_language = getattr(info, "language", None) or language or "unknown"
    logger.info(
        f"[ASR] Done in {time.time()-t0:.1f}s | "
        f"{len(segments)} segments | detected: {detected_language} (faster-whisper)"
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return segments, detected_language


def transcribe_audio(
    wav_path: Path,
    config: dict[str, Any],
    source_lang_override: Optional[str],
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], str]:
    # stages.asr.backend: "openai-whisper" (default, unchanged path below) or
    # "faster-whisper" (experiment, see transcribe_audio_faster)
    if _use_faster_whisper(config["stages"]["asr"]):
        return transcribe_audio_faster(wav_path, config, source_lang_override, logger)

    import whisper

    asr_cfg     = config["stages"]["asr"]
    model_cache = Path(config["paths"]["models_cache"])
    model_cache.mkdir(parents=True, exist_ok=True)

    device     = asr_cfg.get("device", "cuda")
    model_name = asr_cfg.get("model", "medium")

    # Language resolution priority:
    # 1. --source_lang arg (from job_meta.json via server.py) — highest priority
    # 2. config.yaml language.source
    # 3. auto-detect (None)
    language = source_lang_override
    if language and language.lower() in ("auto", ""):
        language = None
    if not language:
        cfg_lang = config.get("language", {}).get("source", "auto")
        language = None if cfg_lang in ("auto", "") else cfg_lang

    logger.info(
        f"[ASR] Loading Whisper '{model_name}' | device={device} | fp16=True | "
        f"language={'auto-detect' if language is None else language}"
    )
    t0 = time.time()

    # in_memory=True → loads entire model into GPU RAM
    # No CPU fallback, no disk paging → significantly faster on RTX 2050
    model = whisper.load_model(
        model_name,
        device=device,
        download_root=str(model_cache),
        in_memory=True,
    )

    vram = torch.cuda.memory_allocated() / 1024 ** 2
    logger.info(f"[ASR] Loaded in {time.time()-t0:.1f}s | VRAM: {vram:.0f} MB")

    logger.info(f"[ASR] Transcribing: {wav_path.name} ...")
    t0 = time.time()
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

    del model
    torch.cuda.empty_cache()
    gc.collect()
    vram_after = torch.cuda.memory_allocated() / 1024 ** 2
    logger.info(f"[ASR] Whisper unloaded | VRAM: {vram_after:.1f} MB")

    return segments, detected_language


# ---------------------------------------------------------------------------
# Segment -> dialogue conversion (+ consecutive duplicate filter)
# ---------------------------------------------------------------------------

def _ms_to_timestamp(ms: int) -> str:
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
    dialogues:  list[dict[str, Any]] = []
    skipped    = 0
    dedup_skip = 0

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if _is_punct_only(text):
            skipped += 1
            continue

        # Consecutive duplicate filter — kills hallucination loops
        if len(dialogues) >= 2:
            cur   = text.strip().lower()
            prev1 = dialogues[-1]["text"].strip().lower()
            prev2 = dialogues[-2]["text"].strip().lower()
            if cur == prev1 == prev2:
                dedup_skip += 1
                continue

        start_ms = int(seg["start"] * 1000)
        end_ms   = int(seg["end"]   * 1000 + 0.999)
        if end_ms <= start_ms:
            end_ms = start_ms + 1

        nlx = _is_nonlexical(text, nonlexical_tokens)

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
                    "speaker_id": "speaker_00",
                })

        idx = len(dialogues) + 1
        dialogues.append({
            "id":            f"dialogue_{idx:04d}",
            "speaker_id":    "speaker_00",
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
    if dedup_skip:
        logger.info(f"[DEDUP] Removed {dedup_skip} consecutive duplicate segment(s)")

    return dialogues


# ---------------------------------------------------------------------------
# Build output JSON
# ---------------------------------------------------------------------------

def build_transcription_json(
    job_id: str,
    wav_path: Path,
    dialogues: list[dict[str, Any]],
    detected_language: str,
    source_lang_used: Optional[str],
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
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
        "source_language":   source_lang_used or config.get("language", {}).get("source", "auto"),
        "detected_language": detected_language,
        "asr_model":         config.get("stages", {}).get("asr", {}).get("model", "medium"),   # which Whisper model made this transcription
        "audio_path":        str(wav_path),
        "total_duration_ms": total_duration_ms,
        "dialogue_count":    len(dialogues),
        "nonlexical_count":  nonlexical_count,
        "speaker_count":     1,
        "created_at":        datetime.now(timezone.utc).isoformat(),
        "dialogues":         dialogues,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logging(args.job_id, config)

    # The model picked in the UI for this job wins over config.yaml
    if args.asr_model:
        config.setdefault("stages", {}).setdefault("asr", {})["model"] = args.asr_model.strip()

    logger.info(f"=== Stage 01 ASR started | job_id={args.job_id} ===")
    logger.info(f"[ASR] Model: {config['stages']['asr'].get('model', 'medium')}"
                f"{' (chosen in the UI)' if args.asr_model else ' (from config.yaml)'}")
    if args.source_lang:
        logger.info(f"[ASR] Source language from UI: {args.source_lang}")

    manifest = load_manifest(args.job_id, config)

    if manifest["stages"][STAGE_KEY]["status"] == "completed":
        logger.info("Stage 01_asr already completed. Exiting.")
        return

    try:
        job_dir    = Path(config["paths"]["jobs"]) / args.job_id
        vocals_wav = jp.vocals_wav(job_dir)
        full_wav   = jp.original_wav(job_dir)

        if vocals_wav.exists():
            wav_path = vocals_wav
            logger.info(f"[ASR] Audio: {wav_path.name}")
        elif full_wav.exists():
            wav_path = full_wav
            logger.warning(f"[ASR] No vocals WAV — falling back to full audio: {wav_path.name}")
        else:
            raise FileNotFoundError(f"No audio found for job {args.job_id}. Run Stage 00 first.")

        logger.info("[ASR] Transcribing ...")
        segments, detected_language = transcribe_audio(
            wav_path, config, args.source_lang, logger
        )

        if not segments:
            raise ValueError("Transcription produced zero segments.")

        logger.info("[ASR] Building dialogue units ...")
        nonlexical_tokens = _load_nonlexical_tokens(config)
        dialogues         = _segments_to_dialogues(segments, nonlexical_tokens, logger)

        if not dialogues:
            raise ValueError("Zero dialogue units produced.")

        transcription = build_transcription_json(
            args.job_id, wav_path, dialogues,
            detected_language, args.source_lang,
            config, logger,
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