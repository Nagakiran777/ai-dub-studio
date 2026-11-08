"""
Stage 01 — Single-dialogue ASR rerun
=====================================
Re-transcribes ONE dialogue's audio slice (used by the web app's per-dialogue
"Re-transcribe" button — e.g. after dragging a dialogue's boundaries, or to
fix a Whisper mistake) without re-running the whole Stage 01 batch. Uses the
same Whisper model/decode params as run.py, just scoped to one slice of
{job_id}_vocals.wav using the dialogue's CURRENT start_ms/end_ms.

Updates "text" for this dialogue in both 01_transcription.json and
03_translations.json (translations.json also carries the original text
per schema) — every other dialogue is left untouched.

Usage:
    python rerun_single.py --job_id <id> --dialogue_id <id> --config <path>
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
import jobpaths as jp          # every job keeps ALL its files inside jobs/<job_id>/

STAGE_NAME = "01_asr_rerun"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 01: single-dialogue ASR rerun")
    p.add_argument("--job_id",      required=True)
    p.add_argument("--dialogue_id", required=True)
    p.add_argument("--config",      required=True)
    p.add_argument("--model",       required=False, default=None,
                   help="Whisper model for THIS dialogue only: 'medium' or 'large-v3' "
                        "(independent of the job's model; overrides stages.asr.model)")
    return p.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(job_id: str, logs_dir: str) -> logging.Logger:
    log_path = Path(logs_dir) / f"{job_id}_{STAGE_NAME}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(STAGE_NAME)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(str(log_path), encoding="utf-8"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def release_gpu(logger: logging.Logger) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"GPU released. VRAM now: {torch.cuda.memory_allocated()/1e6:.1f} MB")


def resolve_language(source_lang_override: Optional[str], config: dict) -> Optional[str]:
    language = source_lang_override
    if language and language.lower() in ("auto", ""):
        language = None
    if not language:
        cfg_lang = config.get("language", {}).get("source", "auto")
        language = None if cfg_lang in ("auto", "") else cfg_lang
    return language


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    project_root = Path(args.config).parent.resolve()
    os.chdir(project_root)

    jobs_dir  = config["paths"]["jobs"]
    jp.migrate_legacy_job(project_root, jobs_dir, args.job_id)
    job_dir_  = jp.job_dir(project_root, jobs_dir, args.job_id)
    logs_dir  = str(jp.logs_dir(job_dir_))
    audio_dir = jp.audio_dir(job_dir_)
    logger = setup_logging(args.job_id, logs_dir)

    logger.info("=" * 60)
    logger.info(f"Stage 01 rerun — dialogue {args.dialogue_id} (job {args.job_id})")
    logger.info("=" * 60)

    trans_path = Path(jobs_dir) / args.job_id / "01_transcription.json"
    if not trans_path.exists():
        logger.error(f"Not found: {trans_path}")
        sys.exit(1)
    with open(trans_path, "r", encoding="utf-8") as f:
        transcription = json.load(f)
    dialogues_by_id = {d["id"]: d for d in transcription["dialogues"]}
    dialogue = dialogues_by_id.get(args.dialogue_id)
    if dialogue is None:
        logger.error(f"Dialogue '{args.dialogue_id}' not found in {trans_path}")
        sys.exit(1)

    start_ms, end_ms = int(dialogue["start_ms"]), int(dialogue["end_ms"])
    if end_ms <= start_ms:
        logger.error(f"[{args.dialogue_id}] Invalid range {start_ms}-{end_ms}ms")
        sys.exit(1)

    vocals_path = jp.vocals_wav(job_dir_)
    if not vocals_path.exists():
        logger.error(f"_vocals.wav not found at {vocals_path}")
        sys.exit(1)

    audio, sr = torchaudio.load(str(vocals_path))
    s0 = max(0, int(start_ms / 1000 * sr))
    s1 = min(audio.shape[1], int(end_ms / 1000 * sr))
    if s1 <= s0:
        logger.error(f"[{args.dialogue_id}] Empty slice after clamping to audio length")
        sys.exit(1)
    sl = audio[:, s0:s1]
    if sl.shape[0] > 1:
        sl = sl.mean(dim=0, keepdim=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp")
    sf.write(tmp.name, sl.squeeze(0).numpy(), sr)

    mp = Path(jobs_dir) / args.job_id / "job_meta.json"
    source_lang_override = None
    if mp.exists():
        with open(mp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        source_lang_override = meta.get("source_lang")
    language = resolve_language(source_lang_override, config)

    asr_cfg = config["stages"]["asr"]
    device  = asr_cfg.get("device", "cuda")
    # a model picked for THIS dialogue wins; otherwise config.yaml (the job's global model is only
    # used by the full ASR stage — the two choices are independent)
    model_name = (args.model or asr_cfg.get("model", "medium")).strip()
    logger.info(f"[{args.dialogue_id}] Re-transcribing with Whisper model: {model_name}"
                f"{' (chosen for this dialogue)' if args.model else ' (from config.yaml)'}")
    model_cache = Path(config["paths"]["models_cache"]); model_cache.mkdir(parents=True, exist_ok=True)

    if str(asr_cfg.get("backend", "openai-whisper")).strip().lower() in ("faster-whisper", "faster_whisper", "fasterwhisper"):
        # Same backend as the main stage, so a re-transcribe matches the full run.
        from faster_whisper import WhisperModel
        compute = asr_cfg.get("faster_whisper_compute_type", "float16" if device == "cuda" else "int8")
        logger.info(f"Loading faster-whisper '{model_name}' | device={device} | compute_type={compute} | language={language or 'auto-detect'}")
        fw_model = WhisperModel(model_name, device=device, compute_type=compute, download_root=str(model_cache))
        try:
            seg_iter, _info = fw_model.transcribe(
                tmp.name,
                language=language,
                word_timestamps=False,
                temperature=asr_cfg.get("temperature", 0.0),
                log_prob_threshold=asr_cfg.get("logprob_threshold", -1.0),
                no_speech_threshold=asr_cfg.get("no_speech_threshold", 0.6),
                condition_on_previous_text=asr_cfg.get("condition_on_previous_text", False),
                compression_ratio_threshold=asr_cfg.get("compression_ratio_threshold", 2.4),
                beam_size=int(asr_cfg.get("faster_whisper_beam_size", 1)),
                vad_filter=False,
            )
            new_text = " ".join(seg.text.strip() for seg in seg_iter).strip()
        finally:
            del fw_model
            release_gpu(logger)
            try: os.unlink(tmp.name)
            except Exception: pass
    else:
        import whisper
        logger.info(f"Loading Whisper '{model_name}' | device={device} | language={language or 'auto-detect'}")
        model = whisper.load_model(model_name, device=device, download_root=str(model_cache), in_memory=True)

        try:
            result = model.transcribe(
                tmp.name,
                language=language,
                word_timestamps=False,
                verbose=False,
                fp16=(device == "cuda"),
                temperature=asr_cfg.get("temperature", 0.0),
                logprob_threshold=asr_cfg.get("logprob_threshold", -1.0),
                no_speech_threshold=asr_cfg.get("no_speech_threshold", 0.6),
                condition_on_previous_text=asr_cfg.get("condition_on_previous_text", False),
                compression_ratio_threshold=asr_cfg.get("compression_ratio_threshold", 2.4),
            )
            new_text = " ".join(seg["text"].strip() for seg in result.get("segments", [])).strip()
            if not new_text:
                new_text = (result.get("text") or "").strip()
        finally:
            del model
            release_gpu(logger)
            try: os.unlink(tmp.name)
            except Exception: pass

    if not new_text:
        logger.error(f"[{args.dialogue_id}] Whisper returned empty text for this slice")
        sys.exit(1)

    logger.info(f"[{args.dialogue_id}] New text: {new_text!r}")

    dialogue["text"] = new_text
    dialogue["asr_model"] = model_name        # remember which model produced this line's text
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(transcription, f, indent=2, ensure_ascii=False)

    transl_path = Path(jobs_dir) / args.job_id / "03_translations.json"
    if transl_path.exists():
        with open(transl_path, "r", encoding="utf-8") as f:
            translations = json.load(f)
        for d in translations.get("dialogues", []):
            if d["id"] == args.dialogue_id:
                d["text"] = new_text
        with open(transl_path, "w", encoding="utf-8") as f:
            json.dump(translations, f, indent=2, ensure_ascii=False)

    logger.info(f"[{args.dialogue_id}] Done")


if __name__ == "__main__":
    main()
