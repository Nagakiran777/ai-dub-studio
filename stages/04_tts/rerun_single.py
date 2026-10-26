"""
Stage 04 — Single-dialogue TTS rerun
=====================================
Re-synthesizes ONE dialogue (used by the web app's "Re-dub" button) without
re-running the whole Stage 04 batch. Shares the same ref-audio resolution
logic as run.py:
  1. manual_ref_audio override (user-picked clip from the UI ref picker)
  2. the dialogue's own slice from {job_id}_vocals.wav
  3. speaker_profiles/speaker_XX.wav fallback

Loads OmniVoice, synthesizes the one dialogue, writes its WAV, and updates
(or inserts) just that dialogue's entry in tts_manifest.json — every other
entry is left untouched.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

OMNIVOICE_MODEL_ID  = "drbaph/OmniVoice-bf16"
OMNIVOICE_OUTPUT_SR = 24000
VOCALS_SR           = 44100
STAGE_NAME          = "04_tts_rerun"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 04: single-dialogue TTS rerun")
    p.add_argument("--job_id",      required=True)
    p.add_argument("--dialogue_id", required=True)
    p.add_argument("--config",      required=True)
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


def load_omnivoice_model(device: str, logger: logging.Logger):
    from omnivoice import OmniVoice
    logger.info(f"Loading OmniVoice: {OMNIVOICE_MODEL_ID} on {device}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    return OmniVoice.from_pretrained(OMNIVOICE_MODEL_ID, device_map=device if device == "cuda" else "cpu", dtype=dtype)


def extract_dialogue_ref(vocals_audio, vocals_sr, start_ms, end_ms, min_ref_duration_s, logger, dialogue_id):
    start_sample = max(0, int(start_ms / 1000 * vocals_sr))
    end_sample   = min(int(end_ms / 1000 * vocals_sr), vocals_audio.shape[1])
    if end_sample <= start_sample:
        return None
    slice_tensor = vocals_audio[:, start_sample:end_sample]
    if slice_tensor.shape[0] > 1:
        slice_tensor = slice_tensor.mean(dim=0, keepdim=True)
    duration_s = slice_tensor.shape[1] / vocals_sr
    if duration_s < min_ref_duration_s:
        logger.debug(f"[{dialogue_id}] slice {duration_s:.2f}s < min {min_ref_duration_s:.1f}s")
        return None
    return slice_tensor.squeeze(0).numpy()


def resolve_fallback_ref(speaker_id: str, profiles_dir: Path, logger, dialogue_id: str) -> Path:
    primary = profiles_dir / f"{speaker_id}.wav"
    if primary.exists():
        return primary
    logger.warning(f"[{dialogue_id}] {primary.name} not found — falling back to speaker_00.wav")
    fallback = profiles_dir / "speaker_00.wav"
    if fallback.exists():
        return fallback
    raise RuntimeError(f"No speaker profile for '{speaker_id}' and no speaker_00.wav fallback in {profiles_dir}.")


def prepare_ref_audio(audio_np: np.ndarray, src_sr: int, logger, dialogue_id: str) -> str:
    import librosa
    if src_sr != OMNIVOICE_OUTPUT_SR:
        audio_np = librosa.resample(audio_np.astype(np.float32), orig_sr=src_sr, target_sr=OMNIVOICE_OUTPUT_SR)
    max_samples = OMNIVOICE_OUTPUT_SR * 10
    if len(audio_np) > max_samples:
        audio_np = audio_np[:max_samples]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp")
    sf.write(tmp.name, audio_np.astype(np.float32), OMNIVOICE_OUTPUT_SR)
    return tmp.name


def is_silent(audio: np.ndarray, threshold: float = 1e-4) -> bool:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) < threshold


def resolve_ref(dialogue: dict, dialogues_by_id: dict, vocals_audio, vocals_sr, profiles_dir: Path,
                 min_ref_duration: float, logger) -> tuple[np.ndarray, int, str, str | None]:
    """Same 3-tier resolution as run.py's synthesize_all: manual override →
    dialogue's own slice → speaker profile. Returns (ref_np, ref_sr, ref_source, ref_text)."""
    d_id = dialogue["id"]
    speaker_id = dialogue.get("speaker_id", "speaker_00")

    manual_ref_id = dialogue.get("manual_ref_audio")
    if manual_ref_id:
        ref_dlg = dialogues_by_id.get(manual_ref_id)
        if ref_dlg:
            ref_np = extract_dialogue_ref(vocals_audio, vocals_sr, int(ref_dlg["start_ms"]), int(ref_dlg["end_ms"]),
                                           min_ref_duration_s=0.3, logger=logger, dialogue_id=d_id)
            if ref_np is not None:
                logger.info(f"[{d_id}] Using manual ref override: {manual_ref_id}")
                return ref_np, vocals_sr, f"manual_ref:{manual_ref_id}", (ref_dlg.get("text") or "").strip() or None
            logger.warning(f"[{d_id}] manual_ref_audio '{manual_ref_id}' too short — falling back")
        else:
            logger.warning(f"[{d_id}] manual_ref_audio '{manual_ref_id}' not found — falling back")

    ref_np = extract_dialogue_ref(vocals_audio, vocals_sr, int(dialogue["start_ms"]), int(dialogue["end_ms"]),
                                   min_ref_duration_s=min_ref_duration, logger=logger, dialogue_id=d_id)
    if ref_np is not None:
        return ref_np, vocals_sr, "dialogue_slice", (dialogue.get("text") or "").strip() or None

    fallback_path = resolve_fallback_ref(speaker_id, profiles_dir, logger, d_id)
    import librosa
    ref_np, ref_sr = librosa.load(str(fallback_path), sr=None, mono=True)
    logger.info(f"[{d_id}] Using fallback ref: {fallback_path.name}")
    return ref_np, ref_sr, f"speaker_profile:{fallback_path.name}", None


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    project_root = Path(args.config).parent.resolve()
    os.chdir(project_root)

    logs_dir  = config["paths"]["logs"]
    jobs_dir  = config["paths"]["jobs"]
    audio_dir = project_root / config["paths"]["data_audio"]
    logger = setup_logging(args.job_id, logs_dir)

    logger.info("=" * 60)
    logger.info(f"Stage 04 rerun — dialogue {args.dialogue_id} (job {args.job_id})")
    logger.info("=" * 60)

    try:
        import omnivoice  # noqa: F401
    except ImportError:
        logger.error("OmniVoice not installed! Run: pip install omnivoice")
        sys.exit(1)

    translations_path = Path(jobs_dir) / args.job_id / "03_translations.json"
    if not translations_path.exists():
        logger.error(f"Not found: {translations_path}")
        sys.exit(1)
    with open(translations_path, "r", encoding="utf-8") as f:
        translations = json.load(f)
    dialogues = translations["dialogues"]
    dialogues_by_id = {d["id"]: d for d in dialogues}

    dialogue = dialogues_by_id.get(args.dialogue_id)
    if dialogue is None:
        logger.error(f"Dialogue '{args.dialogue_id}' not found in {translations_path}")
        sys.exit(1)

    text = (dialogue.get("translation") or "").strip()
    if not text:
        logger.error(f"[{args.dialogue_id}] Empty translation — nothing to synthesize")
        sys.exit(1)

    vocals_path = audio_dir / f"{args.job_id}_vocals.wav"
    if not vocals_path.exists():
        logger.error(f"_vocals.wav not found at {vocals_path}")
        sys.exit(1)
    vocals_audio, vocals_sr = torchaudio.load(str(vocals_path))

    profiles_dir = Path(jobs_dir) / args.job_id / "speaker_profiles"
    tts_dir = Path(jobs_dir) / args.job_id / "04_tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    tts_cfg = config["stages"]["tts"]
    device  = config["hardware"]["device"] if config["hardware"]["gpu_available"] else "cpu"
    min_ref_duration = float(tts_cfg.get("min_ref_duration_s", 1.0))

    speaker_id = dialogue.get("speaker_id", "speaker_00")
    entry: dict[str, Any]
    model = load_omnivoice_model(device, logger)
    try:
        try:
            ref_np, ref_sr, ref_source, ref_text = resolve_ref(
                dialogue, dialogues_by_id, vocals_audio, vocals_sr, profiles_dir, min_ref_duration, logger
            )
        except RuntimeError as e:
            logger.error(f"[{args.dialogue_id}] No ref audio available: {e}")
            entry = {"id": args.dialogue_id, "speaker_id": speaker_id, "wav_path": None,
                     "success": False, "error": "no_ref_audio", "tts_text": text}
        else:
            prepared_ref = prepare_ref_audio(ref_np, ref_sr, logger, args.dialogue_id)
            try:
                audio_list = model.generate(text=text, ref_audio=prepared_ref, ref_text=ref_text)
                audio = np.array(audio_list[0], dtype=np.float32)
                if is_silent(audio):
                    raise RuntimeError("silent_output")
                synthesized_duration_ms = len(audio) / OMNIVOICE_OUTPUT_SR * 1000
                wav_path = tts_dir / f"{args.dialogue_id}.wav"
                sf.write(str(wav_path), audio, OMNIVOICE_OUTPUT_SR, subtype="PCM_16")
                logger.info(f"[{args.dialogue_id}] Re-synthesized: {synthesized_duration_ms:.0f}ms | ref={ref_source}")
                entry = {
                    "id": args.dialogue_id, "speaker_id": speaker_id, "wav_path": str(wav_path),
                    "original_duration_ms": float(dialogue["end_ms"] - dialogue["start_ms"]),
                    "synthesized_duration_ms": round(synthesized_duration_ms, 1),
                    "ref_source": ref_source, "tts_text": text, "success": True,
                }
            except Exception as e:
                logger.error(f"[{args.dialogue_id}] OmniVoice synthesis failed: {e}")
                entry = {"id": args.dialogue_id, "speaker_id": speaker_id, "wav_path": None,
                          "success": False, "error": str(e), "tts_text": text, "ref_source": ref_source}
            finally:
                try: os.unlink(prepared_ref)
                except Exception: pass
    finally:
        del model
        release_gpu(logger)

    # Update just this one entry in tts_manifest.json — leave everything else alone
    manifest_path = tts_dir / "tts_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            tts_manifest = json.load(f)
    else:
        tts_manifest = {"job_id": args.job_id, "model": OMNIVOICE_MODEL_ID,
                         "sample_rate": OMNIVOICE_OUTPUT_SR, "dialogues": []}

    found = False
    for i, e in enumerate(tts_manifest.get("dialogues", [])):
        if e.get("id") == args.dialogue_id:
            tts_manifest["dialogues"][i] = entry
            found = True
            break
    if not found:
        tts_manifest.setdefault("dialogues", []).append(entry)

    tts_manifest["dialogue_count"] = len(tts_manifest["dialogues"])
    tts_manifest["success_count"]  = sum(1 for e in tts_manifest["dialogues"] if e.get("success"))
    tts_manifest["failed_count"]   = tts_manifest["dialogue_count"] - tts_manifest["success_count"]
    tts_manifest["updated_at"]     = datetime.now(timezone.utc).isoformat()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(tts_manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"tts_manifest.json updated for {args.dialogue_id}")

    if not entry.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
