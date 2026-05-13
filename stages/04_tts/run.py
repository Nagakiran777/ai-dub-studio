"""
Stage 04 — TTS Synthesis
========================
Synthesizes English speech for each dialogue using XTTS-v2 with
per-speaker voice profiles from Stage 01b.

Pure voice cloning — no pitch/emotion post-processing.
Only time-stretch is applied to fit the original dialogue time slot.

Input:
  - jobs/{job_id}/03_translations.json
  - jobs/{job_id}/speaker_profiles/speaker_XX.wav

Output:
  - jobs/{job_id}/04_tts/dialogue_XXXX.wav
  - jobs/{job_id}/04_tts/tts_manifest.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

STAGE_KEY = "04_tts"
STAGE_NAME = "04_tts"
TTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE = "en"
XTTS_OUTPUT_SR = 22050


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 04: TTS Synthesis")
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def setup_logging(job_id: str, logs_dir: str) -> logging.Logger:
    log_path = Path(logs_dir) / f"{job_id}_{STAGE_NAME}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(STAGE_NAME)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Log: {log_path}")
    return logger


# ---------------------------------------------------------------------------
# MANIFEST HELPERS
# ---------------------------------------------------------------------------

def load_manifest(job_id: str, jobs_dir: str) -> dict[str, Any]:
    path = Path(jobs_dir) / job_id / "manifest.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict[str, Any], job_id: str, jobs_dir: str) -> None:
    path = Path(jobs_dir) / job_id / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def mark_stage_complete(
    manifest: dict, job_id: str, jobs_dir: str, logger: logging.Logger
) -> None:
    manifest["stages"][STAGE_KEY]["status"] = "completed"
    manifest["stages"][STAGE_KEY]["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["stages"][STAGE_KEY]["failed_at"] = None
    save_manifest(manifest, job_id, jobs_dir)
    logger.info(f"Stage {STAGE_KEY} marked COMPLETED.")


def mark_stage_failed(
    manifest: dict, job_id: str, jobs_dir: str, logger: logging.Logger
) -> None:
    manifest["stages"][STAGE_KEY]["status"] = "failed"
    manifest["stages"][STAGE_KEY]["failed_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest, job_id, jobs_dir)
    logger.error(f"Stage {STAGE_KEY} marked FAILED.")


# ---------------------------------------------------------------------------
# GPU RELEASE
# ---------------------------------------------------------------------------

def release_gpu(logger: logging.Logger | None = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if logger:
        vram_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        logger.info(f"GPU released. VRAM now: {vram_mb:.1f} MB")


# ---------------------------------------------------------------------------
# TIME STRETCH
# ---------------------------------------------------------------------------

def apply_time_stretch(
    audio: np.ndarray,
    synthesized_duration_ms: float,
    target_duration_ms: float,
    min_ratio: float,
    max_ratio: float,
    logger: logging.Logger,
    dialogue_id: str,
) -> tuple[np.ndarray, float, bool]:
    """
    Time-stretch audio to fit within the original dialogue time slot.

    stretch_ratio = target / synthesized
    > 1.0 → synthesized is shorter than slot (slow down)
    < 1.0 → synthesized is longer than slot (speed up)

    librosa rate = 1 / stretch_ratio
    """
    import librosa

    if target_duration_ms <= 0 or synthesized_duration_ms <= 0:
        return audio, 1.0, True

    stretch_ratio = target_duration_ms / synthesized_duration_ms
    within_limit = min_ratio <= stretch_ratio <= max_ratio

    # Less than 3% difference — not worth stretching
    if abs(stretch_ratio - 1.0) < 0.03:
        logger.debug(f"[{dialogue_id}] No stretch needed (ratio={stretch_ratio:.3f})")
        return audio, stretch_ratio, True

    if not within_limit:
        logger.warning(
            f"[{dialogue_id}] Stretch ratio {stretch_ratio:.3f} outside "
            f"[{min_ratio}, {max_ratio}] — applying anyway"
        )
    else:
        logger.info(
            f"[{dialogue_id}] Stretch: {synthesized_duration_ms:.0f}ms → "
            f"{target_duration_ms:.0f}ms (ratio={stretch_ratio:.3f})"
        )

    try:
        librosa_rate = 1.0 / stretch_ratio
        stretched = librosa.effects.time_stretch(
            audio.astype(np.float32),
            rate=librosa_rate
        )
        return stretched, stretch_ratio, within_limit
    except Exception as e:
        logger.warning(f"[{dialogue_id}] Time stretch failed: {e} — using original")
        return audio, 1.0, True


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def is_silent(audio: np.ndarray, threshold: float = 1e-4) -> bool:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) < threshold


def resolve_speaker_profile(
    speaker_id: str,
    profiles_dir: Path,
    logger: logging.Logger,
    dialogue_id: str,
) -> Path:
    """
    Return speaker profile WAV path.
    Falls back to speaker_00.wav if requested speaker is missing.
    """
    primary = profiles_dir / f"{speaker_id}.wav"
    if primary.exists():
        return primary

    logger.warning(
        f"[{dialogue_id}] {primary.name} not found — falling back to speaker_00.wav"
    )
    fallback = profiles_dir / "speaker_00.wav"
    if fallback.exists():
        return fallback

    raise RuntimeError(
        f"Speaker profile missing for '{speaker_id}' AND speaker_00.wav "
        f"not found in {profiles_dir}."
    )


# ---------------------------------------------------------------------------
# CORE SYNTHESIS LOOP
# ---------------------------------------------------------------------------

def synthesize_all(
    dialogues: list[dict[str, Any]],
    tts_dir: Path,
    profiles_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    """
    Load XTTS-v2 once, synthesize all dialogues, apply time-stretch,
    then release GPU.
    """
    from TTS.api import TTS

    tts_cfg = config["stages"]["tts"]
    device = config["hardware"]["device"] if config["hardware"]["gpu_available"] else "cpu"
    min_stretch = float(tts_cfg.get("min_stretch_ratio", 0.7))
    max_stretch = float(tts_cfg.get("max_stretch_ratio", 1.5))

    # Load XTTS-v2
    logger.info(f"Loading XTTS-v2: {TTS_MODEL_ID}")
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"VRAM before load: {torch.cuda.memory_allocated()/1e6:.1f} MB")

    tts = TTS(model_name=TTS_MODEL_ID, progress_bar=True).to(device)

    if device == "cuda":
        logger.info(f"VRAM after load: {torch.cuda.memory_allocated()/1e6:.1f} MB")

    manifest_entries: list[dict[str, Any]] = []

    try:
        for dialogue in dialogues:
            d_id: str = dialogue["id"]
            speaker_id: str = dialogue.get("speaker_id", "speaker_00")
            start_ms: float = dialogue["start_ms"]
            end_ms: float = dialogue["end_ms"]
            original_duration_ms = float(end_ms - start_ms)
            emotion: str = dialogue.get("emotion", "neutral") or "neutral"
            intensity: float = float(dialogue.get("intensity", 0.5) or 0.5)
            text: str = (dialogue.get("translation") or "").strip()

            logger.info(
                f"[{d_id}] speaker={speaker_id} | emotion={emotion} | "
                f"{original_duration_ms:.0f}ms | '{text[:50]}'"
            )

            # Skip empty translation
            if not text:
                logger.error(f"[{d_id}] Empty translation — skipping")
                manifest_entries.append({
                    "id": d_id,
                    "speaker_id": speaker_id,
                    "wav_path": None,
                    "success": False,
                    "error": "empty_translation",
                    "emotion": emotion,
                    "intensity": intensity,
                    "tts_text": text,
                })
                continue

            # Resolve speaker profile
            try:
                ref_audio_path = resolve_speaker_profile(
                    speaker_id, profiles_dir, logger, d_id
                )
            except RuntimeError as e:
                logger.error(f"[{d_id}] {e}")
                manifest_entries.append({
                    "id": d_id,
                    "speaker_id": speaker_id,
                    "wav_path": None,
                    "success": False,
                    "error": "missing_speaker_profile",
                    "emotion": emotion,
                    "intensity": intensity,
                    "tts_text": text,
                })
                continue

            # XTTS synthesis
            try:
                wav_list = tts.tts(
                    text=text,
                    speaker_wav=str(ref_audio_path),
                    language=XTTS_LANGUAGE,
                )
            except Exception as e:
                logger.error(f"[{d_id}] XTTS failed: {e}")
                manifest_entries.append({
                    "id": d_id,
                    "speaker_id": speaker_id,
                    "wav_path": None,
                    "success": False,
                    "error": f"xtts_failed: {e}",
                    "emotion": emotion,
                    "intensity": intensity,
                    "reference_audio": str(ref_audio_path),
                    "tts_text": text,
                })
                continue

            audio = np.array(wav_list, dtype=np.float32)
            synthesized_duration_ms = len(audio) / XTTS_OUTPUT_SR * 1000

            # Validate not silent
            if is_silent(audio):
                logger.error(f"[{d_id}] Output is silent")
                manifest_entries.append({
                    "id": d_id,
                    "speaker_id": speaker_id,
                    "wav_path": None,
                    "success": False,
                    "error": "silent_output",
                    "emotion": emotion,
                    "intensity": intensity,
                    "reference_audio": str(ref_audio_path),
                    "tts_text": text,
                })
                continue

            logger.info(
                f"[{d_id}] Synthesized: {synthesized_duration_ms:.0f}ms | "
                f"RMS={np.sqrt(np.mean(audio**2)):.4f}"
            )

            # Time stretch to fit original slot
            audio, stretch_ratio, stretch_within_limit = apply_time_stretch(
                audio=audio,
                synthesized_duration_ms=synthesized_duration_ms,
                target_duration_ms=original_duration_ms,
                min_ratio=min_stretch,
                max_ratio=max_stretch,
                logger=logger,
                dialogue_id=d_id,
            )

            final_duration_ms = len(audio) / XTTS_OUTPUT_SR * 1000

            # Save WAV
            wav_path = tts_dir / f"{d_id}.wav"
            sf.write(str(wav_path), audio, XTTS_OUTPUT_SR, subtype="PCM_16")

            logger.info(
                f"[{d_id}] Saved: {wav_path.name} | "
                f"final={final_duration_ms:.0f}ms | "
                f"stretch={stretch_ratio:.3f}"
            )

            manifest_entries.append({
                "id": d_id,
                "speaker_id": speaker_id,
                "wav_path": str(wav_path),
                "original_duration_ms": round(original_duration_ms, 1),
                "synthesized_duration_ms": round(synthesized_duration_ms, 1),
                "target_duration_ms": round(original_duration_ms, 1),
                "final_duration_ms": round(final_duration_ms, 1),
                "stretch_ratio": round(stretch_ratio, 4),
                "stretch_within_limit": stretch_within_limit,
                "emotion": emotion,
                "intensity": round(intensity, 4),
                "reference_audio": str(ref_audio_path),
                "tts_text": text,
                "success": True,
            })

    finally:
        logger.info("Releasing XTTS-v2 from GPU...")
        del tts
        release_gpu(logger)

    return manifest_entries


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    project_root = Path(args.config).parent.resolve()
    os.chdir(project_root)

    logs_dir = config["paths"]["logs"]
    jobs_dir = config["paths"]["jobs"]

    logger = setup_logging(args.job_id, logs_dir)
    logger.info("=" * 60)
    logger.info("Stage 04 — TTS Synthesis (pure voice cloning)")
    logger.info(f"Job ID: {args.job_id}")
    logger.info("=" * 60)

    # Restart-safe check
    manifest = load_manifest(args.job_id, jobs_dir)
    if manifest["stages"][STAGE_KEY]["status"] == "completed":
        logger.info("Already completed. Skipping.")
        return

    if manifest["stages"]["03_translation"]["status"] != "completed":
        raise RuntimeError("Stage 03_translation not completed. Run Stage 03 first.")

    # Reset manifest status so we re-run cleanly
    manifest["stages"][STAGE_KEY]["status"] = "running"
    save_manifest(manifest, args.job_id, jobs_dir)

    # Load translations
    translations_path = Path(jobs_dir) / args.job_id / "03_translations.json"
    if not translations_path.exists():
        raise FileNotFoundError(f"Not found: {translations_path}")

    with open(translations_path, "r", encoding="utf-8") as f:
        translations = json.load(f)

    dialogues = translations["dialogues"]
    logger.info(f"Loaded {len(dialogues)} dialogues")

    # Prepare directories
    tts_dir = Path(jobs_dir) / args.job_id / "04_tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    profiles_dir = Path(jobs_dir) / args.job_id / "speaker_profiles"
    if not profiles_dir.exists():
        raise RuntimeError(f"Speaker profiles not found: {profiles_dir}")

    profiles_available = sorted(profiles_dir.glob("speaker_*.wav"))
    logger.info(f"Speaker profiles: {[p.name for p in profiles_available]}")

    # Run synthesis
    try:
        manifest_entries = synthesize_all(
            dialogues=dialogues,
            tts_dir=tts_dir,
            profiles_dir=profiles_dir,
            config=config,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        mark_stage_failed(manifest, args.job_id, jobs_dir, logger)
        raise
    finally:
        release_gpu(logger)

    # Write tts_manifest.json
    successful = [e for e in manifest_entries if e.get("success")]
    failed = [e for e in manifest_entries if not e.get("success")]

    tts_manifest = {
        "job_id": args.job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": XTTS_OUTPUT_SR,
        "dialogue_count": len(manifest_entries),
        "success_count": len(successful),
        "failed_count": len(failed),
        "dialogues": manifest_entries,
    }

    manifest_path = tts_dir / "tts_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(tts_manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"tts_manifest.json written: {manifest_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("Stage 04 Summary")
    logger.info(f"  Total:      {len(manifest_entries)}")
    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Failed:     {len(failed)}")
    logger.info("=" * 60)

    if failed:
        logger.warning(f"Failed IDs: {[e['id'] for e in failed]}")

    mark_stage_complete(manifest, args.job_id, jobs_dir, logger)
    logger.info("Stage 04 complete.")


if __name__ == "__main__":
    main()