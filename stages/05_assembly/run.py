"""
Stage 05 — Audio Assembly
=========================
Assembles the final dubbed MP4 by:
  - Upsampling original audio (16kHz) to working rate (22050Hz)
  - Silencing original audio completely during each dubbed speech slot
  - Applying 10ms fades at every replacement boundary (no clicks/pops)
  - RMS-normalizing all TTS WAVs globally to match original audio loudness
  - Overlaying time-stretched TTS WAVs at correct timestamps
  - Falling back to original audio for failed/missing TTS entries
  - Muxing the new audio track into the original video via ffmpeg (no re-encode)

Inputs:
  jobs/{job_id}/04_tts/tts_manifest.json   — TTS WAV paths + timestamps + success flags
  data/audio/{job_id}.wav                  — original full audio WITH BGM (16000Hz)
  data/input/{prefix}.mp4                  — original video file
  jobs/{job_id}/job_meta.json              — may contain input_video path

Output:
  data/outputs/{job_id}_dubbed.mp4

CPU-only — no torch, no VRAM usage.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKING_SR: int = 22050      # Hz — all mixing done at this rate
FADE_MS: int = 10            # ms fade in/out at every boundary
RMS_FLOOR: float = 1e-6      # avoid division by zero in RMS calculation


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    """Configure file + stderr logging for this stage."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage_05_assembly")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Config + manifest helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def update_manifest(
    manifest_path: Path,
    stage_key: str,
    status: str,
    logger: logging.Logger,
) -> None:
    """Set manifest stage status to 'completed' or 'failed' with timestamp."""
    from datetime import datetime, timezone

    if not manifest_path.exists():
        logger.warning(f"Manifest not found at {manifest_path} — skipping update.")
        return

    manifest = load_json(manifest_path)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    if stage_key not in manifest.get("stages", {}):
        manifest.setdefault("stages", {})[stage_key] = {}

    manifest["stages"][stage_key]["status"] = status
    if status == "completed":
        manifest["stages"][stage_key]["completed_at"] = now_str
        manifest["stages"][stage_key]["failed_at"] = None
    else:
        manifest["stages"][stage_key]["failed_at"] = now_str

    save_json(manifest, manifest_path)
    logger.info(f"Manifest updated: {stage_key} → {status}")


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def find_video_file(job_id: str, project_root: Path, logger: logging.Logger) -> Path:
    """
    Locate the original input video.
    Priority:
      1. jobs/{job_id}/job_meta.json  →  input_video field
      2. data/input/  →  MP4 whose stem starts with the job_id prefix
    """
    job_meta_path = project_root / "jobs" / job_id / "job_meta.json"
    if job_meta_path.exists():
        meta = load_json(job_meta_path)
        if "input_video" in meta and meta["input_video"]:
            candidate = Path(meta["input_video"])
            if not candidate.is_absolute():
                candidate = project_root / candidate
            if candidate.exists():
                logger.info(f"Video found via job_meta.json: {candidate}")
                return candidate
            else:
                logger.warning(
                    f"job_meta.json references '{meta['input_video']}' but file not found. "
                    "Falling back to data/input/ scan."
                )

    prefix = job_id.split("_")[0]
    input_dir = project_root / "data" / "input"
    candidates = sorted(input_dir.glob(f"{prefix}*.mp4"))
    if candidates:
        logger.info(f"Video found via data/input/ scan: {candidates[0]}")
        return candidates[0]

    raise RuntimeError(
        f"Original video not found for job '{job_id}'. "
        f"Searched job_meta.json and data/input/{prefix}*.mp4. "
        "Please ensure the source video is in data/input/."
    )


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def ms_to_samples(ms: float, sr: int) -> int:
    """Convert milliseconds to sample count (floor)."""
    return int(ms * sr / 1000.0)


def apply_fade_in(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """Linear fade-in over fade_samples at the start. Returns a copy."""
    result = audio.copy()
    n = min(fade_samples, len(result))
    if n > 0:
        result[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return result


def apply_fade_out(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """Linear fade-out over fade_samples at the end. Returns a copy."""
    result = audio.copy()
    n = min(fade_samples, len(result))
    if n > 0:
        result[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return result


def load_wav_at_sr(path: Path, target_sr: int, logger: logging.Logger) -> np.ndarray:
    """
    Load a WAV file, resample to target_sr if needed.
    Always returns a 1-D float32 mono array.
    """
    audio, orig_sr = librosa.load(str(path), sr=None, mono=True, dtype=np.float32)
    if orig_sr != target_sr:
        logger.debug(f"Resampling {path.name}: {orig_sr}Hz → {target_sr}Hz")
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (root mean square) energy of an audio array."""
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# RMS global gain computation
# ---------------------------------------------------------------------------

def compute_global_tts_gain(
    dialogues: list[dict],
    project_root: Path,
    sr: int,
    original_rms: float,
    logger: logging.Logger,
) -> float:
    """
    Compute a single global gain factor so all TTS clips match the original
    audio's RMS loudness.

    Steps:
      1. Load all successful TTS WAVs
      2. Concatenate into one array
      3. Compute combined RMS
      4. gain = original_rms / tts_rms

    Returns 1.0 (no change) if no valid TTS clips are found.
    """
    all_tts_samples: list[np.ndarray] = []

    for entry in dialogues:
        if not entry.get("success", False):
            continue
        wav_path_str = entry.get("wav_path")
        if not wav_path_str:
            continue
        wav_path = Path(wav_path_str)
        if not wav_path.is_absolute():
            wav_path = project_root / wav_path
        if not wav_path.exists():
            continue
        try:
            clip = load_wav_at_sr(wav_path, sr, logger)
            all_tts_samples.append(clip)
        except Exception as exc:
            logger.warning(f"Skipping {wav_path.name} during RMS scan: {exc}")

    if not all_tts_samples:
        logger.warning("No valid TTS clips found for RMS calculation — gain set to 1.0")
        return 1.0

    combined = np.concatenate(all_tts_samples)
    tts_rms = compute_rms(combined)

    if tts_rms < RMS_FLOOR:
        logger.warning(f"TTS combined RMS too low ({tts_rms:.6f}) — gain set to 1.0")
        return 1.0

    gain = original_rms / tts_rms
    logger.info(
        f"RMS matching — original: {original_rms:.6f}  tts: {tts_rms:.6f}  "
        f"gain: {gain:.4f}  ({gain:.2f}x)"
    )
    return float(gain)


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------

def assemble(
    tts_manifest: dict,
    translations: dict,
    original_audio: np.ndarray,
    project_root: Path,
    sr: int,
    fade_ms: int,
    tts_gain: float,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Build the full dubbed audio track.

    Per dialogue slot:
      success=True  → silence original in slot entirely, overlay RMS-matched TTS with fades
      success=False → keep original audio unchanged (fallback)

    Returns the assembled float32 mono array at `sr`.
    """
    fade_samples = ms_to_samples(fade_ms, sr)

    # Work on a copy — only modify dubbed slots
    dubbed = original_audio.copy()
    total_samples = len(dubbed)

    # Lookup: dialogue_id → translation entry (for start_ms / end_ms)
    trans_lookup: dict[str, dict] = {
        e["id"]: e for e in translations.get("dialogues", [])
    }

    dialogues = tts_manifest.get("dialogues", [])
    logger.info(f"Processing {len(dialogues)} dialogue entries.")

    for idx, entry in enumerate(dialogues):
        dlg_id: str = entry.get("id", f"dialogue_{idx:04d}")
        speaker_id: str = entry.get("speaker_id", "unknown")
        success: bool = bool(entry.get("success", False))
        wav_path_str: Optional[str] = entry.get("wav_path")

        # --- Resolve timestamps from 03_translations.json ---
        trans_entry = trans_lookup.get(dlg_id)
        if trans_entry is None:
            logger.warning(f"[{dlg_id}] No entry in 03_translations.json — skipping.")
            continue

        start_ms: float = float(trans_entry.get("start_ms", 0))
        end_ms: float = float(trans_entry.get("end_ms", 0))
        slot_start: int = max(0, min(ms_to_samples(start_ms, sr), total_samples))
        slot_end: int = max(slot_start, min(ms_to_samples(end_ms, sr), total_samples))
        slot_len: int = slot_end - slot_start

        if slot_len <= 0:
            logger.warning(
                f"[{dlg_id}] Zero-length slot ({start_ms}–{end_ms} ms) — skipping."
            )
            continue

        # --- Fallback: failed or missing WAV → keep original audio ---
        if not success or not wav_path_str:
            reason = "success=False" if not success else "wav_path is None"
            logger.warning(f"[{dlg_id}] Keeping original audio — reason: {reason}.")
            continue

        wav_path = Path(wav_path_str)
        if not wav_path.is_absolute():
            wav_path = project_root / wav_path

        if not wav_path.exists():
            logger.warning(
                f"[{dlg_id}] TTS WAV not found at {wav_path} — keeping original audio."
            )
            continue

        # --- Load TTS WAV ---
        try:
            tts_audio = load_wav_at_sr(wav_path, sr, logger)
        except Exception as exc:
            logger.warning(
                f"[{dlg_id}] Failed to load TTS WAV ({exc}) — keeping original audio."
            )
            continue

        tts_len = len(tts_audio)

        logger.debug(
            f"[{dlg_id}] speaker={speaker_id}  slot={start_ms:.0f}–{end_ms:.0f}ms  "
            f"slot_samples={slot_len}  tts_samples={tts_len}"
        )

        # --- 1. Apply global RMS gain to match original loudness ---
        tts_audio = tts_audio * tts_gain

        # --- 2. Silence original audio in slot with smooth 10ms transitions ---

        # Fade out original audio just before slot (full volume → 0)
        pre_fade_start = max(0, slot_start - fade_samples)
        pre_fade_len = slot_start - pre_fade_start
        if pre_fade_len > 0:
            dubbed[pre_fade_start:slot_start] *= np.linspace(
                1.0, 0.0, pre_fade_len, dtype=np.float32
            )

        # Zero out the slot itself
        dubbed[slot_start:slot_end] = 0.0

        # Fade in original audio just after slot (0 → full volume)
        post_fade_end = min(total_samples, slot_end + fade_samples)
        post_fade_len = post_fade_end - slot_end
        if post_fade_len > 0:
            dubbed[slot_end:post_fade_end] *= np.linspace(
                0.0, 1.0, post_fade_len, dtype=np.float32
            )

        # --- 3. Prepare TTS: fade in at start ---
        tts_audio = apply_fade_in(tts_audio, fade_samples)

        # --- 4. Overlay TTS — handle shorter vs longer than slot ---
        if tts_len <= slot_len:
            # TTS fits — place at slot_start, fade out at end of clip
            tts_audio = apply_fade_out(tts_audio, fade_samples)
            dubbed[slot_start : slot_start + tts_len] += tts_audio
        else:
            # TTS longer than slot — trim at slot boundary with fade-out
            logger.debug(
                f"[{dlg_id}] TTS ({tts_len} samples) longer than slot "
                f"({slot_len} samples) — trimming."
            )
            tts_trimmed = tts_audio[:slot_len]
            tts_trimmed = apply_fade_out(tts_trimmed, fade_samples)
            dubbed[slot_start:slot_end] += tts_trimmed

    # Final safety clip to [-1, 1]
    np.clip(dubbed, -1.0, 1.0, out=dubbed)

    logger.info("Assembly complete.")
    return dubbed


# ---------------------------------------------------------------------------
# ffmpeg mux
# ---------------------------------------------------------------------------

def mux_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Mux dubbed audio into original video without re-encoding video."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found. Install with:\n  sudo apt update && sudo apt install -y ffmpeg"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",       # video from original — no re-encode
        "-map", "1:a:0",       # audio from dubbed track
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]

    logger.info(f"Running ffmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"ffmpeg stderr:\n{result.stderr}")
        raise RuntimeError(
            f"ffmpeg mux failed (exit code {result.returncode}). See log for details."
        )

    logger.info(f"ffmpeg mux complete → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05 — Audio Assembly")
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    job_id: str = args.job_id
    config_path = Path(args.config)
    project_root = config_path.parent.resolve()

    log_path = project_root / "logs" / f"{job_id}_05_assembly.log"
    logger = setup_logging(log_path)
    logger.info(f"=== Stage 05 — Assembly  |  job_id={job_id} ===")
    logger.info(f"Project root: {project_root}")

    t_start = time.time()
    manifest_path = project_root / "jobs" / job_id / "manifest.json"

    try:
        # --- Config ---
        config = load_config(config_path)
        assembly_cfg = config.get("stages", {}).get("assembly", {})
        sr: int = int(assembly_cfg.get("sample_rate", WORKING_SR))
        fade_ms: int = int(assembly_cfg.get("fade_ms", FADE_MS))
        logger.info(f"Config: sr={sr}Hz  fade_ms={fade_ms}ms")

        # --- Load TTS manifest ---
        tts_manifest_path = project_root / "jobs" / job_id / "04_tts" / "tts_manifest.json"
        if not tts_manifest_path.exists():
            raise FileNotFoundError(f"TTS manifest not found: {tts_manifest_path}")
        tts_manifest = load_json(tts_manifest_path)
        logger.info(
            f"TTS manifest: {tts_manifest.get('dialogue_count','?')} dialogues, "
            f"{tts_manifest.get('success_count','?')} successful."
        )

        # --- Load translations (for start_ms / end_ms) ---
        trans_path = project_root / "jobs" / job_id / "03_translations.json"
        if not trans_path.exists():
            raise FileNotFoundError(f"03_translations.json not found: {trans_path}")
        translations = load_json(trans_path)

        # --- Load original audio ---
        orig_audio_path = project_root / "data" / "audio" / f"{job_id}.wav"
        if not orig_audio_path.exists():
            raise FileNotFoundError(f"Original audio not found: {orig_audio_path}")
        logger.info(f"Loading original audio: {orig_audio_path}")
        original_audio = load_wav_at_sr(orig_audio_path, sr, logger)
        logger.info(
            f"Original audio: {len(original_audio)} samples "
            f"({len(original_audio)/sr:.2f}s) at {sr}Hz"
        )

        # --- Compute original RMS ---
        original_rms = compute_rms(original_audio)
        logger.info(f"Original audio RMS: {original_rms:.6f}")

        # --- Compute global TTS gain ---
        tts_gain = compute_global_tts_gain(
            dialogues=tts_manifest.get("dialogues", []),
            project_root=project_root,
            sr=sr,
            original_rms=original_rms,
            logger=logger,
        )

        # --- Find original video ---
        video_path = find_video_file(job_id, project_root, logger)

        # --- Assemble ---
        logger.info("Starting audio assembly...")
        dubbed_audio = assemble(
            tts_manifest=tts_manifest,
            translations=translations,
            original_audio=original_audio,
            project_root=project_root,
            sr=sr,
            fade_ms=fade_ms,
            tts_gain=tts_gain,
            logger=logger,
        )

        del original_audio
        gc.collect()

        # --- Save temp WAV ---
        temp_audio_path = project_root / "jobs" / job_id / "05_dubbed_audio_temp.wav"
        logger.info(f"Saving dubbed audio: {temp_audio_path}")
        sf.write(str(temp_audio_path), dubbed_audio, sr, subtype="PCM_16")

        del dubbed_audio
        gc.collect()

        # --- ffmpeg mux ---
        output_dir = project_root / "data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}_dubbed.mp4"
        mux_video(video_path, temp_audio_path, output_path, logger)

        # --- Cleanup temp ---
        try:
            temp_audio_path.unlink()
            logger.debug("Temp audio removed.")
        except Exception as exc:
            logger.warning(f"Could not remove temp audio: {exc}")

        # --- Update manifest ---
        update_manifest(manifest_path, "05_assembly", "completed", logger)

        elapsed = time.time() - t_start
        logger.info(f"Stage 05 complete in {elapsed:.1f}s → {output_path}")
        print(f"\n✅ Stage 05 complete → {output_path}", flush=True)

    except Exception as exc:
        logger.exception(f"Stage 05 FAILED: {exc}")
        update_manifest(manifest_path, "05_assembly", "failed", logger)
        sys.exit(1)


if __name__ == "__main__":
    main()