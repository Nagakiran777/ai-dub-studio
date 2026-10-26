"""
Stage 05 — Audio Assembly
=========================
3-track mix:

  Track 1: {job_id}_no_vocals.wav  — BGM + SFX, plays fully untouched
  Track 2: {job_id}_vocals.wav     — original voices + non-verbal sounds
                                     silenced ONLY at slots with successful TTS
                                     screams/grunts/laughs in non-dialogue regions kept
  Track 3: TTS dubbed WAVs         — placed at original start_ms

Rules per dialogue:
  WITH successful TTS:
    → silence vocals from start_ms to max(original_end_ms, tts_end_ms)
    → overlay TTS audio at start_ms
  WITHOUT TTS (failed/skipped):
    → vocals track untouched (original voice stays in mix)
    → no TTS placed

Inputs:
  jobs/{job_id}/04_tts/tts_manifest.json
  jobs/{job_id}/03_translations.json
  data/audio/{job_id}_no_vocals.wav
  data/audio/{job_id}_vocals.wav
  data/input/{prefix}.mp4

Output:
  data/outputs/{job_id}_dubbed.mp4
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

# Same speed-matching formula the web preview uses (stages/_shared/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from dub_speed import effective_speed


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKING_SR: int  = 22050
FADE_MS: int     = 10
RMS_FLOOR: float = 1e-6


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
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
        manifest["stages"][stage_key]["failed_at"]    = None
    else:
        manifest["stages"][stage_key]["failed_at"] = now_str
    save_json(manifest, manifest_path)
    logger.info(f"Manifest updated: {stage_key} → {status}")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_video_file(job_id: str, project_root: Path, logger: logging.Logger) -> Path:
    job_meta_path = project_root / "jobs" / job_id / "job_meta.json"
    if job_meta_path.exists():
        meta = load_json(job_meta_path)
        for key in ("input_video", "input_file"):
            if meta.get(key):
                candidate = Path(meta[key])
                if not candidate.is_absolute():
                    candidate = project_root / candidate
                if candidate.exists():
                    logger.info(f"Video found via job_meta.json [{key}]: {candidate}")
                    return candidate
        logger.warning("job_meta.json video path not found on disk — scanning data/input/")

    prefix    = job_id.split("_")[0]
    input_dir = project_root / "data" / "input"
    candidates = sorted(input_dir.glob(f"{prefix}*.mp4"))
    if candidates:
        logger.info(f"Video found via data/input/ scan: {candidates[0]}")
        return candidates[0]

    raise RuntimeError(
        f"Original video not found for job '{job_id}'. "
        f"Searched job_meta.json and data/input/{prefix}*.mp4."
    )


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def ms_to_samples(ms: float, sr: int) -> int:
    return int(ms * sr / 1000.0)


def load_wav_at_sr(path: Path, target_sr: int, logger: logging.Logger) -> np.ndarray:
    audio, orig_sr = librosa.load(str(path), sr=None, mono=True, dtype=np.float32)
    if orig_sr != target_sr:
        logger.debug(f"Resampling {path.name}: {orig_sr}Hz → {target_sr}Hz")
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio


def change_speed(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Change duration while aiming to preserve pitch.

    ``speed > 1`` makes the clip faster/shorter; ``speed < 1`` makes it
    slower/longer. librosa.effects.time_stretch uses a phase vocoder rather
    than resampling the waveform, so the perceived pitch is retained.
    ``sr`` remains in the signature for backwards compatibility.
    """
    speed = float(speed)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"speed must be a finite positive number; got {speed!r}")
    audio = audio.astype(np.float32, copy=False)
    if abs(speed - 1.0) < 1e-6 or audio.size == 0:
        return audio.copy()
    stretched = librosa.effects.time_stretch(audio, rate=speed)
    return np.asarray(stretched, dtype=np.float32)


def compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def apply_fade_in(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    result = audio.copy()
    n = min(fade_samples, len(result))
    if n > 0:
        result[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return result


def apply_fade_out(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    result = audio.copy()
    n = min(fade_samples, len(result))
    if n > 0:
        result[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return result


# ---------------------------------------------------------------------------
# RMS gain
# ---------------------------------------------------------------------------

def compute_global_tts_gain(
    dialogues: list[dict],
    project_root: Path,
    sr: int,
    reference_rms: float,
    logger: logging.Logger,
) -> float:
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
        logger.warning("No valid TTS clips for RMS — gain set to 1.0")
        return 1.0

    combined = np.concatenate(all_tts_samples)
    tts_rms  = compute_rms(combined)

    if tts_rms < RMS_FLOOR:
        logger.warning(f"TTS RMS too low ({tts_rms:.6f}) — gain set to 1.0")
        return 1.0

    gain = reference_rms / tts_rms
    logger.info(
        f"RMS — reference: {reference_rms:.6f}  tts: {tts_rms:.6f}  "
        f"gain: {gain:.4f} ({gain:.2f}x)"
    )
    return float(gain)


# ---------------------------------------------------------------------------
# Core 3-track assembly
# ---------------------------------------------------------------------------

def assemble(
    tts_manifest: dict,
    translations: dict,
    no_vocals_audio: np.ndarray,
    vocals_audio: np.ndarray,
    project_root: Path,
    sr: int,
    fade_ms: int,
    tts_gain: float,
    logger: logging.Logger,
) -> np.ndarray:
    fade_samples  = ms_to_samples(fade_ms, sr)
    total_samples = len(no_vocals_audio)

    # Align vocals length to no_vocals length
    if len(vocals_audio) < total_samples:
        vocals_audio = np.pad(
            vocals_audio,
            (0, total_samples - len(vocals_audio)),
            mode="constant",
        )
    elif len(vocals_audio) > total_samples:
        vocals_audio = vocals_audio[:total_samples]

    # Track 1: background — no_vocals, never modified
    mix_background = no_vocals_audio.copy()

    # Track 2: vocals — will be silenced only at successful TTS slots
    mix_vocals = vocals_audio.copy()

    # Track 3: TTS dubbed audio — overlaid at dialogue timestamps
    mix_tts = np.zeros(total_samples, dtype=np.float32)

    # Build lookup: dialogue_id → translation entry for timestamps
    trans_lookup: dict[str, dict] = {
        e["id"]: e for e in translations.get("dialogues", [])
    }

    dialogues      = tts_manifest.get("dialogues", [])
    replaced_count = 0
    kept_count     = 0
    skipped_count  = 0

    logger.info(f"Processing {len(dialogues)} TTS manifest entries.")

    for idx, entry in enumerate(dialogues):
        dlg_id       = entry.get("id", f"dialogue_{idx:04d}")
        speaker_id   = entry.get("speaker_id", "unknown")
        success      = bool(entry.get("success", False))
        wav_path_str = entry.get("wav_path")

        trans_entry = trans_lookup.get(dlg_id)
        if trans_entry is None:
            logger.warning(f"[{dlg_id}] Not in 03_translations.json — skipping.")
            skipped_count += 1
            continue

        start_ms        = float(trans_entry.get("start_ms", 0))
        original_end_ms = float(trans_entry.get("end_ms",   0))
        slot_start      = max(0, min(ms_to_samples(start_ms, sr), total_samples))

        # ── No TTS — keep original voice ─────────────────────────────────────
        if not success or not wav_path_str:
            logger.debug(f"[{dlg_id}] No TTS — keeping original voice in mix.")
            kept_count += 1
            continue

        # ── Load TTS WAV ──────────────────────────────────────────────────────
        wav_path = Path(wav_path_str)
        if not wav_path.is_absolute():
            wav_path = project_root / wav_path

        if not wav_path.exists():
            logger.warning(f"[{dlg_id}] TTS WAV missing: {wav_path} — keeping original.")
            kept_count += 1
            continue

        try:
            tts_audio = load_wav_at_sr(wav_path, sr, logger)
        except Exception as exc:
            logger.warning(f"[{dlg_id}] Load failed ({exc}) — keeping original.")
            kept_count += 1
            continue

        # ── Speed-match the dub to the original line ─────────────────────────
        # Manual per-dialogue speed (set in the editor) wins; otherwise the
        # automatic length-matching value. Applied BEFORE measuring length so
        # the silence window below fits the dub as it will actually sound.
        natural_ms = entry.get("synthesized_duration_ms") or (len(tts_audio) / sr * 1000.0)
        speed = effective_speed(
            trans_entry.get("dub_speed"), natural_ms, original_end_ms - start_ms
        )
        if abs(speed - 1.0) > 1e-3:
            tts_audio = change_speed(tts_audio, sr, speed)
            logger.info(
                f"[{dlg_id}] Speed {speed:.2f}x | {natural_ms:.0f}ms → "
                f"{len(tts_audio) / sr * 1000.0:.0f}ms "
                f"(original line {original_end_ms - start_ms:.0f}ms)"
            )

        tts_len         = len(tts_audio)
        tts_duration_ms = tts_len / sr * 1000.0
        tts_end_ms      = start_ms + tts_duration_ms

        # Silence window = max(original_end_ms, tts_end_ms)
        silence_end_ms = max(original_end_ms, tts_end_ms)
        silence_end    = max(
            slot_start,
            min(ms_to_samples(silence_end_ms, sr), total_samples),
        )

        logger.debug(
            f"[{dlg_id}] speaker={speaker_id} | "
            f"original={start_ms:.0f}–{original_end_ms:.0f}ms | "
            f"tts={tts_duration_ms:.0f}ms | "
            f"silence_until={silence_end_ms:.0f}ms"
        )

        # ── Silence vocals track at window (with fades) ───────────────────────
        pre_fade_start = max(0, slot_start - fade_samples)
        pre_fade_len   = slot_start - pre_fade_start
        if pre_fade_len > 0:
            mix_vocals[pre_fade_start:slot_start] *= np.linspace(
                1.0, 0.0, pre_fade_len, dtype=np.float32
            )

        mix_vocals[slot_start:silence_end] = 0.0

        post_fade_end = min(total_samples, silence_end + fade_samples)
        post_fade_len = post_fade_end - silence_end
        if post_fade_len > 0:
            mix_vocals[silence_end:post_fade_end] *= np.linspace(
                0.0, 1.0, post_fade_len, dtype=np.float32
            )

        # ── Place TTS on mix_tts track ────────────────────────────────────────
        tts_audio = tts_audio * tts_gain
        tts_audio = apply_fade_in(tts_audio, fade_samples)
        tts_audio = apply_fade_out(tts_audio, fade_samples)

        tts_place_end = min(total_samples, slot_start + tts_len)
        tts_actual    = tts_audio[: tts_place_end - slot_start]
        mix_tts[slot_start:tts_place_end] += tts_actual

        logger.info(
            f"[{dlg_id}] ✓ Replaced | TTS {tts_duration_ms:.0f}ms "
            f"at {start_ms:.0f}ms | silence until {silence_end_ms:.0f}ms"
        )
        replaced_count += 1

    # ── Sum 3 tracks ──────────────────────────────────────────────────────────
    logger.info(
        f"Mix summary — replaced: {replaced_count} | "
        f"kept original: {kept_count} | skipped: {skipped_count}"
    )
    logger.info("Summing 3 tracks: no_vocals + vocals(patched) + tts ...")

    final = mix_background + mix_vocals + mix_tts
    np.clip(final, -1.0, 1.0, out=final)
    logger.info("Assembly complete.")
    return final


# ---------------------------------------------------------------------------
# ffmpeg mux
# ---------------------------------------------------------------------------

def mux_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
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
        raise RuntimeError(f"ffmpeg mux failed (exit code {result.returncode}).")
    logger.info(f"ffmpeg mux complete → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 05 — Audio Assembly")
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    job_id       = args.job_id
    config_path  = Path(args.config)
    project_root = config_path.parent.resolve()

    log_path = project_root / "logs" / f"{job_id}_05_assembly.log"
    logger   = setup_logging(log_path)
    logger.info(f"=== Stage 05 — Assembly  |  job_id={job_id} ===")
    logger.info(f"Project root: {project_root}")

    t_start       = time.time()
    manifest_path = project_root / "jobs" / job_id / "manifest.json"

    try:
        config       = load_config(config_path)
        assembly_cfg = config.get("stages", {}).get("assembly", {})
        sr: int      = int(assembly_cfg.get("sample_rate", WORKING_SR))
        fade_ms: int = int(assembly_cfg.get("fade_ms", FADE_MS))
        logger.info(f"Config: sr={sr}Hz  fade_ms={fade_ms}ms")

        # Load TTS manifest
        tts_manifest_path = (
            project_root / "jobs" / job_id / "04_tts" / "tts_manifest.json"
        )
        if not tts_manifest_path.exists():
            raise FileNotFoundError(f"TTS manifest not found: {tts_manifest_path}")
        tts_manifest = load_json(tts_manifest_path)
        logger.info(
            f"TTS manifest: {tts_manifest.get('dialogue_count','?')} dialogues, "
            f"{tts_manifest.get('success_count','?')} successful."
        )

        # Load translations
        trans_path = project_root / "jobs" / job_id / "03_translations.json"
        if not trans_path.exists():
            raise FileNotFoundError(f"03_translations.json not found: {trans_path}")
        translations = load_json(trans_path)

        audio_dir = project_root / "data" / "audio"

        # ── Track 1: no_vocals (background BGM+SFX) ──────────────────────────
        no_vocals_path  = audio_dir / f"{job_id}_no_vocals.wav"
        full_audio_path = audio_dir / f"{job_id}.wav"

        if no_vocals_path.exists():
            logger.info(f"Track 1 (background): {no_vocals_path.name}")
            no_vocals_audio = load_wav_at_sr(no_vocals_path, sr, logger)
        else:
            logger.warning(
                "no_vocals.wav not found — falling back to full audio (voice bleed present)."
            )
            if not full_audio_path.exists():
                raise FileNotFoundError(
                    f"Neither no_vocals.wav nor {job_id}.wav found."
                )
            no_vocals_audio = load_wav_at_sr(full_audio_path, sr, logger)

        total_samples = len(no_vocals_audio)
        logger.info(
            f"Background: {total_samples} samples ({total_samples/sr:.2f}s) at {sr}Hz"
        )

        # ── Track 2: vocals (original voices + non-verbal sounds) ────────────
        vocals_path = audio_dir / f"{job_id}_vocals.wav"
        if vocals_path.exists():
            logger.info(f"Track 2 (vocals): {vocals_path.name}")
            vocals_audio = load_wav_at_sr(vocals_path, sr, logger)
        else:
            logger.warning(
                "vocals.wav not found — non-verbal sounds will be missing."
            )
            vocals_audio = np.zeros(total_samples, dtype=np.float32)

        # ── RMS reference ─────────────────────────────────────────────────────
        reference_rms = compute_rms(no_vocals_audio)
        logger.info(f"Reference RMS (no_vocals): {reference_rms:.6f}")

        # ── TTS gain ──────────────────────────────────────────────────────────
        tts_gain = compute_global_tts_gain(
            dialogues=tts_manifest.get("dialogues", []),
            project_root=project_root,
            sr=sr,
            reference_rms=reference_rms,
            logger=logger,
        )
        tts_gain = 1.0

        # ── Find video ────────────────────────────────────────────────────────
        video_path = find_video_file(job_id, project_root, logger)

        # ── Assemble ──────────────────────────────────────────────────────────
        logger.info("Starting 3-track assembly...")
        dubbed_audio = assemble(
            tts_manifest=tts_manifest,
            translations=translations,
            no_vocals_audio=no_vocals_audio,
            vocals_audio=vocals_audio,
            project_root=project_root,
            sr=sr,
            fade_ms=fade_ms,
            tts_gain=tts_gain,
            logger=logger,
        )

        del no_vocals_audio, vocals_audio
        gc.collect()

        # ── Save temp WAV ─────────────────────────────────────────────────────
        temp_audio_path = (
            project_root / "jobs" / job_id / "05_dubbed_audio_temp.wav"
        )
        logger.info(f"Saving dubbed audio: {temp_audio_path}")
        sf.write(str(temp_audio_path), dubbed_audio, sr, subtype="PCM_16")

        del dubbed_audio
        gc.collect()

        # ── ffmpeg mux ────────────────────────────────────────────────────────
        output_dir  = project_root / "data" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}_dubbed.mp4"
        mux_video(video_path, temp_audio_path, output_path, logger)

        # ── Cleanup ───────────────────────────────────────────────────────────
        try:
            temp_audio_path.unlink()
            logger.debug("Temp audio removed.")
        except Exception as exc:
            logger.warning(f"Could not remove temp audio: {exc}")

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