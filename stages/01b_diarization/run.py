#!/usr/bin/env python3
"""
Stage 01b — Speaker Diarization
================================
Approach:
    - Read dialogue timestamps from 01_transcription.json
    - Slice original audio at each dialogue boundary
    - Extract TitaNet speaker embeddings per dialogue
    - Cluster embeddings using AgglomerativeClustering (auto speaker count)
    - Assign speaker IDs to dialogues and words
    - Extract speaker reference profiles from vocals WAV for Stage 04

No NeMo pipeline used — only TitaNet embeddings directly.
No VAD, no full diarization pipeline, no source patching needed.

Author: dubbing_v2 pipeline
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("stage_01b")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 01b — Speaker Diarization")
    p.add_argument("--job_id", required=True)
    p.add_argument("--config", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config / manifest helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def mark_manifest(manifest_path: Path, stage: str, success: bool, note: str = "") -> None:
    manifest = load_manifest(manifest_path)
    ts = datetime.now(timezone.utc).isoformat()
    entry = manifest.get("stages", {}).get(stage, {})
    entry["status"] = "completed"
    entry["completed_at"] = ts
    entry["failed_at"] = None
    if note:
        entry["note"] = note
    manifest.setdefault("stages", {})[stage] = entry
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Audio slice helper
# ---------------------------------------------------------------------------

def load_audio_slice(
    audio: torch.Tensor,
    sr: int,
    start_ms: int,
    end_ms: int,
) -> Optional[torch.Tensor]:
    """
    Slice audio tensor at [start_ms, end_ms].
    Converts to mono if stereo.
    Returns None if slice is too short (< 0.5s).
    """
    start_sample = int(start_ms / 1000 * sr)
    end_sample = int(end_ms / 1000 * sr)
    end_sample = min(end_sample, audio.shape[1])
    start_sample = max(0, start_sample)

    if end_sample <= start_sample:
        return None

    slice_audio = audio[:, start_sample:end_sample]

    # Convert stereo to mono
    if slice_audio.shape[0] > 1:
        slice_audio = slice_audio.mean(dim=0, keepdim=True)

    # Too short for reliable embedding
    if slice_audio.shape[1] < int(0.5 * sr):
        return None

    return slice_audio


# ---------------------------------------------------------------------------
# TitaNet embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    dialogues: List[dict],
    audio: torch.Tensor,
    sr: int,
    speaker_model,
    device: str,
    logger: logging.Logger,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extract TitaNet embeddings for each dialogue.

    Returns:
        embeddings: list of 192-dim numpy arrays
        valid_indices: indices into dialogues list that were successfully processed
    """
    embeddings = []
    valid_indices = []

    for idx, dlg in enumerate(dialogues):
        slice_audio = load_audio_slice(
            audio, sr,
            dlg.get("start_ms", 0),
            dlg.get("end_ms", 0),
        )

        if slice_audio is None:
            duration = (dlg.get("end_ms", 0) - dlg.get("start_ms", 0)) / 1000
            logger.warning(
                "Skipping %s — too short (%.2fs) for reliable embedding",
                dlg["id"], duration,
            )
            continue

        try:
            with torch.no_grad():
                signal = slice_audio.to(device)
                length = torch.tensor([slice_audio.shape[1]]).to(device)
                _, emb = speaker_model.forward(
                    input_signal=signal,
                    input_signal_length=length,
                )
            embeddings.append(emb.cpu().numpy()[0])
            valid_indices.append(idx)
            logger.debug(
                "%s — embedding extracted (%.2fs)",
                dlg["id"],
                (dlg["end_ms"] - dlg["start_ms"]) / 1000,
            )
        except Exception:
            logger.warning(
                "Embedding failed for %s:\n%s",
                dlg["id"], traceback.format_exc(),
            )

    return embeddings, valid_indices


# ---------------------------------------------------------------------------
# Speaker clustering
# ---------------------------------------------------------------------------

def cluster_speakers(
    embeddings: List[np.ndarray],
    max_speakers: int,
    distance_threshold: float,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Cluster embeddings into speaker groups using AgglomerativeClustering.

    Uses distance_threshold for automatic speaker count detection.
    Caps at max_speakers.

    Returns array of integer speaker labels.
    """
    if len(embeddings) == 1:
        logger.info("Only 1 dialogue — assigning speaker_00")
        return np.array([0])

    emb_matrix = normalize(np.array(embeddings))

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(emb_matrix)
    n_detected = len(set(labels))

    logger.info(
        "Clustering: %d embeddings → %d speakers detected (threshold=%.2f)",
        len(embeddings), n_detected, distance_threshold,
    )

    # If over max_speakers, re-cluster with fixed count
    if n_detected > max_speakers:
        logger.warning(
            "Detected %d speakers exceeds max %d — re-clustering with fixed count",
            n_detected, max_speakers,
        )
        clustering = AgglomerativeClustering(
            n_clusters=max_speakers,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(emb_matrix)

    return labels


# ---------------------------------------------------------------------------
# Speaker profile extraction
# ---------------------------------------------------------------------------

def extract_speaker_profiles(
    speaker_segments: Dict[str, List[dict]],
    vocals_audio: torch.Tensor,
    vocals_sr: int,
    profiles_dir: Path,
    min_s: float,
    max_s: float,
    logger: logging.Logger,
) -> Dict[str, Path]:
    """
    Extract reference audio per speaker from vocals WAV.

    For each speaker:
        - Collect all their dialogue segments from vocals
        - Sort by duration descending (longest = cleanest)
        - Concatenate up to max_s seconds
        - Save as 16kHz mono WAV

    Returns dict of speaker_id -> saved WAV path.
    """
    profiles_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    # Convert vocals to mono numpy if needed
    vocals_np = vocals_audio.numpy()
    if vocals_np.ndim > 1:
        vocals_np = vocals_np.mean(axis=0)

    max_samples = int(max_s * vocals_sr)
    min_samples = int(min_s * vocals_sr)

    for speaker_id, segments in speaker_segments.items():
        # Sort by duration descending
        segments_sorted = sorted(
            segments,
            key=lambda s: s["end_ms"] - s["start_ms"],
            reverse=True,
        )

        chunks: List[np.ndarray] = []
        total_samples = 0

        for seg in segments_sorted:
            if total_samples >= max_samples:
                break
            start_sample = int(seg["start_ms"] / 1000 * vocals_sr)
            end_sample = int(seg["end_ms"] / 1000 * vocals_sr)
            start_sample = max(0, min(start_sample, len(vocals_np) - 1))
            end_sample = max(start_sample + 1, min(end_sample, len(vocals_np)))
            chunk = vocals_np[start_sample:end_sample]
            remaining = max_samples - total_samples
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
            chunks.append(chunk)
            total_samples += len(chunk)

        if not chunks:
            logger.warning(
                "No audio chunks for %s — skipping profile (all segments too short)",
                speaker_id,
            )
            continue

        profile = np.concatenate(chunks)
        duration = len(profile) / vocals_sr

        # Skip if effectively silent / zero duration
        if len(profile) == 0 or duration < 0.1:
            logger.warning(
                "%s profile has zero/near-zero duration — skipping",
                speaker_id,
            )
            continue

        if duration < min_s:
            logger.warning(
                "%s has only %.2fs of speech (min %.1fs) — using all available",
                speaker_id, duration, min_s,
            )

        # Resample to 16kHz if needed (Stage 04 TTS expects 16kHz)
        target_sr = 16000
        if vocals_sr != target_sr:
            profile_tensor = torch.tensor(profile).unsqueeze(0)
            profile_tensor = torchaudio.functional.resample(
                profile_tensor, orig_freq=vocals_sr, new_freq=target_sr
            )
            profile = profile_tensor.squeeze(0).numpy()
            logger.debug(
                "%s resampled from %dHz to %dHz",
                speaker_id, vocals_sr, target_sr,
            )

        out_path = profiles_dir / f"{speaker_id}.wav"
        sf.write(str(out_path), profile, target_sr, subtype="PCM_16")
        logger.info(
            "Profile saved: %s — %.2fs at %dHz",
            out_path.name, len(profile) / target_sr, target_sr,
        )
        saved[speaker_id] = out_path

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diarization(job_id: str, config_path: str) -> None:
    cfg = load_config(config_path)
    project_root = Path(config_path).parent.resolve()

    jobs_dir = project_root / cfg["paths"]["jobs"]
    job_dir = jobs_dir / job_id
    logs_dir = project_root / cfg["paths"]["logs"]
    audio_dir = project_root / cfg["paths"]["data_audio"]

    transcription_path = job_dir / "01_transcription.json"
    manifest_path = job_dir / "manifest.json"
    profiles_dir = job_dir / "speaker_profiles"
    log_path = logs_dir / f"{job_id}_01b_diarization.log"

    logger = setup_logging(log_path)
    logger.info("=" * 60)
    logger.info("Stage 01b — Speaker Diarization")
    logger.info("Job    : %s", job_id)
    logger.info("Config : %s", config_path)
    logger.info("=" * 60)

    # Restart-safe
    manifest = load_manifest(manifest_path)
    if manifest.get("stages", {}).get("01b_diarization", {}).get("status") == "completed":
        logger.info("Already completed — skipping.")
        return

    # Load transcription
    if not transcription_path.exists():
        logger.error("01_transcription.json not found: %s", transcription_path)
        sys.exit(1)

    with open(transcription_path, "r", encoding="utf-8") as f:
        transcription = json.load(f)

    dialogues = transcription.get("dialogues", [])
    logger.info("Loaded %d dialogues", len(dialogues))

    if not dialogues:
        logger.warning("No dialogues found — nothing to diarize.")
        mark_manifest(manifest_path, "01b_diarization", success=True, note="no dialogues")
        return

    # Locate original audio (for embeddings)
    original_audio_path = audio_dir / f"{job_id}.wav"
    if not original_audio_path.exists():
        logger.error("Original audio not found: %s", original_audio_path)
        sys.exit(1)

    # Locate vocals audio (for speaker profiles)
    vocals_path = audio_dir / f"{job_id}_vocals.wav"
    if not vocals_path.exists():
        logger.warning("Vocals WAV not found — using original audio for profiles too")
        vocals_path = original_audio_path

    # Diarization config
    diar_cfg = cfg.get("stages", {}).get("diarization", {})
    device_str = diar_cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        device_str = "cpu"

    max_speakers = int(diar_cfg.get("max_num_speakers", 8))
    distance_threshold = float(diar_cfg.get("distance_threshold", 0.7))
    min_profile_s = float(diar_cfg.get("speaker_profile_min_s", 3.0))
    max_profile_s = float(diar_cfg.get("speaker_profile_max_s", 10.0))

    # -----------------------------------------------------------------------
    # Load audio
    # -----------------------------------------------------------------------
    logger.info("Loading original audio: %s", original_audio_path)
    audio, sr = torchaudio.load(str(original_audio_path))
    logger.info("Audio shape: %s, sr: %d", audio.shape, sr)

    logger.info("Loading vocals audio: %s", vocals_path)
    vocals_audio, vocals_sr = torchaudio.load(str(vocals_path))
    if vocals_audio.shape[0] > 1:
        vocals_audio = vocals_audio.mean(dim=0, keepdim=True)
    vocals_np = vocals_audio.squeeze(0)

    # -----------------------------------------------------------------------
    # Load TitaNet
    # -----------------------------------------------------------------------
    diarization_success = False
    try:
        logger.info("Loading TitaNet speaker embedding model...")
        import nemo.collections.asr as nemo_asr
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("titanet_large")
        speaker_model.eval()
        speaker_model = speaker_model.to(device_str)
        logger.info("TitaNet loaded on %s", device_str)

        # -------------------------------------------------------------------
        # Extract embeddings
        # -------------------------------------------------------------------
        embeddings, valid_indices = extract_embeddings(
            dialogues=dialogues,
            audio=audio,
            sr=sr,
            speaker_model=speaker_model,
            device=device_str,
            logger=logger,
        )

        logger.info(
            "Embeddings extracted: %d / %d dialogues",
            len(embeddings), len(dialogues),
        )

        if len(embeddings) == 0:
            raise RuntimeError("No embeddings extracted — all dialogues too short")

        # -------------------------------------------------------------------
        # Cluster
        # -------------------------------------------------------------------
        labels = cluster_speakers(
            embeddings=embeddings,
            max_speakers=max_speakers,
            distance_threshold=distance_threshold,
            logger=logger,
        )

        # Map valid_indices → speaker_id
        idx_to_speaker: Dict[int, str] = {
            valid_indices[i]: f"speaker_{labels[i]:02d}"
            for i in range(len(valid_indices))
        }

        # Dialogues that were skipped (too short) → assign nearest speaker
        all_assigned = set(valid_indices)
        for idx, dlg in enumerate(dialogues):
            if idx not in all_assigned:
                # Find nearest valid dialogue by time
                if valid_indices:
                    dlg_mid = (dlg["start_ms"] + dlg["end_ms"]) / 2
                    nearest = min(
                        valid_indices,
                        key=lambda vi: abs(
                            (dialogues[vi]["start_ms"] + dialogues[vi]["end_ms"]) / 2 - dlg_mid
                        ),
                    )
                    idx_to_speaker[idx] = idx_to_speaker[nearest]
                    logger.debug(
                        "%s too short — assigned speaker from nearest: %s",
                        dlg["id"], idx_to_speaker[idx],
                    )
                else:
                    idx_to_speaker[idx] = "speaker_00"

        # -------------------------------------------------------------------
        # Assign speaker IDs to dialogues and words
        # -------------------------------------------------------------------
        for idx, dlg in enumerate(dialogues):
            spk = idx_to_speaker[idx]
            dlg["speaker_id"] = spk
            for word in dlg.get("words", []):
                word["speaker_id"] = spk

        unique_speakers = sorted(set(d["speaker_id"] for d in dialogues))
        transcription["speaker_count"] = len(unique_speakers)
        logger.info("Unique speakers: %d → %s", len(unique_speakers), unique_speakers)

        # -------------------------------------------------------------------
        # Extract speaker profiles from vocals
        # -------------------------------------------------------------------
        speaker_segments: Dict[str, List[dict]] = {}
        for dlg in dialogues:
            spk = dlg["speaker_id"]
            speaker_segments.setdefault(spk, []).append({
                "start_ms": dlg["start_ms"],
                "end_ms": dlg["end_ms"],
            })

        """ saved_profiles = extract_speaker_profiles(
            speaker_segments=speaker_segments,
            vocals_audio=vocals_np.unsqueeze(0),
            vocals_sr=vocals_sr,
            profiles_dir=profiles_dir,
            min_s=min_profile_s,
            max_s=max_profile_s,
            logger=logger,
        )
        logger.info("Profiles saved: %s", list(saved_profiles.keys()))"""
        logger.info(
            "Speaker profile extraction skipped — "
            "will be built after UI confirmation."
        )

        diarization_success = True

    except Exception:
        logger.error(
            "Diarization failed — falling back to speaker_00:\n%s",
            traceback.format_exc(),
        )
        for dlg in dialogues:
            dlg["speaker_id"] = "speaker_00"
            for word in dlg.get("words", []):
                word["speaker_id"] = "speaker_00"
        transcription["speaker_count"] = 1

    finally:
        try:
            del speaker_model
        except NameError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug("GPU memory released.")

    # -----------------------------------------------------------------------
    # Write updated transcription
    # -----------------------------------------------------------------------
    transcription["dialogues"] = dialogues
    with open(transcription_path, "w", encoding="utf-8") as f:
        json.dump(transcription, f, indent=2, ensure_ascii=False)
    logger.info("01_transcription.json updated: %s", transcription_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    unique_final = sorted(set(d["speaker_id"] for d in dialogues))
    logger.info("=" * 60)
    logger.info("STAGE 01b SUMMARY")
    logger.info("  Success        : %s", diarization_success)
    logger.info("  Total dialogues: %d", len(dialogues))
    logger.info("  Speakers found : %d → %s", len(unique_final), unique_final)
    logger.info("=" * 60)

    note = "" if diarization_success else "failed — fallback to speaker_00"
    mark_manifest(manifest_path, "01b_diarization", success=True, note=note)
    logger.info("Manifest updated: 01b_diarization → completed")
    logger.info("Stage 01b complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_diarization(job_id=args.job_id, config_path=args.config)