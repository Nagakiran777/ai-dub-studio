"""
Stage 02: Emotion Detection
============================
Reads 01_transcription.json, runs dual-model emotion analysis on each
dialogue unit, and writes 02_emotions.json.

Models (unchanged — swap via config when ready):
  - Acoustic: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
      → continuous VAD output, remapped to 8-class categorical scores
  - Text: j-hartmann/emotion-english-distilroberta-base
      → 7-class categorical (mapped to 8-class by adding calm=0)

Architecture:
  - Two-pass: acoustic model (CUDA) → release → text model (CUDA) → release
  - VRAM never has both models simultaneously
  - Input audio: vocals WAV from transcription audio_path, fallback to full WAV
  - VAD scores: mapped from winning emotion label via fixed table (not predicted)
  - Intensity: acoustic_conf * 0.6 + text_conf * 0.4
  - Fusion: 60% acoustic + 40% text weighted average
  - speaker_id carried through from 01_transcription.json
  - Every dialogue always gets an emotion entry — stage never crashes
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_KEY = "02_emotion"
STAGE_LOG_NAME = "02_emotion"

# 8-class emotion set — calm added over previous 7-class set
EMOTIONS: List[str] = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

# j-hartmann label strings → our canonical 8-class set
HARTMANN_MAP: Dict[str, str] = {
    "neutral":  "neutral",
    "joy":      "happy",
    "sadness":  "sad",
    "anger":    "angry",
    "fear":     "fearful",
    "disgust":  "disgust",
    "surprise": "surprised",
    # calm not in j-hartmann — will be 0.0 from text model always
}

# Fixed VAD mapping from emotion label
# valence/arousal/dominance all in [-1, 1]
VAD_MAP: Dict[str, Dict[str, float]] = {
    "neutral":   {"valence":  0.0, "arousal":  0.0, "dominance":  0.0},
    "calm":      {"valence":  0.3, "arousal": -0.4, "dominance":  0.1},
    "happy":     {"valence":  0.8, "arousal":  0.6, "dominance":  0.4},
    "sad":       {"valence": -0.7, "arousal": -0.3, "dominance": -0.4},
    "angry":     {"valence": -0.6, "arousal":  0.8, "dominance":  0.7},
    "fearful":   {"valence": -0.6, "arousal":  0.7, "dominance": -0.5},
    "disgust":   {"valence": -0.5, "arousal":  0.3, "dominance":  0.2},
    "surprised": {"valence":  0.2, "arousal":  0.7, "dominance":  0.0},
}

ACOUSTIC_WEIGHT = 0.6
TEXT_WEIGHT = 0.4

MIN_AUDIO_DURATION_S: float = 0.5
MIN_TEXT_LEN: int = 3
MODEL_SAMPLE_RATE: int = 16_000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("stage02_emotion")


def setup_logging(job_id: str, logs_dir: Path) -> None:
    """Configure file + console logging."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{job_id}_{STAGE_LOG_NAME}.log"
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    logger.info("Logging initialised → %s", log_file)


# ---------------------------------------------------------------------------
# Config / manifest helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Config loaded from %s", config_path)
    return cfg


def load_manifest(job_id: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = Path(cfg["paths"]["jobs"]) / job_id / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(manifest: Dict[str, Any], job_id: str,
                  cfg: Dict[str, Any]) -> None:
    manifest_path = Path(cfg["paths"]["jobs"]) / job_id / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def mark_stage_complete(job_id: str, manifest: Dict[str, Any],
                        cfg: Dict[str, Any]) -> None:
    manifest["stages"][STAGE_KEY]["status"] = "completed"
    manifest["stages"][STAGE_KEY]["completed_at"] = (
        datetime.now(timezone.utc).isoformat()
    )
    manifest["stages"][STAGE_KEY]["failed_at"] = None
    save_manifest(manifest, job_id, cfg)
    logger.info("Stage %s marked COMPLETE.", STAGE_KEY)


def mark_stage_failed(job_id: str, manifest: Dict[str, Any],
                      cfg: Dict[str, Any]) -> None:
    manifest["stages"][STAGE_KEY]["status"] = "failed"
    manifest["stages"][STAGE_KEY]["failed_at"] = (
        datetime.now(timezone.utc).isoformat()
    )
    save_manifest(manifest, job_id, cfg)
    logger.error("Stage %s marked FAILED.", STAGE_KEY)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def release_gpu() -> None:
    """Release all GPU memory — call after every model use."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("GPU cache cleared.")


def get_device(cfg: Dict[str, Any]) -> torch.device:
    if cfg["hardware"]["gpu_available"] and torch.cuda.is_available():
        device = torch.device("cuda")
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        logger.info("CUDA device found. Total VRAM: %dMB", vram_mb)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — using CPU.")
    return device


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def resolve_audio_path(transcription: Dict[str, Any],
                       cfg: Dict[str, Any], job_id: str) -> Path:
    """
    Resolve the audio file to use.
    Priority: vocals WAV from transcription audio_path → full WAV fallback.
    """
    audio_path_str = transcription.get("audio_path", "")
    audio_path = Path(audio_path_str)

    if audio_path.exists():
        logger.info("Using audio: %s", audio_path)
        return audio_path

    full_wav = Path(cfg["paths"]["data_audio"]) / f"{job_id}.wav"
    if full_wav.exists():
        logger.warning(
            "Vocals WAV not found at %s — falling back to full WAV: %s",
            audio_path, full_wav,
        )
        return full_wav

    raise FileNotFoundError(
        f"No audio found. Tried: {audio_path}, {full_wav}"
    )


def load_audio_slice(audio_path: Path, start_ms: int,
                     end_ms: int) -> Tuple[np.ndarray, int]:
    """
    Load a mono 16kHz slice from a WAV file.
    Returns (waveform float32 mono, sample_rate).
    """
    import soundfile as sf
    import librosa

    with sf.SoundFile(str(audio_path)) as f:
        native_sr = f.samplerate
        start_sample = max(0, int(start_ms / 1000 * native_sr))
        end_sample = min(len(f), int(end_ms / 1000 * native_sr))
        f.seek(start_sample)
        data = f.read(end_sample - start_sample, dtype="float32", always_2d=False)

    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample to 16kHz if needed
    if native_sr != MODEL_SAMPLE_RATE:
        data = librosa.resample(
            data, orig_sr=native_sr, target_sr=MODEL_SAMPLE_RATE
        )

    return data, MODEL_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Prosody extraction
# ---------------------------------------------------------------------------

def extract_prosody(waveform: np.ndarray, sr: int,
                    words: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Extract prosodic features from a waveform slice.

    pitch_mean_hz    — mean F0 over voiced frames (librosa pyin)
    pitch_std_hz     — std of F0 over voiced frames
    speech_rate_wps  — words per second from word timestamps
    energy_mean      — RMS energy of the slice
    """
    import librosa

    result: Dict[str, Optional[float]] = {
        "pitch_mean_hz":   None,
        "pitch_std_hz":    None,
        "speech_rate_wps": None,
        "energy_mean":     None,
    }

    duration_s = len(waveform) / sr

    # RMS energy
    try:
        result["energy_mean"] = round(
            float(np.sqrt(np.mean(waveform ** 2))), 6
        )
    except Exception as exc:
        logger.warning("Energy extraction failed: %s", exc)

    # Pitch via pyin
    try:
        if duration_s >= 0.1:
            f0, voiced_flag, _ = librosa.pyin(
                waveform,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr,
            )
            voiced_f0 = f0[voiced_flag]
            if voiced_f0.size > 0:
                result["pitch_mean_hz"] = round(float(np.mean(voiced_f0)), 2)
                result["pitch_std_hz"]  = round(float(np.std(voiced_f0)), 2)
    except Exception as exc:
        logger.warning("Pitch extraction failed: %s", exc)

    # Speech rate from word timestamps
    try:
        if words and duration_s > 0:
            result["speech_rate_wps"] = round(len(words) / duration_s, 3)
    except Exception as exc:
        logger.warning("Speech rate extraction failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Acoustic model
# ---------------------------------------------------------------------------

def load_acoustic_model(model_name: str, cache_dir: Path,
                        device: torch.device) -> Tuple[Any, Any]:
    """Load acoustic emotion model onto device."""
    from transformers import AutoProcessor, AutoModelForAudioClassification

    logger.info("Loading acoustic model: %s", model_name)
    processor = AutoProcessor.from_pretrained(
        model_name, cache_dir=str(cache_dir)
    )
    model = AutoModelForAudioClassification.from_pretrained(
        model_name, cache_dir=str(cache_dir)
    )
    model = model.to(device).half()
    model.eval()

    if device.type == "cuda":
        vram_mb = torch.cuda.memory_allocated() // (1024 ** 2)
        logger.info("Acoustic model loaded. VRAM: ~%dMB", vram_mb)
    return model, processor


def run_acoustic_inference(
    waveform: np.ndarray,
    sr: int,
    model: Any,
    processor: Any,
    device: torch.device,
) -> Optional[Dict[str, float]]:
    """
    Run acoustic model on waveform.
    Returns raw VAD dict {valence, arousal, dominance} clipped to [-1,1].
    NOTE: np.clip used directly — no sigmoid remapping (that was the v1 bug).
    """
    try:
        inputs = processor(
            waveform, sampling_rate=sr, return_tensors="pt", padding=True
        )
        inputs = {
            k: v.to(device).half() if v.dtype == torch.float32 else v.to(device)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            logits = model(**inputs).logits  # shape (1, 3) — VAD

        vad = np.clip(logits.squeeze().float().cpu().numpy(), -1.0, 1.0)
        return {
            "valence":   round(float(vad[0]), 4),
            "arousal":   round(float(vad[1]), 4),
            "dominance": round(float(vad[2]), 4),
        }
    except Exception as exc:
        logger.warning("Acoustic inference failed: %s", exc)
        return None


def vad_to_categorical(vad: Dict[str, float]) -> Dict[str, float]:
    """
    Convert continuous VAD to 8-class probability-like scores.
    Soft multiplicative scoring — compatible with text model softmax for fusion.
    """
    v = vad["valence"]
    a = vad["arousal"]
    d = vad["dominance"]

    scores: Dict[str, float] = {
        "neutral":   max(0.0, (1 - abs(a)) / 2) * max(0.0, (1 - abs(v)) / 2),
        "calm":      max(0.0, (-a + 1) / 2) * max(0.0, (v + 1) / 2),
        "happy":     max(0.0, (a + 1) / 2) * max(0.0, (v + 1) / 2),
        "sad":       max(0.0, (-a + 1) / 2) * max(0.0, (-v + 1) / 2),
        "angry":     max(0.0, (a + 1) / 2) * max(0.0, (-v + 1) / 2) * max(0.0, (d + 1) / 2),
        "fearful":   max(0.0, (a + 1) / 2) * max(0.0, (-v + 1) / 2) * max(0.0, (-d + 1) / 2),
        "disgust":   max(0.0, (d + 1) / 2) * max(0.0, (-v + 1) / 2) * max(0.0, 1 - abs(a)),
        "surprised": max(0.0, (a + 1) / 2) * max(0.0, 1 - abs(v)),
    }

    total = sum(scores.values())
    if total > 0:
        return {k: round(val / total, 4) for k, val in scores.items()}
    scores = {k: 0.0 for k in EMOTIONS}
    scores["neutral"] = 1.0
    return scores


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------

def load_text_model(model_name: str, cache_dir: Path,
                    device: torch.device) -> Any:
    """Load text emotion classification pipeline."""
    from transformers import pipeline as hf_pipeline

    logger.info("Loading text model: %s", model_name)
    pipe = hf_pipeline(
        "text-classification",
        model=model_name,
        top_k=None,
        device=0 if device.type == "cuda" else -1,
        model_kwargs={"cache_dir": str(cache_dir)},
    )
    if device.type == "cuda":
        vram_mb = torch.cuda.memory_allocated() // (1024 ** 2)
        logger.info("Text model loaded. VRAM: ~%dMB", vram_mb)
    return pipe


def run_text_inference(text: str, pipe: Any) -> Optional[Dict[str, float]]:
    """
    Run text emotion model. Returns 8-class scores.
    calm is always 0.0 — j-hartmann has no calm class.
    """
    try:
        raw: List[Dict[str, Any]] = pipe(text)[0]
        scores: Dict[str, float] = {label: 0.0 for label in EMOTIONS}
        for item in raw:
            mapped = HARTMANN_MAP.get(item["label"].lower())
            if mapped:
                scores[mapped] = round(float(item["score"]), 4)
        return scores
    except Exception as exc:
        logger.warning("Text inference failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fuse(
    acoustic_scores: Optional[Dict[str, float]],
    text_scores: Optional[Dict[str, float]],
) -> Tuple[str, float, str, float, str, float]:
    """
    Fuse acoustic + text scores (60/40 weighted average).

    Returns
    -------
    final_emotion    : str
    intensity        : float  (acoustic_conf*0.6 + text_conf*0.4, clamped [0,1])
    acoustic_emotion : str
    acoustic_conf    : float
    text_emotion     : str
    text_conf        : float
    """
    fallback: Dict[str, float] = {k: 0.0 for k in EMOTIONS}
    fallback["neutral"] = 1.0

    if acoustic_scores is None and text_scores is None:
        logger.warning("Both models failed — defaulting to neutral.")
        return "neutral", 0.5, "neutral", 0.5, "neutral", 0.5

    if acoustic_scores is not None:
        acoustic_emotion = max(acoustic_scores, key=lambda k: acoustic_scores[k])
        acoustic_conf = round(float(acoustic_scores[acoustic_emotion]), 4)
    else:
        logger.warning("Acoustic failed — using text only.")
        acoustic_scores = fallback
        acoustic_emotion = "neutral"
        acoustic_conf = 0.0

    if text_scores is not None:
        text_emotion = max(text_scores, key=lambda k: text_scores[k])
        text_conf = round(float(text_scores[text_emotion]), 4)
    else:
        logger.warning("Text failed — using acoustic only.")
        text_scores = fallback
        text_emotion = "neutral"
        text_conf = 0.0

    fused: Dict[str, float] = {
        label: round(
            ACOUSTIC_WEIGHT * acoustic_scores[label] +
            TEXT_WEIGHT * text_scores[label],
            4,
        )
        for label in EMOTIONS
    }

    final_emotion = max(fused, key=lambda k: fused[k])
    intensity = round(
        acoustic_conf * ACOUSTIC_WEIGHT + text_conf * TEXT_WEIGHT, 4
    )
    intensity = max(0.0, min(1.0, intensity))

    return (
        final_emotion, intensity,
        acoustic_emotion, acoustic_conf,
        text_emotion, text_conf,
    )


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------

def acoustic_pass(
    dialogues: List[Dict[str, Any]],
    audio_path: Path,
    model_name: str,
    cache_dir: Path,
    device: torch.device,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Pass 1: Acoustic model — loads, runs all dialogues, releases VRAM.
    Returns: dialogue_id → {vad, categorical, waveform, sr} or None
    """
    model, processor = load_acoustic_model(model_name, cache_dir, device)
    results: Dict[str, Optional[Dict[str, Any]]] = {}

    for idx, dlg in enumerate(dialogues):
        dlg_id = dlg["id"]
        logger.info(
            "[%d/%d] Acoustic — %s", idx + 1, len(dialogues), dlg_id
        )

        try:
            waveform, sr = load_audio_slice(
                audio_path, dlg["start_ms"], dlg["end_ms"]
            )
        except Exception as exc:
            logger.error("  Audio slice failed for %s: %s", dlg_id, exc)
            results[dlg_id] = None
            continue

        duration_s = (dlg["end_ms"] - dlg["start_ms"]) / 1000.0

        if duration_s < MIN_AUDIO_DURATION_S:
            logger.warning(
                "  Too short (%.3fs) for %s — skipping acoustic.", duration_s, dlg_id
            )
            results[dlg_id] = {
                "vad": None, "categorical": None,
                "waveform": waveform, "sr": sr,
            }
            continue

        vad = run_acoustic_inference(waveform, sr, model, processor, device)
        categorical = vad_to_categorical(vad) if vad is not None else None

        if vad:
            top = max(categorical, key=lambda k: categorical[k]) if categorical else "?"
            logger.debug(
                "  VAD v=%.3f a=%.3f d=%.3f → %s",
                vad["valence"], vad["arousal"], vad["dominance"], top,
            )

        results[dlg_id] = {
            "vad": vad,
            "categorical": categorical,
            "waveform": waveform,
            "sr": sr,
        }

    logger.info("Releasing acoustic model from VRAM.")
    del model, processor
    release_gpu()
    return results


def text_pass(
    dialogues: List[Dict[str, Any]],
    model_name: str,
    cache_dir: Path,
    device: torch.device,
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Pass 2: Text model — loads, runs all dialogues, releases VRAM.
    Returns: dialogue_id → 8-class scores or None
    """
    pipe = load_text_model(model_name, cache_dir, device)
    results: Dict[str, Optional[Dict[str, float]]] = {}

    for idx, dlg in enumerate(dialogues):
        dlg_id = dlg["id"]
        logger.info("[%d/%d] Text — %s", idx + 1, len(dialogues), dlg_id)

        text = dlg.get("text", "").strip()
        if len(text) < MIN_TEXT_LEN:
            logger.warning("  Text too short for %s.", dlg_id)
            results[dlg_id] = None
            continue

        results[dlg_id] = run_text_inference(text, pipe)

    logger.info("Releasing text model from VRAM.")
    del pipe
    release_gpu()
    return results


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def build_output(
    transcription: Dict[str, Any],
    acoustic_results: Dict[str, Optional[Dict[str, Any]]],
    text_results: Dict[str, Optional[Dict[str, float]]],
    job_id: str,
) -> Dict[str, Any]:
    """Assemble final 02_emotions.json."""
    output_dialogues: List[Dict[str, Any]] = []

    for dlg in transcription["dialogues"]:
        dlg_id     = dlg["id"]
        speaker_id = dlg.get("speaker_id", "speaker_00")
        words      = dlg.get("words", [])

        ar = acoustic_results.get(dlg_id)
        tr = text_results.get(dlg_id)

        acoustic_scores = ar["categorical"] if ar else None
        vad_raw         = ar["vad"]         if ar else None
        waveform        = ar.get("waveform") if ar else None
        sr              = ar.get("sr", MODEL_SAMPLE_RATE) if ar else MODEL_SAMPLE_RATE
        text_scores     = tr

        # Fuse
        (
            final_emotion, intensity,
            acoustic_emotion, acoustic_conf,
            text_emotion, text_conf,
        ) = fuse(acoustic_scores, text_scores)

        # VAD from fixed map — clean, consistent, no model noise
        vad_mapped = VAD_MAP.get(final_emotion, VAD_MAP["neutral"])

        # Prosody
        prosody: Dict[str, Optional[float]] = {
            "pitch_mean_hz":   None,
            "pitch_std_hz":    None,
            "speech_rate_wps": None,
            "energy_mean":     None,
        }
        if waveform is not None:
            try:
                prosody = extract_prosody(waveform, sr, words)
            except Exception as exc:
                logger.warning("Prosody failed for %s: %s", dlg_id, exc)

        entry = {
            "id":                  dlg_id,
            "speaker_id":          speaker_id,
            "start_ms":            dlg["start_ms"],
            "end_ms":              dlg["end_ms"],
            "text":                dlg["text"],
            "emotion":             final_emotion,
            "intensity":           intensity,
            "valence":             round(vad_mapped["valence"],   4),
            "arousal":             round(vad_mapped["arousal"],   4),
            "dominance":           round(vad_mapped["dominance"], 4),
            "prosody":             prosody,
            "acoustic_emotion":    acoustic_emotion,
            "acoustic_confidence": acoustic_conf,
            "text_emotion":        text_emotion,
            "text_confidence":     text_conf,
        }

        logger.info(
            "  %s [%s] → %s (%.3f) | acoustic=%s(%.2f) text=%s(%.2f)",
            dlg_id, speaker_id, final_emotion, intensity,
            acoustic_emotion, acoustic_conf, text_emotion, text_conf,
        )

        output_dialogues.append(entry)

    return {
        "job_id":         job_id,
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "dialogue_count": len(output_dialogues),
        "dialogues":      output_dialogues,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 02 — Emotion Detection")
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    setup_logging(args.job_id, Path(cfg["paths"]["logs"]))
    logger.info("=== Stage 02 Emotion Detection START  job_id=%s ===", args.job_id)

    manifest = load_manifest(args.job_id, cfg)

    if manifest["stages"][STAGE_KEY]["status"] == "completed":
        logger.info("Already completed. Skipping.")
        return

    jobs_dir    = Path(cfg["paths"]["jobs"]) / args.job_id
    cache_dir   = Path(cfg["paths"]["models_cache"])
    transc_path = jobs_dir / "01_transcription.json"
    output_path = jobs_dir / "02_emotions.json"

    em_cfg              = cfg["stages"]["emotion"]
    acoustic_model_name = em_cfg["acoustic_model"]
    text_model_name     = em_cfg["text_model"]
    device              = get_device(cfg)

    try:
        logger.info("Loading transcription from %s", transc_path)
        with open(transc_path, "r", encoding="utf-8") as fh:
            transcription = json.load(fh)

        dialogues = transcription["dialogues"]
        logger.info(
            "Loaded %d dialogues | speakers: %s | detected_lang: %s",
            len(dialogues),
            transcription.get("speaker_count", "?"),
            transcription.get("detected_language", "?"),
        )

        audio_path = resolve_audio_path(transcription, cfg, args.job_id)

        # Pass 1 — Acoustic (CUDA)
        logger.info("--- Pass 1: Acoustic model ---")
        acoustic_results = acoustic_pass(
            dialogues, audio_path, acoustic_model_name, cache_dir, device
        )

        # Pass 2 — Text (CUDA)
        logger.info("--- Pass 2: Text model ---")
        text_results = text_pass(
            dialogues, text_model_name, cache_dir, device
        )

        # Assemble
        logger.info("--- Assembling output ---")
        output = build_output(
            transcription, acoustic_results, text_results, args.job_id
        )

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        logger.info("02_emotions.json written → %s", output_path)

        mark_stage_complete(args.job_id, manifest, cfg)
        logger.info("=== Stage 02 Emotion Detection COMPLETE ===")

    except Exception as exc:
        logger.exception("Stage 02 failed: %s", exc)
        mark_stage_failed(args.job_id, manifest, cfg)
        raise
    finally:
        release_gpu()


if __name__ == "__main__":
    main()