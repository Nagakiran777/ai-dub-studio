"""
Stage 04 — TTS Synthesis
========================
Synthesizes dubbed speech for each dialogue using OmniVoice.

Reference audio strategy:
  - Primary: slice {job_id}_vocals.wav at dialogue timestamps
             used directly if duration >= min_ref_duration_s (default 1.0s)
  - Fallback: speaker_profiles/speaker_XX.wav
             used when dialogue slice is too short (< 1.0s)

No time stretching — OmniVoice output plays at natural duration.
Emotion is carried naturally via the reference audio slice itself.

Input:
  - jobs/{job_id}/03_translations.json
  - data/audio/{job_id}_vocals.wav        (primary ref audio source)
  - jobs/{job_id}/speaker_profiles/       (fallback ref audio)

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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
import jobpaths as jp          # every job keeps ALL its files inside jobs/<job_id>/

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

STAGE_KEY  = "04_tts"
STAGE_NAME = "04_tts"

OMNIVOICE_MODEL_ID = "drbaph/OmniVoice-bf16"
OMNIVOICE_OUTPUT_SR = 24000

# Vocals WAV sample rate (from Stage 00 BS-Roformer)
VOCALS_SR = 44100


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 04: TTS Synthesis (OmniVoice)")
    parser.add_argument("--job_id",  required=True)
    parser.add_argument("--config",  required=True)
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
    manifest["stages"][STAGE_KEY]["status"]       = "completed"
    manifest["stages"][STAGE_KEY]["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["stages"][STAGE_KEY]["failed_at"]    = None
    save_manifest(manifest, job_id, jobs_dir)
    logger.info(f"Stage {STAGE_KEY} marked COMPLETED.")


def mark_stage_failed(
    manifest: dict, job_id: str, jobs_dir: str, logger: logging.Logger
) -> None:
    manifest["stages"][STAGE_KEY]["status"]    = "failed"
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
# OMNIVOICE LOADER
# ---------------------------------------------------------------------------

def load_omnivoice_model(device: str, logger: logging.Logger):
    from omnivoice import OmniVoice

    logger.info(f"Loading OmniVoice: {OMNIVOICE_MODEL_ID}")
    logger.info(f"Device: {device}")

    if device == "cuda":
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_free  = (torch.cuda.get_device_properties(0).total_memory
                      - torch.cuda.memory_allocated()) / 1e9
        logger.info(f"VRAM total: {vram_total:.1f}GB | free: {vram_free:.1f}GB")
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = OmniVoice.from_pretrained(
        OMNIVOICE_MODEL_ID,
        device_map=device if device == "cuda" else "cpu",
        dtype=dtype,
    )

    if device == "cuda":
        vram_after = torch.cuda.memory_allocated() / 1e6
        logger.info(f"VRAM after OmniVoice load: {vram_after:.1f} MB")

    return model


# ---------------------------------------------------------------------------
# REF AUDIO — dialogue slice from vocals WAV
# ---------------------------------------------------------------------------

def extract_dialogue_ref(
    vocals_audio: torch.Tensor,
    vocals_sr: int,
    start_ms: int,
    end_ms:   int,
    min_ref_duration_s: float,
    logger: logging.Logger,
    dialogue_id: str,
) -> tuple[np.ndarray | None, str]:
    """
    Slice vocals_audio at [start_ms, end_ms], convert to mono float32.

    Returns:
        (audio_np, reason) where audio_np is None if slice is too short.
        reason: "dialogue_slice" or "too_short"
    """
    start_sample = int(start_ms / 1000 * vocals_sr)
    end_sample   = int(end_ms   / 1000 * vocals_sr)
    end_sample   = min(end_sample, vocals_audio.shape[1])
    start_sample = max(0, start_sample)

    if end_sample <= start_sample:
        return None, "too_short"

    slice_tensor = vocals_audio[:, start_sample:end_sample]

    # Stereo → mono
    if slice_tensor.shape[0] > 1:
        slice_tensor = slice_tensor.mean(dim=0, keepdim=True)

    duration_s = slice_tensor.shape[1] / vocals_sr

    if duration_s < min_ref_duration_s:
        logger.debug(
            f"[{dialogue_id}] Dialogue slice {duration_s:.2f}s < "
            f"min {min_ref_duration_s:.1f}s — will use fallback ref"
        )
        return None, "too_short"

    audio_np = slice_tensor.squeeze(0).numpy()
    logger.debug(
        f"[{dialogue_id}] Dialogue slice extracted: {duration_s:.2f}s"
    )
    return audio_np, "dialogue_slice"


# ---------------------------------------------------------------------------
# REF AUDIO — fallback speaker profile
# ---------------------------------------------------------------------------

def resolve_fallback_ref(
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
# PREPARE REF AUDIO for OmniVoice
# ---------------------------------------------------------------------------

def prepare_ref_audio(
    audio_np: np.ndarray,
    src_sr: int,
    logger: logging.Logger,
    dialogue_id: str,
) -> str:
    """
    Resample to OMNIVOICE_OUTPUT_SR (24kHz), write to temp WAV.
    OmniVoice expects 3–10s clips — we trim if over 10s.
    Under 3s clips are used as-is (short dialogue slices, fallback for short profiles).
    Returns path to temp WAV file.
    """
    import librosa

    # Resample to 24kHz if needed
    if src_sr != OMNIVOICE_OUTPUT_SR:
        audio_np = librosa.resample(
            audio_np.astype(np.float32),
            orig_sr=src_sr,
            target_sr=OMNIVOICE_OUTPUT_SR,
        )
        logger.debug(f"[{dialogue_id}] Resampled ref {src_sr}Hz → {OMNIVOICE_OUTPUT_SR}Hz")

    # Trim if over 10s
    max_samples = OMNIVOICE_OUTPUT_SR * 10
    if len(audio_np) > max_samples:
        audio_np = audio_np[:max_samples]
        logger.debug(f"[{dialogue_id}] Trimmed ref audio to 10s")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp")
    sf.write(tmp.name, audio_np.astype(np.float32), OMNIVOICE_OUTPUT_SR)
    return tmp.name


# ---------------------------------------------------------------------------
# REF AUDIO — user-uploaded files + authoritative transcript lookup
# ---------------------------------------------------------------------------

USER_REF_PREFIX = "user:"          # manual_ref_audio = "user:<ref_id>"
MAX_REF_SECONDS = 10.0             # prepare_ref_audio() trims to this

SHORT_REF_FLOOR_S  = 0.15    # anything shorter than this is just a click/breath — not usable as a voice sample
SHORT_REF_REPEATS  = 3       # a ref under the minimum length is repeated this many times (at least)


def repeat_short_ref(audio: np.ndarray, sr: int, text: str | None, min_s: float,
                     gap_ms: int = 60, base_repeats: int = SHORT_REF_REPEATS):
    """Make a too-short reference long enough: the SAME clip repeated `base_repeats` times
    (more only if even that is still under `min_s`), with a tiny gap between copies, and the
    transcript repeated the same number of times so it still describes the audio exactly.
    Returns (audio, text_or_None, repeats)."""
    one = len(audio) / float(sr)
    n = base_repeats
    while n < 10 and (n * one + (n - 1) * gap_ms / 1000.0) < min_s:
        n += 1
    gap = np.zeros(int(sr * gap_ms / 1000.0), dtype=audio.dtype)
    parts = []
    for i in range(n):
        parts.append(audio)
        if i < n - 1:
            parts.append(gap)
    out = np.concatenate(parts)
    txt = " ".join([text.strip()] * n) if text and text.strip() else None
    return out, txt, n


def load_original_text_lookup(jobs_dir: str, job_id: str) -> dict[str, str]:
    """dialogue_id -> ORIGINAL text, read from 01_transcription.json.

    The `text` copy inside 03_translations.json goes stale when the user edits
    the original text in the editor (that edit only touches the transcription
    file). Using the stale copy as ref_text gave OmniVoice a transcript that did
    not match the reference audio, which hurts voice cloning.
    """
    p = Path(jobs_dir) / job_id / "01_transcription.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return {d["id"]: (d.get("text") or "") for d in data.get("dialogues", []) if "id" in d}


def resolve_user_ref(job_dir: Path, manual_ref_id: str, logger: logging.Logger, dialogue_id: str):
    """Load a reference the user uploaded (jobs/<job>/user_refs/<id>.wav).

    Returns (audio_np, sr, ref_text_or_None) or (None, None, None) if missing.
    """
    ref_id = manual_ref_id[len(USER_REF_PREFIX):]
    if not ref_id or not all(c.isalnum() or c in "_-" for c in ref_id):
        logger.warning(f"[{dialogue_id}] Bad user ref id '{manual_ref_id}' — falling back")
        return None, None, None
    refs_dir = job_dir / "user_refs"
    wav_path = refs_dir / f"{ref_id}.wav"
    if not wav_path.exists():
        logger.warning(f"[{dialogue_id}] User ref '{ref_id}' not found — falling back")
        return None, None, None
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception as e:
        logger.warning(f"[{dialogue_id}] Could not read user ref '{ref_id}': {e} — falling back")
        return None, None, None
    ref_text = None
    idx_path = refs_dir / "index.json"
    if idx_path.exists():
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                for r in json.load(f).get("refs", []):
                    if r.get("id") == ref_id:
                        ref_text = (r.get("text") or "").strip() or None
        except Exception:
            pass
    return audio, int(sr), ref_text


# ---------------------------------------------------------------------------
# OmniVoice generation settings (language / num_step / position_temperature)
# ---------------------------------------------------------------------------

# Our language codes (the ones in the editor) that OmniVoice spells differently
OMNI_LANG_ALIASES = {
    "ar": "arb", "ne": "npi", "jw": "jv", "tl": "fil", "iw": "he", "or": "ory",
    "zh-cn": "zh", "zh-tw": "zh", "zh-hans": "zh", "zh-hant": "zh",
}
OMNI_LANG_NAMES = {"mg": "plateau malagasy"}     # code → OmniVoice language NAME when it has no matching code
_LANG_WARNED: set = set()


def resolve_omni_language(code, logger: logging.Logger):
    """Our language code ('te', 'hi', 'zh-CN', 'auto' …) → OmniVoice language id, or None.

    None = language-agnostic mode (OmniVoice guesses from the text, as before). That is used
    for 'auto' and for languages OmniVoice has no id for — a warning is logged once per code.
    """
    c = (code or "").strip().lower().replace("_", "-")
    if not c or c in ("auto", "none", "unknown"):
        return None
    try:
        from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID
    except Exception:
        return OMNI_LANG_ALIASES.get(c, c)       # can't validate here — pass it on, OmniVoice resolves it
    cand = [OMNI_LANG_ALIASES.get(c, c)]
    base = c.split("-")[0]
    cand.append(OMNI_LANG_ALIASES.get(base, base))
    for x in cand:
        if x in LANG_IDS:
            return x
    for x in (c, base):
        nm = OMNI_LANG_NAMES.get(x)
        if nm and nm in LANG_NAME_TO_ID:
            return LANG_NAME_TO_ID[nm]
    if c not in _LANG_WARNED:
        _LANG_WARNED.add(c)
        logger.warning(f"Language '{code}' has no OmniVoice language id — using language-agnostic mode for it.")
    return None


def job_target_language(job_dir: Path, config: dict):
    """The job's target language: job_meta.json (set in the editor), else config.yaml language.target."""
    try:
        with open(Path(job_dir) / "job_meta.json", "r", encoding="utf-8") as f:
            t = json.load(f).get("target_lang")
        if t and str(t).strip().lower() != "auto":
            return t
    except Exception:
        pass
    return (config.get("language", {}) or {}).get("target")


def generation_kwargs(tts_cfg: dict) -> dict:
    """Generation settings taken from config.yaml (stages.tts). A key that is missing or empty is
    simply not passed, so OmniVoice keeps its own default for it."""
    kw = {}
    if tts_cfg.get("num_step") not in (None, ""):
        kw["num_step"] = int(tts_cfg["num_step"])
    if tts_cfg.get("position_temperature") not in (None, ""):
        kw["position_temperature"] = float(tts_cfg["position_temperature"])
    return kw


def omni_generate(model, logger: logging.Logger, **kwargs):
    """model.generate(...) — and if the installed OmniVoice is too old to know `language`,
    say so once and carry on without it instead of failing every line."""
    try:
        return model.generate(**kwargs)
    except TypeError as e:
        if "language" in kwargs and "language" in str(e):
            if "language_unsupported" not in _LANG_WARNED:
                _LANG_WARNED.add("language_unsupported")
                logger.warning("This OmniVoice version does not accept `language` — continuing without it. "
                               "(pip install -U omnivoice to enable it)")
            kwargs.pop("language", None)
            return model.generate(**kwargs)
        raise


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def is_silent(audio: np.ndarray, threshold: float = 1e-4) -> bool:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) < threshold


def check_omnivoice_installed(logger: logging.Logger) -> None:
    try:
        import omnivoice  # noqa: F401
    except ImportError:
        logger.error("OmniVoice not installed!")
        logger.error("Run: pip install omnivoice")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CORE SYNTHESIS LOOP
# ---------------------------------------------------------------------------

def synthesize_all(
    dialogues: list[dict[str, Any]],
    tts_dir: Path,
    profiles_dir: Path,
    vocals_audio: torch.Tensor,
    vocals_sr: int,
    config: dict[str, Any],
    logger: logging.Logger,
    job_dir: Path | None = None,
    text_lookup: dict[str, str] | None = None,
    default_language: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load OmniVoice once, synthesize all dialogues, release GPU.

    For each dialogue:
      1. Try to extract a ref slice from vocals_audio at dialogue timestamps
      2. If slice >= min_ref_duration_s → use as ref (carries emotion naturally)
      3. If slice too short → use speaker_profiles/speaker_XX.wav as fallback
    """
    tts_cfg          = config["stages"]["tts"]
    device           = config["hardware"]["device"] if config["hardware"]["gpu_available"] else "cpu"
    min_ref_duration = float(tts_cfg.get("min_ref_duration_s", 1.0))
    gen_kw           = generation_kwargs(tts_cfg)          # num_step / position_temperature from config.yaml

    model = load_omnivoice_model(device, logger)
    logger.info(
        f"OmniVoice settings: {gen_kw or 'defaults'} | "
        f"language: job target '{default_language or 'auto'}' (a dialogue's own target language wins)"
    )

    dialogues_by_id = {d["id"]: d for d in dialogues}
    manifest_entries: list[dict[str, Any]] = []
    temp_files: list[str] = []
    text_lookup = text_lookup or {}

    def original_text(d: dict[str, Any]) -> str | None:
        # 01_transcription.json is authoritative; 03_translations.json's copy can be stale
        t = text_lookup.get(d.get("id"), d.get("text"))
        return (t or "").strip() or None

    try:
        for dialogue in dialogues:
            d_id       = dialogue["id"]
            speaker_id = dialogue.get("speaker_id", "speaker_00")
            start_ms   = int(dialogue["start_ms"])
            end_ms     = int(dialogue["end_ms"])
            text       = (dialogue.get("translation") or "").strip()

            logger.info(
                f"[{d_id}] speaker={speaker_id} | "
                f"{start_ms}ms–{end_ms}ms | '{text[:60]}'"
            )

            # Skip empty translation
            if not text:
                logger.error(f"[{d_id}] Empty translation — skipping")
                manifest_entries.append({
                    "id": d_id, "speaker_id": speaker_id,
                    "wav_path": None, "success": False,
                    "error": "empty_translation", "tts_text": text,
                })
                continue

            # ── Step 1: manual ref override (user-picked clip from UI) ─────
            ref_np, ref_src_sr, ref_source, ref_text = None, vocals_sr, None, None
            manual_ref_id = dialogue.get("manual_ref_audio")
            if manual_ref_id and str(manual_ref_id).startswith(USER_REF_PREFIX):
                # ── user-uploaded reference file ───────────────────────────
                if job_dir is not None:
                    u_np, u_sr, u_text = resolve_user_ref(job_dir, str(manual_ref_id), logger, d_id)
                    if u_np is not None:
                        ref_np, ref_src_sr, ref_text = u_np, u_sr, u_text
                        ref_source = f"manual_ref:{manual_ref_id}"
                        logger.info(f"[{d_id}] Using user-uploaded ref: {manual_ref_id}")
            elif manual_ref_id:
                ref_dlg = dialogues_by_id.get(manual_ref_id)
                if ref_dlg:
                    ref_np, _reason = extract_dialogue_ref(
                        vocals_audio=vocals_audio, vocals_sr=vocals_sr,
                        start_ms=int(ref_dlg["start_ms"]), end_ms=int(ref_dlg["end_ms"]),
                        min_ref_duration_s=0.3, logger=logger, dialogue_id=d_id,
                    )
                    if ref_np is not None:
                        ref_source = f"manual_ref:{manual_ref_id}"
                        ref_text   = original_text(ref_dlg)
                        logger.info(f"[{d_id}] Using manual ref override: {manual_ref_id}")
                    else:
                        logger.warning(f"[{d_id}] manual_ref_audio '{manual_ref_id}' too short — falling back")
                else:
                    logger.warning(f"[{d_id}] manual_ref_audio '{manual_ref_id}' not found — falling back")

            # ── Step 2: try dialogue's own slice as ref (if no manual ref) ──
            if ref_np is None:
                ref_np, slice_reason = extract_dialogue_ref(
                    vocals_audio=vocals_audio,
                    vocals_sr=vocals_sr,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    min_ref_duration_s=min_ref_duration,
                    logger=logger,
                    dialogue_id=d_id,
                )
                if ref_np is not None:
                    ref_source = "dialogue_slice"
                    # the dialogue's own slice says exactly the dialogue's own text
                    ref_text = original_text(dialogue)
                else:
                    # Under the minimum length: don't give up on the line's own voice —
                    # repeat the short clip (and its transcript) so it becomes long enough.
                    short_np, _ = extract_dialogue_ref(
                        vocals_audio=vocals_audio, vocals_sr=vocals_sr, start_ms=start_ms, end_ms=end_ms,
                        min_ref_duration_s=SHORT_REF_FLOOR_S, logger=logger, dialogue_id=d_id,
                    )
                    if short_np is not None:
                        ref_text = original_text(dialogue)
                        ref_np, ref_text, reps = repeat_short_ref(short_np, vocals_sr, ref_text, min_ref_duration)
                        ref_src_sr = vocals_sr
                        ref_source = f"dialogue_slice_x{reps}"
                        logger.info(f"[{d_id}] Short line ({len(short_np)/vocals_sr:.2f}s): ref repeated {reps}x"
                                    f"{' with transcript repeated' if ref_text else ''}")

            # ── Step 3: fallback to speaker profile if still nothing ────────
            if ref_np is None:
                try:
                    fallback_path = resolve_fallback_ref(
                        speaker_id, profiles_dir, logger, d_id
                    )
                    import librosa
                    ref_np, ref_src_sr = librosa.load(
                        str(fallback_path), sr=None, mono=True
                    )
                    ref_source = f"speaker_profile:{fallback_path.name}"
                    ref_text   = None   # profile = many lines joined; no single transcript
                    logger.info(
                        f"[{d_id}] Using fallback ref: {fallback_path.name}"
                    )
                except RuntimeError as e:
                    logger.error(f"[{d_id}] No ref audio available: {e}")
                    manifest_entries.append({
                        "id": d_id, "speaker_id": speaker_id,
                        "wav_path": None, "success": False,
                        "error": "no_ref_audio", "tts_text": text,
                    })
                    continue

            # prepare_ref_audio() keeps only the first 10 s; the transcript would
            # then describe more audio than OmniVoice receives, so drop it and
            # let OmniVoice transcribe what it is actually given.
            if ref_text and len(ref_np) / float(ref_src_sr) > MAX_REF_SECONDS:
                logger.info(f"[{d_id}] Ref is longer than {MAX_REF_SECONDS:.0f}s and will be trimmed — ref_text dropped")
                ref_text = None

            # ── Step 3: prepare ref for OmniVoice ──────────────────────────
            try:
                prepared_ref = prepare_ref_audio(
                    ref_np, ref_src_sr, logger, d_id
                )
                temp_files.append(prepared_ref)
            except Exception as e:
                logger.error(f"[{d_id}] Failed to prepare ref audio: {e}")
                manifest_entries.append({
                    "id": d_id, "speaker_id": speaker_id,
                    "wav_path": None, "success": False,
                    "error": f"ref_prep_failed: {e}", "tts_text": text,
                })
                continue

            # ── Step 4: OmniVoice synthesis ─────────────────────────────────
            # ref_text = transcript of whichever clip is actually being used as
            # ref_audio (resolved above) — passing it skips OmniVoice's internal
            # Whisper re-transcription of the ref clip (faster, and avoids
            # cloning off a possibly-mistranscribed ref). Left unset for the
            # speaker-profile fallback since that's a concatenation of many
            # different lines with no single accurate transcript.
            # Language of the TEXT being spoken = the dialogue's own target language if the user
            # set one in the editor, otherwise the job's target language.
            d_lang = dialogue.get("target_lang")
            if not d_lang or str(d_lang).strip().lower() == "auto":
                d_lang = default_language
            omni_lang = resolve_omni_language(d_lang, logger)
            logger.info(f"[{d_id}] language={omni_lang or 'auto'} | {gen_kw or 'default settings'}")
            try:
                audio_list = omni_generate(
                    model, logger,
                    text=text,
                    language=omni_lang,
                    ref_audio=prepared_ref,
                    ref_text=ref_text,
                    **gen_kw,
                )
                audio = np.array(audio_list[0], dtype=np.float32)
            except Exception as e:
                logger.error(f"[{d_id}] OmniVoice synthesis failed: {e}")
                manifest_entries.append({
                    "id": d_id, "speaker_id": speaker_id,
                    "wav_path": None, "success": False,
                    "error": f"omnivoice_failed: {e}", "tts_text": text,
                    "ref_source": ref_source,
                })
                continue

            # Validate not silent
            if is_silent(audio):
                logger.error(f"[{d_id}] Output is silent")
                manifest_entries.append({
                    "id": d_id, "speaker_id": speaker_id,
                    "wav_path": None, "success": False,
                    "error": "silent_output", "tts_text": text,
                    "ref_source": ref_source,
                })
                continue

            synthesized_duration_ms = len(audio) / OMNIVOICE_OUTPUT_SR * 1000
            logger.info(
                f"[{d_id}] Synthesized: {synthesized_duration_ms:.0f}ms | "
                f"ref={ref_source} | RMS={np.sqrt(np.mean(audio**2)):.4f}"
            )

            # Save WAV at natural duration — no time stretching
            wav_path = tts_dir / f"{d_id}.wav"
            sf.write(str(wav_path), audio, OMNIVOICE_OUTPUT_SR, subtype="PCM_16")
            logger.info(f"[{d_id}] Saved: {wav_path.name}")

            manifest_entries.append({
                "id":                      d_id,
                "speaker_id":              speaker_id,
                "wav_path":                f"04_tts/{wav_path.name}",   # relative to the job folder (portable)
                "original_duration_ms":    float(end_ms - start_ms),
                "synthesized_duration_ms": round(synthesized_duration_ms, 1),
                "ref_source":              ref_source,
                "tts_text":                text,
                "language":                omni_lang,        # what OmniVoice was told (None = auto)
                "success":                 True,
            })

    finally:
        for tmp in temp_files:
            try:
                os.unlink(tmp)
            except Exception:
                pass
        logger.info("Releasing OmniVoice from GPU...")
        del model
        release_gpu(logger)

    return manifest_entries


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    project_root = Path(args.config).parent.resolve()
    os.chdir(project_root)

    jobs_dir = config["paths"]["jobs"]
    jp.migrate_legacy_job(project_root, jobs_dir, args.job_id)
    job_dir_ = jp.job_dir(project_root, jobs_dir, args.job_id)
    logs_dir = str(jp.logs_dir(job_dir_))

    logger = setup_logging(args.job_id, logs_dir)
    logger.info("=" * 60)
    logger.info("Stage 04 — TTS Synthesis (OmniVoice)")
    logger.info(f"Job ID  : {args.job_id}")
    logger.info(f"Model   : {OMNIVOICE_MODEL_ID}")
    logger.info(f"Ref mode: dialogue slice from _vocals.wav (fallback: speaker profile)")
    logger.info("=" * 60)

    check_omnivoice_installed(logger)

    # Restart-safe
    manifest = load_manifest(args.job_id, jobs_dir)
    if manifest["stages"][STAGE_KEY]["status"] == "completed":
        logger.info("Already completed. Skipping.")
        return

    if manifest["stages"]["03_translation"]["status"] != "completed":
        raise RuntimeError("Stage 03_translation not completed. Run Stage 03 first.")

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

    # Load vocals WAV for ref audio slicing
    vocals_path = jp.vocals_wav(job_dir_)
    if not vocals_path.exists():
        logger.warning(
            f"_vocals.wav not found at {vocals_path} — "
            "all dialogues will use speaker profile fallback"
        )
        vocals_audio = None
        vocals_sr    = None
    else:
        logger.info(f"Loading vocals WAV for ref slicing: {vocals_path}")
        vocals_audio, vocals_sr = torchaudio.load(str(vocals_path))
        logger.info(
            f"Vocals: shape={vocals_audio.shape}, sr={vocals_sr}"
        )

    # Prepare directories
    tts_dir = Path(jobs_dir) / args.job_id / "04_tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    profiles_dir = Path(jobs_dir) / args.job_id / "speaker_profiles"
    if not profiles_dir.exists():
        logger.warning(
            f"Speaker profiles dir not found: {profiles_dir} — "
            "fallback ref will fail if dialogue slices are too short"
        )

    profiles_available = sorted(profiles_dir.glob("speaker_*.wav")) if profiles_dir.exists() else []
    logger.info(f"Speaker profiles available: {[p.name for p in profiles_available]}")

    # If vocals not loaded, set dummy tensor so extract_dialogue_ref always returns None
    if vocals_audio is None:
        vocals_audio = torch.zeros(1, 1)
        vocals_sr    = VOCALS_SR

    # Run synthesis
    try:
        manifest_entries = synthesize_all(
            dialogues=dialogues,
            tts_dir=tts_dir,
            profiles_dir=profiles_dir,
            vocals_audio=vocals_audio,
            vocals_sr=vocals_sr,
            config=config,
            logger=logger,
            job_dir=Path(jobs_dir) / args.job_id,
            text_lookup=load_original_text_lookup(jobs_dir, args.job_id),
            default_language=job_target_language(Path(jobs_dir) / args.job_id, config),
        )
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        mark_stage_failed(manifest, args.job_id, jobs_dir, logger)
        raise
    finally:
        release_gpu(logger)

    # Write tts_manifest.json
    successful = [e for e in manifest_entries if e.get("success")]
    failed     = [e for e in manifest_entries if not e.get("success")]

    tts_manifest = {
        "job_id":          args.job_id,
        "created_at":      datetime.now(timezone.utc).isoformat(),
        "model":           OMNIVOICE_MODEL_ID,
        "sample_rate":     OMNIVOICE_OUTPUT_SR,
        "dialogue_count":  len(manifest_entries),
        "success_count":   len(successful),
        "failed_count":    len(failed),
        "dialogues":       manifest_entries,
    }

    manifest_path = tts_dir / "tts_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(tts_manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"tts_manifest.json written: {manifest_path}")

    logger.info("=" * 60)
    logger.info("Stage 04 Summary")
    logger.info(f"  Total:      {len(manifest_entries)}")
    logger.info(f"  Successful: {len(successful)}")
    logger.info(f"  Failed:     {len(failed)}")
    if failed:
        logger.warning(f"  Failed IDs: {[e['id'] for e in failed]}")
    logger.info("=" * 60)

    mark_stage_complete(manifest, args.job_id, jobs_dir, logger)
    logger.info("Stage 04 complete.")


if __name__ == "__main__":
    main()