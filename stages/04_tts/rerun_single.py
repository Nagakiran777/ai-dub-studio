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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
import jobpaths as jp          # every job keeps ALL its files inside jobs/<job_id>/

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


USER_REF_PREFIX = "user:"          # manual_ref_audio = "user:<ref_id>"
MAX_REF_SECONDS = 10.0             # prepare_ref_audio() trims to this


def load_original_text_lookup(jobs_dir: str, job_id: str) -> dict[str, str]:
    """dialogue_id -> ORIGINAL text from 01_transcription.json (authoritative —
    the copy in 03_translations.json goes stale when the user edits the text)."""
    p = Path(jobs_dir) / job_id / "01_transcription.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return {d["id"]: (d.get("text") or "") for d in data.get("dialogues", []) if "id" in d}


def resolve_user_ref(job_dir: Path, manual_ref_id: str, logger, dialogue_id: str):
    """Load a reference the user uploaded (jobs/<job>/user_refs/<id>.wav).
    Returns (audio_np, sr, ref_text_or_None) or (None, None, None) if missing."""
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


def resolve_ref(dialogue: dict, dialogues_by_id: dict, vocals_audio, vocals_sr, profiles_dir: Path,
                 min_ref_duration: float, logger, job_dir: Path | None = None,
                 text_lookup: dict | None = None) -> tuple[np.ndarray, int, str, str | None]:
    """Same 3-tier resolution as run.py's synthesize_all: manual override →
    dialogue's own slice → speaker profile. Returns (ref_np, ref_sr, ref_source, ref_text)."""
    d_id = dialogue["id"]
    speaker_id = dialogue.get("speaker_id", "speaker_00")
    text_lookup = text_lookup or {}

    def original_text(d: dict) -> str | None:
        t = text_lookup.get(d.get("id"), d.get("text"))
        return (t or "").strip() or None

    manual_ref_id = dialogue.get("manual_ref_audio")
    if manual_ref_id and str(manual_ref_id).startswith(USER_REF_PREFIX):
        if job_dir is not None:
            u_np, u_sr, u_text = resolve_user_ref(job_dir, str(manual_ref_id), logger, d_id)
            if u_np is not None:
                logger.info(f"[{d_id}] Using user-uploaded ref: {manual_ref_id}")
                return u_np, u_sr, f"manual_ref:{manual_ref_id}", u_text
    elif manual_ref_id:
        ref_dlg = dialogues_by_id.get(manual_ref_id)
        if ref_dlg:
            ref_np = extract_dialogue_ref(vocals_audio, vocals_sr, int(ref_dlg["start_ms"]), int(ref_dlg["end_ms"]),
                                           min_ref_duration_s=0.3, logger=logger, dialogue_id=d_id)
            if ref_np is not None:
                logger.info(f"[{d_id}] Using manual ref override: {manual_ref_id}")
                return ref_np, vocals_sr, f"manual_ref:{manual_ref_id}", original_text(ref_dlg)
            logger.warning(f"[{d_id}] manual_ref_audio '{manual_ref_id}' too short — falling back")
        else:
            logger.warning(f"[{d_id}] manual_ref_audio '{manual_ref_id}' not found — falling back")

    ref_np = extract_dialogue_ref(vocals_audio, vocals_sr, int(dialogue["start_ms"]), int(dialogue["end_ms"]),
                                   min_ref_duration_s=min_ref_duration, logger=logger, dialogue_id=d_id)
    if ref_np is not None:
        return ref_np, vocals_sr, "dialogue_slice", original_text(dialogue)

    # Under the minimum length: repeat the short clip (and its transcript) so it is long enough.
    short_np = extract_dialogue_ref(vocals_audio, vocals_sr, int(dialogue["start_ms"]), int(dialogue["end_ms"]),
                                    min_ref_duration_s=SHORT_REF_FLOOR_S, logger=logger, dialogue_id=d_id)
    if short_np is not None:
        rep_np, rep_text, reps = repeat_short_ref(short_np, vocals_sr, original_text(dialogue), min_ref_duration)
        logger.info(f"[{d_id}] Short line ({len(short_np)/vocals_sr:.2f}s): ref repeated {reps}x")
        return rep_np, vocals_sr, f"dialogue_slice_x{reps}", rep_text

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

    jobs_dir  = config["paths"]["jobs"]
    jp.migrate_legacy_job(project_root, jobs_dir, args.job_id)
    job_dir_  = jp.job_dir(project_root, jobs_dir, args.job_id)
    logs_dir  = str(jp.logs_dir(job_dir_))
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

    vocals_path = jp.vocals_wav(job_dir_)
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
                dialogue, dialogues_by_id, vocals_audio, vocals_sr, profiles_dir, min_ref_duration, logger,
                job_dir=Path(jobs_dir) / args.job_id,
                text_lookup=load_original_text_lookup(jobs_dir, args.job_id),
            )
        except RuntimeError as e:
            logger.error(f"[{args.dialogue_id}] No ref audio available: {e}")
            entry = {"id": args.dialogue_id, "speaker_id": speaker_id, "wav_path": None,
                     "success": False, "error": "no_ref_audio", "tts_text": text}
        else:
            # prepare_ref_audio() keeps only the first 10 s; a transcript of more
            # audio than OmniVoice receives would mismatch, so drop it then.
            if ref_text and len(ref_np) / float(ref_sr) > MAX_REF_SECONDS:
                logger.info(f"[{args.dialogue_id}] Ref longer than {MAX_REF_SECONDS:.0f}s will be trimmed — ref_text dropped")
                ref_text = None
            prepared_ref = prepare_ref_audio(ref_np, ref_sr, logger, args.dialogue_id)
            try:
                d_lang = dialogue.get("target_lang")
                if not d_lang or str(d_lang).strip().lower() == "auto":
                    d_lang = job_target_language(job_dir_, config)
                omni_lang = resolve_omni_language(d_lang, logger)
                gen_kw = generation_kwargs(tts_cfg)
                logger.info(f"[{args.dialogue_id}] language={omni_lang or 'auto'} | {gen_kw or 'default settings'}")
                audio_list = omni_generate(model, logger, text=text, language=omni_lang, ref_audio=prepared_ref,
                                           ref_text=ref_text, **gen_kw)
                audio = np.array(audio_list[0], dtype=np.float32)
                if is_silent(audio):
                    raise RuntimeError("silent_output")
                synthesized_duration_ms = len(audio) / OMNIVOICE_OUTPUT_SR * 1000
                wav_path = tts_dir / f"{args.dialogue_id}.wav"
                sf.write(str(wav_path), audio, OMNIVOICE_OUTPUT_SR, subtype="PCM_16")
                logger.info(f"[{args.dialogue_id}] Re-synthesized: {synthesized_duration_ms:.0f}ms | ref={ref_source}")
                entry = {
                    "id": args.dialogue_id, "speaker_id": speaker_id, "wav_path": f"04_tts/{wav_path.name}",   # relative to the job folder (portable)
                    "original_duration_ms": float(dialogue["end_ms"] - dialogue["start_ms"]),
                    "synthesized_duration_ms": round(synthesized_duration_ms, 1),
                    "ref_source": ref_source, "tts_text": text, "language": omni_lang, "success": True,
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
