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
    → silence vocals over the ORIGINAL bar (start_ms..original_end_ms); the dub is placed independently at its own dub_start_ms, trimmed/sped per dialogue, and volume-matched to that line
    → overlay TTS audio at start_ms
  WITHOUT TTS (failed/skipped):
    → vocals track untouched (original voice stays in mix)
    → no TTS placed

Inputs:
  jobs/{job_id}/04_tts/tts_manifest.json
  jobs/{job_id}/03_translations.json
  jobs/<job_id>/audio/no_vocals.wav
  jobs/<job_id>/audio/vocals.wav
  jobs/<job_id>/source/<video>

Output:
  jobs/<job_id>/output/{job_id}_dubbed.mkv|mp4
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
import tempfile
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import yaml

# Same speed-matching formula the web preview uses (stages/_shared/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
from dub_speed import effective_speed, dub_geometry, dub_parts_geometry
import jobpaths as jp          # every job keeps ALL its files inside jobs/<job_id>/


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

def _find_tts_wav(job_dir: Path, stored: str):
    """TTS clip for a manifest entry. `stored` may be job-relative ("04_tts/x.wav"), an old
    project-relative / absolute path from another machine — find the real file either way."""
    return jp.resolve_job_file(Path(job_dir), stored, "04_tts")


def find_video_file(job_id: str, project_root: Path, logger: logging.Logger) -> Path:
    """The job's video lives in jobs/<job_id>/source/ (older jobs: the path in job_meta.json)."""
    jd = project_root / "jobs" / job_id
    found = jp.find_source_video(jd)
    if found is not None:
        logger.info(f"Video: {found}")
        return found
    raise RuntimeError(f"Original video not found for job '{job_id}' (looked in {jp.source_dir(jd)}).")


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


SPEED_METHOD = "atempo"        # set from config: assembly.speed_method — atempo | rubberband | librosa


def _atempo_chain(speed: float) -> str:
    """ffmpeg's atempo only accepts 0.5–2.0 per stage; chain stages for anything beyond."""
    stages, s = [], float(speed)
    while s > 2.0:
        stages.append(2.0); s /= 2.0
    while s < 0.5:
        stages.append(0.5); s /= 0.5
    stages.append(s)
    return ",".join(f"atempo={x:.6f}" for x in stages)


def _ffmpeg_stretch(audio: np.ndarray, sr: int, speed: float, method: str) -> np.ndarray:
    flt = _atempo_chain(speed) if method == "atempo" else \
        f"rubberband=tempo={speed:.6f}:pitch=1:formant=preserved:transients=smooth:pitchq=quality"
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(src), audio, sr, subtype="FLOAT")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-filter:a", flt,
                            "-c:a", "pcm_f32le", str(dst)], capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists():
            raise RuntimeError((r.stderr or "ffmpeg failed").strip().splitlines()[-1][:160])
        out, _ = sf.read(str(dst), dtype="float32", always_2d=False)
    return np.asarray(out, dtype=np.float32)


def change_speed(audio: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Change duration while preserving pitch.

    ``speed > 1`` makes the clip faster/shorter; ``speed < 1`` slower/longer.

    The default engine is ffmpeg's ``atempo`` (WSOLA time-domain stretching) — the same kind
    of algorithm the browser uses for the preview, so a sped-up line sounds like it did in
    the preview. librosa's phase vocoder (the old engine) smears speech into a robotic,
    phasey sound, so it is only the last-resort fallback now. ``rubberband`` is available
    when your ffmpeg has it. Configure with assembly.speed_method.
    """
    speed = float(speed)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"speed must be a finite positive number; got {speed!r}")
    audio = audio.astype(np.float32, copy=False)
    if abs(speed - 1.0) < 1e-6 or audio.size == 0:
        return audio.copy()
    order = [SPEED_METHOD] + [m for m in ("atempo", "rubberband") if m != SPEED_METHOD]
    if shutil.which("ffmpeg"):
        for method in order:
            if method == "librosa":
                break
            try:
                return _ffmpeg_stretch(audio, sr, speed, method)
            except Exception as exc:
                logging.getLogger("assembly").warning(f"{method} time-stretch failed ({exc}) — trying the next method")
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
        wav_path = _find_tts_wav(project_root, wav_path_str)
        if wav_path is None:
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
# Per-dialogue volume matching
# ---------------------------------------------------------------------------

def active_rms(audio: np.ndarray, sr: int, frame_ms: int = 25) -> float:
    """RMS of the speaking part of a clip.

    Frames much quieter than the clip's loudest frame (pauses, breaths,
    trailing silence) are ignored, so a clip with long gaps isn't judged
    quieter than it really sounds when someone is talking.
    """
    if audio.size == 0:
        return 0.0
    n = max(1, int(sr * frame_ms / 1000))
    if audio.size < n:
        return compute_rms(audio)
    m = audio.size // n
    frames = audio[: m * n].reshape(m, n).astype(np.float64)
    fr = np.sqrt((frames ** 2).mean(axis=1))
    peak = float(fr.max())
    if peak < RMS_FLOOR:
        return 0.0
    keep = fr >= 0.1 * peak
    return float(np.sqrt((frames[keep] ** 2).mean()))


def dialogue_volume_gain(
    ref_slice: np.ndarray,
    tts_audio: np.ndarray,
    sr: int,
    gain_min: float,
    gain_max: float,
) -> tuple[float, str]:
    """Gain that makes THIS dub as loud as THIS original dialogue.

    The reference is the original (vocals) audio of that one dialogue — not
    the whole video — so a whispered line stays a whisper and a shout stays a
    shout. Returns (gain, note).
    """
    ref_rms = active_rms(ref_slice, sr)
    tts_rms = active_rms(tts_audio, sr)
    if ref_rms < 1e-4:
        return 1.0, "original slice silent — gain 1.0"
    if tts_rms < RMS_FLOOR:
        return 1.0, "dub silent — gain 1.0"
    gain = float(np.clip(ref_rms / tts_rms, gain_min, gain_max))
    note = f"orig RMS {ref_rms:.4f} / dub RMS {tts_rms:.4f}"
    if float(np.max(np.abs(tts_audio))) * gain > 0.98:
        note += " (only the loudest peaks are soft-limited — the level is kept)"
    return gain, note


def soft_limit(x: np.ndarray, knee: float = 0.9) -> np.ndarray:
    """Keep the gain EXACT: instead of turning the whole clip down when a peak would clip,
    only the few samples above the knee are rounded off smoothly."""
    over = np.abs(x) > knee
    if not over.any():
        return x
    y = x.copy()
    a = np.abs(x[over])
    y[over] = np.sign(x[over]) * (knee + (1.0 - knee) * np.tanh((a - knee) / (1.0 - knee)))
    return y


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
    match_volume: bool = False,
    gain_min: float = 0.1,
    gain_max: float = 8.0,
    layer_gains: tuple = (1.0, 1.0, 1.0),     # (no_vocals, vocals, dub) — the Mixer sliders
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

        # ── No TTS (or the user removed this dub) — keep original voice ──────
        if success and wav_path_str and trans_entry.get("dub_deleted"):
            logger.info(f"[{dlg_id}] Dub deleted by user — keeping original voice.")
            kept_count += 1
            continue

        if not success or not wav_path_str:
            logger.debug(f"[{dlg_id}] No TTS — keeping original voice in mix.")
            kept_count += 1
            continue

        # ── Load TTS WAV ──────────────────────────────────────────────────────
        wav_path = _find_tts_wav(project_root, wav_path_str)
        if wav_path is None:
            logger.warning(f"[{dlg_id}] TTS WAV missing: {wav_path_str} — keeping original.")
            kept_count += 1
            continue

        try:
            tts_full = load_wav_at_sr(wav_path, sr, logger)
        except Exception as exc:
            logger.warning(f"[{dlg_id}] Load failed ({exc}) — keeping original.")
            kept_count += 1
            continue

        # ── Dub geometry (shared with the web UI) ─────────────────────────────
        # The DUB is independent of the ORIGINAL bar. A dub may be cut into several
        # pieces (split in the Dubbed timeline); every piece has its own position,
        # trim, speed and can be deleted — all pieces play parts of the same clip.
        natural_ms = entry.get("synthesized_duration_ms") or (len(tts_full) / sr * 1000.0)
        parts = [p for p in dub_parts_geometry(natural_ms, trans_entry, start_ms, original_end_ms)
                 if not p["deleted"]]
        if not parts:
            logger.info(f"[{dlg_id}] Dub deleted by user — keeping original voice.")
            kept_count += 1
            continue

        # Silence window = the ORIGINAL dialogue bar only.
        silence_end_ms = original_end_ms
        silence_end    = max(slot_start, min(ms_to_samples(silence_end_ms, sr), total_samples))

        # ── Loudness ──────────────────────────────────────────────────────────
        # NO automatic normalization: every dub plays at its natural level, multiplied only by
        # what the user set — the dialogue's own volume (dub_volume) and the Dubbed layer slider.
        # That is exactly what the preview plays, so the export sounds like the preview.
        # (assembly.match_dialogue_volume: true brings back the old automatic matching.)
        dub_vol = trans_entry.get("dub_volume")
        dub_vol = 1.0 if dub_vol is None else max(0.0, min(4.0, float(dub_vol)))
        line_gain = 1.0
        if match_volume:
            line_gain, gain_note = dialogue_volume_gain(
                vocals_audio[slot_start:silence_end], tts_full, sr, gain_min, gain_max
            )
            logger.info(f"[{dlg_id}] Volume ×{line_gain:.2f} ({gain_note})")

        placed = 0
        for part in parts:
            i0 = max(0, int(part["trim_in_ms"]  / 1000.0 * sr))
            i1 = min(len(tts_full), int(part["trim_out_ms"] / 1000.0 * sr))
            if i1 - i0 < int(0.05 * sr):
                logger.warning(f"[{dlg_id}/{part['part_id']}] Trimmed piece is empty — skipped.")
                continue
            piece = tts_full[i0:i1]
            if i0 > 0 or i1 < len(tts_full):
                logger.info(f"[{dlg_id}/{part['part_id']}] Trim {part['trim_in_ms']:.0f}–{part['trim_out_ms']:.0f}ms of {natural_ms:.0f}ms")

            speed = part["speed"]
            if abs(speed - 1.0) > 1e-3:
                before_ms = len(piece) / sr * 1000.0
                piece = change_speed(piece, sr, speed)
                logger.info(f"[{dlg_id}/{part['part_id']}] Speed {speed:.2f}x | {before_ms:.0f}ms → {len(piece) / sr * 1000.0:.0f}ms")

            dub_start_ms = part["dub_start_ms"]
            dub_slot     = max(0, min(ms_to_samples(dub_start_ms, sr), total_samples))
            piece = piece * tts_gain * line_gain * dub_vol
            if match_volume:
                piece = soft_limit(piece)
            piece = apply_fade_in(piece, fade_samples)
            piece = apply_fade_out(piece, fade_samples)
            place_end = min(total_samples, dub_slot + len(piece))
            if place_end <= dub_slot:
                logger.warning(f"[{dlg_id}/{part['part_id']}] Placed past the end of the audio — skipped.")
                continue
            mix_tts[dub_slot:place_end] += piece[: place_end - dub_slot]
            placed += 1
            logger.info(f"[{dlg_id}/{part['part_id']}] ✓ placed {len(piece)/sr*1000:.0f}ms at {dub_start_ms:.0f}ms")

        if placed == 0:
            logger.warning(f"[{dlg_id}] Nothing could be placed — keeping original voice.")
            kept_count += 1
            continue

        # ── Silence the ORIGINAL voice under the original bar (with fades) ────
        pre_fade_start = max(0, slot_start - fade_samples)
        pre_fade_len   = slot_start - pre_fade_start
        if pre_fade_len > 0:
            mix_vocals[pre_fade_start:slot_start] *= np.linspace(1.0, 0.0, pre_fade_len, dtype=np.float32)
        mix_vocals[slot_start:silence_end] = 0.0
        post_fade_end = min(total_samples, silence_end + fade_samples)
        post_fade_len = post_fade_end - silence_end
        if post_fade_len > 0:
            mix_vocals[silence_end:post_fade_end] *= np.linspace(0.0, 1.0, post_fade_len, dtype=np.float32)

        logger.info(f"[{dlg_id}] ✓ Replaced | {placed} piece(s) | original silenced {start_ms:.0f}–{silence_end_ms:.0f}ms")
        replaced_count += 1

    # ── Sum 3 tracks ──────────────────────────────────────────────────────────
    logger.info(
        f"Mix summary — replaced: {replaced_count} | "
        f"kept original: {kept_count} | skipped: {skipped_count}"
    )
    logger.info("Summing 3 tracks: no_vocals + vocals(patched) + tts ...")

    g_bg, g_voc, g_dub = layer_gains
    if (g_bg, g_voc, g_dub) != (1.0, 1.0, 1.0):
        logger.info(f"Layer volumes — no_vocals ×{g_bg:.2f} | vocals ×{g_voc:.2f} | dub ×{g_dub:.2f}")
    final = mix_background * g_bg + mix_vocals * g_voc + mix_tts * g_dub
    np.clip(final, -1.0, 1.0, out=final)
    logger.info("Assembly complete.")
    return final


# ---------------------------------------------------------------------------
# ffmpeg mux
# ---------------------------------------------------------------------------

# (display name, ISO 639-2 code) for the track title/language tag of the dub
LANG_TAGS = {
    "en": ("English", "eng"), "te": ("Telugu", "tel"), "hi": ("Hindi", "hin"), "ta": ("Tamil", "tam"),
    "kn": ("Kannada", "kan"), "ml": ("Malayalam", "mal"), "mr": ("Marathi", "mar"), "bn": ("Bengali", "ben"),
    "gu": ("Gujarati", "guj"), "pa": ("Punjabi", "pan"), "ur": ("Urdu", "urd"), "or": ("Odia", "ori"),
    "as": ("Assamese", "asm"), "ne": ("Nepali", "nep"), "si": ("Sinhala", "sin"),
    "es": ("Spanish", "spa"), "fr": ("French", "fra"), "de": ("German", "deu"), "it": ("Italian", "ita"),
    "pt": ("Portuguese", "por"), "ru": ("Russian", "rus"), "ja": ("Japanese", "jpn"), "ko": ("Korean", "kor"),
    "zh": ("Chinese", "zho"), "ar": ("Arabic", "ara"), "tr": ("Turkish", "tur"), "id": ("Indonesian", "ind"),
    "vi": ("Vietnamese", "vie"), "th": ("Thai", "tha"), "pl": ("Polish", "pol"), "nl": ("Dutch", "nld"),
    "sv": ("Swedish", "swe"), "uk": ("Ukrainian", "ukr"), "fa": ("Persian", "fas"), "he": ("Hebrew", "heb"),
    "el": ("Greek", "ell"), "ms": ("Malay", "msa"), "fil": ("Filipino", "fil"), "sw": ("Swahili", "swa"),
}

# Codecs the .mp4 container can hold. Anything else (opus/vorbis/flac/dts audio, srt/ass/PGS
# subtitles, vp8/wmv video, attached fonts …) goes in .mkv, which can hold everything.
MP4_VIDEO = {"h264", "hevc", "mpeg4", "av1", "vp9"}
MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}
MP4_SUBS  = {"mov_text", "tx3g"}


def probe_streams(path: Path) -> list[dict]:
    if not shutil.which("ffprobe"):
        return []
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_type,codec_name,pix_fmt",
             "-of", "json", str(path)], capture_output=True, text=True, timeout=120)
        return json.loads(r.stdout or "{}").get("streams", [])
    except Exception:
        return []


def choose_container(streams: list[dict]) -> str:
    """'mp4' when everything in the source fits it (smallest, most compatible), else 'mkv'."""
    if not streams:
        return "mkv"
    for s in streams:
        t, c = s.get("codec_type"), s.get("codec_name")
        if t == "video" and c not in MP4_VIDEO: return "mkv"
        if t == "audio" and c not in MP4_AUDIO: return "mkv"
        if t == "subtitle" and c not in MP4_SUBS: return "mkv"
        if t == "attachment": return "mkv"
    return "mp4"


def _run_ffmpeg(cmd: list[str], logger: logging.Logger) -> bool:
    logger.info(f"Running ffmpeg: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        logger.warning(f"ffmpeg failed (exit {r.returncode}):\n{(r.stderr or '')[-1500:]}")
        return False
    return True


def mux_video(
    video_path: Path,
    audio_path: Path,
    output_base: Path,            # path WITHOUT extension; the container picks .mp4 or .mkv
    logger: logging.Logger,
    target_lang: str = "",
) -> Path:
    """Build the final video: EVERYTHING from the original (video, every audio track,
    every subtitle, chapters, attached fonts) is kept untouched and the AI dub is
    appended as one extra audio track, flagged as the default one.

    Streams are copied, not re-encoded, so quality is unchanged and the file is only
    as much bigger as the added dub track. .mp4 when the source fits it, otherwise .mkv.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found.")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    streams = probe_streams(video_path)
    container = choose_container(streams)
    n_audio = sum(1 for s in streams if s.get("codec_type") == "audio")
    n_subs  = sum(1 for s in streams if s.get("codec_type") == "subtitle")
    name, iso = LANG_TAGS.get((target_lang or "").lower().split("-")[0], ((target_lang or "Dubbed").upper() or "Dubbed", (target_lang or "und")[:3]))
    title = f"{name} (AI dubbed)"
    out = output_base.with_suffix(f".{container}")
    logger.info(
        f"Source has {n_audio} audio track(s) and {n_subs} subtitle track(s) → "
        f"output container .{container}; adding '{title}' as audio track #{n_audio + 1}"
    )

    # Put the AI dubbed audio FIRST.
    # This improves compatibility with players that automatically
    # select the first audio stream.

    maps = [
        "-map", "0:v",
        "-map", "1:a:0",   # Audio 0: AI dubbed audio
        "-map", "0:a?",    # Remaining audio: original tracks
        "-map", "0:s?"
    ]

    if container == "mkv":
        maps += ["-map", "0:t?"]

    # Dubbed audio metadata (first audio stream)
    tags = [
        "-c:a:0", "aac",
        "-b:a:0", "192k",
        "-metadata:s:a:0", f"title={title}",
        "-metadata:s:a:0", f"language={iso}",
        "-metadata:s:a:0", f"handler_name={title}",
    ]

    # Set dubbed audio as DEFAULT
    disp = [
        "-disposition:a:0", "default"
    ]

    # Disable default on all original audio tracks
    for i in range(n_audio):
        output_index = i + 1
        disp += [
            f"-disposition:a:{output_index}", "0"
        ]
    tail = ["-movflags", "+faststart"] if container == "mp4" else []

    base = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path), *maps, "-map_chapters", "0"]
    attempts = [
        # 1) copy everything (normal case)
        base + ["-c", "copy", *tags, *disp, *tail, str(out)],
        # 2) the picture can't be copied → re-encode only the picture, keep audio + subtitles
        base + ["-c", "copy", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                *tags, *disp, *tail, str(out)],
    ]
    ok = False
    for i, cmd in enumerate(attempts, 1):
        if _run_ffmpeg(cmd, logger):
            ok = True; break
        logger.warning(f"Mux attempt {i} failed" + (" — trying a more tolerant one" if i < len(attempts) else ""))
    if not ok:
        # 3) last resort: picture + dub only (the old behaviour) so an export never fails outright
        out = output_base.with_suffix(".mp4")
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
        if not _run_ffmpeg(cmd, logger):
            raise RuntimeError("ffmpeg mux failed.")
        logger.warning("Fell back to picture + dub only — other audio tracks and subtitles were NOT kept.")
        container = "mp4"

    # only one finished file per job
    for ext in ("mp4", "mkv"):
        other = output_base.with_suffix(f".{ext}")
        if other != out and other.exists():
            try: other.unlink()
            except Exception: pass
    logger.info(f"ffmpeg mux complete → {out}")
    return out


def make_assembled_preview(video_path: Path, audio_path: Path, job_dir: Path, logger: logging.Logger) -> None:
    """A browser-playable copy of the REAL result (original picture + the finished dub
    audio) for the editor's player. Browsers can't switch audio tracks or play every
    container, so the full multi-track file is for downloading and this one is for viewing."""
    try:
        out = job_dir / "assembled_preview.mp4"
        tmp = job_dir / "assembled_preview.part.mp4"
        src = job_dir / "preview.mp4"                      # made by the web app for browser-unfriendly sources
        if not src.exists():
            src = video_path
        v = next((s for s in probe_streams(src) if s.get("codec_type") == "video"), {})
        if v.get("codec_name") == "h264" and v.get("pix_fmt") in ("yuv420p", "yuvj420p"):
            vargs = ["-c:v", "copy"]
        else:
            vargs = ["-vf", "scale=-2:'min(720,ih)',format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
        cmd = ["ffmpeg", "-y", "-i", str(src), "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
               *vargs, "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(tmp)]
        if _run_ffmpeg(cmd, logger) and tmp.exists():
            tmp.replace(out)
            logger.info(f"Assembled preview ready → {out}")
        else:
            tmp.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning(f"Could not build the assembled preview (the export itself is fine): {exc}")


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

    job_dir_ = project_root / "jobs" / job_id
    jp.migrate_legacy_job(project_root, "jobs", job_id)
    log_path = jp.logs_dir(job_dir_) / f"{job_id}_05_assembly.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger   = setup_logging(log_path)
    logger.info(f"=== Stage 05 — Assembly  |  job_id={job_id} ===")
    logger.info(f"Project root: {project_root}")

    t_start       = time.time()
    manifest_path = project_root / "jobs" / job_id / "manifest.json"

    try:
        config       = load_config(config_path)
        assembly_cfg = config.get("stages", {}).get("assembly", {})
        sr: int      = int(assembly_cfg.get("sample_rate", WORKING_SR))
        global SPEED_METHOD
        SPEED_METHOD = str(assembly_cfg.get("speed_method", "atempo")).strip().lower()
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

        # ── Track 1: no_vocals (background BGM+SFX) ──────────────────────────
        no_vocals_path  = jp.no_vocals_wav(job_dir_)
        full_audio_path = jp.original_wav(job_dir_)

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
        vocals_path = jp.vocals_wav(job_dir_)
        if vocals_path.exists():
            logger.info(f"Track 2 (vocals): {vocals_path.name}")
            vocals_audio = load_wav_at_sr(vocals_path, sr, logger)
        else:
            logger.warning(
                "vocals.wav not found — non-verbal sounds will be missing."
            )
            vocals_audio = np.zeros(total_samples, dtype=np.float32)

        # No automatic loudness normalization of any kind: what you hear in the preview is what
        # gets exported (the old global RMS gain was computed and then discarded — removed).
        tts_gain = 1.0

        # Mixer sliders (jobs/<id>/mix.json): level of the background, vocals and dubbed layers
        layer_gains = (1.0, 1.0, 1.0)
        try:
            mixp = job_dir_ / "mix.json"
            if mixp.exists():
                mx = load_json(mixp)
                lg = lambda k: max(0.0, min(4.0, float(mx.get(k, 1.0))))
                layer_gains = (lg("no_vocals"), lg("vocals"), lg("dub"))
        except Exception as exc:
            logger.warning(f"mix.json could not be read ({exc}) — layer volumes left at 100%")

        # ── Find video ────────────────────────────────────────────────────────
        video_path = find_video_file(job_id, project_root, logger)

        # ── Assemble ──────────────────────────────────────────────────────────
        logger.info("Starting 3-track assembly...")
        dubbed_audio = assemble(
            tts_manifest=tts_manifest,
            translations=translations,
            no_vocals_audio=no_vocals_audio,
            vocals_audio=vocals_audio,
            project_root=job_dir_,
            sr=sr,
            fade_ms=fade_ms,
            tts_gain=tts_gain,
            logger=logger,
            match_volume=bool(assembly_cfg.get("match_dialogue_volume", False)),
            layer_gains=layer_gains,
            gain_min=float(assembly_cfg.get("volume_gain_min", 0.1)),
            gain_max=float(assembly_cfg.get("volume_gain_max", 8.0)),
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
        job_dir     = job_dir_
        output_dir  = jp.output_dir(job_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target_lang = ""
        try:
            target_lang = (load_json(job_dir / "job_meta.json").get("target_lang")
                           or config.get("language", {}).get("target", ""))
        except Exception:
            target_lang = config.get("language", {}).get("target", "")
        output_path = mux_video(video_path, temp_audio_path, output_dir / f"{job_id}_dubbed", logger, target_lang)
        make_assembled_preview(video_path, temp_audio_path, job_dir, logger)

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