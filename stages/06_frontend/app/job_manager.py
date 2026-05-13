"""
DubStudio Pro — Job Manager.
All JSON read/write operations are centralised here.
No Qt imports — pure Python / dataclass layer.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from app.models.job_model import (
    Job, JobMeta, Dialogue, WordToken, StageStatus, TTSEntry
)
from app.models.speaker_model import SpeakerRegistry, DEFAULT_COLORS

PROJECT_ROOT = "/mnt/d/$/my/dubbing_V2"


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def jobs_dir() -> Path:
    return Path(PROJECT_ROOT) / "jobs"


def job_dir(job_id: str) -> Path:
    return jobs_dir() / job_id


def input_dir() -> Path:
    return Path(PROJECT_ROOT) / "data" / "input"


def audio_dir() -> Path:
    return Path(PROJECT_ROOT) / "data" / "audio"


def outputs_dir() -> Path:
    return Path(PROJECT_ROOT) / "data" / "outputs"


def meta_path(job_id: str) -> Path:
    return job_dir(job_id) / "job_meta.json"


def manifest_path(job_id: str) -> Path:
    return job_dir(job_id) / "manifest.json"


def transcription_path(job_id: str) -> Path:
    return job_dir(job_id) / "01_transcription.json"


def translations_path(job_id: str) -> Path:
    return job_dir(job_id) / "03_translations.json"


def tts_manifest_path(job_id: str) -> Path:
    return job_dir(job_id) / "04_tts" / "tts_manifest.json"


def ui_state_path(job_id: str) -> Path:
    return job_dir(job_id) / "ui_state.json"


def original_wav_path(job_id: str) -> Path:
    return audio_dir() / f"{job_id}.wav"


def dubbed_output_path(job_id: str) -> Path:
    return outputs_dir() / f"{job_id}_dubbed.mp4"


def speaker_profiles_dir(job_id: str) -> Path:
    return job_dir(job_id) / "speaker_profiles"


# ──────────────────────────────────────────────────────────────────────────────
# Job listing
# ──────────────────────────────────────────────────────────────────────────────

def list_jobs() -> List[str]:
    """Return sorted list of job_ids (newest first)."""
    d = jobs_dir()
    if not d.exists():
        return []
    ids = [p.name for p in d.iterdir()
           if p.is_dir() and (p / "job_meta.json").exists()]
    return sorted(ids, reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Job creation
# ──────────────────────────────────────────────────────────────────────────────

def create_job(video_path: str) -> str:
    """Create a new job from a video path. Returns job_id."""
    base = Path(video_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{base}_{ts}"

    jdir = job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "04_tts").mkdir(exist_ok=True)
    (jdir / "speaker_profiles").mkdir(exist_ok=True)

    meta = {
        "job_id": job_id,
        "input_video": str(video_path),
        "input_file": str(video_path),
        "created_at": datetime.now().isoformat(),
    }
    _write_json(meta_path(job_id), meta)

    stages = {
        "00_vocals":       {"status": "pending", "completed_at": None, "failed_at": None},
        "01_asr":          {"status": "pending", "completed_at": None, "failed_at": None},
        "01b_diarization": {"status": "pending", "completed_at": None, "failed_at": None},
        "02_emotion":      {"status": "pending", "completed_at": None, "failed_at": None},
        "03_translation":  {"status": "pending", "completed_at": None, "failed_at": None},
        "04_tts":          {"status": "pending", "completed_at": None, "failed_at": None},
        "05_assembly":     {"status": "pending", "completed_at": None, "failed_at": None},
        "06_frontend":     {"status": "pending", "completed_at": None, "failed_at": None},
    }
    _write_json(manifest_path(job_id), {"job_id": job_id, "stages": stages})

    return job_id


# ──────────────────────────────────────────────────────────────────────────────
# Job loading
# ──────────────────────────────────────────────────────────────────────────────

def load_job(job_id: str) -> Optional[Job]:
    """Load a full Job object from disk. Returns None if not found."""
    mp = meta_path(job_id)
    if not mp.exists():
        return None

    raw_meta = _read_json(mp)
    meta = JobMeta(
        job_id=raw_meta.get("job_id", job_id),
        input_video=raw_meta.get("input_video", raw_meta.get("input_file", "")),
        input_file=raw_meta.get("input_file", raw_meta.get("input_video", "")),
        created_at=raw_meta.get("created_at", ""),
    )

    # ── stages ──
    stages: Dict[str, StageStatus] = {}
    if manifest_path(job_id).exists():
        raw_manifest = _read_json(manifest_path(job_id))
        for sid, sv in raw_manifest.get("stages", {}).items():
            stages[sid] = StageStatus(
                stage_id=sid,
                status=sv.get("status", "pending"),
                completed_at=sv.get("completed_at"),
                failed_at=sv.get("failed_at"),
            )

    # ── speaker registry ──
    registry = SpeakerRegistry()

    # ── ui_state sidecar ──
    speaker_names: Dict[str, str] = {}
    speaker_colors: Dict[str, str] = {}
    dialogue_overrides: Dict[str, Dict[str, Any]] = {}
    if ui_state_path(job_id).exists():
        ui = _read_json(ui_state_path(job_id))
        speaker_names = ui.get("speaker_names", {})
        dialogue_overrides = ui.get("dialogue_overrides", {})
        for sid, ov in ui.get("speaker_overrides", {}).items():
            if "color" in ov:
                speaker_colors[sid] = ov["color"]
            if "display_name" in ov:
                speaker_names[sid] = ov["display_name"]

    # ── transcription ──
    dialogues: List[Dialogue] = []
    if transcription_path(job_id).exists():
        raw_t = _read_json(transcription_path(job_id))
        # KEY: always "dialogues" — never "segments"
        for raw_d in raw_t.get("dialogues", []):
            words = [
                WordToken(
                    word=w.get("word", ""),
                    start_ms=w.get("start_ms", 0),
                    end_ms=w.get("end_ms", 0),
                    speaker_id=w.get("speaker_id", ""),
                )
                for w in raw_d.get("words", [])
            ]
            d = Dialogue(
                id=raw_d["id"],
                speaker_id=raw_d.get("speaker_id", ""),
                start_ms=raw_d.get("start_ms", 0),
                end_ms=raw_d.get("end_ms", 0),
                start_time=raw_d.get("start_time", ""),
                end_time=raw_d.get("end_time", ""),
                text=raw_d.get("text", ""),
                is_nonlexical=raw_d.get("is_nonlexical", False),
                words=words,
            )
            registry.ensure_speaker(d.speaker_id)
            dialogues.append(d)

    # ── translations (overlay on dialogues) ──
    if translations_path(job_id).exists():
        raw_tr = _read_json(translations_path(job_id))
        tr_map = {td["id"]: td for td in raw_tr.get("dialogues", [])}
        for d in dialogues:
            if d.id in tr_map:
                td = tr_map[d.id]
                d.emotion = td.get("emotion", "")
                d.intensity = td.get("intensity", 0.0)
                d.translation = td.get("translation", d.text)
                d.is_passthrough = td.get("is_passthrough", False)
                d.dub_enabled = td.get("dub_enabled", True)
                d.source_lang = td.get("source_lang", "en")
                d.target_lang = td.get("target_lang", "en")

    # ── apply ui_state dialogue overrides ──
    for did, ov in dialogue_overrides.items():
        for d in dialogues:
            if d.id == did:
                if "dub_enabled" in ov:
                    d.dub_enabled = ov["dub_enabled"]
                if "translation" in ov:
                    d.translation = ov["translation"]
                if "text" in ov:
                    d.text = ov["text"]
                if "source_lang" in ov:
                    d.source_lang = ov["source_lang"]
                if "target_lang" in ov:
                    d.target_lang = ov["target_lang"]

    # ── tts manifest ──
    tts_entries: Dict[str, TTSEntry] = {}
    if tts_manifest_path(job_id).exists():
        raw_tts = _read_json(tts_manifest_path(job_id))
        for te in raw_tts.get("dialogues", []):
            entry = TTSEntry(
                id=te["id"],
                speaker_id=te.get("speaker_id", ""),
                wav_path=te.get("wav_path", ""),
                original_duration_ms=te.get("original_duration_ms", 0.0),
                synthesized_duration_ms=te.get("synthesized_duration_ms", 0.0),
                target_duration_ms=te.get("target_duration_ms", 0.0),
                final_duration_ms=te.get("final_duration_ms", 0.0),
                stretch_ratio=te.get("stretch_ratio", 1.0),
                emotion=te.get("emotion", ""),
                intensity=te.get("intensity", 0.0),
                tts_text=te.get("tts_text", ""),
                success=te.get("success", False),
                reference_audio=te.get("reference_audio", ""),
                stretch_within_limit=te.get("stretch_within_limit", True),
            )
            tts_entries[entry.id] = entry

    registry.load(speaker_names, speaker_colors)
    # Ensure all dialogue speakers are registered
    for d in dialogues:
        registry.ensure_speaker(d.speaker_id)

    names, colors = registry.to_dicts()

    return Job(
        meta=meta,
        dialogues=dialogues,
        stages=stages,
        tts_entries=tts_entries,
        speaker_names=names,
        speaker_colors=colors,
        dialogue_overrides=dialogue_overrides,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Saving
# ──────────────────────────────────────────────────────────────────────────────

def save_ui_state(job: Job) -> None:
    """Write ui_state.json sidecar — not read by pipeline."""
    state = {
        "speaker_names": job.speaker_names,
        "dialogue_overrides": job.dialogue_overrides,
        "speaker_overrides": {
            sid: {"display_name": job.speaker_names.get(sid, sid),
                  "color": job.speaker_colors.get(sid, "#E8A020")}
            for sid in job.speaker_names
        },
    }
    _write_json(ui_state_path(job.job_id), state)


def save_translations(job: Job) -> None:
    """Write 03_translations.json from current dialogue state."""
    dialogues_out = []
    for d in job.dialogues:
        dialogues_out.append({
            "id": d.id,
            "speaker_id": d.speaker_id,
            "start_ms": d.start_ms,
            "end_ms": d.end_ms,
            "text": d.text,
            "emotion": d.emotion,
            "intensity": d.intensity,
            "translation": d.translation if d.translation else d.text,
            "is_passthrough": d.is_passthrough,
            "dub_enabled": d.dub_enabled,
            "source_lang": d.source_lang,
            "target_lang": d.target_lang,
        })
    out = {
        "job_id": job.job_id,
        "source_lang": job.dialogues[0].source_lang if job.dialogues else "en",
        "target_lang": job.dialogues[0].target_lang if job.dialogues else "en",
        "dialogue_count": len(dialogues_out),
        "dialogues": dialogues_out,
    }
    _write_json(translations_path(job.job_id), out)


def save_transcription(job: Job) -> None:
    """Write 01_transcription.json from current dialogue state."""
    dialogues_out = []
    for d in job.dialogues:
        dialogues_out.append({
            "id": d.id,
            "speaker_id": d.speaker_id,
            "start_ms": d.start_ms,
            "end_ms": d.end_ms,
            "start_time": _ms_to_timestr(d.start_ms),
            "end_time": _ms_to_timestr(d.end_ms),
            "text": d.text,
            "is_nonlexical": d.is_nonlexical,
            "words": [{"word": w.word, "start_ms": w.start_ms, "end_ms": w.end_ms,
                        "speaker_id": w.speaker_id} for w in d.words],
        })
    raw = {}
    if transcription_path(job.job_id).exists():
        raw = _read_json(transcription_path(job.job_id))
    raw["dialogues"] = dialogues_out
    raw["dialogue_count"] = len(dialogues_out)
    _write_json(transcription_path(job.job_id), raw)


def update_manifest_stage(job_id: str, stage_id: str, status: str,
                           completed_at: Optional[str] = None,
                           failed_at: Optional[str] = None) -> None:
    mp = manifest_path(job_id)
    if not mp.exists():
        return
    raw = _read_json(mp)
    if stage_id not in raw.get("stages", {}):
        raw.setdefault("stages", {})[stage_id] = {}
    raw["stages"][stage_id]["status"] = status
    raw["stages"][stage_id]["completed_at"] = completed_at
    raw["stages"][stage_id]["failed_at"] = failed_at
    _write_json(mp, raw)


# ──────────────────────────────────────────────────────────────────────────────
# Speaker profile audio extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_speaker_profiles(job: Job, progress_cb=None) -> Dict[str, float]:
    """
    Extract speaker profile WAVs from the original stereo WAV.
    Returns dict of {speaker_id: duration_seconds}.
    Raises if soundfile / scipy not available or WAV not found.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    wav_path = original_wav_path(job.job_id)
    if not wav_path.exists():
        raise FileNotFoundError(f"Original WAV not found: {wav_path}")

    data, sr = sf.read(str(wav_path), always_2d=True)
    # data shape: (samples, channels)

    profiles_dir = speaker_profiles_dir(job.job_id)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    TARGET_SR = 16000
    SILENCE_SAMPLES = int(0.1 * TARGET_SR)
    silence = np.zeros(SILENCE_SAMPLES, dtype=np.float32)

    speaker_clips: Dict[str, list] = {}
    for d in job.dialogues:
        if d.is_nonlexical:
            continue
        start_s = int(d.start_ms / 1000 * sr)
        end_s = int(d.end_ms / 1000 * sr)
        clip = data[start_s:end_s]
        if clip.shape[0] == 0:
            continue
        # stereo → mono
        mono = clip.mean(axis=1).astype(np.float32)
        # resample to 16000
        resampled = resample_poly(mono, TARGET_SR, sr).astype(np.float32)
        speaker_clips.setdefault(d.speaker_id, []).append(resampled)

    durations: Dict[str, float] = {}
    speakers = list(speaker_clips.keys())
    for i, spk in enumerate(speakers):
        if progress_cb:
            progress_cb(i, len(speakers), spk)
        clips = speaker_clips[spk]
        parts = []
        for c in clips:
            parts.append(c)
            parts.append(silence)
        combined = np.concatenate(parts).astype(np.float32)
        out_path = profiles_dir / f"{spk}.wav"
        sf.write(str(out_path), combined, TARGET_SR, subtype="PCM_16")
        durations[spk] = len(combined) / TARGET_SR

    return durations


def append_clip_to_speaker_profile(job: Job, dialogue: Dialogue) -> None:
    """Append a dialogue audio clip to a speaker's profile WAV."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    wav_path = original_wav_path(job.job_id)
    if not wav_path.exists():
        return

    data, sr = sf.read(str(wav_path), always_2d=True)
    start_s = int(dialogue.start_ms / 1000 * sr)
    end_s = int(dialogue.end_ms / 1000 * sr)
    clip = data[start_s:end_s]
    if clip.shape[0] == 0:
        return

    TARGET_SR = 16000
    mono = clip.mean(axis=1).astype(np.float32)
    resampled = resample_poly(mono, TARGET_SR, sr).astype(np.float32)

    profiles_dir = speaker_profiles_dir(job.job_id)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    out_path = profiles_dir / f"{dialogue.speaker_id}.wav"

    SILENCE = np.zeros(int(0.1 * TARGET_SR), dtype=np.float32)
    if out_path.exists():
        existing, _ = sf.read(str(out_path))
        existing = existing.astype(np.float32)
        combined = np.concatenate([existing, SILENCE, resampled])
    else:
        combined = resampled

    sf.write(str(out_path), combined.astype(np.float32), TARGET_SR, subtype="PCM_16")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _ms_to_timestr(ms: int) -> str:
    total_s = ms // 1000
    frac = ms % 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{frac:03d}"