"""
Job folder layout — single source of truth (standard library only).

EVERYTHING that belongs to one job lives inside jobs/<job_id>/ , so a job can be zipped,
shared, and dropped into another AI Dub Studio to continue exactly where it stopped:

    jobs/<job_id>/
        job_meta.json  manifest.json
        01_transcription.json  03_translations.json
        source/              the original video (and nothing else)
        audio/               original.wav  vocals.wav  no_vocals.wav
        output/              <job_id>_dubbed.mkv|mp4        (the exported video)
        logs/                one log per stage
        00_vocals_chunks/  04_tts/  speaker_profiles/  user_refs/ ...   (stage work files)
        preview.mp4  assembled_preview.mp4                  (browser viewing copies, re-creatable)

Nothing in a job refers to an absolute path of the machine it was made on.

Older jobs (audio in data/audio, video in data/input, exports in data/outputs, logs in logs/)
are moved into this layout automatically by migrate_legacy_job().
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

SOURCE_DIR = "source"
AUDIO_DIR  = "audio"
OUTPUT_DIR = "output"
LOGS_DIR   = "logs"
LAYOUT_MARK = ".layout_v2"

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v", ".ts", ".wmv", ".mpg", ".mpeg",
              ".3gp", ".ogv", ".m2ts", ".mts", ".asf", ".divx", ".vob", ".mxf", ".f4v", ".rm", ".rmvb"}

log = logging.getLogger("jobpaths")


# ── locations ────────────────────────────────────────────────────────────────
def job_dir(project_root, jobs_dir, job_id: str) -> Path:
    jd = Path(jobs_dir)
    if not jd.is_absolute():
        jd = Path(project_root) / jd
    return jd / job_id

def audio_dir(jd: Path) -> Path:      return Path(jd) / AUDIO_DIR
def source_dir(jd: Path) -> Path:     return Path(jd) / SOURCE_DIR
def output_dir(jd: Path) -> Path:     return Path(jd) / OUTPUT_DIR
def logs_dir(jd: Path) -> Path:       return Path(jd) / LOGS_DIR
def original_wav(jd: Path) -> Path:   return audio_dir(jd) / "original.wav"
def vocals_wav(jd: Path) -> Path:     return audio_dir(jd) / "vocals.wav"
def no_vocals_wav(jd: Path) -> Path:  return audio_dir(jd) / "no_vocals.wav"

def ensure_layout(jd: Path) -> None:
    for sub in (SOURCE_DIR, AUDIO_DIR, OUTPUT_DIR, LOGS_DIR):
        (Path(jd) / sub).mkdir(parents=True, exist_ok=True)


def find_source_video(jd: Path, meta: Optional[dict] = None) -> Optional[Path]:
    """The job's video: whatever is in source/ (portable). Falls back to an old absolute
    path in job_meta.json for jobs that were never migrated."""
    sd = source_dir(jd)
    if sd.is_dir():
        vids = sorted(p for p in sd.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
        if vids:
            return vids[0]
    if meta is None:
        mp = Path(jd) / "job_meta.json"
        if mp.exists():
            try: meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception: meta = {}
    for key in ("input_video", "input_file"):
        v = (meta or {}).get(key)
        if not v:
            continue
        p = Path(v)
        if p.exists():
            return p
        rel = Path(jd) / v                       # relative like "source/movie.mp4"
        if rel.exists():
            return rel
    return None


def resolve_job_file(jd: Path, stored: str, default_sub: str = "") -> Optional[Path]:
    """A path written into a job file (e.g. tts_manifest wav_path) may be absolute on another
    machine, relative to the project, or relative to the job. Find the real file by trying
    each, finally by file name inside default_sub."""
    if not stored:
        return None
    p = Path(stored)
    if p.exists():
        return p
    cand = Path(jd) / stored
    if cand.exists():
        return cand
    cand = Path(jd) / default_sub / p.name
    return cand if cand.exists() else None


# ── legacy migration ─────────────────────────────────────────────────────────
def _move(src: Path, dst: Path) -> bool:
    if not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        log.warning(f"could not move {src} -> {dst}: {e}")
        return False

def _link_or_copy(src: Path, dst: Path) -> bool:
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)                         # instant, no extra disk space (same drive)
        return True
    except Exception:
        pass
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        log.warning(f"could not copy {src} -> {dst}: {e}")
        return False


def migrate_legacy_job(project_root, jobs_dir, job_id: str,
                       legacy_audio="data/audio", legacy_input="data/input",
                       legacy_output="data/outputs", legacy_logs="logs") -> bool:
    """Move a pre-v2 job's files into its own folder. Safe to call any number of times."""
    jd = job_dir(project_root, jobs_dir, job_id)
    if not jd.is_dir() or (jd / LAYOUT_MARK).exists():
        return False
    root = Path(project_root)
    ensure_layout(jd)
    changed = False

    # audio
    la = root / legacy_audio
    for old, new in ((f"{job_id}.wav", original_wav(jd)), (f"{job_id}_vocals.wav", vocals_wav(jd)),
                     (f"{job_id}_no_vocals.wav", no_vocals_wav(jd))):
        changed |= _move(la / old, new)

    # exported video(s)
    lo = root / legacy_output
    for ext in ("mp4", "mkv"):
        changed |= _move(lo / f"{job_id}_dubbed.{ext}", output_dir(jd) / f"{job_id}_dubbed.{ext}")

    # logs
    ll = root / legacy_logs
    if ll.is_dir():
        for f in ll.glob(f"{job_id}_*.log"):
            changed |= _move(f, logs_dir(jd) / f.name)

    # source video (copy/link — other jobs may still point at the same file)
    meta_p = jd / "job_meta.json"
    meta = {}
    if meta_p.exists():
        try: meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception: meta = {}
    if find_source_video(jd, {}) is None:
        for key in ("input_video", "input_file"):
            v = meta.get(key)
            if v and Path(v).is_file():
                if _link_or_copy(Path(v), source_dir(jd) / Path(v).name):
                    changed = True
                break
    vid = find_source_video(jd, {})
    if vid is not None and meta_p.exists():
        rel = f"{SOURCE_DIR}/{vid.name}"
        meta["input_video"] = rel
        meta["input_file"]  = rel
        meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    (jd / LAYOUT_MARK).write_text("2", encoding="utf-8")
    if changed:
        log.info(f"[jobpaths] migrated job {job_id} into its own folder")
    return changed


def normalize_imported_job(jd: Path) -> None:
    """After unzipping a job from another machine: make job_meta.json point inside the folder."""
    jd = Path(jd)
    ensure_layout(jd)
    meta_p = jd / "job_meta.json"
    if meta_p.exists():
        try: meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception: meta = {}
        vid = find_source_video(jd, {})
        if vid is not None:
            meta["input_video"] = f"{SOURCE_DIR}/{vid.name}"
            meta["input_file"]  = f"{SOURCE_DIR}/{vid.name}"
        meta_p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (jd / LAYOUT_MARK).write_text("2", encoding="utf-8")


if __name__ == "__main__":      # python jobpaths.py <project_root>  → migrate every job once
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    jobs = root / "jobs"
    n = 0
    for d in sorted(jobs.iterdir()) if jobs.is_dir() else []:
        if d.is_dir() and not d.name.startswith("_") and (d / "job_meta.json").exists():
            n += int(migrate_legacy_job(root, "jobs", d.name))
    print(f"migrated {n} job(s)")
