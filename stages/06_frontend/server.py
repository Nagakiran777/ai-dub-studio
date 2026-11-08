"""
DubStudio Pro — FastAPI Backend v6
All fixes:
  - dialogue/add endpoint
  - tts/rerun works correctly (saves translation first)
  - Translation uses deep_translator with retry + error handling
  - Language passed from job_meta to stage CLI args
  - build_profiles uses stdlib wave only (no numpy)
  - vocal audio served as exact slice
  - speaker_dialogues endpoint for ref audio
  - Run vs Resume based on manifest state
"""
from __future__ import annotations
import asyncio, gc, io, json, logging, os, re, shutil, subprocess, sys, time, uuid, wave as wave_mod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn, yaml
# Single source of truth for dub speed matching — shared with Stage 05 so the
# preview and the exported video always agree. Lives in stages/_shared/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))
try:
    from dub_speed import auto_speed, effective_speed, dub_geometry, dub_parts_geometry
    import jobpaths as jp          # every job keeps ALL its files inside jobs/<job_id>/
    import dub_volume
except ImportError as _e:  # pragma: no cover
    raise RuntimeError("stages/_shared/dub_speed.py not found — copy the _shared folder next to the other stage folders") from _e
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/mnt/d/$/my/dubbing_V2")
CONFIG_PATH  = PROJECT_ROOT / "config.yaml"
JOBS_DIR     = PROJECT_ROOT / "jobs"
INBOX_DIR    = JOBS_DIR / "_inbox"          # uploaded videos waiting to become a job (no root-level data/ folder any more)
LEGACY_INPUT = PROJECT_ROOT / "data" / "input"   # old videos folder — still listed if it exists
STAGES_DIR   = PROJECT_ROOT / "stages"
CONDA_BIN    = "/home/jani/miniconda3/bin/conda"
CONDA_PATH   = "/home/jani/miniconda3/bin:/home/jani/miniconda3/condabin"

STAGES = [
    {"id": "00_vocals",       "label": "Vocals Extraction",  "env": "dub_vocals"},
    {"id": "01_asr",          "label": "Speech Recognition", "env": "dub_asr"},
    {"id": "01b_diarization", "label": "Speaker Detection",  "env": "dub_diar"},
    {"id": "03_translation",  "label": "Translation",        "env": "dub_translate"},
    {"id": "04_tts",          "label": "Voice Synthesis",    "env": "dub_tts"},
    {"id": "05_assembly",     "label": "Final Assembly",     "env": "dub_assembly"},
]
CHECKPOINTS = {}  # Run All no longer pauses mid-pipeline for anything — it runs
                   # straight through 00→01→01b→03→04 and stops once, ready for
                   # export/review. Speaker renaming and every dialogue edit are
                   # always available in the UI regardless of pipeline state.

app = FastAPI(title="DubStudio Pro API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline_tasks: Dict[str, asyncio.Task] = {}
ws_connections: Dict[str, List[WebSocket]] = {}
pipeline_state: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_PATH) as f: return yaml.safe_load(f)

def load_json_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f: return json.load(f)

def save_json_file(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

_MIGRATED: set = set()
def get_job_dir(job_id: str) -> Path:
    """jobs/<job_id>/ — the one folder that holds everything of this job. A job made by an
    older version (audio in data/audio, export in data/outputs …) is moved in on first use."""
    jd = JOBS_DIR / job_id
    if job_id not in _MIGRATED and jd.is_dir() and not job_id.startswith("_"):
        _MIGRATED.add(job_id)
        try: jp.migrate_legacy_job(PROJECT_ROOT, "jobs", job_id)
        except Exception as e: logging.getLogger("server").warning(f"migrate {job_id}: {e}")
    return jd

def job_audio(job_id: str, kind: str) -> Path:
    jd = get_job_dir(job_id)
    return {"vocals": jp.vocals_wav, "no_vocals": jp.no_vocals_wav, "original": jp.original_wav}[kind](jd)

def get_manifest(job_id: str) -> dict:
    p = get_job_dir(job_id) / "manifest.json"
    return load_json_file(p) if p.exists() else {}

def save_manifest(job_id: str, m: dict):
    save_json_file(get_job_dir(job_id) / "manifest.json", m)

async def broadcast(job_id: str, msg: dict):
    dead = []
    for ws in ws_connections.get(job_id, []):
        try: await ws.send_json(msg)
        except: dead.append(ws)
    for d in dead: ws_connections.get(job_id, []).remove(d)

# ---------------------------------------------------------------------------
# WebSocket
@app.websocket("/ws/{job_id}")
async def ws_endpoint(ws: WebSocket, job_id: str):
    await ws.accept()
    ws_connections.setdefault(job_id, []).append(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        if ws in ws_connections.get(job_id, []): ws_connections[job_id].remove(ws)

# ---------------------------------------------------------------------------
# Jobs
@app.get("/api/jobs")
def list_jobs():
    if not JOBS_DIR.exists(): return []
    jobs = []
    for d in sorted(JOBS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or d.name.startswith("_"): continue
        mp = d / "job_meta.json"
        if mp.exists():
            meta = load_json_file(mp)
            jobs.append({"job_id": d.name, "created_at": meta.get("created_at",""), "input_video": meta.get("input_video","")})
    return jobs

@app.get("/api/jobs/{job_id}/manifest")
def get_manifest_api(job_id: str): return get_manifest(job_id)

@app.get("/api/jobs/{job_id}/meta")
def get_meta(job_id: str):
    p = get_job_dir(job_id) / "job_meta.json"
    if not p.exists(): raise HTTPException(404)
    return load_json_file(p)

class NewJobReq(BaseModel):
    input_video: str

@app.post("/api/jobs/create")
def create_job(req: NewJobReq):
    vp = Path(req.input_video)
    if not vp.exists(): raise HTTPException(400, f"Video not found: {req.input_video}")
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{vp.stem}_{ts}"
    jd     = JOBS_DIR / job_id
    jd.mkdir(parents=True, exist_ok=True)
    jp.ensure_layout(jd)
    # The video moves INTO the job (source/), so the job folder holds everything and can be zipped.
    dest = jp.source_dir(jd) / vp.name
    try:
        if INBOX_DIR in vp.resolve().parents: shutil.move(str(vp), str(dest))    # was just uploaded → no duplicate
        else: shutil.copy2(str(vp), str(dest))                                     # picked from an existing folder → keep the original
    except Exception as e:
        raise HTTPException(500, f"Could not copy the video into the job folder: {e}")
    (jd / jp.LAYOUT_MARK).write_text("2", encoding="utf-8")
    rel = f"{jp.SOURCE_DIR}/{dest.name}"
    save_json_file(jd / "job_meta.json", {
        "job_id": job_id, "input_video": rel, "input_file": rel,
        "created_at": datetime.now().isoformat(), "status": "created",
        "source_lang": "auto", "target_lang": "en",
        "asr_model": "medium",          # Whisper model for the speech-recognition stage (medium = recommended)
    })
    save_json_file(jd / "manifest.json", {
        "job_id": job_id, "stages": {
            s["id"]: {"status": "pending", "completed_at": None, "failed_at": None} for s in STAGES
        }
    })
    return {"job_id": job_id}

class LangReq(BaseModel):
    source_lang: str = "auto"
    target_lang: str = "en"

@app.post("/api/jobs/{job_id}/language")
async def save_language(job_id: str, req: LangReq):
    mp = get_job_dir(job_id) / "job_meta.json"
    if not mp.exists(): raise HTTPException(404)
    meta = load_json_file(mp)
    meta["source_lang"] = req.source_lang
    meta["target_lang"]  = req.target_lang
    save_json_file(mp, meta)
    return {"ok": True}

@app.get("/api/input_videos")
def list_videos():
    out = []
    for folder in (INBOX_DIR, LEGACY_INPUT):
        if folder.exists():
            out += [{"name": f.name, "path": str(f)} for f in sorted(folder.iterdir()) if f.suffix.lower() in jp.VIDEO_EXTS]
    return out

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    dest = INBOX_DIR / Path(file.filename or "upload.mp4").name
    with open(dest, "wb") as f: shutil.copyfileobj(file.file, f)
    return {"name": file.filename, "path": str(dest)}

# ---------------------------------------------------------------------------
# Data files
@app.get("/api/jobs/{job_id}/transcription")
def get_transcription(job_id: str):
    p = get_job_dir(job_id) / "01_transcription.json"
    if not p.exists(): return {"dialogues": [], "ready": False}
    d = load_json_file(p); d["ready"] = True; return d

@app.get("/api/jobs/{job_id}/translations")
def get_translations(job_id: str):
    p = get_job_dir(job_id) / "03_translations.json"
    if not p.exists(): return {"dialogues": [], "ready": False}
    d = load_json_file(p); d["ready"] = True; return d

# ---------------------------------------------------------------------------
# Translation — skip / file-based (manual or any-chat-AI) flow.
# "Skip" and "import" both just need to produce a valid 03_translations.json
# and mark 03_translation completed — Stage 04 already skips any dialogue
# whose translation is empty, and Stage 05 already leaves the original vocals
# untouched wherever there's no successful TTS take, so an empty translation
# naturally falls back to the original voice with no extra logic needed.
def _write_translations_from_transcription(job_id: str, skipped: bool) -> dict:
    jd = get_job_dir(job_id)
    trans = load_json_file(jd / "01_transcription.json")
    meta = load_json_file(jd / "job_meta.json") if (jd / "job_meta.json").exists() else {}
    out = {
        "job_id": job_id,
        "source_lang": meta.get("source_lang", "auto"),
        "target_lang": meta.get("target_lang", "en"),
        "dialogue_count": len(trans.get("dialogues", [])),
        "skipped": skipped,
        "dialogues": [
            {"id": d["id"], "speaker_id": d.get("speaker_id","speaker_00"),
             "start_ms": d["start_ms"], "end_ms": d["end_ms"],
             "text": d.get("text",""), "translation": "", "is_passthrough": False}
            for d in trans.get("dialogues", [])
        ],
    }
    save_json_file(jd / "03_translations.json", out)
    m = get_manifest(job_id)
    m["stages"]["03_translation"]["status"] = "completed"
    m["stages"]["03_translation"]["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(job_id, m)
    return out

@app.post("/api/jobs/{job_id}/translation/skip")
async def skip_translation(job_id: str):
    """User will type every translation by hand — creates translations.json
    with empty translation fields (ready to edit in DialogueEditor) and marks
    the stage complete so Stage 04 can proceed."""
    if not (get_job_dir(job_id) / "01_transcription.json").exists():
        raise HTTPException(400, "Run ASR (Stage 01) first")
    _write_translations_from_transcription(job_id, skipped=True)
    await broadcast(job_id, {"type": "stage_completed", "stage": "03_translation"})
    return {"ok": True}

@app.get("/api/jobs/{job_id}/translation/export_file")
def export_translation_file(job_id: str):
    """Generates one plain-text file: instructions + the full episode's
    dialogue in order, meant to be pasted whole into any chat AI (or sent to
    Gemini automatically, once that path exists). One line per dialogue in
    the reply format the importer below expects."""
    jd = get_job_dir(job_id)
    p = jd / "01_transcription.json"
    if not p.exists(): raise HTTPException(404, "No transcription yet")
    trans = load_json_file(p)
    meta = load_json_file(jd / "job_meta.json") if (jd / "job_meta.json").exists() else {}
    src, tgt = meta.get("source_lang","the source language"), meta.get("target_lang","the target language")

    lines = [
        "You are a professional dubbing dialogue translator.",
        f"Translate the dialogue below from {src} to {tgt}.",
        "Read the WHOLE script first so you understand the story, each speaker's",
        "personality, their relationships, and the weight of each scene — then",
        "translate for natural spoken meaning and emotion, not literal word-for-word.",
        "You may quietly fix obvious grammar/spelling mistakes in the original text.",
        "",
        "Reply with translations in batches of about 20-30 lines at a time (ask for",
        "the next batch if the conversation continues) so nothing gets cut off.",
        "For EACH line, reply in EXACTLY this format, one per line, nothing else added:",
        "<id>|<translated line>",
        "",
        "Dialogue script (id, speaker, start-end timestamp, original text):",
    ]
    for d in trans.get("dialogues", []):
        lines.append(f'[{d["id"]}] {d.get("speaker_id","speaker_00")} ({d.get("start_time","")}-{d.get("end_time","")}): {d.get("text","")}')

    content = "\n".join(lines)
    return StreamingResponse(io.BytesIO(content.encode("utf-8")), media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_translation_prompt.txt"'})

@app.post("/api/jobs/{job_id}/translation/import_file")
async def import_translation_file(job_id: str, file: UploadFile = File(...)):
    """Parses a pasted-back AI response (or hand-edited file) in the
    '<id>|<translation>' format and fills in translations.json. Unmatched
    lines are reported back rather than silently dropped."""
    jd = get_job_dir(job_id)
    if not (jd / "03_translations.json").exists():
        _write_translations_from_transcription(job_id, skipped=False)
    text = (await file.read()).decode("utf-8", errors="replace")

    data = load_json_file(jd / "03_translations.json")
    by_id = {d["id"]: d for d in data.get("dialogues", [])}
    matched, unmatched = 0, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "|" not in line: continue
        left, _, right = line.partition("|")
        did = left.strip().strip("[]")
        translation = right.strip()
        if did in by_id and translation:
            by_id[did]["translation"] = translation
            matched += 1
        elif did:
            unmatched.append(did)

    data["skipped"] = False
    save_json_file(jd / "03_translations.json", data)
    m = get_manifest(job_id)
    m["stages"]["03_translation"]["status"] = "completed"
    m["stages"]["03_translation"]["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(job_id, m)
    await broadcast(job_id, {"type": "stage_completed", "stage": "03_translation"})
    return {"ok": True, "matched": matched, "unmatched_ids": unmatched,
            "total_dialogues": len(by_id)}

@app.get("/api/jobs/{job_id}/tts_manifest")
def get_tts(job_id: str):
    p = get_job_dir(job_id) / "04_tts" / "tts_manifest.json"
    if not p.exists(): return {"dialogues": [], "ready": False}
    d = load_json_file(p); d["ready"] = True
    tp = get_job_dir(job_id) / "03_translations.json"
    tr = {e["id"]: e for e in load_json_file(tp).get("dialogues", [])} if tp.exists() else {}
    for e in d.get("dialogues", []):
        t = tr.get(e.get("id"))
        if not t or not e.get("success"): continue
        orig = float(t.get("end_ms", 0)) - float(t.get("start_ms", 0))
        dub  = e.get("synthesized_duration_ms")
        manual = t.get("dub_speed")
        # One shared calculation (stages/_shared/dub_speed.py) for placement, trim,
        # speed and final length — Stage 05 uses the very same function.
        g = dub_geometry(dub, t, float(t.get("start_ms", 0)), float(t.get("end_ms", 0)))
        e["original_ms"]   = orig
        e["speed_auto"]    = g["speed_fit"]      # value behind the "Fit to original" button
        e["speed_manual"]  = manual
        e["speed"]         = g["speed"]          # default 1.0 unless the user set one
        e["natural_ms"]    = g["natural_ms"]
        e["trim_in_ms"]    = g["trim_in_ms"]
        e["trim_out_ms"]   = g["trim_out_ms"]
        e["kept_ms"]       = g["kept_ms"]
        e["dub_start_ms"]  = g["dub_start_ms"]
        e["dub_len_ms"]    = g["dub_len_ms"]
        e["dub_end_ms"]    = g["dub_end_ms"]
        e["dub_deleted"]   = g["deleted"]
        e["dub_moved"]     = g["moved"]
        e["dub_trimmed"]   = g["trimmed"]
        # every piece the dub has been cut into (piece "p0" = the dialogue's own dub)
        e["parts"]         = dub_parts_geometry(dub, t, float(t.get("start_ms", 0)), float(t.get("end_ms", 0)))
        # loudness: the gain that makes this dub as loud as ITS original line — same rule as the export
        # dub_volume = what the user set for THIS dialogue (1.0 = natural level). Nothing is
        # normalized automatically; volume_gain is only the value behind "Match to original".
        dv = t.get("dub_volume")
        e["dub_volume"] = 1.0 if dv is None else max(0.0, min(4.0, float(dv)))
        vg = _volume_gain(job_id, e, t)
        if vg:
            e["volume_gain"] = vg["gain"]; e["orig_rms"] = vg["orig_rms"]; e["dub_rms"] = vg["dub_rms"]
    return d

_VOL_CACHE: Dict[tuple, Optional[dict]] = {}
def _volume_gain(job_id: str, e: dict, t: dict) -> Optional[dict]:
    """Cached per (original window, dub file, vocals file) so a refresh doesn't re-measure everything."""
    jd = get_job_dir(job_id)
    wav = jp.resolve_job_file(jd, e.get("wav_path") or "", "04_tts")
    voc = jp.vocals_wav(jd)
    if wav is None or not voc.exists(): return None
    try:
        key = (job_id, e.get("id"), int(t.get("start_ms", 0)), int(t.get("end_ms", 0)),
               wav.stat().st_mtime_ns, voc.stat().st_mtime_ns)
    except OSError:
        return None
    if key not in _VOL_CACHE:
        _VOL_CACHE[key] = dub_volume.dialogue_gain(voc, float(t.get("start_ms", 0)), float(t.get("end_ms", 0)), wav)
    return _VOL_CACHE[key]

# ---------------------------------------------------------------------------
# Dialogue CRUD
class DlgUpdateReq(BaseModel):
    dialogue_id: str; field: str; value: Any

# Field routing — which file(s) a given dialogue field belongs to.
# Prevents e.g. "translation" edits leaking into 01_transcription.json and
# vice versa. Fields not listed here are treated as shared (written to both,
# same as before) since start_ms/end_ms/speaker_id legitimately live in both files.
TRANSCRIPTION_ONLY_FIELDS = {"text", "is_nonlexical", "words"}
TRANSLATIONS_ONLY_FIELDS  = {"translation", "is_passthrough", "manual_ref_audio", "source_lang", "target_lang", "dub_speed",
                             # dub placement / trim / removal — independent of the original bar, never stale the TTS
                             "dub_start_ms", "dub_trim_start_ms", "dub_trim_end_ms", "dub_deleted", "dub_parts", "dub_volume"}
# Fields that, if a TTS take already exists for this dialogue, make that take
# no longer match (different words, different ref-audio slice, different
# speaker/voice) — flagged as stale rather than silently left looking "done".
FIELDS_THAT_STALE_TTS = {"text", "start_ms", "end_ms", "speaker_id"}

def mark_tts_stale(job_id: str, dialogue_id: str) -> None:
    p = get_job_dir(job_id) / "04_tts" / "tts_manifest.json"
    if not p.exists(): return
    data = load_json_file(p)
    changed = False
    for e in data.get("dialogues", []):
        if e.get("id") == dialogue_id and e.get("success") and not e.get("stale"):
            e["stale"] = True
            changed = True
    if changed: save_json_file(p, data)

def reset_dub_trims(job_id: str, dialogue_id: Optional[str] = None) -> None:
    """A freshly synthesized clip has a different length, so trim points that were
    set on the OLD clip no longer mean anything — clear them (speed and position stay)."""
    p = get_job_dir(job_id) / "03_translations.json"
    if not p.exists(): return
    data = load_json_file(p); changed = False
    for d in data.get("dialogues", []):
        if dialogue_id is not None and d.get("id") != dialogue_id: continue
        for k in ("dub_trim_start_ms", "dub_trim_end_ms"):
            if d.get(k) is not None:
                d[k] = None; changed = True
        if d.get("dub_parts"):          # pieces were cut from the OLD clip — gone with it
            d["dub_parts"] = None; changed = True
    if changed: save_json_file(p, data)

def remove_tts_entry(job_id: str, dialogue_id: str) -> None:
    """Used when a dialogue id stops existing (merged away) — its TTS take
    is for content that no longer has a home."""
    jd = get_job_dir(job_id)
    p = jd / "04_tts" / "tts_manifest.json"
    if p.exists():
        data = load_json_file(p)
        before = len(data.get("dialogues", []))
        data["dialogues"] = [e for e in data.get("dialogues", []) if e.get("id") != dialogue_id]
        if len(data["dialogues"]) != before: save_json_file(p, data)
    wav = jd / "04_tts" / f"{dialogue_id}.wav"
    if wav.exists():
        try: wav.unlink()
        except Exception: pass

def freeze_dub_start(job_id: str, dialogue_id: str, fields: Dict[str, Any]) -> None:
    """The dub bar and the original bar are independent. A dub that has never
    been placed by hand simply sits at the original's start — so when the
    ORIGINAL start is about to move, pin the dub where it is now; otherwise
    dragging the original bar would silently drag the dub with it."""
    if "start_ms" not in fields: return
    jd = get_job_dir(job_id)
    tp, mp = jd / "03_translations.json", jd / "04_tts" / "tts_manifest.json"
    if not tp.exists() or not mp.exists(): return
    if not any(e.get("id") == dialogue_id and e.get("success") for e in load_json_file(mp).get("dialogues", [])): return
    data = load_json_file(tp); changed = False
    for d in data.get("dialogues", []):
        if d.get("id") != dialogue_id or d.get("dub_start_ms") is not None: continue
        try: new_start = float(fields["start_ms"])
        except (TypeError, ValueError): continue
        if float(d.get("start_ms", 0)) != new_start:
            d["dub_start_ms"] = d.get("start_ms", 0); changed = True
    if changed: save_json_file(tp, data)

@app.post("/api/jobs/{job_id}/dialogue/update")
async def update_dialogue(job_id: str, req: DlgUpdateReq):
    freeze_dub_start(job_id, req.dialogue_id, {req.field: req.value})
    targets = ["01_transcription.json", "03_translations.json"]
    if req.field in TRANSCRIPTION_ONLY_FIELDS:
        targets = ["01_transcription.json"]
    elif req.field in TRANSLATIONS_ONLY_FIELDS:
        targets = ["03_translations.json"]
    for fname in targets:
        p = get_job_dir(job_id) / fname
        if not p.exists(): continue
        data = load_json_file(p)
        for d in data.get("dialogues", []):
            if d["id"] == req.dialogue_id: d[req.field] = req.value
        save_json_file(p, data)
    if req.field in FIELDS_THAT_STALE_TTS:
        mark_tts_stale(job_id, req.dialogue_id)
    await broadcast(job_id, {"type": "dialogue_updated", "dialogue_id": req.dialogue_id})
    return {"ok": True}

class DlgUpdateManyReq(BaseModel):
    dialogue_id: str
    fields: Dict[str, Any]

@app.post("/api/jobs/{job_id}/dialogue/update_many")
async def update_dialogue_many(job_id: str, req: DlgUpdateManyReq):
    """Apply several field changes to one dialogue in a single write per file
    and a single broadcast — used by drag-and-drop (time and/or speaker)."""
    freeze_dub_start(job_id, req.dialogue_id, req.fields)
    jd = get_job_dir(job_id)
    for fname in ["01_transcription.json", "03_translations.json"]:
        p = jd / fname
        if not p.exists(): continue
        data = load_json_file(p)
        changed = False
        for d in data.get("dialogues", []):
            if d["id"] != req.dialogue_id: continue
            for field, value in req.fields.items():
                if fname == "01_transcription.json" and field in TRANSLATIONS_ONLY_FIELDS: continue
                if fname == "03_translations.json" and field in TRANSCRIPTION_ONLY_FIELDS: continue
                d[field] = value; changed = True
        if changed: save_json_file(p, data)
    if any(f in FIELDS_THAT_STALE_TTS for f in req.fields):
        mark_tts_stale(job_id, req.dialogue_id)
    await broadcast(job_id, {"type": "dialogue_updated", "dialogue_id": req.dialogue_id})
    return {"ok": True}

class BulkSetReq(BaseModel):
    transcription_dialogues: Optional[List[dict]] = None
    translations_dialogues:  Optional[List[dict]] = None

@app.post("/api/jobs/{job_id}/dialogue/bulk_set")
async def bulk_set_dialogues(job_id: str, req: BulkSetReq):
    """Overwrite the full dialogues array in one or both files at once.
    Used by undo/redo so a reverted client-side snapshot is persisted
    server-side in a single call instead of one request per changed field."""
    jd = get_job_dir(job_id)
    if req.transcription_dialogues is not None:
        p = jd / "01_transcription.json"
        if p.exists():
            data = load_json_file(p)
            data["dialogues"] = req.transcription_dialogues
            save_json_file(p, data)
    if req.translations_dialogues is not None:
        p = jd / "03_translations.json"
        if p.exists():
            data = load_json_file(p)
            data["dialogues"] = req.translations_dialogues
            save_json_file(p, data)
    await broadcast(job_id, {"type": "dialogue_updated", "dialogue_id": None})
    return {"ok": True}

class AddDlgReq(BaseModel):
    id: str; speaker_id: str; start_ms: int; end_ms: int; text: str

@app.post("/api/jobs/{job_id}/dialogue/add")
async def add_dialogue(job_id: str, req: AddDlgReq):
    """Insert a new dialogue into transcription (and translations if exists), sorted by start_ms."""
    new_dlg = {
        "id": req.id, "speaker_id": req.speaker_id,
        "start_ms": req.start_ms, "end_ms": req.end_ms,
        "start_time": "", "end_time": "",
        "text": req.text, "is_nonlexical": False, "words": [],
    }
    # Transcription
    tp = get_job_dir(job_id) / "01_transcription.json"
    if tp.exists():
        data = load_json_file(tp)
        dlgs = data.get("dialogues", [])
        dlgs.append(new_dlg)
        dlgs.sort(key=lambda d: d["start_ms"])
        data["dialogues"] = dlgs
        save_json_file(tp, data)
    # Translations
    trp = get_job_dir(job_id) / "03_translations.json"
    if trp.exists():
        data = load_json_file(trp)
        dlgs = data.get("dialogues", [])
        dlgs.append({**new_dlg, "translation": req.text, "is_passthrough": True})
        dlgs.sort(key=lambda d: d["start_ms"])
        data["dialogues"] = dlgs
        save_json_file(trp, data)
    await broadcast(job_id, {"type": "dialogue_added"})
    return {"ok": True}

class MergeReq(BaseModel):
    dialogue_id_a: str; dialogue_id_b: str

@app.post("/api/jobs/{job_id}/dialogue/merge")
async def merge_dialogues(job_id: str, req: MergeReq):
    def merge_file(path, tf):
        if not path.exists(): return
        data = load_json_file(path); dlgs = data.get("dialogues", [])
        ia = next((i for i,d in enumerate(dlgs) if d["id"]==req.dialogue_id_a), None)
        ib = next((i for i,d in enumerate(dlgs) if d["id"]==req.dialogue_id_b), None)
        if ia is None or ib is None: return
        a, b = dlgs[ia], dlgs[ib]
        merged = {**a, "end_ms": b["end_ms"], "end_time": b.get("end_time",""),
                  tf: f"{a.get(tf,'')} {b.get(tf,'')}".strip(),
                  "words": a.get("words",[]) + b.get("words",[])}
        new_dlgs = [d for d in dlgs if d["id"] not in (req.dialogue_id_a, req.dialogue_id_b)]
        new_dlgs.insert(min(ia,ib), merged)
        data["dialogues"] = new_dlgs; save_json_file(path, data)
    merge_file(get_job_dir(job_id)/"01_transcription.json", "text")
    merge_file(get_job_dir(job_id)/"03_translations.json",  "translation")
    # The merged dialogue now covers different audio/text than either half
    # did alone — its old take (if any) no longer matches; the absorbed id
    # stops existing entirely, so its take goes with it.
    mark_tts_stale(job_id, req.dialogue_id_a)
    remove_tts_entry(job_id, req.dialogue_id_b)
    await broadcast(job_id, {"type": "dialogues_merged"})
    return {"ok": True}

class DeleteReq(BaseModel):
    dialogue_id: str

@app.post("/api/jobs/{job_id}/dialogue/delete")
async def delete_dialogue(job_id: str, req: DeleteReq):
    for fname in ["01_transcription.json", "03_translations.json"]:
        p = get_job_dir(job_id) / fname
        if not p.exists(): continue
        data = load_json_file(p)
        data["dialogues"] = [d for d in data.get("dialogues",[]) if d["id"] != req.dialogue_id]
        save_json_file(p, data)
    await broadcast(job_id, {"type": "dialogue_deleted", "dialogue_id": req.dialogue_id})
    return {"ok": True}

class SplitReq(BaseModel):
    dialogue_id: str; split_ms: int

@app.post("/api/jobs/{job_id}/dialogue/split")
async def split_dialogue(job_id: str, req: SplitReq):
    p = get_job_dir(job_id) / "01_transcription.json"
    if not p.exists(): raise HTTPException(404)
    data = load_json_file(p); dlgs = data.get("dialogues",[])
    idx  = next((i for i,d in enumerate(dlgs) if d["id"]==req.dialogue_id), None)
    if idx is None: raise HTTPException(404)
    orig = dlgs[idx]; ts = datetime.now().strftime("%f")
    a = {**orig, "id": f"{orig['id']}_a{ts}", "end_ms": req.split_ms}
    b = {**orig, "id": f"{orig['id']}_b{ts}", "start_ms": req.split_ms}
    dlgs[idx:idx+1] = [a, b]; data["dialogues"] = dlgs; save_json_file(p, data)
    # Also split translations if exist
    tp = get_job_dir(job_id) / "03_translations.json"
    if tp.exists():
        tdata = load_json_file(tp); tdlgs = tdata.get("dialogues",[])
        tidx  = next((i for i,d in enumerate(tdlgs) if d["id"]==req.dialogue_id), None)
        if tidx is not None:
            torig = tdlgs[tidx]
            ta = {**torig, "id": f"{orig['id']}_a{ts}", "end_ms": req.split_ms}
            tb = {**torig, "id": f"{orig['id']}_b{ts}", "start_ms": req.split_ms}
            tdlgs[tidx:tidx+1] = [ta, tb]; tdata["dialogues"] = tdlgs; save_json_file(tp, tdata)
    await broadcast(job_id, {"type": "dialogue_split"})
    return {"ok": True}

class SetRefReq(BaseModel):
    dialogue_id: str; ref_audio: str

@app.post("/api/jobs/{job_id}/dialogue/set_ref")
async def set_ref(job_id: str, req: SetRefReq):
    p = get_job_dir(job_id) / "03_translations.json"
    if not p.exists(): raise HTTPException(404)
    data = load_json_file(p)
    for d in data.get("dialogues",[]):
        if d["id"] == req.dialogue_id: d["manual_ref_audio"] = req.ref_audio
    save_json_file(p, data)
    # Without this broadcast, the frontend's cached translations state never
    # learns the ref changed — switching to another dialogue and back would
    # re-derive the picker's selection from stale data and show "Auto"
    # again even though the real value was saved correctly on disk.
    await broadcast(job_id, {"type": "dialogue_updated", "dialogue_id": req.dialogue_id})
    return {"ok": True}

class SetRefSpeakerReq(BaseModel):
    speaker_id: str; ref_audio: str

@app.post("/api/jobs/{job_id}/dialogue/set_ref_speaker")
async def set_ref_speaker(job_id: str, req: SetRefSpeakerReq):
    """Use one reference (e.g. an uploaded voice file) for EVERY line of a speaker."""
    p = get_job_dir(job_id) / "03_translations.json"
    if not p.exists(): raise HTTPException(404)
    data = load_json_file(p); n = 0
    for d in data.get("dialogues", []):
        if d.get("speaker_id") == req.speaker_id:
            d["manual_ref_audio"] = req.ref_audio; n += 1
    save_json_file(p, data)
    await broadcast(job_id, {"type": "dialogue_updated", "dialogue_id": None})
    return {"ok": True, "updated": n}

# ---------------------------------------------------------------------------
# User-uploaded reference voices — jobs/<job>/user_refs/<id>.wav + index.json
# A dialogue uses one by setting manual_ref_audio = "user:<id>".
REF_ID_RE      = re.compile(r"^[A-Za-z0-9_-]+$")
USER_REF_MAX_S = 30            # only the first 30 s of an upload is kept (TTS uses ≤10 s anyway)

def _user_refs_dir(job_id: str) -> Path: return get_job_dir(job_id) / "user_refs"

def _load_user_refs(job_id: str) -> List[dict]:
    p = _user_refs_dir(job_id) / "index.json"
    return load_json_file(p).get("refs", []) if p.exists() else []

def _save_user_refs(job_id: str, refs: List[dict]) -> None:
    d = _user_refs_dir(job_id); d.mkdir(parents=True, exist_ok=True)
    save_json_file(d / "index.json", {"refs": refs})

def _user_ref_public(job_id: str, r: dict) -> dict:
    return {**r, "value": f"user:{r['id']}", "url": f"/api/jobs/{job_id}/user_refs/{r['id']}/audio"}

@app.get("/api/jobs/{job_id}/user_refs")
def list_user_refs(job_id: str):
    return [_user_ref_public(job_id, r) for r in _load_user_refs(job_id)]

@app.post("/api/jobs/{job_id}/user_refs")
async def upload_user_ref(job_id: str, file: UploadFile = File(...), text: str = Form("")):
    """Accepts any audio (or video) file ffmpeg can read — wav, mp3, m4a, flac, ogg,
    opus, aac, wma, mp4, mkv … — and stores it as mono 24 kHz WAV for OmniVoice."""
    if not get_job_dir(job_id).exists(): raise HTTPException(404, "Unknown job")
    if not shutil.which("ffmpeg"):
        raise HTTPException(500, "ffmpeg was not found on this machine, so the file can't be converted")
    d = _user_refs_dir(job_id); d.mkdir(parents=True, exist_ok=True)
    ref_id = f"uref_{uuid.uuid4().hex[:8]}"
    ext = Path(file.filename or "").suffix[:10] or ".bin"
    raw = d / f"{ref_id}_upload{ext}"
    out = d / f"{ref_id}.wav"
    try:
        with open(raw, "wb") as f: shutil.copyfileobj(file.file, f)
        proc = await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-i", str(raw), "-vn", "-ac", "1", "-ar", "24000",
             "-t", str(USER_REF_MAX_S), "-c:a", "pcm_s16le", str(out)],
            capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            tail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
            raise HTTPException(400, f"That file couldn't be read as audio ({tail[0][:140]})")
        with wave_mod.open(str(out), "rb") as wf:
            dur_ms = int(wf.getnframes() / float(wf.getframerate()) * 1000)
        if dur_ms < 500:
            out.unlink(missing_ok=True)
            raise HTTPException(400, "That audio is shorter than 0.5 s — use a clip with a few seconds of clear speech")
    finally:
        raw.unlink(missing_ok=True)
    refs = _load_user_refs(job_id)
    refs.append({"id": ref_id, "name": file.filename or ref_id, "duration_ms": dur_ms,
                 "text": (text or "").strip(), "created_at": datetime.now(timezone.utc).isoformat()})
    _save_user_refs(job_id, refs)
    return {"ok": True, **_user_ref_public(job_id, refs[-1])}

@app.get("/api/jobs/{job_id}/user_refs/{ref_id}/audio")
def serve_user_ref(job_id: str, ref_id: str):
    if not REF_ID_RE.match(ref_id): raise HTTPException(400)
    p = _user_refs_dir(job_id) / f"{ref_id}.wav"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/wav")

class UserRefTextReq(BaseModel):
    text: str

@app.post("/api/jobs/{job_id}/user_refs/{ref_id}/text")
def set_user_ref_text(job_id: str, ref_id: str, req: UserRefTextReq):
    """What is said in the uploaded clip (optional — skips OmniVoice's own transcription of it)."""
    refs = _load_user_refs(job_id)
    for r in refs:
        if r["id"] == ref_id: r["text"] = req.text.strip()
    _save_user_refs(job_id, refs)
    return {"ok": True}

@app.delete("/api/jobs/{job_id}/user_refs/{ref_id}")
async def delete_user_ref(job_id: str, ref_id: str):
    if not REF_ID_RE.match(ref_id): raise HTTPException(400)
    (_user_refs_dir(job_id) / f"{ref_id}.wav").unlink(missing_ok=True)
    _save_user_refs(job_id, [r for r in _load_user_refs(job_id) if r["id"] != ref_id])
    # any dialogue that pointed at it goes back to Auto
    p = get_job_dir(job_id) / "03_translations.json"
    if p.exists():
        data = load_json_file(p); changed = False
        for d in data.get("dialogues", []):
            if d.get("manual_ref_audio") == f"user:{ref_id}":
                d["manual_ref_audio"] = ""; changed = True
        if changed: save_json_file(p, data)
    await broadcast(job_id, {"type": "dialogue_updated", "dialogue_id": None})
    return {"ok": True}

# ---------------------------------------------------------------------------
# Speakers
@app.get("/api/jobs/{job_id}/speakers")
def get_speakers(job_id: str):
    p = get_job_dir(job_id) / "01_transcription.json"
    spks = {}
    if p.exists():
        data = load_json_file(p)
        for d in data.get("dialogues",[]):
            s = d.get("speaker_id","speaker_00"); spks[s] = spks.get(s,0) + 1
    # Manually-declared speakers with no dialogues yet (via speakers/add) —
    # so a newly-added character shows up as an empty row to drop lines onto,
    # instead of only ever appearing once a dialogue already references them.
    mp = get_job_dir(job_id) / "job_meta.json"
    if mp.exists():
        meta = load_json_file(mp)
        for s in meta.get("extra_speakers", []):
            spks.setdefault(s, 0)
    return [{"id": k, "dialogue_count": v} for k,v in sorted(spks.items())]

@app.post("/api/jobs/{job_id}/speakers/add")
async def add_speaker(job_id: str):
    p = get_job_dir(job_id) / "01_transcription.json"
    existing = set()
    if p.exists():
        for d in load_json_file(p).get("dialogues", []):
            existing.add(d.get("speaker_id","speaker_00"))
    mp = get_job_dir(job_id) / "job_meta.json"
    meta = load_json_file(mp) if mp.exists() else {}
    existing.update(meta.get("extra_speakers", []))
    n = 0
    while f"speaker_{n:02d}" in existing: n += 1
    new_id = f"speaker_{n:02d}"
    meta.setdefault("extra_speakers", []).append(new_id)
    save_json_file(mp, meta)
    await broadcast(job_id, {"type": "speaker_renamed"})  # reuses the same refresh path
    return {"ok": True, "speaker_id": new_id}

@app.get("/api/jobs/{job_id}/speaker_profiles")
def get_profiles(job_id: str):
    pd = get_job_dir(job_id) / "speaker_profiles"
    if not pd.exists(): return []
    return [{"name": f.stem, "path": str(f)} for f in sorted(pd.glob("*.wav"))]

@app.get("/api/jobs/{job_id}/speaker_dialogues/{speaker_id}")
def get_speaker_dialogue_refs(job_id: str, speaker_id: str):
    """Return dialogues >= 1s from this speaker for use as ref audio."""
    p = get_job_dir(job_id) / "01_transcription.json"
    if not p.exists(): return []
    data = load_json_file(p); refs = []
    for d in data.get("dialogues", []):
        if d.get("speaker_id") == speaker_id:
            dur = (d["end_ms"] - d["start_ms"]) / 1000
            if dur >= 1.0:
                refs.append({
                    "id":       d["id"],
                    "text":     d.get("text","")[:60],
                    "duration": round(dur, 1),
                    "url":      f"/api/jobs/{job_id}/vocals_audio/{d['id']}",
                })
    return refs

class RenameReq(BaseModel):
    old_id: str; new_id: str

@app.post("/api/jobs/{job_id}/speakers/rename")
async def rename_speaker(job_id: str, req: RenameReq):
    jd = get_job_dir(job_id)
    for fname in ["01_transcription.json", "03_translations.json"]:
        fp = jd / fname
        if not fp.exists(): continue
        data = load_json_file(fp)
        for d in data.get("dialogues", []):
            if d.get("speaker_id") == req.old_id:
                d["speaker_id"] = req.new_id
            for w in d.get("words", []):
                if w.get("speaker_id") == req.old_id:
                    w["speaker_id"] = req.new_id
        save_json_file(fp, data)
    old_wav = jd / "speaker_profiles" / f"{req.old_id}.wav"
    new_wav = jd / "speaker_profiles" / f"{req.new_id}.wav"
    if old_wav.exists(): old_wav.rename(new_wav)
    mp = jd / "job_meta.json"
    if mp.exists():
        meta = load_json_file(mp)
        extras = meta.get("extra_speakers", [])
        if req.old_id in extras:
            extras[extras.index(req.old_id)] = req.new_id
            save_json_file(mp, meta)
    await broadcast(job_id, {"type": "speaker_renamed", "old_id": req.old_id, "new_id": req.new_id})
    return {"ok": True}

@app.post("/api/jobs/{job_id}/speakers/build_profiles")
async def build_profiles(job_id: str):
    """Build speaker profiles using stdlib wave only — no numpy/torch needed."""
    jd         = get_job_dir(job_id)
    trans_path = jd / "01_transcription.json"
    if not trans_path.exists(): raise HTTPException(404, "transcription not found")
    vocals_path = job_audio(job_id, "vocals")
    if not vocals_path.exists(): raise HTTPException(404, f"vocals WAV not found — run Stage 00 first")

    data      = load_json_file(trans_path)
    dialogues = data.get("dialogues", [])
    profiles_dir = jd / "speaker_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    spk_map: Dict[str, list] = {}
    for dlg in dialogues:
        spk = dlg.get("speaker_id","speaker_00")
        spk_map.setdefault(spk, []).append(dlg)

    saved = []
    try:
        with wave_mod.open(str(vocals_path), 'rb') as wf:
            sr      = wf.getframerate()
            nch     = wf.getnchannels()
            sw      = wf.getsampwidth()
            n_total = wf.getnframes()
            MAX_FR  = int(10.0 * sr)
            MIN_FR  = int(1.0  * sr)

            for spk_id, segs in spk_map.items():
                segs_sorted    = sorted(segs, key=lambda s: s["end_ms"]-s["start_ms"], reverse=True)
                collected      = b""
                total_frames   = 0
                for seg in segs_sorted:
                    if total_frames >= MAX_FR: break
                    s0 = max(0, int(seg["start_ms"]/1000*sr))
                    s1 = min(n_total, int(seg["end_ms"]/1000*sr))
                    if s1-s0 < MIN_FR: continue
                    remaining = MAX_FR - total_frames
                    wf.setpos(s0)
                    raw = wf.readframes(min(s1-s0, remaining))
                    collected    += raw
                    total_frames += min(s1-s0, remaining)
                if not collected: continue
                out = profiles_dir / f"{spk_id}.wav"
                with wave_mod.open(str(out), 'wb') as ow:
                    ow.setnchannels(nch); ow.setsampwidth(sw); ow.setframerate(sr)
                    ow.writeframes(collected)
                saved.append(spk_id)
    except Exception as e:
        raise HTTPException(500, f"Profile build failed: {e}")

    await broadcast(job_id, {"type": "profiles_built", "speakers": saved})
    return {"ok": True, "saved": saved}

# ---------------------------------------------------------------------------
# TTS rerun — saves translation update first
class RerunReq(BaseModel):
    dialogue_id: str

@app.post("/api/jobs/{job_id}/tts/rerun")
async def rerun_tts(job_id: str, req: RerunReq):
    asyncio.create_task(_rerun_tts(job_id, req.dialogue_id))
    return {"ok": True}

async def _rerun_tts(job_id: str, dlg_id: str):
    await broadcast(job_id, {"type": "tts_rerun_started", "dialogue_id": dlg_id})
    script = STAGES_DIR / "04_tts" / "rerun_single.py"
    if not script.exists():
        await broadcast(job_id, {"type": "tts_rerun_failed", "dialogue_id": dlg_id, "error": "rerun_single.py not found"})
        return
    env = os.environ.copy(); env["PATH"] = f"{CONDA_PATH}:{env.get('PATH','')}"
    cmd = [CONDA_BIN,"run","--no-capture-output","-n","dub_tts","python",str(script),
           "--job_id",job_id,"--dialogue_id",dlg_id,"--config",str(CONFIG_PATH)]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
           stderr=asyncio.subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))
    async for line in proc.stdout:
        await broadcast(job_id, {"type":"log","stage":"04_tts_rerun","text":line.decode(errors="replace").rstrip()})
    await proc.wait()
    if proc.returncode == 0: reset_dub_trims(job_id, dlg_id)
    t = "tts_rerun_complete" if proc.returncode==0 else "tts_rerun_failed"
    await broadcast(job_id, {"type": t, "dialogue_id": dlg_id})

# Whisper models the user can pick. "medium" is the default and the recommended one;
# "large-v3" is the optional heavier choice when the transcription isn't good.
ASR_MODELS      = ("medium", "large-v3")
ASR_MODEL_DEFAULT = "medium"

def _clean_asr_model(m: Optional[str]) -> str:
    m = (m or "").strip().lower()
    return m if m in ASR_MODELS else ASR_MODEL_DEFAULT

class AsrModelReq(BaseModel):
    model: str

@app.get("/api/jobs/{job_id}/asr_model")
def get_asr_model(job_id: str):
    mp = get_job_dir(job_id) / "job_meta.json"
    if not mp.exists(): raise HTTPException(404)
    return {"model": _clean_asr_model(load_json_file(mp).get("asr_model")), "models": list(ASR_MODELS), "default": ASR_MODEL_DEFAULT}

@app.post("/api/jobs/{job_id}/asr_model")
def set_asr_model(job_id: str, req: AsrModelReq):
    """The model for the WHOLE speech-recognition stage of this job (all dialogues)."""
    if req.model.strip().lower() not in ASR_MODELS: raise HTTPException(400, f"Model must be one of {ASR_MODELS}")
    mp = get_job_dir(job_id) / "job_meta.json"
    if not mp.exists(): raise HTTPException(404)
    meta = load_json_file(mp); meta["asr_model"] = req.model.strip().lower(); save_json_file(mp, meta)
    return {"ok": True, "model": meta["asr_model"]}

class AsrRerunReq(BaseModel):
    dialogue_id: str
    model: Optional[str] = None      # for THIS dialogue only; independent of the job's model

@app.post("/api/jobs/{job_id}/asr/rerun")
async def rerun_asr(job_id: str, req: AsrRerunReq):
    asyncio.create_task(_rerun_asr(job_id, req.dialogue_id, _clean_asr_model(req.model) if req.model else None))
    return {"ok": True}

async def _rerun_asr(job_id: str, dlg_id: str, model: Optional[str] = None):
    await broadcast(job_id, {"type": "asr_rerun_started", "dialogue_id": dlg_id})
    script = STAGES_DIR / "01_asr" / "rerun_single.py"
    if not script.exists():
        await broadcast(job_id, {"type": "asr_rerun_failed", "dialogue_id": dlg_id, "error": "rerun_single.py not found"})
        return
    env = os.environ.copy(); env["PATH"] = f"{CONDA_PATH}:{env.get('PATH','')}"
    cmd = [CONDA_BIN,"run","--no-capture-output","-n","dub_asr","python",str(script),
           "--job_id",job_id,"--dialogue_id",dlg_id,"--config",str(CONFIG_PATH)]
    if model: cmd += ["--model", model]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
           stderr=asyncio.subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))
    async for line in proc.stdout:
        await broadcast(job_id, {"type":"log","stage":"01_asr_rerun","text":line.decode(errors="replace").rstrip()})
    await proc.wait()
    if proc.returncode == 0:
        # Text changed — any existing TTS take for this dialogue no longer
        # matches what's being read out, so flag it rather than silently
        # leaving a "TTS ✓" badge on now-mismatched audio.
        mark_tts_stale(job_id, dlg_id)
        await broadcast(job_id, {"type": "asr_rerun_complete", "dialogue_id": dlg_id})
    else:
        await broadcast(job_id, {"type": "asr_rerun_failed", "dialogue_id": dlg_id})

# ---------------------------------------------------------------------------
# Pipeline
@app.post("/api/jobs/{job_id}/pipeline/run")
async def run_pipeline(job_id: str):
    if job_id in pipeline_tasks and not pipeline_tasks[job_id].done():
        raise HTTPException(400, "Already running")
    ev = asyncio.Event()
    pipeline_state[job_id] = {"paused_at": None, "resume_event": ev}
    pipeline_tasks[job_id] = asyncio.create_task(_run_pipeline(job_id))
    return {"ok": True}

@app.post("/api/jobs/{job_id}/pipeline/resume")
async def resume_pipeline(job_id: str):
    s = pipeline_state.get(job_id)
    if s and s.get("resume_event"):
        # A pipeline task in THIS process is actively paused at a checkpoint —
        # just unblock it.
        s["resume_event"].set()
        return {"ok": True, "mode": "unpaused"}
    # No live in-memory pause (server restarted, or job was never started this
    # session). Fall back to (re)starting the pipeline — _run_pipeline already
    # skips any stage the manifest marks "completed", so this correctly
    # continues from wherever the job actually left off instead of silently
    # doing nothing.
    if job_id in pipeline_tasks and not pipeline_tasks[job_id].done():
        raise HTTPException(400, "Already running")
    await broadcast(job_id, {"type": "pipeline_state_recovered"})
    ev = asyncio.Event()
    pipeline_state[job_id] = {"paused_at": None, "resume_event": ev}
    pipeline_tasks[job_id] = asyncio.create_task(_run_pipeline(job_id))
    return {"ok": True, "mode": "restarted"}

@app.post("/api/jobs/{job_id}/pipeline/stop")
async def stop_pipeline(job_id: str):
    t = pipeline_tasks.get(job_id)
    if t and not t.done(): t.cancel()
    pipeline_state.pop(job_id, None)
    await broadcast(job_id, {"type": "pipeline_stopped", "reason": "user stopped"})
    return {"ok": True}

@app.get("/api/jobs/{job_id}/pipeline/state")
def pipeline_state_api(job_id: str):
    s = pipeline_state.get(job_id, {})
    return {"running": job_id in pipeline_tasks and not pipeline_tasks[job_id].done(),
            "paused_at": s.get("paused_at")}

async def _run_pipeline(job_id: str):
    state = pipeline_state[job_id]
    await broadcast(job_id, {"type": "pipeline_started"})
    for stage in STAGES:
        sid = stage["id"]
        if sid == "05_assembly":
            # Assembly only ever runs via the explicit /export call — stop
            # cleanly here and tell the frontend it's ready, rather than
            # broadcasting a "paused" state with no matching UI (that used
            # to leave the Run/Resume button hidden with nothing to click).
            await broadcast(job_id, {"type": "ready_for_export"})
            return
        if sid == "03_translation":
            # No automatic translator anymore (Google Translate was too
            # unreliable) — translation only completes via the user typing it
            # themselves or importing a response file. If that's already
            # done (they handled it before hitting Run All), carry on; if
            # not, stop cleanly here rather than trying to auto-run anything.
            manifest = get_manifest(job_id)
            if manifest.get("stages",{}).get(sid,{}).get("status") != "completed":
                await broadcast(job_id, {"type": "waiting_on_translation"})
                return
            continue
        manifest = get_manifest(job_id)
        if manifest.get("stages",{}).get(sid,{}).get("status") == "completed":
            await broadcast(job_id, {"type":"stage_skipped","stage":sid}); continue
        await broadcast(job_id, {"type":"stage_started","stage":sid})
        ok = await _run_stage(job_id, sid, stage["env"])
        if not ok:
            await broadcast(job_id, {"type":"stage_failed","stage":sid})
            await broadcast(job_id, {"type":"pipeline_stopped","reason":f"{sid} failed"}); return
        manifest = get_manifest(job_id)
        manifest["stages"][sid]["status"]       = "completed"
        manifest["stages"][sid]["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest(job_id, manifest)
        await broadcast(job_id, {"type":"stage_completed","stage":sid})
        if sid in CHECKPOINTS:
            cp = CHECKPOINTS[sid]
            state["paused_at"]    = cp
            state["resume_event"] = asyncio.Event()
            await broadcast(job_id, {"type":"pipeline_paused","checkpoint":cp,"stage":sid})
            await state["resume_event"].wait()
            state["paused_at"] = None
            await broadcast(job_id, {"type":"pipeline_resumed","stage":sid})
    await broadcast(job_id, {"type":"pipeline_complete"})

class RunStageReq(BaseModel):
    stage_id: str

@app.post("/api/jobs/{job_id}/pipeline/run_stage")
async def run_single_stage_endpoint(job_id: str, req: RunStageReq):
    """Run exactly one stage, ignoring manifest-completed skip logic (i.e.
    force re-run), without touching any other stage. Used by the per-stage
    Run buttons so the user isn't forced through the whole pipeline."""
    if job_id in pipeline_tasks and not pipeline_tasks[job_id].done():
        raise HTTPException(400, "Already running")
    if req.stage_id == "03_translation":
        raise HTTPException(400, "Translation no longer runs automatically — type it yourself or import a file")
    valid = {s["id"]: s for s in STAGES}
    if req.stage_id not in valid:
        raise HTTPException(400, "Unknown stage")
    pipeline_tasks[job_id] = asyncio.create_task(_run_single_stage(job_id, req.stage_id))
    return {"ok": True}

def reset_stages_from(job_id: str, sid: str) -> List[str]:
    """Re-running a stage: mark it AND every stage after it as pending in manifest.json.

    Every stage script is restart-safe — it looks at the manifest and, if its own stage says
    "completed", exits at once ("already completed"). So a re-run must first flip the manifest
    back to pending, otherwise the script just quits and nothing is re-done. Stages after it are
    built on its OLD output, so they go back to pending too; stages before it are never touched.
    """
    manifest = get_manifest(job_id)
    order = [s["id"] for s in STAGES]
    reset = []
    for later in order[order.index(sid):]:
        entry = manifest["stages"].setdefault(later, {})
        entry["status"] = "pending"
        entry["completed_at"] = None
        entry["failed_at"] = None
        reset.append(later)
    save_manifest(job_id, manifest)
    return reset

def purge_stage_outputs(job_id: str, sid: str) -> None:
    """Some stages also skip work when their output files already exist (vocals separation checks
    for finished vocals/no_vocals files and resumes from saved chunks). A forced re-run must start
    from scratch, so those leftovers go first. Everything else a stage writes is simply overwritten."""
    jd = get_job_dir(job_id)
    if sid == "00_vocals":
        for p in (jp.vocals_wav(jd), jp.no_vocals_wav(jd), jd / "00_vocals_progress.json"):
            try: p.unlink(missing_ok=True)
            except Exception: pass
        shutil.rmtree(jd / "00_vocals_chunks", ignore_errors=True)

async def _run_single_stage(job_id: str, sid: str):
    stage = next(s for s in STAGES if s["id"] == sid)
    # Flip the manifest to pending FIRST (this stage + all later ones) — otherwise the stage script
    # sees "completed" and exits without doing anything.
    reset = reset_stages_from(job_id, sid)
    purge_stage_outputs(job_id, sid)
    await broadcast(job_id, {"type": "stages_reset", "stages": reset})
    await broadcast(job_id, {"type": "stage_started", "stage": sid})
    ok = await _run_stage(job_id, sid, stage["env"])
    if not ok:
        await broadcast(job_id, {"type": "stage_failed", "stage": sid})
        await broadcast(job_id, {"type": "pipeline_stopped", "reason": f"{sid} failed"})
        return
    manifest = get_manifest(job_id)
    manifest["stages"][sid]["status"]       = "completed"
    manifest["stages"][sid]["completed_at"] = datetime.now(timezone.utc).isoformat()
    # Downstream stages were built on this stage's OLD output — their results
    # are now stale, so mark them pending again (re-running stage 3 should
    # invalidate stage 4/5, but never touch 1/2 before it).
    order = [s["id"] for s in STAGES]
    pos = order.index(sid)
    invalidated = []
    for later_sid in order[pos+1:]:
        entry = manifest["stages"].get(later_sid)
        if entry and entry.get("status") == "completed":
            entry["status"] = "pending"
            entry["completed_at"] = None
            invalidated.append(later_sid)
    save_manifest(job_id, manifest)
    await broadcast(job_id, {"type": "stage_completed", "stage": sid, "invalidated": invalidated})
    await broadcast(job_id, {"type": "pipeline_stopped", "reason": "single stage complete"})

async def _run_stage(job_id: str, sid: str, env_name: str) -> bool:
    script = STAGES_DIR / sid / "run.py"
    if not script.exists():
        await broadcast(job_id, {"type":"log","stage":sid,"text":f"Script not found: {script}"}); return False
    env = os.environ.copy(); env["PATH"] = f"{CONDA_PATH}:{env.get('PATH','')}"
    cmd = [CONDA_BIN,"run","--no-capture-output","-n",env_name,"python",str(script),
           "--job_id",job_id,"--config",str(CONFIG_PATH)]
    # Pass language from job_meta
    mp = get_job_dir(job_id) / "job_meta.json"
    if mp.exists():
        meta = load_json_file(mp)
        src  = meta.get("source_lang","")
        tgt  = meta.get("target_lang","")
        if sid == "01_asr" and src:          cmd += ["--source_lang", src]
        if sid == "01_asr":                  cmd += ["--asr_model", _clean_asr_model(meta.get("asr_model"))]   # the job's model (default: medium)
        if sid == "03_translation":
            if src: cmd += ["--source_lang", src]
            if tgt: cmd += ["--target_lang",  tgt]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
           stderr=asyncio.subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))
    async for line in proc.stdout:
        await broadcast(job_id, {"type":"log","stage":sid,"text":line.decode(errors="replace").rstrip()})
    await proc.wait()
    if proc.returncode == 0 and sid == "04_tts": reset_dub_trims(job_id)
    return proc.returncode == 0

# ---------------------------------------------------------------------------
# Export
@app.post("/api/jobs/{job_id}/export")
async def export_video(job_id: str):
    if job_id in pipeline_tasks and not pipeline_tasks[job_id].done():
        raise HTTPException(400,"Pipeline still running")
    pipeline_tasks[job_id] = asyncio.create_task(_run_export(job_id))
    return {"ok": True}

async def _run_export(job_id: str):
    await broadcast(job_id, {"type":"export_started"})
    ok = await _run_stage(job_id, "05_assembly", "dub_assembly")
    if ok:
        m = get_manifest(job_id)
        m["stages"]["05_assembly"]["status"]       = "completed"
        m["stages"]["05_assembly"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_manifest(job_id, m)
        out = _final_output(job_id)
        await broadcast(job_id, {"type":"export_complete","download_url":f"/api/jobs/{job_id}/download","exists":out is not None})
    else:
        await broadcast(job_id, {"type":"export_failed"})

def _final_output(job_id: str) -> Optional[Path]:
    """The exported file — .mkv or .mp4 depending on what the original needed (newest wins)."""
    od = jp.output_dir(get_job_dir(job_id))
    found = [p for p in (od / f"{job_id}_dubbed.mkv", od / f"{job_id}_dubbed.mp4") if p.exists()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None

@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    p = _final_output(job_id)
    if not p: raise HTTPException(404)
    return FileResponse(str(p), media_type="video/x-matroska" if p.suffix == ".mkv" else "video/mp4", filename=p.name)

# ---------------------------------------------------------------------------
# Mixer — how loud the three layers are (background, vocals, dubbed). Used by the preview
# AND by the export (Stage 05 reads the same file), so both sound the same.
MIX_DEFAULT = {"no_vocals": 1.0, "vocals": 1.0, "dub": 1.0}

class MixReq(BaseModel):
    no_vocals: Optional[float] = None
    vocals:    Optional[float] = None
    dub:       Optional[float] = None

def _read_mix(job_id: str) -> dict:
    p = get_job_dir(job_id) / "mix.json"
    mix = dict(MIX_DEFAULT)
    if p.exists():
        try:
            for k, v in load_json_file(p).items():
                if k in mix: mix[k] = max(0.0, min(4.0, float(v)))
        except Exception: pass
    return mix

@app.get("/api/jobs/{job_id}/mix")
def get_mix(job_id: str): return _read_mix(job_id)

@app.post("/api/jobs/{job_id}/mix")
def set_mix(job_id: str, req: MixReq):
    if not (get_job_dir(job_id) / "job_meta.json").exists(): raise HTTPException(404)
    mix = _read_mix(job_id)
    for k in MIX_DEFAULT:
        v = getattr(req, k)
        if v is not None: mix[k] = max(0.0, min(4.0, float(v)))
    save_json_file(get_job_dir(job_id) / "mix.json", mix)
    return mix

# ---------------------------------------------------------------------------
# Zip the present work / import a zip — a job folder is self-contained, so it can be shared
# and resumed anywhere. Browser viewing copies are left out (they are re-made on demand).
import tempfile, zipfile
from starlette.background import BackgroundTask

ZIP_SKIP_NAMES   = {"preview.mp4", "assembled_preview.mp4", "05_dubbed_audio_temp.wav", jp.LAYOUT_MARK}
ZIP_STORED_EXTS  = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".zip"}

@app.get("/api/jobs/{job_id}/zip")
def zip_job(job_id: str):
    """The whole job folder as one .zip — video, audio tracks, every JSON, TTS clips, speaker
    profiles, uploaded refs, the export. Drop it into any AI Dub Studio (Import) to continue."""
    jd = get_job_dir(job_id)
    if not (jd / "job_meta.json").exists(): raise HTTPException(404, "Unknown job")
    tmp = tempfile.NamedTemporaryFile(prefix=f"{job_id}_", suffix=".zip", delete=False)
    tmp.close()
    try:
        with zipfile.ZipFile(tmp.name, "w", allowZip64=True) as zf:
            for f in sorted(jd.rglob("*")):
                if not f.is_file(): continue
                rel = f.relative_to(jd)
                if f.name in ZIP_SKIP_NAMES or f.name.endswith(".part.mp4") or "__pycache__" in rel.parts: continue
                method = zipfile.ZIP_STORED if f.suffix.lower() in ZIP_STORED_EXTS else zipfile.ZIP_DEFLATED
                zf.write(f, arcname=f"{job_id}/{rel.as_posix()}", compress_type=method)
    except Exception as e:
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(500, f"Could not create the zip: {e}")
    return FileResponse(tmp.name, media_type="application/zip", filename=f"{job_id}.zip",
                        background=BackgroundTask(lambda: Path(tmp.name).unlink(missing_ok=True)))

@app.post("/api/jobs/import_zip")
async def import_job_zip(file: UploadFile = File(...)):
    """Unpack a job zip made by 'Zip this job' into jobs/ and return its id, ready to continue."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="_import_", dir=str(INBOX_DIR)))
    try:
        zpath = work / "upload.zip"
        with open(zpath, "wb") as f: shutil.copyfileobj(file.file, f)
        if not zipfile.is_zipfile(zpath): raise HTTPException(400, "That file is not a zip")
        ex = work / "x"; ex.mkdir()
        with zipfile.ZipFile(zpath) as zf:
            for m in zf.infolist():                       # refuse paths that would escape the folder
                target = (ex / m.filename).resolve()
                if ex.resolve() not in target.parents and target != ex.resolve():
                    raise HTTPException(400, "The zip contains an unsafe path")
            await asyncio.to_thread(zf.extractall, str(ex))
        meta_files = sorted(ex.rglob("job_meta.json"), key=lambda p: len(p.parts))
        if not meta_files: raise HTTPException(400, "No job_meta.json inside — this is not an AI Dub Studio job zip")
        src = meta_files[0].parent
        meta = load_json_file(meta_files[0])
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", str(meta.get("job_id") or src.name or "imported_job")) or "imported_job"
        job_id, n = base, 1
        while (JOBS_DIR / job_id).exists():               # never overwrite a job that already exists here
            n += 1; job_id = f"{base}_imported{'' if n == 2 else n}"
        shutil.move(str(src), str(JOBS_DIR / job_id))
        jd = JOBS_DIR / job_id
        jp.normalize_imported_job(jd)
        meta = load_json_file(jd / "job_meta.json"); meta["job_id"] = job_id; save_json_file(jd / "job_meta.json", meta)
        mp = jd / "manifest.json"
        if mp.exists():
            m = load_json_file(mp); m["job_id"] = job_id; save_json_file(mp, m)
        _MIGRATED.add(job_id)
        return {"ok": True, "job_id": job_id}
    finally:
        shutil.rmtree(work, ignore_errors=True)

# The editor's player shows the REAL assembled result once an export exists — until the
# dialogues/dubs are edited again (then the export is out of date and the preview takes over).
@app.get("/api/jobs/{job_id}/assembled_status")
def assembled_status(job_id: str):
    jd = get_job_dir(job_id)
    pv = jd / "assembled_preview.mp4"
    if not pv.exists() or _final_output(job_id) is None:
        return {"available": False, "stale": False, "version": 0}
    t = pv.stat().st_mtime
    watched = [jd / "01_transcription.json", jd / "03_translations.json", jd / "04_tts" / "tts_manifest.json", jd / "mix.json"]
    stale = any(p.exists() and p.stat().st_mtime > t + 1.0 for p in watched)
    return {"available": True, "stale": stale, "version": int(t)}

@app.get("/api/jobs/{job_id}/assembled_video")
def assembled_video(job_id: str):
    pv = get_job_dir(job_id) / "assembled_preview.mp4"
    if not pv.exists(): raise HTTPException(404)
    return FileResponse(str(pv), media_type="video/mp4")

# ---------------------------------------------------------------------------
# Audio / Video serving
# Browsers only play a few codec/container combinations (H.264 in mp4 is the safe one).
# Anything else — webm variants, mkv, avi, wmv, flv, HEVC, mpeg-ts, … — is converted ONCE into
# a lightweight H.264/AAC preview (jobs/<id>/preview.mp4). It is only for the editor's player:
# the dubbing stages and the export always use the original file.
_PREVIEW_STATE: Dict[str, dict] = {}     # job_id -> {"state": "processing"|"failed", "error": str}

def _input_video_path(job_id: str) -> Path:
    jd = get_job_dir(job_id)
    if not (jd / "job_meta.json").exists(): raise HTTPException(404)
    vp = jp.find_source_video(jd)
    if vp is None: raise HTTPException(404, "Original video file is missing (expected in the job's source/ folder)")
    return vp

def _probe_streams(vp: Path) -> dict:
    """{"v": codec|None, "a": codec|None, "pix": str, "fmt": str} via ffprobe ({} if unavailable)."""
    if not shutil.which("ffprobe"): return {}
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,pix_fmt:format=format_name",
                            "-of", "json", str(vp)], capture_output=True, text=True, timeout=60)
        j = json.loads(r.stdout or "{}")
    except Exception:
        return {}
    out = {"v": None, "a": None, "pix": "", "fmt": (j.get("format") or {}).get("format_name", "")}
    for s in j.get("streams", []):
        if s.get("codec_type") == "video" and out["v"] is None: out["v"], out["pix"] = s.get("codec_name"), s.get("pix_fmt", "")
        if s.get("codec_type") == "audio" and out["a"] is None: out["a"] = s.get("codec_name")
    return out

def _browser_can_play_directly(vp: Path) -> bool:
    if vp.suffix.lower() not in (".mp4", ".m4v", ".mov"): return False
    p = _probe_streams(vp)
    return bool(p) and p["v"] == "h264" and p["pix"] in ("yuv420p", "yuvj420p", "") and p["a"] in (None, "aac", "mp3")

def _preview_path(job_id: str) -> Path: return get_job_dir(job_id) / "preview.mp4"

def _make_preview(job_id: str, vp: Path) -> None:
    out = _preview_path(job_id); tmp = out.with_suffix(".part.mp4")
    cmd = ["ffmpeg", "-y", "-i", str(vp), "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
           "-vf", "scale=-2:'min(720,ih)',format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-movflags", "+faststart", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists():
            tail = (r.stderr or "").strip().splitlines()[-1:] or ["ffmpeg failed"]
            _PREVIEW_STATE[job_id] = {"state": "failed", "error": tail[0][:200]}
            tmp.unlink(missing_ok=True); return
        tmp.replace(out)
        _PREVIEW_STATE.pop(job_id, None)
    except Exception as e:
        _PREVIEW_STATE[job_id] = {"state": "failed", "error": str(e)[:200]}

@app.get("/api/jobs/{job_id}/video_status")
async def video_status(job_id: str):
    """state: ready (play /video) | processing (converting, poll again) | failed."""
    vp = _input_video_path(job_id)
    if _preview_path(job_id).exists() or await asyncio.to_thread(_browser_can_play_directly, vp):
        return {"state": "ready"}
    st = _PREVIEW_STATE.get(job_id)
    if st and st["state"] == "processing": return {"state": "processing"}
    if st and st["state"] == "failed":     return {"state": "failed", "error": st.get("error", "")}
    if not shutil.which("ffmpeg"):
        return {"state": "failed", "error": "ffmpeg was not found, so this video can't be converted for preview"}
    _PREVIEW_STATE[job_id] = {"state": "processing"}
    asyncio.get_running_loop().run_in_executor(None, _make_preview, job_id, vp)
    return {"state": "processing"}

@app.post("/api/jobs/{job_id}/video_status/retry")
async def video_retry(job_id: str):
    _PREVIEW_STATE.pop(job_id, None); _preview_path(job_id).unlink(missing_ok=True)
    return await video_status(job_id)

@app.get("/api/jobs/{job_id}/video")
def serve_video(job_id: str):
    vp = _input_video_path(job_id)
    pv = _preview_path(job_id)
    if pv.exists(): return FileResponse(str(pv), media_type="video/mp4")
    if _browser_can_play_directly(vp): return FileResponse(str(vp), media_type="video/mp4")
    raise HTTPException(409, "Preview is still being prepared")

@app.get("/api/jobs/{job_id}/tts_audio/{dialogue_id}")
def serve_tts(job_id: str, dialogue_id: str):
    p = get_job_dir(job_id) / "04_tts" / f"{dialogue_id}.wav"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/wav")

@app.get("/api/jobs/{job_id}/speaker_audio/{speaker_id}")
def serve_speaker(job_id: str, speaker_id: str):
    p = get_job_dir(job_id) / "speaker_profiles" / f"{speaker_id}.wav"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/wav")

@app.get("/api/jobs/{job_id}/audio_track/{track}")
def serve_audio_track(job_id: str, track: str):
    """Serve a full mixdown-stage audio track. track: no_vocals | vocals | original.
    Used by the frontend's dub-preview toggle to play a background layer
    while TTS clips are triggered on top, without waiting for Stage 05."""
    if track not in ("no_vocals", "vocals", "original"): raise HTTPException(404, "Unknown track")
    p = job_audio(job_id, track)
    if not p.exists(): raise HTTPException(404)
    return FileResponse(str(p), media_type="audio/wav")

@app.get("/api/jobs/{job_id}/vocals_audio/{dialogue_id}")
def serve_vocals_clip(job_id: str, dialogue_id: str):
    """Serve exact [start_ms, end_ms] slice of vocals WAV."""
    p = get_job_dir(job_id) / "01_transcription.json"
    if not p.exists(): raise HTTPException(404)
    data = load_json_file(p)
    dlg  = next((d for d in data.get("dialogues",[]) if d["id"]==dialogue_id), None)
    if not dlg: raise HTTPException(404)
    vp = job_audio(job_id, "vocals")
    if not vp.exists(): raise HTTPException(404)
    try:
        with wave_mod.open(str(vp), 'rb') as wf:
            sr       = wf.getframerate()
            nch      = wf.getnchannels()
            sw       = wf.getsampwidth()
            n_total  = wf.getnframes()
            s0       = max(0, min(int(dlg["start_ms"]/1000*sr), n_total))
            s1       = max(s0, min(int(dlg["end_ms"]/1000*sr), n_total))
            wf.setpos(s0)
            frames   = wf.readframes(s1 - s0)
        buf = io.BytesIO()
        with wave_mod.open(buf, 'wb') as ow:
            ow.setnchannels(nch); ow.setsampwidth(sw); ow.setframerate(sr)
            ow.writeframes(frames)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav",
            headers={"Content-Disposition": f"inline; filename={dialogue_id}.wav"})
    except Exception as e:
        raise HTTPException(500, f"Slice failed: {e}")

# ---------------------------------------------------------------------------
FRONTEND_BUILD = Path(__file__).parent / "frontend" / "dist"
if FRONTEND_BUILD.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")