#!/usr/bin/env python3
"""
Stage 03: Translation
=====================
Fixes vs previous:
  - deep_translator GoogleTranslator with retry (3 attempts, exponential backoff)
  - Strips HTML error responses (Google 500 errors returned as HTML text)
  - Accepts --source_lang and --target_lang from server.py (via job_meta.json)
  - Same-language passthrough when source == target
  - Batch with delay to avoid rate limiting
"""

import argparse, json, logging, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

STAGE_KEY  = "03_translation"
STAGE_NAME = "03_translation"
API_DELAY  = 0.3   # seconds between calls
MAX_RETRY  = 3     # retries per dialogue


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job_id",      required=True)
    p.add_argument("--config",      required=True)
    p.add_argument("--source_lang", required=False, default=None)
    p.add_argument("--target_lang", required=False, default=None)
    return p.parse_args()


def load_config(path):
    with open(path, encoding="utf-8") as f: return yaml.safe_load(f)


def setup_logging(job_id, logs_dir):
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{job_id}_{STAGE_NAME}.log")
    logger   = logging.getLogger(STAGE_NAME)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%dT%H:%M:%S")
    fh  = logging.FileHandler(log_path, encoding="utf-8"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    ch  = logging.StreamHandler(sys.stdout);               ch.setLevel(logging.INFO);  ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def load_manifest(job_id, jobs_dir):
    with open(os.path.join(jobs_dir, job_id, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)

def save_manifest(manifest, job_id, jobs_dir):
    with open(os.path.join(jobs_dir, job_id, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

def mark_complete(job_id, manifest, jobs_dir, logger):
    manifest["stages"][STAGE_KEY]["status"]       = "completed"
    manifest["stages"][STAGE_KEY]["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["stages"][STAGE_KEY]["failed_at"]    = None
    save_manifest(manifest, job_id, jobs_dir)
    logger.info(f"Manifest → {STAGE_KEY}: completed")

def mark_failed(job_id, manifest, jobs_dir, logger):
    manifest["stages"][STAGE_KEY]["status"]    = "failed"
    manifest["stages"][STAGE_KEY]["failed_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest, job_id, jobs_dir)
    logger.error(f"Manifest → {STAGE_KEY}: failed")


def is_html_error(text: str) -> bool:
    """Detect when Google returns an HTML error page instead of translation."""
    t = text.strip().lower()
    return (t.startswith("<!doctype") or t.startswith("<html") or
            "error 500" in t or "that's an error" in t or
            "please try again" in t or len(text) > 2000)


def translate_with_retry(text: str, translator, logger, dialogue_id: str, max_retry: int = MAX_RETRY):
    """
    Translate with exponential backoff retry.
    Returns (translated_text, is_passthrough).
    """
    if not text.strip() or len(text.strip()) < 2:
        return text, True

    for attempt in range(1, max_retry + 1):
        try:
            result = translator.translate(text)

            # Check for HTML error response
            if not result or not result.strip():
                logger.warning(f"[{dialogue_id}] Empty result (attempt {attempt})")
                time.sleep(attempt * 1.5)
                continue

            if is_html_error(result):
                logger.warning(f"[{dialogue_id}] HTML error response from Google (attempt {attempt}) — retrying")
                time.sleep(attempt * 2.0)
                continue

            return result.strip(), False

        except Exception as e:
            err_str = str(e)
            logger.warning(f"[{dialogue_id}] Translation error attempt {attempt}: {err_str[:80]}")
            if attempt < max_retry:
                time.sleep(attempt * 2.0)

    logger.error(f"[{dialogue_id}] All {max_retry} attempts failed — using original text")
    return text, True


def build_translator(source_lang: str, target_lang: str, logger):
    from deep_translator import GoogleTranslator
    # Normalise language codes
    src = source_lang if source_lang and source_lang not in ("auto","") else "auto"
    tgt = target_lang if target_lang and target_lang not in ("","auto") else "en"

    try:
        translator = GoogleTranslator(source=src, target=tgt)
        logger.info(f"GoogleTranslator: {src} → {tgt}")
        return translator, src, tgt
    except Exception as e:
        logger.warning(f"Translator init failed ({e}) — falling back to auto→{tgt}")
        return GoogleTranslator(source="auto", target=tgt), "auto", tgt


def main():
    args   = parse_args()
    config = load_config(args.config)
    logger = setup_logging(args.job_id, config["paths"]["logs"])

    logger.info("=" * 60)
    logger.info(f"Stage 03 Translation | job_id: {args.job_id}")
    logger.info("=" * 60)

    jobs_dir = config["paths"]["jobs"]
    manifest = load_manifest(args.job_id, jobs_dir)

    if manifest["stages"][STAGE_KEY]["status"] == "completed":
        logger.info("Already completed. Skipping.")
        return

    # Load transcription directly
    trans_path = os.path.join(jobs_dir, args.job_id, "01_transcription.json")
    with open(trans_path, encoding="utf-8") as f:
        transcription = json.load(f)

    dialogues         = transcription.get("dialogues", [])
    detected_language = transcription.get("detected_language", "")
    total             = len(dialogues)
    logger.info(f"Loaded {total} dialogues | detected: {detected_language}")

    # ── Language resolution ──────────────────────────────────────────────
    # Priority: CLI arg (from job_meta via server) → ASR detected → config → auto
    if args.source_lang and args.source_lang not in ("auto",""):
        source_lang = args.source_lang
        logger.info(f"Source language from UI: {source_lang}")
    elif detected_language and detected_language not in ("auto","unknown",""):
        source_lang = detected_language
        logger.info(f"Source language from ASR: {source_lang}")
    else:
        cfg_lang    = config.get("language",{}).get("source","auto")
        source_lang = cfg_lang if cfg_lang not in ("auto","") else "auto"
        logger.info(f"Source language from config: {source_lang}")

    if args.target_lang and args.target_lang not in ("","auto"):
        target_lang = args.target_lang
        logger.info(f"Target language from UI: {target_lang}")
    else:
        target_lang = config.get("language",{}).get("target","en") or "en"
        logger.info(f"Target language from config: {target_lang}")

    # ── Same-language passthrough ────────────────────────────────────────
    src_norm = source_lang.lower().strip()
    tgt_norm = target_lang.lower().strip()
    passthrough_all = (
        src_norm == tgt_norm or
        (src_norm in ("en","english") and tgt_norm in ("en","english")) or
        src_norm == "auto"  # can't translate if we don't know source
    )

    if passthrough_all:
        logger.info(f"Passthrough mode ({source_lang} → {target_lang}) — no translation needed")
        translator = None
    else:
        translator, source_lang, target_lang = build_translator(source_lang, target_lang, logger)

    # ── Process ──────────────────────────────────────────────────────────
    output_dialogues = []
    pass_count = trans_count = err_count = 0

    for idx, dlg in enumerate(dialogues):
        did  = dlg["id"]
        text = dlg.get("text", "")
        logger.info(f"[{idx+1}/{total}] {did} | {dlg.get('speaker_id','?')} | '{text[:50]}'")

        if passthrough_all or translator is None:
            translation    = text
            is_passthrough = True
            pass_count    += 1
        else:
            translation, is_passthrough = translate_with_retry(text, translator, logger, did)
            if is_passthrough: err_count  += 1
            else:               trans_count += 1
            if idx < total - 1: time.sleep(API_DELAY)

        if translation != text:
            logger.info(f"  → '{translation[:60]}'")

        output_dialogues.append({
            "id":             did,
            "speaker_id":     dlg.get("speaker_id","speaker_00"),
            "start_ms":       dlg["start_ms"],
            "end_ms":         dlg["end_ms"],
            "text":           text,
            "translation":    translation,
            "is_passthrough": is_passthrough,
        })

    # ── Write output ─────────────────────────────────────────────────────
    output_path = os.path.join(jobs_dir, args.job_id, "03_translations.json")
    output = {
        "job_id":         args.job_id,
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "source_lang":    source_lang,
        "target_lang":    target_lang,
        "dialogue_count": len(output_dialogues),
        "dialogues":      output_dialogues,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Written: {output_path}")
    logger.info(f"Summary: {pass_count} passthrough | {trans_count} translated | {err_count} errors")
    mark_complete(args.job_id, manifest, jobs_dir, logger)
    logger.info("Stage 03 complete ✓")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)