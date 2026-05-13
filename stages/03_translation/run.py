#!/usr/bin/env python3
"""
Stage 03: Translation
=====================
Translates each dialogue from the detected source language to English
using Google Translate via the deep-translator library.

If source language is already English, all dialogues are passed through
as-is without any API calls.

Input:  jobs/{job_id}/02_emotions.json
        jobs/{job_id}/01_transcription.json  (for detected_language)
Output: jobs/{job_id}/03_translations.json

Behavior:
- Reads detected_language from 01_transcription.json
- English source → is_passthrough: true, no API call
- Non-English source → Google Translate → is_passthrough: false
- API failures → original text used, logged, pipeline never crashes
- Stage is restart-safe: skips if manifest shows 03_translation=completed
- No ML model, no VRAM used — pure Python + API
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGE_KEY: str = "03_translation"
STAGE_NAME: str = "03_translation"

# Delay between Google Translate API calls — avoids rate limiting
API_CALL_DELAY_S: float = 0.1

# Minimum text length to attempt translation — shorter texts passed through
MIN_TRANSLATE_LENGTH: int = 2

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 03: Translation")
    parser.add_argument("--job_id", required=True, help="Unique job identifier")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(job_id: str, logs_dir: str) -> logging.Logger:
    """
    Creates a logger writing to both stdout and a dedicated log file.
    Log file: logs/{job_id}_03_translation.log
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{job_id}_{STAGE_NAME}.log")

    logger = logging.getLogger(STAGE_NAME)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logging initialized → {log_path}")
    return logger


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def load_manifest(job_id: str, jobs_dir: str) -> dict[str, Any]:
    path = os.path.join(jobs_dir, job_id, "manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(
    manifest: dict[str, Any], job_id: str, jobs_dir: str
) -> None:
    path = os.path.join(jobs_dir, job_id, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def mark_stage_complete(
    job_id: str,
    manifest: dict[str, Any],
    jobs_dir: str,
    logger: logging.Logger,
) -> None:
    manifest["stages"][STAGE_KEY]["status"] = "completed"
    manifest["stages"][STAGE_KEY]["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["stages"][STAGE_KEY]["failed_at"] = None
    save_manifest(manifest, job_id, jobs_dir)
    logger.info(f"Manifest updated: {STAGE_KEY} → completed")


def mark_stage_failed(
    job_id: str,
    manifest: dict[str, Any],
    jobs_dir: str,
    logger: logging.Logger,
) -> None:
    manifest["stages"][STAGE_KEY]["status"] = "failed"
    manifest["stages"][STAGE_KEY]["failed_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest, job_id, jobs_dir)
    logger.error(f"Manifest updated: {STAGE_KEY} → failed")


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_emotions_json(job_id: str, jobs_dir: str) -> dict[str, Any]:
    path = os.path.join(jobs_dir, job_id, "02_emotions.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transcription_json(job_id: str, jobs_dir: str) -> dict[str, Any]:
    path = os.path.join(jobs_dir, job_id, "01_transcription.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Translator setup
# ---------------------------------------------------------------------------


def build_translator(source_lang: str, logger: logging.Logger) -> Any:
    """
    Builds a GoogleTranslator instance for the given source language.

    Falls back to source="auto" if the language code is not recognized
    by deep-translator. Always targets English.

    Args:
        source_lang: ISO 639-1 language code (e.g. "en", "te", "hi")
        logger: Logger instance

    Returns:
        GoogleTranslator instance
    """
    from deep_translator import GoogleTranslator

    try:
        translator = GoogleTranslator(source=source_lang, target="en")
        # Test that the language is valid by checking supported languages
        supported = GoogleTranslator.get_supported_languages(as_dict=True)
        # deep-translator uses full names as keys, codes as values
        # Check if source_lang is in the values
        if source_lang not in supported.values() and source_lang != "auto":
            logger.warning(
                f"Language code '{source_lang}' may not be supported by GoogleTranslator. "
                f"Falling back to source='auto'."
            )
            translator = GoogleTranslator(source="auto", target="en")
        return translator
    except Exception as e:
        logger.warning(
            f"Failed to build translator for '{source_lang}': {e}. "
            f"Falling back to source='auto'."
        )
        return GoogleTranslator(source="auto", target="en")


# ---------------------------------------------------------------------------
# Translation core
# ---------------------------------------------------------------------------


def translate_text(
    text: str,
    translator: Any,
    logger: logging.Logger,
    dialogue_id: str,
) -> tuple[str, bool]:
    """
    Translates a single text string to English using Google Translate.

    Returns (translated_text, is_passthrough).
    On any failure, returns (original_text, True) — pipeline never crashes.

    Args:
        text: Source text to translate
        translator: GoogleTranslator instance
        logger: Logger instance
        dialogue_id: For logging context

    Returns:
        (translation, is_passthrough) tuple
    """
    # Too short to translate meaningfully
    if len(text.strip()) < MIN_TRANSLATE_LENGTH:
        logger.debug(f"{dialogue_id}: text too short, passing through: '{text}'")
        return text, True

    try:
        result = translator.translate(text)

        if not result or not result.strip():
            logger.warning(
                f"{dialogue_id}: translation returned empty — using original text"
            )
            return text, True

        return result.strip(), False

    except Exception as e:
        logger.error(
            f"{dialogue_id}: translation failed ({e}) — using original text"
        )
        return text, True


# ---------------------------------------------------------------------------
# Output record builder
# ---------------------------------------------------------------------------


def build_output_record(
    dialogue: dict[str, Any],
    translation: str,
    is_passthrough: bool,
) -> dict[str, Any]:
    """
    Constructs a single dialogue entry for 03_translations.json.

    Reads emotion as a plain string and intensity as a plain float —
    as per the new schema from Stage 02. Does NOT access emotion as a dict.

    Args:
        dialogue: Raw dialogue dict from 02_emotions.json
        translation: Translated (or passthrough) text
        is_passthrough: True if no translation was performed

    Returns:
        Output dialogue record matching 03_translations.json schema
    """
    return {
        "id": dialogue["id"],
        "speaker_id": dialogue.get("speaker_id", "speaker_00"),
        "start_ms": dialogue["start_ms"],
        "end_ms": dialogue["end_ms"],
        "text": dialogue["text"],
        # emotion is a plain string in new schema — NOT a dict
        "emotion": dialogue.get("emotion", "neutral"),
        # intensity is a plain float at top level — NOT inside emotion dict
        "intensity": dialogue.get("intensity", 0.0),
        "translation": translation,
        "is_passthrough": is_passthrough,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    jobs_dir: str = config["paths"]["jobs"]
    logs_dir: str = config["paths"]["logs"]

    logger = setup_logging(args.job_id, logs_dir)

    logger.info("=" * 60)
    logger.info(f"Stage 03 — Translation | job_id: {args.job_id}")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Restart-safe check
    # -----------------------------------------------------------------------
    manifest = load_manifest(args.job_id, jobs_dir)

    if manifest["stages"][STAGE_KEY]["status"] == "completed":
        logger.info("Stage already completed. Skipping (restart-safe).")
        return

    # -----------------------------------------------------------------------
    # Verify prerequisite
    # -----------------------------------------------------------------------
    if manifest["stages"]["02_emotion"]["status"] != "completed":
        raise RuntimeError(
            "Stage 02_emotion has not completed. Run Stage 02 first."
        )

    # -----------------------------------------------------------------------
    # Load inputs
    # -----------------------------------------------------------------------
    logger.info("Loading 02_emotions.json...")
    emotions_data = load_emotions_json(args.job_id, jobs_dir)
    dialogues: list[dict[str, Any]] = emotions_data["dialogues"]
    total_dialogues = len(dialogues)
    logger.info(f"Loaded {total_dialogues} dialogues")

    logger.info("Loading 01_transcription.json for detected_language...")
    transcription_data = load_transcription_json(args.job_id, jobs_dir)
    detected_language: str = transcription_data.get("detected_language", "en")
    logger.info(f"Detected source language: '{detected_language}'")

    # -----------------------------------------------------------------------
    # Determine if translation is needed
    # -----------------------------------------------------------------------
    source_is_english = detected_language.lower() in ("en", "english")

    if source_is_english:
        logger.info(
            "Source language is English — all dialogues will be passed through. "
            "No API calls will be made."
        )
        translator = None
    else:
        logger.info(
            f"Source language is '{detected_language}' — "
            f"translating all dialogues to English via Google Translate."
        )
        translator = build_translator(detected_language, logger)

    # -----------------------------------------------------------------------
    # Process each dialogue
    # -----------------------------------------------------------------------
    output_dialogues: list[dict[str, Any]] = []
    passthrough_count = 0
    translated_count = 0
    error_count = 0

    for idx, dialogue in enumerate(dialogues):
        did = dialogue["id"]
        text = dialogue.get("text", "")

        logger.info(
            f"[{idx + 1}/{total_dialogues}] {did} | "
            f"speaker: {dialogue.get('speaker_id', 'unknown')} | "
            f"emotion: {dialogue.get('emotion', 'unknown')} | "
            f"'{text[:50]}'"
        )

        if source_is_english:
            # Pass through — no translation needed
            translation = text
            is_passthrough = True
            passthrough_count += 1
            logger.debug(f"{did}: passthrough (English source)")

        else:
            # Translate to English
            translation, is_passthrough = translate_text(
                text, translator, logger, did
            )

            if is_passthrough:
                # translate_text only sets is_passthrough=True on failure
                error_count += 1
                logger.warning(f"{did}: fell back to original text")
            else:
                translated_count += 1
                logger.info(f"{did} → '{translation[:60]}'")

            # Polite delay between API calls
            if idx < total_dialogues - 1 and not source_is_english:
                time.sleep(API_CALL_DELAY_S)

        output_dialogues.append(
            build_output_record(dialogue, translation, is_passthrough)
        )

    # -----------------------------------------------------------------------
    # Write 03_translations.json
    # -----------------------------------------------------------------------
    output_dir = os.path.join(jobs_dir, args.job_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "03_translations.json")

    output_data: dict[str, Any] = {
        "job_id": args.job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_lang": detected_language,
        "target_lang": "en",
        "dialogue_count": len(output_dialogues),
        "dialogues": output_dialogues,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"03_translations.json written → {output_path}")
        logger.info(
            f"Summary: {passthrough_count} passthrough, "
            f"{translated_count} translated, "
            f"{error_count} errors (fell back to original)"
        )

    except Exception as e:
        logger.error(f"Failed to write output JSON: {e}")
        mark_stage_failed(args.job_id, manifest, jobs_dir, logger)
        raise

    # -----------------------------------------------------------------------
    # Mark complete
    # -----------------------------------------------------------------------
    mark_stage_complete(args.job_id, manifest, jobs_dir, logger)
    logger.info("Stage 03 complete ✓")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Top-level catch — ensures non-zero exit code on failure
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)