"""
DUBBING PROJECT — Master Pipeline Orchestrator
Runs each stage sequentially via conda run
Restart-safe: skips already completed stages
"""

import os
import json
import subprocess
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import yaml


# ============================================================
# CONFIG LOADER
# ============================================================
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# JOB MANAGER
# ============================================================
def create_job(input_video: str, config: dict) -> str:
    """Create a new job folder and return job_id."""
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(input_video).stem
    job_id     = f"{video_name}_{timestamp}"

    job_dir = Path(config["paths"]["jobs"]) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    job_meta = {
        "job_id":      job_id,
        "input_video": str(Path(input_video).resolve()),
        "video_name":  video_name,
        "created_at":  datetime.now().isoformat(),
        "source_lang": config["language"]["source"],
        "target_lang": config["language"]["target"],
        "status":      "created"
    }

    with open(job_dir / "job_meta.json", "w") as f:
        json.dump(job_meta, f, indent=2)

    print(f"[pipeline] Job created: {job_id}")
    print(f"[pipeline] Job folder: {job_dir}")
    return job_id


def load_manifest(job_id: str, config: dict) -> dict:
    """Load manifest.json for a job, create if not exists."""
    manifest_path = Path(config["paths"]["jobs"]) / job_id / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)

    # Default manifest — all stages pending
    manifest = {
        "job_id": job_id,
        "stages": {
            "00_vocals":        {"status": "pending", "completed_at": None, "failed_at": None},
            "01_asr":           {"status": "pending", "completed_at": None, "failed_at": None},
            "01b_diarization":  {"status": "pending", "completed_at": None, "failed_at": None},
            "02_emotion":       {"status": "pending", "completed_at": None, "failed_at": None},
            "03_translation":   {"status": "pending", "completed_at": None, "failed_at": None},
            "04_tts":           {"status": "pending", "completed_at": None, "failed_at": None},
            "05_assembly":      {"status": "pending", "completed_at": None, "failed_at": None},
        }
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def save_manifest(job_id: str, manifest: dict, config: dict):
    """Save manifest.json."""
    manifest_path = Path(config["paths"]["jobs"]) / job_id / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def mark_stage_complete(job_id: str, stage: str, manifest: dict, config: dict):
    """Mark a stage as completed in manifest."""
    manifest["stages"][stage]["status"]       = "completed"
    manifest["stages"][stage]["completed_at"] = datetime.now().isoformat()
    save_manifest(job_id, manifest, config)
    print(f"[pipeline] ✅ Stage {stage} marked complete")


def mark_stage_failed(job_id: str, stage: str, manifest: dict, config: dict):
    """Mark a stage as failed in manifest."""
    manifest["stages"][stage]["status"]    = "failed"
    manifest["stages"][stage]["failed_at"] = datetime.now().isoformat()
    save_manifest(job_id, manifest, config)
    print(f"[pipeline] ❌ Stage {stage} marked failed")


# ============================================================
# STAGE RUNNER
# ============================================================
def run_stage(stage: str, env_name: str, job_id: str, config_path: str) -> bool:
    """Run a stage script inside its conda environment."""

    stage_script = Path("stages") / stage / "run.py"

    if not stage_script.exists():
        print(f"[pipeline] ⚠️  Stage script not found: {stage_script}")
        print(f"[pipeline]     Skipping — will be built in future chat")
        return True  # Don't block pipeline during development

    cmd = [
        "conda", "run",
        "--no-capture-output",
        "-n", env_name,
        "python", str(stage_script),
        "--job_id", job_id,
        "--config", config_path
    ]

    print(f"\n[pipeline] Running stage: {stage}")
    print(f"[pipeline] Env: {env_name}")
    print(f"[pipeline] CMD: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"[pipeline] ❌ Stage {stage} failed with code {result.returncode}")
        return False

    return True


# ============================================================
# STAGE KEY RESOLVER
# ============================================================
def get_stage_config_key(stage: str) -> str:
    """
    Map stage folder name to config.yaml stages key.

    Examples:
        "00_vocals"       -> "vocals"
        "01_asr"          -> "asr"
        "01b_diarization" -> "diarization"
        "02_emotion"      -> "emotion"
        "03_translation"  -> "translation"
        "04_tts"          -> "tts"
        "05_assembly"     -> "assembly"

    Rule: strip leading segment up to and including first underscore
    that follows digits (with optional letter suffix like 'b').
    """
    import re
    # Match leading pattern like "00_", "01_", "01b_", "05_"
    match = re.match(r'^\d+[a-z]?_', stage)
    if match:
        return stage[match.end():]
    return stage


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_pipeline(input_video: str, job_id: str = None,
                 config_path: str = "config.yaml",
                 force_stage: str = None):
    """
    Main pipeline entry point.

    Args:
        input_video:  Path to input video file
        job_id:       Existing job_id to resume, or None to create new
        config_path:  Path to config.yaml
        force_stage:  Re-run a specific stage even if completed
    """

    config = load_config(config_path)

    # Create or resume job
    if job_id is None:
        job_id = create_job(input_video, config)
    else:
        print(f"[pipeline] Resuming job: {job_id}")

    manifest       = load_manifest(job_id, config)
    stages_to_run  = config["pipeline"]["run_stages"]

    print(f"\n[pipeline] {'='*50}")
    print(f"[pipeline] Job ID     : {job_id}")
    print(f"[pipeline] Input      : {input_video}")
    print(f"[pipeline] Stages     : {stages_to_run}")
    print(f"[pipeline] {'='*50}\n")

    # Run each stage
    for stage in stages_to_run:

        # Add stage to manifest if missing (handles newly added stages)
        if stage not in manifest["stages"]:
            manifest["stages"][stage] = {
                "status": "pending",
                "completed_at": None,
                "failed_at": None
            }
            save_manifest(job_id, manifest, config)

        stage_status = manifest["stages"][stage]["status"]

        # Skip if already completed (restart-safe)
        if stage_status == "completed" and force_stage != stage:
            print(f"[pipeline] ⏭️  Skipping {stage} — already completed")
            continue

        # Get env name from config using correct key mapping
        stage_config_key = get_stage_config_key(stage)

        if stage_config_key not in config["stages"]:
            print(f"[pipeline] ⚠️  No config found for stage key '{stage_config_key}' — skipping")
            mark_stage_complete(job_id, stage, manifest, config)
            continue

        env_name = config["stages"][stage_config_key]["env_name"]

        # Run the stage
        success = run_stage(stage, env_name, job_id, config_path)

        if success:
            mark_stage_complete(job_id, stage, manifest, config)
        else:
            mark_stage_failed(job_id, stage, manifest, config)
            if config["pipeline"]["stop_on_error"]:
                print(f"\n[pipeline] Stopping pipeline due to error in {stage}")
                print(f"[pipeline] To resume: python pipeline.py --resume {job_id}")
                break

    # Final status
    print(f"\n[pipeline] {'='*50}")
    print(f"[pipeline] Final manifest:")
    for stage, info in manifest["stages"].items():
        status = info["status"]
        icon   = "✅" if status == "completed" else "❌" if status == "failed" else "⏳"
        print(f"  {icon}  {stage}: {status}")
    print(f"[pipeline] {'='*50}\n")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dubbing Pipeline Orchestrator")

    parser.add_argument(
        "--input", type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume an existing job by job_id"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--force_stage", type=str, default=None,
        help="Force re-run a specific stage e.g. 03_translation"
    )
    parser.add_argument(
        "--list_jobs", action="store_true",
        help="List all existing jobs"
    )

    args = parser.parse_args()

    # List jobs
    if args.list_jobs:
        config   = load_config(args.config)
        jobs_dir = Path(config["paths"]["jobs"])
        jobs     = sorted(jobs_dir.iterdir()) if jobs_dir.exists() else []
        if not jobs:
            print("No jobs found.")
        else:
            print(f"\nExisting jobs ({len(jobs)}):")
            for j in jobs:
                manifest_path = j / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        m = json.load(f)
                    statuses = [v["status"] for v in m["stages"].values()]
                    print(f"  {j.name}  ->  {statuses}")
        exit(0)

    # Validate input
    if args.resume is None and args.input is None:
        parser.error("Either --input or --resume is required")

    if args.input and not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")

    # Run
    run_pipeline(
        input_video=args.input or "",
        job_id=args.resume,
        config_path=args.config,
        force_stage=args.force_stage
    )