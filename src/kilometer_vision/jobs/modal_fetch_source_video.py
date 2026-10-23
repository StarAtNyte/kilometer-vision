"""Fetch a single organizer-referenced Walking Tours video into Modal storage.

This is only for rebuilding lost derived motion artifacts; it does not alter
questions, labels, or submissions.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg").pip_install("yt-dlp")
app = modal.App("kilometervision-fetch-source")


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=90 * 60)
def fetch(video_id: str, split: str = "validation") -> dict:
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    if video_id not in manifest:
        raise ValueError(f"video ID not present in {split} manifest: {video_id}")
    target_dir = Path("/data/shared/walking-tours") / video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{video_id}.mp4"
    if not target.exists() or target.stat().st_size < 1_000_000:
        command = [
            "yt-dlp", "--no-playlist", "--no-part", "--retries", "3",
            "--extractor-args", "youtube:player_client=android",
            "-f", "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "--merge-output-format", "mp4", "-o", str(target),
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        subprocess.run(command, check=True, timeout=75 * 60)
    if target.stat().st_size < 1_000_000:
        raise RuntimeError(f"download is unexpectedly small: {target.stat().st_size}")
    VOLUME.commit()
    return {"video_id": video_id, "path": str(target), "bytes": target.stat().st_size}


@app.local_entrypoint()
def main(video_id: str, split: str = "validation"):
    print(fetch.remote(video_id, split))
