"""Controlled runner for the explicit-FPS Gemini validation probes.

This intentionally runs only a bounded validation subset and never uploads a
submission. Modal stores each answer/audit record under the existing probe
namespace; this wrapper only orchestrates calls and prints a compact summary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _family(question: dict) -> str:
    text = str(question.get("question", "")).lower()
    if "maps correctly" in text:
        return "map"
    if "direction" in text:
        return "compass"
    if "image show" in text:
        return "landmark"
    return "distance"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("all", "compass", "map", "distance", "landmark"), default="compass")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--key-start", type=int, default=0)
    parser.add_argument("--profile", default="gpudashboard0")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(".cache/validation.json") if args.split == "validation" else Path(".cache/test_public_20260811.json")
    manifest = json.loads(manifest_path.read_text())
    targets = []
    for video_id, questions in manifest.items():
        for question_id, question in questions.items():
            if args.family == "all" or _family(question) == args.family:
                targets.append((video_id, question_id, _family(question)))
    targets = targets[: max(0, args.limit)]
    for offset, (video_id, question_id, family) in enumerate(targets):
        command = [
            "modal", "run", "modal_gemini_direct_video_probe.py",
            "--video-id", video_id, "--question-id", str(question_id),
            "--split", args.split, "--model", args.model,
            "--fps", "2", "--scale", "320", "--thinking-level", "low",
            "--key-index", str(args.key_start + offset),
        ]
        print(f"[{offset + 1}/{len(targets)}] {family} {video_id}/{question_id}")
        if args.dry_run:
            print(" ".join(command))
            continue
        subprocess.run(command, check=False, env={**__import__("os").environ, "MODAL_PROFILE": args.profile})


if __name__ == "__main__":
    main()
