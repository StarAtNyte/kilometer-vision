"""Cheap remote triage of test map-question ambiguity."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import modal

from kilometer_vision.map_solver import ordered_trace_candidate_scores
from kilometer_vision.question_planner import classify_question_facets

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "opencv-python-headless")
         .add_local_python_source('kilometer_vision'))
app = modal.App("kilometervision-map-triage")


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=30 * 60)
def triage(split: str = "test") -> dict:
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    archive = ROOT / "manifests" / f"{split}_images.zip"
    rows = []
    with zipfile.ZipFile(archive) as images:
        for video_id, questions in manifest.items():
            motion_path = ROOT / "experiments" / "dpvo-motion" / split / \
                "fps1-focal0.9-w256-b256-p16" / f"{video_id}.json"
            if not motion_path.exists():
                motion_path = ROOT / "motion" / split / f"{video_id}.json"
            motions = json.loads(motion_path.read_text()) if motion_path.exists() else {}
            for question_id, question in questions.items():
                if "map" not in classify_question_facets(question):
                    continue
                image_candidate = len(question.get("image_paths", [])) == 5
                motion = motions.get(str(question_id), {})
                trace = motion.get("trajectory_2d", [])
                if not trace and motion.get("steps"):
                    from kilometer_vision.map_solver import motion_polyline
                    trace = motion_polyline(motion["steps"])
                scores = ordered_trace_candidate_scores(question, trace, archive) if len(trace) > 1 else []
                values = sorted(float(item["score"]) for item in scores)
                prediction = min(range(len(scores)), key=lambda i: float(scores[i]["score"])) if scores else None
                rows.append({
                    "video_id": video_id, "question_id": str(question_id),
                    "prediction": prediction,
                    "top_score": values[0] if values else None,
                    "second_score": values[1] if len(values) > 1 else None,
                    "margin": values[1] - values[0] if len(values) > 1 else None,
                    "motion_available": bool(trace),
                    "image_candidate": image_candidate,
                })
    summary = {
        "split": split, "questions": len(rows),
        "image_candidate_questions": sum(r["image_candidate"] for r in rows),
        "route_instruction_questions": sum(not r["image_candidate"] for r in rows),
        "rows_with_motion": sum(r["motion_available"] for r in rows),
        "margin_lt_0.005": sum(r["image_candidate"] and r["margin"] is not None and r["margin"] < 0.005 for r in rows),
        "margin_lt_0.01": sum(r["image_candidate"] and r["margin"] is not None and r["margin"] < 0.01 for r in rows),
        "margin_lt_0.02": sum(r["image_candidate"] and r["margin"] is not None and r["margin"] < 0.02 for r in rows),
        "rows": rows,
    }
    output = ROOT / "experiments" / "map-triage" / f"{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    VOLUME.commit()
    return summary


@app.local_entrypoint()
def main(split: str = "test") -> None:
    result = triage.remote(split)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
