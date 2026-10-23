"""Fast p16-only map ranking audit for the available test trajectories."""
from __future__ import annotations

import json
from pathlib import Path

import modal

from kilometer_vision.map_solver import ordered_trace_candidate_scores
from kilometer_vision.question_planner import classify_question_facets

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
VARIANT = "fps1-focal0.9-w256-b256-p16"
IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install("opencv-python-headless", "numpy") \
    .add_local_python_source('kilometer_vision')
app = modal.App("kilometervision-map-geometry-light")


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=60 * 60)
def evaluate(split: str = "test"):
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    archive = ROOT / "manifests" / f"{split}_images.zip"
    records = []
    for video_id, questions in manifest.items():
        motion_path = ROOT / "experiments" / "dpvo-motion" / split / VARIANT / f"{video_id}.json"
        if not motion_path.exists():
            continue
        motions = json.loads(motion_path.read_text())
        for question_id, question in questions.items():
            if "map" not in classify_question_facets(question):
                continue
            trace = motions.get(str(question_id), {}).get("trajectory_2d", [])
            if len(trace) < 4:
                continue
            scores = ordered_trace_candidate_scores(question, trace, archive, reflection_penalty=0.0)
            if not scores:
                continue
            ranking = sorted(range(len(scores)), key=lambda i: float(scores[i]["score"]))
            records.append({"video_id": video_id, "question_id": str(question_id),
                            "prediction": ranking[0], "ranking": ranking,
                            "scores": scores, "variant": VARIANT,
                            "ground_truth_used_for_inference": False})
    result = {"split": split, "variant": VARIANT, "records": records,
              "questions": len(records), "ground_truth_used_for_inference": False}
    output = ROOT / "experiments" / "map-geometry-light" / f"{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit(); return {"split": split, "questions": len(records), "artifact": str(output)}


@app.local_entrypoint()
def main(split: str = "test"):
    print(evaluate.remote(split))
