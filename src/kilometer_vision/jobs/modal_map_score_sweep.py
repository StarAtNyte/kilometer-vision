"""Sweep deterministic ordered-geometry score variants on Modal.

Validation answers are used only for reporting after predictions are made;
test answers are never read.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from kilometer_vision.map_solver import ordered_trace_candidate_scores
from kilometer_vision.question_planner import classify_question_facets

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install("opencv-python-headless", "numpy") \
    .add_local_python_source('kilometer_vision')
APP = modal.App("kilometervision-map-score-sweep")

VARIANTS = {
    "base": dict(reflection_penalty=0.0, turn_weight=0.35),
    "turn_high": dict(reflection_penalty=0.0, turn_weight=0.70),
    "turn_low": dict(reflection_penalty=0.0, turn_weight=0.15),
    "frechet": dict(reflection_penalty=0.0, turn_weight=0.35, frechet_weight=0.20),
    "chamfer": dict(reflection_penalty=0.0, turn_weight=0.35, chamfer_weight=0.20),
    "compact": dict(reflection_penalty=0.0, turn_weight=0.35, compactness_weight=0.12),
    "reflect04": dict(reflection_penalty=0.04, turn_weight=0.35),
}


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=60 * 60)
def run(split: str = "validation"):
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    archive = ROOT / "manifests" / f"{split}_images.zip"
    output = {name: {"correct": 0, "questions": 0, "predictions": {}} for name in VARIANTS}
    for video_id, questions in manifest.items():
        motion_path = ROOT / "experiments" / "dpvo-motion" / split / "fps1-focal0.9-w256-b256-p16" / f"{video_id}.json"
        if not motion_path.exists():
            continue
        motions = json.loads(motion_path.read_text())
        for qid, question in questions.items():
            if "map" not in classify_question_facets(question):
                continue
            trace = motions.get(str(qid), {}).get("trajectory_2d", [])
            if len(trace) < 4:
                continue
            for name, kwargs in VARIANTS.items():
                scores = ordered_trace_candidate_scores(question, trace, archive, **kwargs)
                if not scores:
                    continue
                ranking = sorted(range(len(scores)), key=lambda i: float(scores[i]["score"]))
                pred = ranking[0]
                row = output[name]
                row["questions"] += 1
                row["correct"] += int(pred == question.get("answer_id"))
                row["predictions"][f"{video_id}/{qid}"] = {"prediction": pred, "ranking": ranking,
                                                              "margin": float(scores[ranking[1]]["score"] - scores[pred]["score"]) }
    result = {"split": split, "variants": output, "ground_truth_used_for_inference": False}
    target = ROOT / "experiments" / "map-score-sweep" / f"{split}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return {"split": split, "summary": {k: {"correct": v["correct"], "questions": v["questions"]} for k, v in output.items()}, "artifact": str(target)}


@APP.local_entrypoint()
def main(split: str = "validation"):
    print(run.remote(split))
