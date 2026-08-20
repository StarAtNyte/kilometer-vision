"""Candidate-relative ordered geometry experiments for map questions."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import zipfile

import modal

from map_solver import endpoint_marker_features, mask_chamfer_distance, motion_polyline, ordered_candidate_scores, ordered_trace_candidate_scores, polyline_mask, route_mask, score_candidates, skeleton_diameter_path
from question_planner import classify_question_facets


ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("opencv-python-headless", "numpy")
    .add_local_python_source("map_solver", "question_planner")
)
app = modal.App("kilometervision-map-geometry")


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=30 * 60)
def evaluate(split: str = "validation") -> dict:
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    archive = ROOT / "manifests" / f"{split}_images.zip"
    records = []
    for video_id, questions in manifest.items():
        motion_path = ROOT / "motion" / split / f"{video_id}.json"
        motions = json.loads(motion_path.read_text()) if motion_path.exists() else {}
        for question_id, question in questions.items():
            if "map" not in classify_question_facets(question):
                continue
            motion = motions.get(str(question_id), {})
            hu = score_candidates(question, motion, archive)
            ordered = ordered_candidate_scores(question, motion, archive)
            vggt_variants = {}
            for variant in ("global-frames30", "global-frames60", "chunk16-overlap8-frames120"):
                path = ROOT / "experiments" / "vggt-motion" / split / variant / f"{video_id}.json"
                if not path.exists():
                    continue
                payload = json.loads(path.read_text())
                trace = payload.get(str(question_id), {}).get("trajectory_2d", [])
                if trace:
                    vggt_variants[variant] = ordered_trace_candidate_scores(question, trace, archive)
            marker_features = []
            candidate_sha256 = []
            with zipfile.ZipFile(archive) as images:
                members = images.namelist()
                for reference in question.get("image_paths", []):
                    matches = [name for name in members if Path(name).name == Path(str(reference)).name]
                    if len(matches) == 1:
                        image_bytes = images.read(matches[0])
                        candidate_sha256.append(hashlib.sha256(image_bytes).hexdigest())
                        path = skeleton_diameter_path(route_mask(image_bytes))
                        marker_features.append(endpoint_marker_features(image_bytes, path))
                    else:
                        candidate_sha256.append(None)
                        marker_features.append([])
            hu_prediction = min(range(len(hu)), key=hu.__getitem__) if hu else None
            ordered_prediction = min(range(len(ordered)), key=lambda index: float(ordered[index]["score"])) if ordered else None
            vggt_predictions = {
                variant: min(range(len(scores)), key=lambda index: float(scores[index]["score"]))
                for variant, scores in vggt_variants.items() if scores
            }
            vggt_consensus = None
            if len(vggt_variants) == 3:
                rank_sums = [0] * 5
                for scores in vggt_variants.values():
                    order = sorted(range(5), key=lambda index: float(scores[index]["score"]))
                    for rank, index in enumerate(order):
                        rank_sums[index] += rank
                vggt_consensus = min(range(5), key=rank_sums.__getitem__)
            dpvo_variants = {}
            dpvo_pixel_predictions = {}
            for variant in ("fps0.5-focal0.9-w256-b256-p16", "fps1.5-focal0.9-w256-b256-p16", "fps1-focal0.75-w256-b256-p16", "fps2-focal0.9-w256-b256-p8", "fps1-focal0.9-w256-b256-p16"):
                path = ROOT / "experiments" / "dpvo-motion" / split / variant / f"{video_id}.json"
                if not path.exists():
                    continue
                payload = json.loads(path.read_text())
                trace = payload.get(str(question_id), {}).get("trajectory_2d", [])
                if trace:
                    # PCA/SVD projection has an arbitrary axis sign, so mirror
                    # orientation is not evidence against a VO trajectory.
                    dpvo_variants[variant] = ordered_trace_candidate_scores(
                        question, trace, archive, reflection_penalty=0.0
                    )
                    trace_mask = polyline_mask(trace)
                    with zipfile.ZipFile(archive) as images:
                        pixel_scores = []
                        for reference in question.get("image_paths", []):
                            matches = [name for name in images.namelist() if Path(name).name == Path(str(reference)).name]
                            pixel_scores.append(mask_chamfer_distance(trace_mask, route_mask(images.read(matches[0]))) if len(matches) == 1 else float("inf"))
                    if pixel_scores:
                        dpvo_pixel_predictions[variant] = min(range(len(pixel_scores)), key=pixel_scores.__getitem__)
            dpvo_predictions = {
                variant: min(range(len(scores)), key=lambda index: float(scores[index]["score"]))
                for variant, scores in dpvo_variants.items() if scores
            }
            dpvo_turn_predictions = {}
            dpvo_trace_path = ROOT / "experiments" / "dpvo-motion" / split / "fps1-focal0.9-w256-b256-p16" / f"{video_id}.json"
            if dpvo_trace_path.exists():
                payload = json.loads(dpvo_trace_path.read_text())
                trace = payload.get(str(question_id), {}).get("trajectory_2d", [])
                if trace:
                    for weight in (0.1, 0.35, 0.5, 1.0, 2.0):
                        scores = ordered_trace_candidate_scores(question, trace, archive, reflection_penalty=0.0, turn_weight=weight)
                        if scores:
                            dpvo_turn_predictions[str(weight)] = min(range(len(scores)), key=lambda index: float(scores[index]["score"]))
            slam_predictions = {}
            slam_path = ROOT / "experiments" / "vggt-slam2" / f"{video_id}.json"
            if slam_path.exists():
                import numpy as np
                slam_payload = json.loads(slam_path.read_text())
                for variant, payload in slam_payload.get("variants", {}).items():
                    values = np.asarray([row["translation"] for row in payload.get("poses", [])], dtype="float64")
                    if len(values) >= 2:
                        values -= values.mean(axis=0)
                        _, _, axes = np.linalg.svd(values, full_matrices=False)
                        trace = (values @ axes[:2].T).tolist()
                        scores = ordered_trace_candidate_scores(question, trace, archive, reflection_penalty=0.0)
                        if scores:
                            slam_predictions[variant] = min(range(len(scores)), key=lambda index: float(scores[index]["score"]))
            da3_predictions = {}
            da3_root = ROOT / "experiments" / "da3-streaming-motion" / split
            if da3_root.exists():
                for path in da3_root.glob(f"*/{video_id}.json"):
                    payload = json.loads(path.read_text())
                    for row in payload.get("records", []):
                        if str(row.get("question_id")) != str(question_id):
                            continue
                        for name in ("trajectory_2d", "pedestrian_trajectory_2d"):
                            trace = row.get(name, [])
                            scores = ordered_trace_candidate_scores(question, trace, archive, reflection_penalty=0.0) if trace else []
                            if scores:
                                da3_predictions[f"{path.parent.name}-{name}"] = min(range(len(scores)), key=lambda index: float(scores[index]["score"]))
            frechet_scores = ordered_trace_candidate_scores(
                question, motion_polyline(motion.get("steps", [])), archive,
                reflection_penalty=0.0, frechet_weight=0.25
            ) if motion.get("steps") else []
            frechet_prediction = min(range(len(frechet_scores)), key=lambda index: float(frechet_scores[index]["score"])) if frechet_scores else None
            guided_scores = ordered_trace_candidate_scores(
                question, motion_polyline(motion.get("steps", [])), archive,
                reflection_penalty=0.0, frechet_weight=0.25, marker_guided=True
            ) if motion.get("steps") else []
            guided_prediction = min(range(len(guided_scores)), key=lambda index: float(guided_scores[index]["score"])) if guided_scores else None
            compact_scores = ordered_trace_candidate_scores(
                question, motion_polyline(motion.get("steps", [])), archive,
                reflection_penalty=0.0, frechet_weight=0.25, compactness_weight=0.25
            ) if motion.get("steps") else []
            compact_prediction = min(range(len(compact_scores)), key=lambda index: float(compact_scores[index]["score"])) if compact_scores else None
            compact_weight_predictions = {}
            if motion.get("steps"):
                for weight in (0.05, 0.1, 0.25, 0.5):
                    scores = ordered_trace_candidate_scores(
                        question, motion_polyline(motion.get("steps", [])), archive,
                        reflection_penalty=0.0, frechet_weight=0.25, compactness_weight=weight
                    )
                    if scores:
                        compact_weight_predictions[str(weight)] = min(range(len(scores)), key=lambda index: float(scores[index]["score"]))
            turn_weight_predictions = {}
            if motion.get("steps"):
                for weight in (0.1, 0.35, 0.5, 1.0, 2.0):
                    scores = ordered_trace_candidate_scores(
                        question, motion_polyline(motion.get("steps", [])), archive,
                        reflection_penalty=0.0, turn_weight=weight
                    )
                    if scores:
                        turn_weight_predictions[str(weight)] = min(range(len(scores)), key=lambda index: float(scores[index]["score"]))
            answer_id = question.get("answer_id")
            records.append(
                {
                    "video_id": video_id,
                    "question_id": str(question_id),
                    "question": question.get("question"),
                    "options": question.get("options"),
                    "image_paths": question.get("image_paths"),
                    "motion_status": motion.get("status"),
                    "accepted_steps": sum(bool(step.get("accepted")) for step in motion.get("steps", [])),
                    "net_yaw_degrees": motion.get("summary", {}).get("net_yaw_degrees"),
                    "hu_scores": hu,
                    "hu_prediction": hu_prediction,
                    "ordered_scores": ordered,
                    "endpoint_marker_features": marker_features,
                    "candidate_sha256": candidate_sha256,
                    "ordered_prediction": ordered_prediction,
                    "frechet_scores": frechet_scores,
                    "frechet_prediction": frechet_prediction,
                    "vggt_scores": vggt_variants,
                    "vggt_predictions": vggt_predictions,
                    "vggt_consensus_prediction": vggt_consensus,
                    "dpvo_scores": dpvo_variants,
                    "dpvo_predictions": dpvo_predictions,
                    "dpvo_pixel_predictions": dpvo_pixel_predictions,
                    "dpvo_turn_predictions": dpvo_turn_predictions,
                    "vggt_slam_predictions": slam_predictions,
                    "da3_predictions": da3_predictions,
                    "answer_id": answer_id,
                    "hu_correct": hu_prediction == answer_id if answer_id is not None else None,
                    "ordered_correct": ordered_prediction == answer_id if answer_id is not None else None,
                    "frechet_correct": frechet_prediction == answer_id if answer_id is not None else None,
                    "guided_correct": guided_prediction == answer_id if answer_id is not None else None,
                    "compact_correct": compact_prediction == answer_id if answer_id is not None else None,
                    "compact_weight_predictions": compact_weight_predictions,
                    "turn_weight_predictions": turn_weight_predictions,
                    "vggt_consensus_correct": vggt_consensus == answer_id if answer_id is not None else None,
                    "dpvo_correct": {
                        variant: prediction == answer_id if answer_id is not None else None
                        for variant, prediction in dpvo_predictions.items()
                    },
                    "vggt_slam_correct": {
                        variant: prediction == answer_id if answer_id is not None else None
                        for variant, prediction in slam_predictions.items()
                    },
                    "da3_correct": {
                        variant: prediction == answer_id if answer_id is not None else None
                        for variant, prediction in da3_predictions.items()
                    },
                }
            )
    summary = {
        "split": split,
        "questions": len(records),
        "hu_correct": sum(row["hu_correct"] is True for row in records),
        "ordered_correct": sum(row["ordered_correct"] is True for row in records),
        "frechet_correct": sum(row["frechet_correct"] is True for row in records),
        "guided_correct": sum(row["guided_correct"] is True for row in records),
        "compact_correct": sum(row["compact_correct"] is True for row in records),
        "compact_weight_correct": {
            weight: sum(row["compact_weight_predictions"].get(weight) == row["answer_id"] for row in records)
            for weight in ("0.05", "0.1", "0.25", "0.5")
        },
        "turn_weight_correct": {
            weight: sum(row["turn_weight_predictions"].get(weight) == row["answer_id"] for row in records)
            for weight in ("0.1", "0.35", "0.5", "1.0", "2.0")
        },
        "vggt_consensus_correct": sum(row["vggt_consensus_correct"] is True for row in records),
        "dpvo_correct": {
            variant: sum(row["dpvo_correct"].get(variant) is True for row in records)
            for variant in ("fps0.5-focal0.9-w256-b256-p16", "fps1.5-focal0.9-w256-b256-p16", "fps1-focal0.75-w256-b256-p16", "fps2-focal0.9-w256-b256-p8", "fps1-focal0.9-w256-b256-p16")
        },
        "dpvo_pixel_correct": {
            variant: sum(row["dpvo_pixel_predictions"].get(variant) == row["answer_id"] for row in records)
            for variant in sorted({variant for row in records for variant in row["dpvo_pixel_predictions"]})
        },
        "dpvo_turn_correct": {
            weight: sum(row["dpvo_turn_predictions"].get(weight) == row["answer_id"] for row in records)
            for weight in ("0.1", "0.35", "0.5", "1.0", "2.0")
        },
        "vggt_slam_correct": {
            variant: sum(row["vggt_slam_correct"].get(variant) is True for row in records)
            for variant in sorted({variant for row in records for variant in row["vggt_slam_correct"]})
        },
        "da3_correct": {
            variant: sum(row["da3_correct"].get(variant) is True for row in records)
            for variant in sorted({variant for row in records for variant in row["da3_correct"]})
        },
        "records": records,
    }
    output = ROOT / "experiments" / "map-ordered-geometry" / f"{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    VOLUME.commit()
    return summary


@app.local_entrypoint()
def main(split: str = "validation") -> None:
    print(json.dumps(evaluate.remote(split), indent=2))
