"""Validation-only sweep for reconnecting thin red map strokes after pin removal."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import modal

from map_solver import (
    _skeleton,
    orient_path_from_start,
    ordered_shape_distance,
    route_marker_positions,
    skeleton_diameter_path,
    skeleton_path_between_markers,
)
from question_planner import classify_question_facets

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("opencv-python-headless", "numpy")
    .add_local_python_source("map_solver", "question_planner")
)
APP = modal.App("kilometervision-map-morphology-sweep")
VARIANT = "fps1-focal0.9-w256-b256-p16"
RADII = (12, 15, 18)
CLOSES = (3, 5, 7, 9, 11)


def _path(raw: bytes, radius: int, close_size: int, guided: bool):
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return [], {}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 90, 110), (12, 255, 255)),
        cv2.inRange(hsv, (168, 90, 110), (180, 255, 255)),
    )
    markers = route_marker_positions(raw)
    for x, tip_y in markers.values():
        cv2.circle(red, (int(round(x)), int(round(tip_y - 11))), radius, 0, -1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    skeleton = _skeleton(red)
    path = skeleton_path_between_markers(skeleton, markers) if guided else []
    path = path or skeleton_diameter_path(skeleton)
    return orient_path_from_start(path, markers), markers


@APP.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=60 * 30)
def run(split: str = "validation"):
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    archive_path = ROOT / "manifests" / f"{split}_images.zip"
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for video_id, questions in manifest.items():
            motion_path = ROOT / "experiments" / "dpvo-motion" / split / VARIANT / f"{video_id}.json"
            if not motion_path.exists():
                continue
            motion = json.loads(motion_path.read_text())
            for question_id, question in questions.items():
                if "map" not in classify_question_facets(question):
                    continue
                trace = motion.get(str(question_id), {}).get("trajectory_2d", [])
                if len(trace) < 4 or len(question.get("image_paths", [])) != 5:
                    continue
                for radius in RADII:
                    for close_size in CLOSES:
                        for guided in (False, True):
                            scores = []
                            for reference in question["image_paths"]:
                                hits = [n for n in names if Path(n).name == Path(str(reference)).name]
                                raw = archive.read(hits[0]) if len(hits) == 1 else b""
                                path, markers = _path(raw, radius, close_size, guided)
                                scores.append(
                                    ordered_shape_distance(
                                        trace,
                                        path,
                                        allow_reverse=not bool(markers),
                                        reflection_penalty=0.0,
                                    )
                                )
                            order = sorted(range(len(scores)), key=lambda i: float(scores[i]["score"]))
                            rows.append(
                                {
                                    "video_id": video_id,
                                    "question_id": str(question_id),
                                    "radius": radius,
                                    "close_size": close_size,
                                    "guided": guided,
                                    "prediction": order[0],
                                    "ranking": order,
                                    "margin": float(scores[order[1]]["score"] - scores[order[0]]["score"]),
                                    "answer_id": question.get("answer_id"),
                                    "ground_truth_used_for_inference": False,
                                }
                            )
    summary = {}
    for radius in RADII:
        for close_size in CLOSES:
            for guided in (False, True):
                key = f"r{radius}-c{close_size}-g{int(guided)}"
                selected = [
                    row for row in rows
                    if row["radius"] == radius and row["close_size"] == close_size and row["guided"] == guided
                ]
                summary[key] = {
                    "questions": len(selected),
                    "correct": sum(row["prediction"] == row["answer_id"] for row in selected),
                }
    output = ROOT / "experiments" / "map-morphology-sweep" / f"{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"split": split, "summary": summary, "rows": rows, "ground_truth_used_for_inference": False}, indent=2) + "\n")
    VOLUME.commit()
    return {"summary": summary, "artifact": str(output)}


@APP.local_entrypoint()
def main(split: str = "validation"):
    print(json.dumps(run.remote(split), indent=2))
