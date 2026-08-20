"""Produce an evidence audit for a labeled split without changing predictions.

This is an evaluation/debugging report, not a tuning loop.  It exposes routing,
question-window coverage, VLM decisions, and final correctness so plumbing bugs are
not mistaken for model limitations.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from question_planner import classify_question, classify_question_facets
from motion_solver import compass_option_index


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_diagnostic(root: Path, split: str = "validation") -> dict[str, Any]:
    manifest = _load(root / "manifests" / f"{split}.json")
    predictions = _load(root / "predictions" / f"{split}_submission.json")
    rows: list[dict[str, Any]] = []
    for video_id, questions in manifest.items():
        predicted = {str(row["id"]): row for row in predictions.get(video_id, [])}
        vlm_path = root / "vlm" / f"{video_id}.json"
        geometry_path = root / "geometry" / f"{video_id}.json"
        vlm = _load(vlm_path) if vlm_path.exists() else {}
        geometry = _load(geometry_path) if geometry_path.exists() else {}
        motion_path = root / "motion" / split / f"{video_id}.json"
        motion = _load(motion_path) if motion_path.exists() else {}
        pose_times = [
            float(frame["timestamp_seconds"])
            for frame in geometry.get("frames", [])
            if isinstance(frame, Mapping) and frame.get("extrinsic") is not None
        ]
        for key, question in questions.items():
            question_id = str(question.get("id", key))
            final = predicted.get(question_id, {})
            record = vlm.get(str(key), {}) if isinstance(vlm, Mapping) else {}
            motion_record = motion.get(str(key), {}) if isinstance(motion, Mapping) else {}
            start, end = question.get("start_seconds"), question.get("end_seconds")
            window_times = []
            if start is not None and end is not None:
                window_times = [time for time in pose_times if float(start) <= time <= float(end)]
            row = {
                "video_id": video_id,
                "question_key": str(key),
                "question_id": question.get("id", key),
                "category": classify_question(question),
                "facets": list(classify_question_facets(question)),
                "start_seconds": start,
                "end_seconds": end,
                "window_duration_seconds": float(end) - float(start) if start is not None and end is not None else None,
                "geometry_pose_count_total": len(pose_times),
                "geometry_pose_count_in_window": len(window_times),
                "geometry_window_covered": bool(window_times),
                "vlm_status": record.get("status"),
                "vlm_selected_index": record.get("selected_index"),
                "vlm_raw_text": record.get("raw_text"),
                "final_selected_index": final.get("answer_id"),
                "motion_status": motion_record.get("status"),
                "motion_net_yaw_degrees": motion_record.get("summary", {}).get("net_yaw_degrees"),
            }
            if motion_record.get("summary", {}).get("net_yaw_degrees") is not None and "compass" in row["facets"]:
                row["motion_selected_index"] = compass_option_index(question, motion_record["summary"]["net_yaw_degrees"])
            if "answer_id" in question:
                row["gold_index"] = question["answer_id"]
                row["correct"] = final.get("answer_id") == question["answer_id"]
                row["vlm_correct"] = record.get("selected_index") == question["answer_id"]
                row["motion_correct"] = row.get("motion_selected_index") == question["answer_id"] if "motion_selected_index" in row else None
            rows.append(row)

    labeled = [row for row in rows if "correct" in row]
    by_facet: dict[str, dict[str, int]] = {}
    for facet in sorted({facet for row in labeled for facet in row["facets"]}):
        selected = [row for row in labeled if facet in row["facets"]]
        by_facet[facet] = {
            "questions": len(selected),
            "final_correct": sum(bool(row["correct"]) for row in selected),
            "vlm_correct": sum(bool(row["vlm_correct"]) for row in selected),
            "with_window_geometry": sum(bool(row["geometry_window_covered"]) for row in selected),
            "motion_correct": sum(row.get("motion_correct") is True for row in selected),
        }
    return {
        "schema_version": 1,
        "split": split,
        "summary": {
            "questions": len(rows),
            "labeled_questions": len(labeled),
            "final_correct": sum(bool(row["correct"]) for row in labeled),
            "vlm_correct": sum(bool(row["vlm_correct"]) for row in labeled),
            "questions_with_window_geometry": sum(bool(row["geometry_window_covered"]) for row in rows),
            "primary_categories": dict(Counter(row["category"] for row in rows)),
            "by_facet": by_facet,
        },
        "questions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_diagnostic(args.root, args.split)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
