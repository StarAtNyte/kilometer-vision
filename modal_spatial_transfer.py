from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import modal


VOLUME_NAME = "perception-test-2026-data"
TRAIN_ROOT = Path("/data/allowed-training/perception-test")
AUDIO_ROOT = Path("/data/2706-kilometer-audio")
VISION_ROOT = Path("/data/2707-kilometer-vision")

app = modal.App("kilometer-vision-spatial-transfer-audit")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11")

# Deliberately narrow: these concepts have a plausible transfer path to route,
# heading, distance, map, temporal-order, or landmark-presence specialists.
SPATIAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "direction": (
        r"\b(left|right|clockwise|counterclockwise|direction|orient(?:ation|ed)?|heading|facing)\b",
        r"\b(north|south|east|west|northeast|northwest|southeast|southwest)\b",
        r"\b(turn(?:ed|ing|s)?|rotate[ds]?|behind|in front of)\b",
    ),
    "distance": (
        r"\b(distance|farther|farthest|nearest|closest|meters?|kilometers?|miles?|feet|blocks?)\b",
        r"\b(how far|closer to|further from)\b",
    ),
    "route_order": (
        r"\b(before|after|first|last|earlier|later|next|previously|subsequently)\b",
        r"\b(on the way|along the route|path|route|walking|walked|passed by)\b",
    ),
    "relative_location": (
        r"\b(where|location|located|position(?:ed)?|beside|between|across from|adjacent|nearby)\b",
        r"\b(left of|right of|in front of|behind|next to|opposite)\b",
    ),
    "landmark_presence": (
        r"\b(seen|visible|appears?|shown|encountered|visited|passed|present)\b",
        r"\b(landmark|building|statue|monument|church|temple|bridge|sign|store|shop)\b",
    ),
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _evaluation_video_ids() -> tuple[set[str], Counter[str]]:
    ids: set[str] = set()
    video_keys: Counter[str] = Counter()
    manifests = VISION_ROOT / "manifests"
    if not manifests.exists():
        return ids, video_keys
    for path in manifests.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        stack: list[Any] = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if "video" in key.lower() and isinstance(value, str):
                        video_keys[key] += 1
                        # Normalize paths/URLs to a stable stem where possible.
                        parsed = urlparse(value)
                        query_id = parse_qs(parsed.query).get("v", [None])[0]
                        candidate = query_id or parsed.path.rsplit("/", 1)[-1]
                        ids.add(Path(candidate).stem)
                    else:
                        stack.append(value)
            elif isinstance(item, list):
                stack.extend(item)
    return ids, video_keys


def _labels(row: dict[str, Any]) -> list[str]:
    fields = [
        str(row.get("question") or ""),
        str(row.get("area") or ""),
        str(row.get("reasoning") or ""),
        " ".join(map(str, row.get("tags") or [])),
    ]
    text = " ".join(fields).lower()
    return [
        label
        for label, patterns in SPATIAL_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    ]


def _prediction_map(fold: str, candidate: str) -> dict[str, dict[str, Any]]:
    path = (
        AUDIO_ROOT
        / "evaluations"
        / "internal"
        / fold
        / f"moss_{candidate}_constrained_predictions.json"
    )
    if not path.exists():
        return {}
    return {row["question_id"]: row for row in json.loads(path.read_text())}


@app.function(image=image, volumes={"/data": volume}, timeout=10 * 60)
def audit() -> dict[str, Any]:
    volume.reload()
    eval_ids, video_keys = _evaluation_video_ids()
    result: dict[str, Any] = {
        "definition": {key: list(value) for key, value in SPATIAL_PATTERNS.items()},
        "evaluation_video_ids_found": len(eval_ids),
        "evaluation_video_fields": dict(video_keys),
        "folds": {},
    }
    for fold in ("train", "dev", "holdout"):
        rows = _load_jsonl(TRAIN_ROOT / "folds" / f"mcq_{fold}.jsonl")
        selected = [(row, _labels(row)) for row in rows]
        selected = [(row, labels) for row, labels in selected if labels]
        overlap = sorted({row["video_id"] for row, _ in selected} & eval_ids)
        report: dict[str, Any] = {
            "all_questions": len(rows),
            "spatial_questions": len(selected),
            "spatial_fraction": len(selected) / len(rows),
            "spatial_videos": len({row["video_id"] for row, _ in selected}),
            "category_counts": dict(Counter(label for _, labels in selected for label in labels)),
            "area_counts": dict(Counter(str(row.get("area") or "unknown") for row, _ in selected)),
            "reasoning_counts": dict(
                Counter(str(row.get("reasoning") or "unknown") for row, _ in selected)
            ),
            "kilometervision_eval_overlap_count": len(overlap),
            "kilometervision_eval_overlap_ids": overlap,
        }
        if fold in {"dev", "holdout"}:
            candidates = {
                name: _prediction_map(fold, name)
                for name in (
                    "base",
                    "mcq-checkpoint-741",
                    "spatial-checkpoint-200",
                    "spatial-checkpoint-219",
                )
            }
            comparison: dict[str, Any] = {}
            for name, predictions in candidates.items():
                matched = [predictions[row["question_id"]] for row, _ in selected if row["question_id"] in predictions]
                comparison[name] = {
                    "available": len(matched),
                    "correct": sum(bool(row["correct"]) for row in matched),
                    "accuracy": (sum(bool(row["correct"]) for row in matched) / len(matched)) if matched else None,
                    "by_category": {
                        label: {
                            "total": len(items),
                            "correct": sum(bool(item["correct"]) for item in items),
                            "accuracy": sum(bool(item["correct"]) for item in items) / len(items),
                        }
                        for label in SPATIAL_PATTERNS
                        if (items := [
                            predictions[row["question_id"]]
                            for row, labels in selected
                            if label in labels and row["question_id"] in predictions
                        ])
                    },
                }
            report["cached_candidate_comparison"] = comparison
        result["folds"][fold] = report
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(audit.remote(), indent=2))
