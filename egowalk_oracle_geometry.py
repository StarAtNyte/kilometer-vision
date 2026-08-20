"""Label-independent geometry stress test on EgoWalk camera poses.

The external dataset supplies metric camera poses; no KilometerVision labels or
test media are read. Synthetic distractors probe reversal, reflection, a turn
change, and segment-order corruption. The test is deliberately oracle-motion:
it validates map scoring separately from visual odometry.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import modal

from map_solver import ordered_shape_distance, resample_polyline


SESSIONS = (
    "2024_07_02__15_27_22", "2024_07_03__09_58_27", "2024_07_04__12_07_13",
    "2024_07_08__09_36_43", "2024_07_09__12_04_48", "2024_07_09__14_58_27",
    "2024_07_11__09_55_53", "2024_07_17__16_53_42", "2024_07_19__16_53_59",
    "2024_08_20__13_03_53",
)
ROOT = Path("/data/2707-kilometer-vision")
EXTERNAL_ROOT = ROOT / "external" / "egowalk"
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyarrow", "huggingface_hub")
    .add_local_python_source("map_solver")
)
app = modal.App("kilometervision-egowalk-oracle")


def _pose_points(rows: list[dict], count: int = 96) -> list[tuple[float, float]]:
    points = [(float(row["cart_x"]), float(row["cart_y"])) for row in rows
              if row.get("cart_x") is not None and row.get("cart_y") is not None]
    if len(points) < 8:
        return []
    sampled = resample_polyline(points, count=count)
    return [(float(x), float(y)) for x, y in sampled]


def _turn_changed(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Replace the latter half's displacement with a perpendicular continuation."""
    if len(points) < 8:
        return points
    import numpy as np
    values = np.asarray(points, dtype="float64")
    pivot = len(values) // 2
    heading = values[pivot] - values[pivot - 4]
    length = max(float(np.linalg.norm(heading)), 1e-6)
    perpendicular = np.asarray([-heading[1], heading[0]]) / length
    tail_distances = np.linalg.norm(values[pivot + 1:] - values[pivot], axis=1)
    changed = values.copy()
    for index, distance in enumerate(tail_distances, pivot + 1):
        changed[index] = changed[pivot] + perpendicular * distance
    return [tuple(row) for row in changed]


def _segment_swapped(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    values = list(points)
    pivot = len(values) // 3
    return values[pivot:2 * pivot] + values[:pivot] + values[2 * pivot:]


def evaluate(rows_by_session: dict[str, list[dict]]) -> dict:
    records = []
    for session, rows in rows_by_session.items():
        truth = _pose_points(rows)
        if not truth:
            continue
        x0, y0 = truth[0]
        reflected = [(2 * x0 - x, y) for x, y in truth]
        candidates = {
            "truth": truth,
            "reversed": list(reversed(truth)),
            "reflected": reflected,
            "turn_changed": _turn_changed(truth),
            "segment_swapped": _segment_swapped(truth),
        }
        directed = {
            name: ordered_shape_distance(truth, candidate, allow_reverse=False,
                                         reflection_penalty=0.04)
            for name, candidate in candidates.items()
        }
        reverse_free = {
            name: ordered_shape_distance(truth, candidate, allow_reverse=True,
                                         reflection_penalty=0.04)
            for name, candidate in candidates.items()
        }
        directed_order = sorted(directed, key=lambda name: directed[name]["score"])
        reverse_free_order = sorted(reverse_free, key=lambda name: reverse_free[name]["score"])
        records.append({
            "session": session, "points": len(truth),
            "directed_scores": {name: value["score"] for name, value in directed.items()},
            "directed_hypotheses": {name: value["hypothesis"] for name, value in directed.items()},
            "directed_order": directed_order,
            "reverse_free_scores": {name: value["score"] for name, value in reverse_free.items()},
            "reverse_free_order": reverse_free_order,
            "directed_truth_rank": directed_order.index("truth") + 1,
            "reverse_free_truth_rank": reverse_free_order.index("truth") + 1,
        })
    return {
        "dataset": "EgoWalk/trajectories",
        "sessions_requested": list(rows_by_session),
        "distractors": ["reversed", "reflected", "turn_changed", "segment_swapped"],
        "motion_source": "ground-truth camera pose; no KilometerVision labels",
        "records": records,
        "summary": {
            "sessions": len(records),
            "directed_truth_top1": sum(row["directed_truth_rank"] == 1 for row in records),
            "reverse_free_truth_top1": sum(row["reverse_free_truth_rank"] == 1 for row in records),
            "reversal_separated": sum(
                row["directed_scores"]["reversed"] > row["directed_scores"]["truth"] + 1e-6
                for row in records
            ),
        },
    }


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=30 * 60)
def run_remote() -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    VOLUME.reload()
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    rows_by_session = {}
    for session in SESSIONS:
        path = hf_hub_download(
            "EgoWalk/trajectories", f"data/{session}.parquet", repo_type="dataset",
            local_dir=EXTERNAL_ROOT,
        )
        rows_by_session[session] = pq.read_table(path).to_pylist()
    result = evaluate(rows_by_session)
    output = ROOT / "experiments" / "external-egowalk" / "oracle-geometry.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    result = run_remote.remote()
    print(json.dumps(result["summary"], indent=2))


def legacy_local_main() -> None:
    """Explicit opt-in legacy helper; the normal entrypoint is Modal-only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("/tmp/egowalk-oracle"))
    parser.add_argument("--output", type=Path, default=Path("experiments/external-egowalk/oracle-geometry.json"))
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    rows_by_session = {}
    args.cache.mkdir(parents=True, exist_ok=True)
    for session in SESSIONS:
        path = hf_hub_download(
            "EgoWalk/trajectories", f"data/{session}.parquet", repo_type="dataset",
            local_dir=args.cache,
        )
        rows_by_session[session] = pq.read_table(path).to_pylist()
    result = evaluate(rows_by_session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    # Keep direct execution from silently consuming local CPU.
    print("Use `modal run egowalk_oracle_geometry.py` for the remote benchmark.")
