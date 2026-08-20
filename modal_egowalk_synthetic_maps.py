"""Modal-only rendered-map gate for the EgoWalk oracle geometry benchmark.

This deliberately uses external camera poses and synthetic five-choice maps. It
tests the image decoder, S/E marker orientation, and ordered map scorer without
touching KilometerVision validation/test labels or stressing the local machine.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import modal

from egowalk_oracle_geometry import SESSIONS, _pose_points, _turn_changed, _segment_swapped
from map_solver import ordered_trace_candidate_scores

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pyarrow", "huggingface_hub", "opencv-python-headless")
    .add_local_python_source("map_solver")
    .add_local_python_source("egowalk_oracle_geometry")
)
app = modal.App("kilometervision-egowalk-synthetic-maps")


def _render(points, marker_start: bool = True) -> bytes:
    import cv2
    import numpy as np

    values = np.asarray(points, dtype="float64")
    lo, hi = values.min(axis=0), values.max(axis=0)
    span = max(float((hi - lo).max()), 1e-6)
    xy = ((values - lo) * (430.0 / span) + 37.0).round().astype("int32")
    image = np.full((512, 512, 3), 245, dtype="uint8")
    cv2.polylines(image, [xy], False, (0, 0, 220), 5, cv2.LINE_AA)
    labels = (("S", "E") if marker_start else ("E", "S"))
    for point, label in zip((xy[0], xy[-1]), labels):
        x, y = map(int, point)
        cv2.circle(image, (x, y), 16, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.putText(image, label, (x - 7, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to render synthetic map")
    return encoded.tobytes()


def _write_archive(path: Path, candidates: dict[str, bytes]) -> dict:
    references = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in candidates.items():
            filename = f"{name}.png"
            archive.writestr(filename, payload)
            references.append(filename)
    return {"image_paths": references}


@app.function(image=IMAGE, volumes={"/data": VOLUME}, timeout=30 * 60)
def run_remote() -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    VOLUME.reload()
    cache = ROOT / "external" / "egowalk"
    output_dir = ROOT / "experiments" / "external-egowalk"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for session in SESSIONS:
        path = hf_hub_download("EgoWalk/trajectories", f"data/{session}.parquet",
                               repo_type="dataset", local_dir=cache)
        truth = _pose_points(pq.read_table(path).to_pylist())
        if not truth:
            continue
        candidates = {
            "truth": truth,
            "reversed": list(reversed(truth)),
            "reflected": [(2 * truth[0][0] - x, y) for x, y in truth],
            "turn_changed": _turn_changed(truth),
            "segment_swapped": _segment_swapped(truth),
        }
        archive_path = Path("/tmp") / f"{session}.zip"
        payloads = {name: _render(points) for name, points in candidates.items()}
        # Keep the geometric route unchanged but swap only S/E for this
        # distractor. The scorer must use markers to reject the wrong traversal
        # direction; rendering a reversed point list with reversed labels would
        # cancel that test.
        payloads["reversed"] = _render(truth, marker_start=False)
        question = _write_archive(archive_path, payloads)
        scores = ordered_trace_candidate_scores(question, truth, archive_path)
        order = sorted(range(len(scores)), key=lambda index: scores[index]["score"])
        records.append({
            "session": session,
            "candidate_names": list(candidates),
            "scores": [float(item["score"]) for item in scores],
            "hypotheses": [item["hypothesis"] for item in scores],
            "markers": [item["marker_positions"] for item in scores],
            "order": [list(candidates)[index] for index in order],
            "truth_rank": order.index(0) + 1,
        })
    result = {
        "dataset": "EgoWalk/trajectories",
        "test": "synthetic rendered five-map candidates from oracle camera poses",
        "ground_truth_used_for_inference": False,
        "records": records,
        "summary": {
            "sessions": len(records),
            "truth_top1": sum(row["truth_rank"] == 1 for row in records),
            "marker_pairs": sum(all(set(item) == {"S", "E"} for item in row["markers"]) for row in records),
        },
    }
    (output_dir / "synthetic-map-gate.json").write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(run_remote.remote()["summary"], indent=2))


if __name__ == "__main__":
    print("Use `modal run modal_egowalk_synthetic_maps.py` for the remote benchmark.")
