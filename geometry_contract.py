"""Materialize DA3 pose output into the real ``geometry_backend`` route contract.

``geometry_backend.py`` already implements ``CameraPose``/``extract_route_signature``,
but nothing previously called it: ``submission.py``'s specialists re-derive a raw
two-frame yaw delta from ``geometry/<video_id>.json`` by hand, and the route/turn
sequence extract_route_signature can produce was never computed at all. This module
is the single place that parses the canonical geometry contract into
``geometry_backend`` objects and writes a route-signature artifact (turn sequence,
segment lengths, uncertainty, and a short human-readable narrative) that specialists
and the VLM evidence-card prompt can both read.

The forward-axis convention used here (``heading = atan2(rotation[0][2],
rotation[2][2])``, driven by ``extract_route_signature``'s ``rotation @ (0, 0, 1)``)
is the same one ``submission.py:_geometry_summary`` already uses; this module does
not change any existing yaw numbers, it exposes the same computation through the
tested contract instead of ad hoc code.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from geometry_backend import (
    CameraPose,
    DepthSummary,
    GeometryFrame,
    SequenceGeometry,
    extract_route_signature,
    fallback_geometry,
)

SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_pose(frame: Any) -> CameraPose:
    timestamp = float(frame.get("timestamp_seconds", 0.0) or 0.0)
    confidence_raw = frame.get("confidence")
    confidence = max(0.0, min(1.0, float(confidence_raw))) if confidence_raw is not None else 0.0
    extrinsic = frame.get("extrinsic")
    if not (isinstance(extrinsic, list) and len(extrinsic) >= 3 and all(isinstance(row, list) and len(row) >= 4 for row in extrinsic[:3])):
        return CameraPose(timestamp, confidence=confidence, available=False)
    rotation = tuple(float(extrinsic[row][col]) for row in range(3) for col in range(3))
    translation = (float(extrinsic[0][3]), float(extrinsic[1][3]), float(extrinsic[2][3]))
    return CameraPose(timestamp, rotation=rotation, translation=translation, confidence=confidence, available=True)


def load_da3_geometry(video_id: str, root: Path | None) -> SequenceGeometry:
    """Parse ``geometry/<video_id>.json`` (or ``.da3.json``) into a real ``SequenceGeometry``.

    Missing, empty, or malformed artifacts fall back to
    ``geometry_backend.fallback_geometry`` with an explicit reason, matching the
    "never invent evidence" contract used everywhere else in this pipeline.
    """
    if root is None:
        return fallback_geometry(video_id, [], "no_volume_root")
    path = Path(root) / "geometry" / f"{video_id}.json"
    if not path.exists():
        path = Path(root) / "geometry" / f"{video_id}.da3.json"
    if not path.exists():
        return fallback_geometry(video_id, [], "missing_geometry")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_frames = payload.get("frames", [])
        if not isinstance(raw_frames, list) or not raw_frames:
            return fallback_geometry(video_id, [], "empty_geometry")
        backend = str(payload.get("backend", "unknown"))
        poses = [_parse_pose(frame) for frame in raw_frames]
        frames = tuple(
            GeometryFrame(
                timestamp_seconds=pose.timestamp_seconds,
                pose=pose,
                depth=DepthSummary(),
                confidence=pose.confidence,
                source=backend,
                status="available" if pose.available else "pose_unavailable",
            )
            for pose in poses
        )
        signature = extract_route_signature(poses)
        return SequenceGeometry(video_id=video_id, backend=backend, status="available", frames=frames, route_signature=signature)
    except (OSError, ValueError, TypeError, IndexError, KeyError):
        return fallback_geometry(video_id, [], "invalid_geometry")


def _describe_turn(delta_degrees: float) -> str:
    direction = "left" if delta_degrees < 0 else "right"
    return f"{direction} ~{abs(round(delta_degrees))}°"


def _displacement_ratio(geometry: SequenceGeometry) -> float | None:
    positions = [frame.pose.translation for frame in geometry.frames if frame.pose.available]
    if len(positions) < 2:
        return None
    path_length = sum(
        sum((b[i] - a[i]) ** 2 for i in range(3)) ** 0.5 for a, b in zip(positions, positions[1:])
    )
    if path_length <= 1e-9:
        return 1.0
    displacement = sum((positions[-1][i] - positions[0][i]) ** 2 for i in range(3)) ** 0.5
    return displacement / path_length


def build_route_signature(video_id: str, root: Path) -> dict[str, Any]:
    """Compute and persist a real turn/heading route signature plus a text narrative.

    The narrative is what feeds the VLM evidence-card prompt (``vlm_solver.py``) as
    geometry context, and is also available to map/distance specialists that
    previously had zero access to DA3's turn sequence.
    """
    geometry = load_da3_geometry(video_id, root)
    signature = geometry.route_signature
    closure_ratio = _displacement_ratio(geometry)
    if not signature.available:
        narrative = "No calibrated camera pose available; geometry is relative-scale evidence only."
    else:
        turns_text = ", ".join(_describe_turn(delta) for delta in signature.turn_angles_degrees) or "no sustained turns"
        path_length = round(sum(signature.segment_lengths), 3)
        narrative = (
            f"{len(signature.turn_angles_degrees)} sustained turn(s) detected: {turns_text}. "
            f"Relative path length {path_length} (unscaled units, not meters); "
            f"pose uncertainty {round(signature.uncertainty, 3)}."
        )
        if closure_ratio is not None:
            narrative += f" Endpoint displacement is {round(closure_ratio, 3)}x the traveled path length."
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "backend": geometry.backend,
        "status": geometry.status,
        "available": signature.available,
        "headings_degrees": list(signature.headings_degrees),
        "turn_angles_degrees": list(signature.turn_angles_degrees),
        "segment_lengths": list(signature.segment_lengths),
        "topology": dict(signature.topology),
        "uncertainty": signature.uncertainty,
        "closure_ratio": closure_ratio,
        "narrative": narrative,
    }
    _atomic_json(Path(root) / "geometry" / f"{video_id}.route_signature.json", result)
    return result


__all__ = ["load_da3_geometry", "build_route_signature"]
