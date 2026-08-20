"""Dependency-light visual novelty and uncertain route-change estimation.

This module deliberately does not infer metric scale or absolute heading.  In the
absence of calibrated camera/IMU information, heading deltas remain ``None`` and
novelty peaks are only *turn candidates* requiring later geometric verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
DEFAULT_SMOOTHING_RADIUS = 2
DEFAULT_MAX_VECTOR_VALUES = 256


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _byte_vector(path: Path, size: int = DEFAULT_MAX_VECTOR_VALUES) -> list[float] | None:
    """Return a deterministic fallback sketch when Pillow is unavailable."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    # Hash-bucket bytes gives a bounded, deterministic signal without pretending
    # to decode arbitrary image formats. Pillow is preferred below.
    buckets = [0.0] * size
    for index, value in enumerate(data):
        buckets[index % size] += value / 255.0
    norm = math.sqrt(sum(value * value for value in buckets))
    return [value / norm for value in buckets] if norm else None


def _image_vector(path: Path, size: int = 16) -> list[float] | None:
    """Read a compact color + edge thumbnail for viewpoint-tolerant retrieval.

    The old grayscale-only sketch made differently colored landmarks collide.
    Channel-wise centering preserves chromatic cues while the low-resolution
    grid keeps this dependency-light and bounded. Byte hashing remains the
    explicit fallback for undecodable files.
    """
    try:
        from PIL import Image, ImageFilter  # type: ignore

        with Image.open(path) as source:
            image = source.convert("RGB").resize((size, size))
            edge = image.convert("L").filter(ImageFilter.FIND_EDGES)
            channels = [list(channel.getdata()) for channel in image.split()] + [list(edge.getdata())]
            values = []
            means = []
            centered = []
            for channel in channels:
                normalized = [float(value) / 255.0 for value in channel]
                mean = sum(normalized) / len(normalized)
                means.append(mean)
                centered.extend(value - mean for value in normalized)
            values.extend(means)
            values.extend(centered)
            norm = math.sqrt(sum(value * value for value in values))
            return [value / norm for value in values] if norm else [0.0] * len(values)
    except (ImportError, OSError, ValueError):
        return _byte_vector(path)


def _distance(left: Sequence[float] | None, right: Sequence[float] | None) -> float | None:
    if left is None or right is None or len(left) != len(right) or not left:
        return None
    # Cosine distance is bounded and robust to overall brightness changes.
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, 0.5 * (1.0 - dot / (left_norm * right_norm))))


def _smooth(values: Sequence[float | None], radius: int) -> list[float | None]:
    result: list[float | None] = []
    for index, value in enumerate(values):
        if value is None:
            result.append(None)
            continue
        nearby = [item for item in values[max(0, index - radius):index + radius + 1] if item is not None]
        result.append(sum(nearby) / len(nearby) if nearby else None)
    return result


def _quantile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 1.0
    position = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[position]


def _resolve_frame(manifest_path: Path, frame_path: str) -> Path:
    candidate = (manifest_path.parent / frame_path).resolve()
    root = manifest_path.parent.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"frame path escapes manifest directory: {frame_path!r}")
    return candidate


def estimate_route(
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    smoothing_radius: int = DEFAULT_SMOOTHING_RADIUS,
) -> dict[str, Any]:
    """Estimate visual novelty and uncertain turn candidates from one frame manifest."""
    if not isinstance(smoothing_radius, int) or smoothing_radius < 0:
        raise ValueError("smoothing_radius must be a non-negative integer")
    source = Path(manifest_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frame manifest must contain a non-empty frames list")

    vectors: list[list[float] | None] = []
    timestamps: list[float] = []
    frame_records: list[dict[str, Any]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("frame entries must be objects")
        timestamp = float(frame["timestamp_seconds"])
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("frame timestamps must be finite and non-negative")
        relative_path = frame.get("path")
        if not isinstance(relative_path, str):
            raise ValueError("frame entries require a relative path")
        path = _resolve_frame(source, relative_path)
        vectors.append(_image_vector(path) if path.is_file() else None)
        timestamps.append(timestamp)
        frame_records.append({"index": frame.get("index", len(frame_records)), "timestamp_seconds": timestamp})

    novelty: list[float | None] = [None]
    novelty.extend(_distance(vectors[index - 1], vectors[index]) for index in range(1, len(vectors)))
    smoothed = _smooth(novelty, smoothing_radius)
    usable = [value for value in smoothed if value is not None]
    # A high novelty threshold proposes candidates only; it is not a turn label.
    threshold = max(0.35, _quantile(usable, 0.85)) if usable else None
    turn_candidates: list[dict[str, Any]] = []
    for index, value in enumerate(smoothed):
        if value is None or threshold is None or value < threshold or index == 0:
            continue
        previous = smoothed[index - 1]
        following = smoothed[index + 1] if index + 1 < len(smoothed) else None
        if previous is not None and value < previous:
            continue
        if following is not None and value < following:
            continue
        turn_candidates.append({
            "frame_index": frame_records[index]["index"],
            "timestamp_seconds": timestamps[index],
            "signal": round(value, 6),
            "uncertainty": 0.8,
            "reason": "visual novelty peak; geometric verification required",
        })

    observations = []
    for index, frame in enumerate(frame_records):
        observations.append({
            **frame,
            "novelty": None if novelty[index] is None else round(novelty[index], 6),
            "smoothed_novelty": None if smoothed[index] is None else round(smoothed[index], 6),
            "heading_delta_degrees": None,
            "uncertainty": 1.0 if vectors[index] is None else 0.8,
            "heading_status": "unavailable_without_calibrated_motion_or_imu",
        })

    result = {
        "schema_version": SCHEMA_VERSION,
        "video_id": payload.get("video_id"),
        "source_manifest": str(source),
        "method": "thumbnail novelty; no metric scale or absolute heading",
        "observations": observations,
        "turn_candidates": turn_candidates,
        "candidate_threshold": None if threshold is None else round(threshold, 6),
    }
    _atomic_json(Path(output_path), result)
    return result


__all__ = ["estimate_route"]
