"""PATHFINDER data bootstrap and metadata audit actions.

This module deliberately downloads only organizer metadata and its referenced image
archives. Walking-tour videos remain in the shared Modal Volume tree and are never
copied by this bootstrap step.

The module is usable with the Modal CLI (when ``modal`` is installed) and with the
standard Python interpreter for metadata/audit unit tests.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import posixpath
import re
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from fractions import Fraction
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qs, urlparse

from submission import generate_split
from route_solver import estimate_route, _image_vector
from geometry_backend import discover_backends, fallback_geometry
from geometry_contract import build_route_signature

try:  # Modal is intentionally optional for local schema tests.
    import modal  # type: ignore
except ImportError:  # pragma: no cover - exercised by normal local test runs
    modal = None  # type: ignore


VOLUME_NAMESPACE = "gpudashboard0"
VOLUME_NAME = "perception-test-2026-data"
VOLUME_REF = f"{VOLUME_NAMESPACE}/{VOLUME_NAME}"
DATA_ROOT = Path("/data/2707-kilometer-vision")
SHARED_VIDEO_ROOT = Path("/data/shared/walking-tours")
CANONICAL_DIRS = (
    "manifests",
    "frames",
    "route-features",
    "ocr",
    "embeddings",
    "trajectories",
    "geometry",
    "evidence",
    "predictions",
    "logs",
)

# The organizer's August 11 release is the current release documented in RESEARCH.md.
# URLs can be overridden for an organizer release without changing code.
SOURCES: dict[str, dict[str, str]] = {
    "validation": {
        "metadata": "https://storage.googleapis.com/dm-perception-test/kilometer_vision_2026_val/kilometervision_valid_20260811.json",
        "images": "https://storage.googleapis.com/dm-perception-test/kilometer_vision_2026_val/kilometervision_valid_images_20260811.zip",
    },
    "test": {
        "metadata": "https://storage.googleapis.com/dm-perception-test/kilometer_vision_2026_val/kilometervision_test_public_20260811.json",
        "images": "https://storage.googleapis.com/dm-perception-test/kilometer_vision_2026_val/kilometervision_test_images_20260811.zip",
    },
}

# These counts are release sanity checks, not tuning parameters. Callers handling a
# future organizer release can pass expected_counts=None or explicit counts.
EXPECTED_COUNTS = {"validation": (13, 18), "test": (54, 986)}
FRAME_SECONDS_TOLERANCE = 1.0
_USE_RELEASE_COUNTS = object()
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class MetadataAudit:
    split: str | None
    video_count: int = 0
    question_count: int = 0
    image_reference_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid"] = self.valid
        return result


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
DEFAULT_FRAME_FPS = 1.0
DEFAULT_MAX_FRAMES = 600
DEFAULT_FRAME_WIDTH = 320
IMAGE_THUMBNAIL_SIZE = 8
QUESTION_FAMILIES = ("compass", "distance", "map", "landmark", "unknown")


def _atomic_json(destination: Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON so readers never observe a partially serialized artifact."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _parse_rate(value: Any) -> float | None:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        rate = float(Fraction(str(value))) if "/" in str(value) else float(value)
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def _probe_with_ffprobe(path: Path, executable: str = "ffprobe") -> dict[str, Any]:
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams or not isinstance(streams[0], Mapping):
        raise ValueError(f"ffprobe found no video stream in {path}")
    stream = streams[0]
    rate = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    duration_value = stream.get("duration")
    if duration_value in (None, "N/A"):
        duration_value = (payload.get("format") or {}).get("duration")
    try:
        duration = float(duration_value)
    except (TypeError, ValueError):
        duration = float("nan")
    try:
        frames = int(stream["nb_frames"]) if stream.get("nb_frames") not in (None, "N/A") else None
    except (TypeError, ValueError):
        frames = None
    try:
        width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
    except (TypeError, ValueError):
        width, height = 0, 0
    if rate is None or not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"ffprobe returned invalid video metadata for {path}")
    if frames is not None and frames <= 0:
        frames = None
    return {
        "duration_seconds": duration,
        "frame_rate": rate,
        "frame_count": frames,
        "width": width,
        "height": height,
        "probe": "ffprobe",
    }


def _probe_with_cv2(path: Path) -> dict[str, Any]:
    """Small fallback for images mounted where ffprobe is unavailable."""
    try:
        import cv2  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on runtime image
        raise RuntimeError("ffprobe is unavailable and OpenCV is not installed") from error
    capture = cv2.VideoCapture(str(path))
    try:
        rate = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if rate <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"OpenCV returned invalid video metadata for {path}")
    return {
        "duration_seconds": frames / rate,
        "frame_rate": rate,
        "frame_count": frames,
        "width": width,
        "height": height,
        "probe": "opencv",
    }


def probe_video(path: str | os.PathLike[str], *, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    """Read validated timing metadata without decoding/copying the source video."""
    source = Path(path)
    try:
        return _probe_with_ffprobe(source, ffprobe_bin)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
        return _probe_with_cv2(source)


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _video_key(source_root: Path, source: Path) -> str:
    relative = source.relative_to(source_root)
    # The organizer layout is <video-id>/<original>.mkv. For a flat synthetic
    # fixture, the filename is the natural stable ID.
    value = relative.parts[0] if len(relative.parts) > 1 else source.stem
    if not _VIDEO_ID_RE.fullmatch(value):
        raise ValueError(f"unsafe source video ID {value!r} for {source}")
    return value


def discover_source_videos(
    source_root: str | os.PathLike[str] = SHARED_VIDEO_ROOT,
    video_ids: Iterable[str] | None = None,
) -> list[tuple[str, Path]]:
    """Discover originals only; no source bytes are written by this function."""
    root = Path(source_root)
    if not root.is_dir():
        raise FileNotFoundError(f"shared video root does not exist: {root}")
    requested = set(video_ids) if video_ids is not None else None
    discovered: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for source in sorted(root.rglob("*")):
        if (not source.is_file() or source.suffix.lower() not in VIDEO_SUFFIXES
                or ".partial" in source.stem):
            continue
        video_id = _video_key(root, source)
        # Filter before duplicate detection: the shared volume may contain a
        # duplicate unrelated artifact, but a targeted build must not fail on it.
        if requested is not None and video_id not in requested:
            continue
        if video_id in seen:
            raise ValueError(f"multiple source videos found for video ID {video_id!r}")
        seen.add(video_id)
        discovered.append((video_id, source))
    return discovered


def _frame_timestamps(metadata: Mapping[str, Any], sample_fps: float, max_frames: int) -> list[tuple[int, float, int]]:
    duration = float(metadata["duration_seconds"])
    source_fps = float(metadata["frame_rate"])
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("video duration and frame rate must be finite and positive")
    count = min(max_frames, max(1, int(math.ceil(duration * sample_fps))))
    timestamps: list[tuple[int, float, int]] = []
    for index in range(count):
        timestamp = min(index / sample_fps, max(0.0, duration - 1.0 / source_fps))
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp > duration + FRAME_SECONDS_TOLERANCE:
            raise ValueError("sample timestamp is outside the probed video duration")
        source_frame = max(0, int(round(timestamp * source_fps)))
        frame_count = metadata.get("frame_count")
        if frame_count is not None and int(frame_count) > 0:
            source_frame = min(source_frame, int(frame_count) - 1)
        timestamps.append((index, timestamp, source_frame))
    return timestamps


def _frame_feature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _extract_frame_ffmpeg(
    source: Path, destination: Path, timestamp: float, *, ffmpeg_bin: str = "ffmpeg", width: int = DEFAULT_FRAME_WIDTH
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep a real image suffix so ffmpeg can infer the muxer before the atomic rename.
    temporary = destination.with_name(destination.name + ".partial.jpg")
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:-2",
        "-q:v",
        "5",
        "-y",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced no frame for {source} at {timestamp:.3f}s")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _extract_frame_cv2(source: Path, destination: Path, timestamp: float, *, width: int = DEFAULT_FRAME_WIDTH) -> None:
    try:
        import cv2  # type: ignore
    except ImportError as error:  # pragma: no cover - depends on runtime image
        raise RuntimeError("ffmpeg is unavailable and OpenCV is not installed") from error
    capture = cv2.VideoCapture(str(source))
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"OpenCV could not decode {source} at {timestamp:.3f}s")
    height, image_width = image.shape[:2]
    scaled_height = max(1, round(height * width / image_width))
    image = cv2.resize(image, (width, scaled_height), interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial.jpg")
    try:
        if not cv2.imwrite(str(temporary), image, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
            raise RuntimeError(f"OpenCV could not write {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_frame(
    source: str | os.PathLike[str], destination: str | os.PathLike[str], timestamp: float, *,
    ffmpeg_bin: str = "ffmpeg", width: int = DEFAULT_FRAME_WIDTH
) -> None:
    """Extract one bounded low-resolution JPEG, preferring ffmpeg and falling back to OpenCV."""
    source_path, destination_path = Path(source), Path(destination)
    try:
        _extract_frame_ffmpeg(source_path, destination_path, timestamp, ffmpeg_bin=ffmpeg_bin, width=width)
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        _extract_frame_cv2(source_path, destination_path, timestamp, width=width)


def _valid_existing_manifest(path: Path, fingerprint: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload.get("source", {}).get("fingerprint") == dict(fingerprint)
            and payload.get("sampling") == dict(config)
            and all((path.parent / frame["path"]).is_file() and (path.parent / frame["path"]).stat().st_size > 0 for frame in payload.get("frames", []))
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def build_video_frames(
    video_id: str,
    source: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] = DATA_ROOT,
    sample_fps: float = DEFAULT_FRAME_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frame_width: int = DEFAULT_FRAME_WIDTH,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, Any]:
    """Build/resume one source's frame manifest and dependency-light byte features."""
    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError(f"unsafe video ID {video_id!r}")
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    if not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")
    if not isinstance(frame_width, int) or frame_width < 16:
        raise ValueError("frame_width must be at least 16 pixels")
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    root_path = Path(root)
    frame_root = root_path / "frames"
    manifest_path = frame_root / f"{video_id}.json"
    features_path = frame_root / f"{video_id}.features.json"
    config = {"sample_fps": sample_fps, "max_frames": max_frames, "frame_width": frame_width}
    fingerprint = _source_fingerprint(source_path)
    if _valid_existing_manifest(manifest_path, fingerprint, config) and features_path.exists():
        try:
            feature_payload = json.loads(features_path.read_text(encoding="utf-8"))
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(feature_payload.get("frames"), list) and len(feature_payload["frames"]) == len(manifest_payload.get("frames", [])):
                return {"video_id": video_id, "status": "skipped", "manifest": str(manifest_path), "frames": len(manifest_payload.get("frames", []))}
        except (OSError, TypeError, AttributeError, json.JSONDecodeError):
            pass

    metadata = probe_video(source_path, ffprobe_bin=ffprobe_bin)
    timestamps = _frame_timestamps(metadata, sample_fps, max_frames)
    frame_dir = frame_root / video_id
    partial_manifest = manifest_path.with_name(manifest_path.name + ".partial")
    frames: list[dict[str, Any]] = []
    if partial_manifest.exists():
        try:
            checkpoint = json.loads(partial_manifest.read_text(encoding="utf-8"))
            candidate_frames = checkpoint.get("frames", [])
            if (checkpoint.get("source", {}).get("fingerprint") == fingerprint
                    and checkpoint.get("sampling") == config
                    and isinstance(candidate_frames, list)):
                frames = candidate_frames
        except (OSError, TypeError, json.JSONDecodeError):
            frames = []
    # A checkpoint may contain a prefix from an interrupted run. Reuse only the
    # exact expected prefix; stale or malformed entries are discarded safely.
    prefix = 0
    for expected, saved in zip(timestamps, frames):
        index, timestamp, source_frame = expected
        if (saved.get("index") != index or
                abs(float(saved.get("timestamp_seconds", -1)) - round(timestamp, 6)) > 1e-6 or
                not (frame_root / saved.get("path", "")).is_file()):
            break
        prefix += 1
    frames = frames[:prefix]
    for index, timestamp, source_frame in timestamps[prefix:]:
        filename = f"frame-{index:06d}.jpg"
        output = frame_dir / filename
        if not (output.is_file() and output.stat().st_size > 0):
            extract_frame(source_path, output, timestamp, ffmpeg_bin=ffmpeg_bin, width=frame_width)
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError(f"frame extraction produced an empty artifact: {output}")
        frame = {
            "index": index,
            "timestamp_seconds": round(timestamp, 6),
            "source_frame": source_frame,
            "path": f"{video_id}/{filename}",
            "feature": _frame_feature(output),
        }
        frames.append(frame)
        checkpoint = {
            "schema_version": 1,
            "video_id": video_id,
            "source": {"path": str(source_path), "fingerprint": fingerprint},
            "probe": metadata,
            "sampling": config,
            "frames": frames,
        }
        _atomic_json(partial_manifest, checkpoint)
    manifest = {
        "schema_version": 1,
        "video_id": video_id,
        "source": {"path": str(source_path), "fingerprint": fingerprint},
        "probe": metadata,
        "sampling": config,
        "frames": frames,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(features_path, {"schema_version": 1, "video_id": video_id, "frames": [{"index": f["index"], "timestamp_seconds": f["timestamp_seconds"], **f["feature"]} for f in frames]})
    if partial_manifest.exists():
        partial_manifest.unlink()
    return {"video_id": video_id, "status": "built", "manifest": str(manifest_path), "features": str(features_path), "frames": len(frames)}


def build_frame_manifests(
    *,
    source_root: str | os.PathLike[str] = SHARED_VIDEO_ROOT,
    root: str | os.PathLike[str] = DATA_ROOT,
    sample_fps: float = DEFAULT_FRAME_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frame_width: int = DEFAULT_FRAME_WIDTH,
    video_ids: Iterable[str] | None = None,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, Any]:
    """Build all question-independent frame artifacts from shared originals."""
    bootstrap_volume(root)
    requested = set(video_ids) if video_ids is not None else None
    discovered = discover_source_videos(source_root, requested)
    if requested is not None:
        unknown = requested - {video_id for video_id, _ in discovered}
        if unknown:
            raise ValueError(f"unknown source video IDs: {sorted(unknown)}")
    results: list[dict[str, Any]] = []
    started = time.time()
    report_path = Path(root) / "logs" / "frame_build.json"
    for video_id, source in discovered:
        try:
            results.append(build_video_frames(video_id, source, root=root, sample_fps=sample_fps, max_frames=max_frames, frame_width=frame_width, ffprobe_bin=ffprobe_bin, ffmpeg_bin=ffmpeg_bin))
        except Exception as error:  # preserve other videos' resumable progress
            results.append({"video_id": video_id, "status": "error", "error": str(error)})
        # Commit after every source so a long batch can resume after a timeout or
        # container loss; the report itself is also an atomic checkpoint.
        report = {"schema_version": 1, "source_root": str(source_root), "sampling": {"sample_fps": sample_fps, "max_frames": max_frames, "frame_width": frame_width}, "videos": results, "elapsed_seconds": round(time.time() - started, 3)}
        _atomic_json(report_path, report)
        _commit_volume(root)
    report = {"schema_version": 1, "source_root": str(source_root), "sampling": {"sample_fps": sample_fps, "max_frames": max_frames, "frame_width": frame_width}, "videos": results, "elapsed_seconds": round(time.time() - started, 3)}
    _atomic_json(report_path, report)
    _commit_volume(root)
    return report


def build_geometry_contracts(
    *, root: str | os.PathLike[str] = DATA_ROOT, video_ids: Iterable[str] | None = None, backend: str = "auto",
) -> dict[str, Any]:
    """Persist explicit geometry contracts; learned backends are opt-in."""
    root_path = Path(root); requested = set(video_ids) if video_ids is not None else None
    available = discover_backends(); results = []
    for manifest in sorted((root_path / "frames").glob("*.json")):
        if manifest.name.endswith(".features.json") or manifest.name.endswith(".partial"): continue
        video_id = manifest.stem
        if requested is not None and video_id not in requested: continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        times = [float(frame["timestamp_seconds"]) for frame in payload.get("frames", [])]
        learned_path = root_path / "geometry" / f"{video_id}.da3.json"
        chosen = backend
        if chosen == "auto":
            chosen = "da3" if learned_path.exists() else next((name for name in ("vggt", "mapanything") if available[name]["available"]), "fallback")
        destination = root_path / "geometry" / f"{video_id}.json"
        if chosen == "da3" and learned_path.exists():
            learned = json.loads(learned_path.read_text(encoding="utf-8"))
            for frame in learned.get("frames", []):
                raw_confidence = frame.get("confidence")
                if raw_confidence is not None:
                    raw = float(raw_confidence)
                    frame["raw_confidence"] = raw
                    frame["confidence"] = raw if 0.0 <= raw <= 1.0 else 1.0 / (1.0 + abs(raw))
            learned["contract"] = {"pose_source": "DA3-LARGE", "scale": "relative", "confidence": "bounded_inverse_abs_with_raw_preserved", "uncertainty": "backend_confidence_or_unknown"}
            _atomic_json(destination, learned)
            results.append({"video_id": video_id, "backend": "da3", "status": "available", "frames": len(learned.get("frames", []))})
        else:
            result = fallback_geometry(video_id, times, reason=f"{chosen}_adapter_not_enabled")
            _atomic_json(destination, result.to_dict())
            results.append({"video_id": video_id, "backend": chosen, "status": result.status, "frames": len(times)})
    report = {"schema_version": 1, "requested_backend": backend, "available_backends": available, "videos": results}
    _atomic_json(root_path / "logs" / "geometry_build.json", report); _commit_volume(root)
    return report


def build_route_signatures(
    *, root: str | os.PathLike[str] = DATA_ROOT, video_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Materialize the real turn/heading route signature for every geometry contract.

    This is the missing link ``GEOMETRY_RESEARCH_2026.md`` flags: ``build_geometry_contracts``
    persists raw DA3 poses, but nothing previously turned them into a route signature.
    """
    root_path = Path(root)
    requested = set(video_ids) if video_ids is not None else None
    results: list[dict[str, Any]] = []
    for manifest in sorted((root_path / "geometry").glob("*.json")):
        name = manifest.name
        if name.endswith(".da3.json") or name.endswith(".route_signature.json") or name.endswith(".partial"):
            continue
        video_id = manifest.stem
        if requested is not None and video_id not in requested:
            continue
        result = build_route_signature(video_id, root_path)
        results.append({"video_id": video_id, "available": result["available"], "turn_count": len(result["turn_angles_degrees"])})
    report = {"schema_version": 1, "videos": results}
    _atomic_json(root_path / "logs" / "route_signature_build.json", report)
    _commit_volume(root)
    return report


def build_visual_indexes(
    *, root: str | os.PathLike[str] = DATA_ROOT, video_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build bounded frame descriptors and optional OCR/place text indexes."""
    root_path = Path(root)
    requested = set(video_ids) if video_ids is not None else None
    results: list[dict[str, Any]] = []
    for manifest in sorted((root_path / "frames").glob("*.json")):
        if manifest.name.endswith(".features.json") or manifest.name.endswith(".partial"):
            continue
        video_id = manifest.stem
        if requested is not None and video_id not in requested:
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        embeddings: list[dict[str, Any]] = []
        ocr_rows: list[dict[str, Any]] = []
        for frame in payload.get("frames", []):
            path = (manifest.parent / frame["path"]).resolve()
            if not path.is_file():
                continue
            vector = _image_vector(path, size=16)
            embeddings.append({"index": frame["index"], "timestamp_seconds": frame["timestamp_seconds"], "vector": vector, "source": frame["path"]})
            text = ""
            try:
                import pytesseract  # type: ignore
                from PIL import Image  # type: ignore
                text = pytesseract.image_to_string(Image.open(path)).strip()
            except Exception:
                text = ""
            ocr_rows.append({"index": frame["index"], "timestamp_seconds": frame["timestamp_seconds"], "text": text, "source": frame["path"]})
        _atomic_json(root_path / "embeddings" / f"{video_id}.json", {"schema_version": 1, "video_id": video_id, "method": "thumbnail descriptor; semantic model optional", "frames": embeddings})
        _atomic_json(root_path / "ocr" / f"{video_id}.json", {"schema_version": 1, "video_id": video_id, "method": "pytesseract when installed", "frames": ocr_rows})
        results.append({"video_id": video_id, "frames": len(embeddings), "ocr_rows": len(ocr_rows)})
    report = {"schema_version": 1, "videos": results}
    _atomic_json(root_path / "logs" / "visual_index.json", report)
    _commit_volume(root)
    return report


def build_route_features(
    *, root: str | os.PathLike[str] = DATA_ROOT, video_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    root_path = Path(root)
    frame_root = root_path / "frames"
    output_root = root_path / "trajectories"
    requested = set(video_ids) if video_ids is not None else None
    manifests = sorted(frame_root.glob("*.json"))
    results: list[dict[str, Any]] = []
    for manifest in manifests:
        if manifest.name.endswith(".features.json") or manifest.name.endswith(".partial"):
            continue
        video_id = manifest.stem
        if requested is not None and video_id not in requested:
            continue
        output = output_root / f"{video_id}.json"
        try:
            result = estimate_route(manifest, output)
            results.append({"video_id": video_id, "status": "built", "output": str(output), "turn_candidates": len(result["turn_candidates"])})
        except Exception as error:
            results.append({"video_id": video_id, "status": "error", "error": str(error)})
    report = {"schema_version": 1, "videos": results, "built": sum(item["status"] == "built" for item in results)}
    _atomic_json(root_path / "logs" / "route_build.json", report)
    _commit_volume(root)
    return report


def bootstrap_volume(root: str | os.PathLike[str] = DATA_ROOT) -> dict[str, Any]:
    """Create the canonical PATHFINDER directories in the mounted Volume."""
    root_path = Path(root)
    created = []
    for relative in CANONICAL_DIRS:
        path = root_path / relative
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
    return {"root": str(root_path), "directories": created}


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    normalized = posixpath.normpath(value)
    return (
        normalized == value
        and normalized not in (".", "..")
        and not normalized.startswith("/")
        and not normalized.startswith("../")
    )


def _iter_records(payload: Any) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
    """Yield ``(video_id, question_key, question)`` for organizer JSON variants."""
    if isinstance(payload, Mapping) and isinstance(payload.get("videos"), (list, dict)):
        payload = payload["videos"]

    if isinstance(payload, Mapping):
        for video_id, questions in payload.items():
            if video_id in {"metadata", "version", "split"}:
                continue
            if isinstance(questions, Mapping):
                iterator = questions.items()
            elif isinstance(questions, list):
                iterator = ((str(i), item) for i, item in enumerate(questions))
            else:
                yield str(video_id), "", {}  # validation will report the shape error
                continue
            for question_key, question in iterator:
                yield str(video_id), str(question_key), question if isinstance(question, Mapping) else {}
        return

    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                yield "", str(index), {}
                continue
            video_id = item.get("video_id") or item.get("videoId") or item.get("video")
            question_id = item.get("id", index)
            yield str(video_id or ""), str(question_id), item
        return

    yield "", "", {}


def _referenced_images(question: Mapping[str, Any]) -> list[Any]:
    paths = question.get("image_paths", [])
    if paths is None:
        return []
    return paths if isinstance(paths, list) else [paths]


def _video_id_from_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    query_id = parse_qs(parsed.query).get("v")
    if query_id and query_id[0]:
        return query_id[0]
    return None


def validate_metadata(
    payload: Any,
    *,
    split: str | None = None,
    image_members: Iterable[str] | None = None,
    expected_counts: tuple[int, int] | None = None,
    frame_tolerance: float = FRAME_SECONDS_TOLERANCE,
) -> MetadataAudit:
    """Validate organizer metadata without interpreting or changing answer labels.

    ``image_members`` should be the names from the matching ZIP. ZIP directory
    prefixes are accepted; image references are compared by basename because the
    organizer metadata stores archive-relative basenames.
    """
    audit = MetadataAudit(split=split)
    videos: set[str] = set()
    question_ids: dict[str, set[str]] = {}
    members = None
    if image_members is not None:
        members = {posixpath.basename(str(member).rstrip("/")) for member in image_members}

    for video_id, question_key, question in _iter_records(payload):
        audit.question_count += 1
        if not video_id or not _VIDEO_ID_RE.fullmatch(video_id):
            audit.errors.append(f"invalid video ID: {video_id!r}")
        else:
            videos.add(video_id)
            question_ids.setdefault(video_id, set())
            if question_key in question_ids[video_id]:
                audit.errors.append(f"duplicate question ID {question_key!r} for video {video_id!r}")
            question_ids[video_id].add(question_key)

        if not question:
            audit.errors.append(f"video {video_id!r} question {question_key!r} is not an object")
            continue
        record_video_id = question.get("video_id") or question.get("videoId")
        if record_video_id is not None and str(record_video_id) != video_id:
            audit.errors.append(f"video ID mismatch for question {question_key!r}: {record_video_id!r}")
        url_video_id = _video_id_from_url(question.get("video_url"))
        if url_video_id is not None and url_video_id != video_id:
            audit.errors.append(f"video_url ID mismatch for video {video_id!r}: {url_video_id!r}")

        options = question.get("options")
        if not isinstance(options, list) or len(options) != 5:
            audit.errors.append(f"video {video_id!r} question {question_key!r} must have exactly five options")
        elif any(not isinstance(option, (str, int, float)) for option in options):
            audit.errors.append(f"video {video_id!r} question {question_key!r} has a non-scalar option")

        refs = _referenced_images(question)
        audit.image_reference_count += len(refs)
        for ref in refs:
            if not _safe_relative_path(ref):
                audit.errors.append(f"unsafe referenced image path for {video_id!r}/{question_key!r}: {ref!r}")
            elif members is not None and posixpath.basename(ref) not in members:
                audit.errors.append(f"missing referenced image {ref!r} for {video_id!r}/{question_key!r}")

        frame_rate = question.get("image/original_frame_rate", question.get("original_frame_rate"))
        frame_fields = ("start_frame", "end_frame", "start_seconds", "end_seconds")
        present = [field_name in question for field_name in frame_fields]
        if any(present) and not all(present):
            audit.errors.append(f"incomplete frame/time fields for {video_id!r}/{question_key!r}")
        elif all(present):
            try:
                fps = float(frame_rate)
                start_frame = float(question["start_frame"])
                end_frame = float(question["end_frame"])
                start_seconds = float(question["start_seconds"])
                end_seconds = float(question["end_seconds"])
                values = (fps, start_frame, end_frame, start_seconds, end_seconds)
                if not all(math.isfinite(value) for value in values) or fps <= 0 or start_frame < 0 or end_frame < start_frame:
                    raise ValueError
                if abs(start_frame / fps - start_seconds) > frame_tolerance or abs(end_frame / fps - end_seconds) > frame_tolerance:
                    audit.errors.append(f"frame/time mismatch for {video_id!r}/{question_key!r}")
            except (TypeError, ValueError):
                audit.errors.append(f"invalid frame/time values for {video_id!r}/{question_key!r}")

    audit.video_count = len(videos)
    if expected_counts is not None:
        expected_videos, expected_questions = expected_counts
        if audit.video_count != expected_videos:
            audit.errors.append(f"expected {expected_videos} videos, found {audit.video_count}")
        if audit.question_count != expected_questions:
            audit.errors.append(f"expected {expected_questions} questions, found {audit.question_count}")
    if audit.video_count == 0:
        audit.errors.append("metadata contains no videos")
    return audit


def _fallback_image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read common image headers without adding a required imaging dependency."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if data[:2] != b"\xff\xd8":
        return None
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in (0xD8, 0xD9):
            continue
        if offset + 2 > len(data):
            break
        length = struct.unpack(">H", data[offset:offset + 2])[0]
        if length < 2 or offset + length > len(data):
            break
        # SOF markers contain precision, height, width after their length.
        if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
            if length >= 7:
                return struct.unpack(">HH", data[offset + 3:offset + 7])
        offset += length
    return None


def image_descriptor(data: bytes) -> dict[str, Any]:
    """Return a stable, small descriptor for image bytes.

    The source bytes are consumed from memory and are never written to the
    PATHFINDER tree. Pillow is optional: dimensions still work for common image
    headers, while the thumbnail/hash fields are populated when Pillow exists.
    """
    descriptor: dict[str, Any] = {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": None,
        "height": None,
        "format": None,
        "thumbnail": None,
        "thumbnail_size": None,
        "average_rgb": None,
        "perceptual_hash": None,
        "pillow": False,
    }
    dimensions = _fallback_image_dimensions(data)
    if dimensions:
        descriptor["width"], descriptor["height"] = dimensions
    try:
        from PIL import Image  # type: ignore
        image_module = Image
        with image_module.open(io.BytesIO(data)) as image:
            descriptor["width"], descriptor["height"] = image.size
            descriptor["format"] = image.format
            rgb = image.convert("RGB")
            resampling = getattr(image_module, "Resampling", image_module)
            bilinear = resampling.BILINEAR
            tiny = rgb.resize((IMAGE_THUMBNAIL_SIZE, IMAGE_THUMBNAIL_SIZE), bilinear)
            pixels = list(tiny.getdata())
            raw = bytes(channel for pixel in pixels for channel in pixel)
            # A fixed 8x8 RGB thumbnail is compact and useful for color retrieval;
            # dHash provides a deterministic perceptual comparison signal.
            wide = rgb.resize((IMAGE_THUMBNAIL_SIZE + 1, IMAGE_THUMBNAIL_SIZE), bilinear).convert("L")
            wide_pixels = list(wide.getdata())
            hash_bits = 0
            for row in range(IMAGE_THUMBNAIL_SIZE):
                for column in range(IMAGE_THUMBNAIL_SIZE):
                    hash_bits = (hash_bits << 1) | int(wide_pixels[row * (IMAGE_THUMBNAIL_SIZE + 1) + column] > wide_pixels[row * (IMAGE_THUMBNAIL_SIZE + 1) + column + 1])
            descriptor.update({
                "thumbnail": base64.b64encode(raw).decode("ascii"),
                "thumbnail_size": [IMAGE_THUMBNAIL_SIZE, IMAGE_THUMBNAIL_SIZE],
                "average_rgb": [round(sum(pixel[channel] for pixel in pixels) / len(pixels), 3) for channel in range(3)],
                "perceptual_hash": f"{hash_bits:016x}",
                "pillow": True,
            })
    except ImportError:
        pass
    except Exception as error:  # malformed media remains auditable, not executable
        descriptor["image_error"] = type(error).__name__
    return descriptor


def _question_text(question: Mapping[str, Any]) -> str:
    values = [question.get(key, "") for key in ("question", "question_text", "text", "prompt")]
    return " ".join(str(value) for value in values if isinstance(value, str))


def _option_text(option: Any) -> str:
    if isinstance(option, Mapping):
        for key in ("text", "label", "value", "option"):
            if isinstance(option.get(key), (str, int, float)):
                return str(option[key])
        return ""
    return str(option) if isinstance(option, (str, int, float)) else ""


def _option_image_paths(option: Any) -> list[str]:
    if not isinstance(option, Mapping):
        return []
    paths: list[str] = []
    for key in ("image_path", "image", "path", "src"):
        value = option.get(key)
        values = value if isinstance(value, list) else [value]
        paths.extend(item for item in values if _safe_relative_path(item))
    return paths


def _compile_option_differences(options: list[Any]) -> dict[str, Any]:
    direction_words = {"north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest", "left", "right", "straight", "back"}
    tokenized: list[dict[str, Any]] = []
    for index, option in enumerate(options):
        text = _option_text(option)
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        tokens = sorted(set(re.findall(r"[a-z]+(?:-[a-z]+)?", normalized)))
        numbers = [float(value) if "." in value else int(value) for value in re.findall(r"\b\d+(?:\.\d+)?\b", normalized)]
        directions = sorted({token.replace("-", "") for token in tokens if token.replace("-", "") in direction_words})
        tokenized.append({
            "index": index,
            "text": text,
            "normalized": normalized,
            "tokens": tokens,
            "numbers": numbers,
            "directions": directions,
            "image_paths": _option_image_paths(option),
        })
    token_sets = [set(item["tokens"]) for item in tokenized]
    common_tokens = set.intersection(*token_sets) if token_sets else set()
    varying_tokens = sorted(set.union(*token_sets) - common_tokens) if token_sets else []
    number_sets = [set(item["numbers"]) for item in tokenized]
    direction_sets = [set(item["directions"]) for item in tokenized]
    common_numbers = set.intersection(*number_sets) if number_sets else set()
    common_directions = set.intersection(*direction_sets) if direction_sets else set()
    varying_numbers = sorted(set.union(*number_sets) - common_numbers) if number_sets else []
    varying_directions = sorted(set.union(*direction_sets) - common_directions) if direction_sets else []
    pairwise: list[dict[str, Any]] = []
    for left in range(len(tokenized)):
        for right in range(left + 1, len(tokenized)):
            left_tokens, right_tokens = set(tokenized[left]["tokens"]), set(tokenized[right]["tokens"])
            pairwise.append({
                "left": left,
                "right": right,
                "added_to_right": sorted(right_tokens - left_tokens),
                "removed_from_left": sorted(left_tokens - right_tokens),
            })
    return {
        "options": tokenized,
        "common_tokens": sorted(common_tokens),
        "varying_tokens": varying_tokens,
        "common_numbers": sorted(common_numbers),
        "varying_numbers": varying_numbers,
        "common_directions": sorted(common_directions),
        "varying_directions": varying_directions,
        "pairwise": pairwise,
    }


def compile_question(video_id: str, question_key: str, question: Mapping[str, Any]) -> dict[str, Any]:
    """Compile routing predicates without consulting organizer answer fields."""
    if not isinstance(question, Mapping):
        raise TypeError("question must be an object")
    text = _question_text(question)
    options = question.get("options", [])
    options = options if isinstance(options, list) else []
    combined = (text + " " + " ".join(_option_text(option) for option in options)).lower()
    hint_text = text.lower() if text.strip() else combined
    refs = [ref for ref in _referenced_images(question) if _safe_relative_path(ref)]
    option_refs = [ref for option in options for ref in _option_image_paths(option)]
    image_refs = sorted(set(refs + option_refs))
    map_hint = bool(re.search(r"\b(map|route|path|street layout|trace|top[- ]?down)\b", hint_text))
    compass_hint = bool(re.search(r"\b(compass|direction|heading|facing|turn|north|south|east|west|left|right)\b", hint_text))
    distance_hint = bool(re.search(r"\b(distance|far|near|meters?|metres?|kilometers?|kilometres?|km|miles?|how long|closer|further)\b", hint_text))
    landmark_hint = bool(re.search(r"\b(landmark|building|sign|monument|location|appear|seen|shown|image|picture|photo|reference)\b", hint_text))
    if map_hint or len(image_refs) >= 5 and len(options) == 5:
        family = "map"
    elif compass_hint:
        family = "compass"
    elif distance_hint:
        family = "distance"
    elif landmark_hint or image_refs:
        family = "landmark"
    else:
        family = "unknown"
    return {
        "schema_version": 1,
        "video_id": video_id,
        "question_id": question_key,
        "family": family,
        "question_text": text,
        "option_count": len(options),
        "image_paths": image_refs,
        "option_differences": _compile_option_differences(options),
    }


def compile_questions(payload: Any, *, split: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    counts = {family: 0 for family in QUESTION_FAMILIES}
    for video_id, question_key, question in _iter_records(payload):
        if not question:
            counts["unknown"] += 1
            compiled.append({"schema_version": 1, "video_id": video_id, "question_id": question_key, "family": "unknown", "question_text": "", "option_count": 0, "image_paths": [], "option_differences": _compile_option_differences([])})
            continue
        item = compile_question(video_id, question_key, question)
        compiled.append(item)
        counts[item["family"]] += 1
    audit = {
        "schema_version": 1,
        "split": split,
        "question_count": len(compiled),
        "family_counts": counts,
        "classified_count": len(compiled) - counts["unknown"],
        "coverage_ratio": round((len(compiled) - counts["unknown"]) / len(compiled), 6) if compiled else 0.0,
        "schema_fields": ["video_id", "question_id", "family", "question_text", "option_count", "image_paths", "option_differences"],
        "ground_truth_fields_read": [],
    }
    return compiled, audit


def _zip_member_for_reference(reference: str, members: list[str]) -> str:
    if not _safe_relative_path(reference):
        raise ValueError(f"unsafe referenced image path: {reference!r}")
    exact = [member for member in members if member == reference]
    if len(exact) == 1:
        return exact[0]
    basename = posixpath.basename(reference)
    matches = [member for member in members if posixpath.basename(member.rstrip("/")) == basename]
    if len(matches) != 1:
        if not matches:
            raise ValueError(f"missing referenced image {reference!r}")
        raise ValueError(f"ambiguous referenced image basename {reference!r}")
    return matches[0]


def index_split_images(
    split: str,
    *,
    root: str | os.PathLike[str] = DATA_ROOT,
) -> dict[str, Any]:
    """Index only referenced ZIP images; never extract or rewrite organizer media."""
    root_path = Path(root)
    manifest_path = root_path / "manifests" / f"{split}.json"
    image_path = root_path / "manifests" / f"{split}_images.zip"
    if not manifest_path.exists() or not image_path.exists():
        raise FileNotFoundError(f"ingest {split!r} before indexing images")
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    members = _read_zip_members(image_path)
    compiled, compiler_audit = compile_questions(payload, split=split)
    references: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in compiled:
        for reference in item["image_paths"]:
            key = (item["video_id"], item["question_id"], reference)
            if key not in seen:
                seen.add(key)
                references.append(key)
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with zipfile.ZipFile(image_path) as archive:
        for video_id, question_id, reference in references:
            try:
                member = _zip_member_for_reference(reference, members)
                data = archive.read(member)
                records.append({"video_id": video_id, "question_id": question_id, "reference": reference, "member": member, "descriptor": image_descriptor(data)})
            except (KeyError, ValueError, OSError, zipfile.BadZipFile) as error:
                errors.append(f"{video_id}/{question_id}: {error}")
    report = {"schema_version": 1, "split": split, "reference_count": len(references), "indexed_count": len(records), "errors": errors, "compiler": compiler_audit, "images": records}
    features_path = root_path / "route-features" / f"{split}_images.json"
    compiler_path = root_path / "route-features" / f"{split}_questions.json"
    audit_path = root_path / "logs" / f"{split}_compiler_audit.json"
    _atomic_json(features_path, report)
    _atomic_json(compiler_path, {"schema_version": 1, "split": split, "questions": compiled})
    _atomic_json(audit_path, {**compiler_audit, "image_reference_count": len(references), "image_index_errors": errors})
    _commit_volume(root_path)
    if errors:
        raise ValueError("image index failed: " + "; ".join(errors))
    return {"split": split, "images": str(features_path), "questions": str(compiler_path), "audit": str(audit_path), "reference_count": len(references), "indexed_count": len(records)}


def _read_zip_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        # testfile also catches truncated downloads before an audit is published.
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"corrupt image archive member: {bad_member}")
        return archive.namelist()


def download_atomic(url: str, destination: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> Path:
    """Download one organizer artifact to ``*.partial``, then atomically rename it."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    partial = target.with_name(target.name + ".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial, target)
    return target


def _commit_volume(root: str | os.PathLike[str]) -> None:
    # A local unit test may import the Modal SDK while writing to a temporary
    # directory. Only the mounted canonical path may issue a remote commit.
    if Path(root) != DATA_ROOT:
        return
    if DATA_VOLUME is not None and hasattr(DATA_VOLUME, "commit"):
        DATA_VOLUME.commit()


def _source(split: str) -> dict[str, str]:
    if split not in SOURCES:
        raise ValueError(f"split must be one of {', '.join(sorted(SOURCES))}")
    return SOURCES[split]


def ingest_split(
    split: str,
    *,
    root: str | os.PathLike[str] = DATA_ROOT,
    sources: Mapping[str, str] | None = None,
    expected_counts: tuple[int, int] | None | object = _USE_RELEASE_COUNTS,
) -> dict[str, Any]:
    """Ingest one metadata JSON and matching image ZIP into the mounted Volume."""
    source = dict(_source(split))
    if sources:
        source.update(sources)
    root_path = Path(root)
    bootstrap_volume(root_path)
    manifest_path = root_path / "manifests" / f"{split}.json"
    image_path = root_path / "manifests" / f"{split}_images.zip"
    download_atomic(source["metadata"], manifest_path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    members = _read_zip_members(download_atomic(source["images"], image_path))
    audit = validate_metadata(
        payload,
        split=split,
        image_members=members,
        expected_counts=EXPECTED_COUNTS.get(split) if expected_counts is _USE_RELEASE_COUNTS else expected_counts,
    )
    _write_audit(root_path, split, audit)
    if not audit.valid:
        raise ValueError("metadata audit failed: " + "; ".join(audit.errors))
    _commit_volume(root_path)
    return {"split": split, "manifest": str(manifest_path), "images": str(image_path), "audit": audit.to_dict()}


def _write_audit(root: Path, split: str, audit: MetadataAudit) -> Path:
    destination = root / "logs" / f"{split}_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # This report intentionally contains counts/errors only; organizer answers stay
    # solely in the downloaded organizer manifest.
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".partial", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(audit.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def predict_split(split: str, *, root: str | os.PathLike[str] = DATA_ROOT) -> dict[str, Any]:
    if split not in SOURCES:
        raise ValueError(f"split must be one of {', '.join(sorted(SOURCES))}")
    root_path = Path(root)
    manifest_path = root_path / "manifests" / f"{split}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"ingest {split!r} before prediction")
    prediction_path = root_path / "predictions" / f"{split}_submission.json"
    evidence_path = root_path / "evidence" / f"{split}_evidence.json"
    audit = generate_split(manifest_path, prediction_path, evidence_path)
    _commit_volume(root)
    return {"split": split, "prediction": str(prediction_path), "evidence": str(evidence_path), "audit": audit}


def audit_split(
    split: str,
    *,
    root: str | os.PathLike[str] = DATA_ROOT,
    expected_counts: tuple[int, int] | None | object = _USE_RELEASE_COUNTS,
) -> dict[str, Any]:
    root_path = Path(root)
    manifest_path = root_path / "manifests" / f"{split}.json"
    image_path = root_path / "manifests" / f"{split}_images.zip"
    if not manifest_path.exists() or not image_path.exists():
        raise FileNotFoundError(f"ingest {split!r} before auditing")
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    audit = validate_metadata(
        payload,
        split=split,
        image_members=_read_zip_members(image_path),
        expected_counts=EXPECTED_COUNTS.get(split) if expected_counts is _USE_RELEASE_COUNTS else expected_counts,
    )
    _write_audit(root_path, split, audit)
    return audit.to_dict()


def run_action(
    action: str,
    split: str | None = None,
    *,
    root: str | os.PathLike[str] = DATA_ROOT,
    source_root: str | os.PathLike[str] = SHARED_VIDEO_ROOT,
    sample_fps: float = DEFAULT_FRAME_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frame_width: int = DEFAULT_FRAME_WIDTH,
    video_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    if action == "bootstrap":
        return bootstrap_volume(root)
    if action in {"ingest", "audit", "predict", "index-split", "compile-split"}:
        if split not in SOURCES:
            raise ValueError("--split validation or --split test is required")
        if action == "ingest":
            return ingest_split(split, root=root)
        if action == "audit":
            return audit_split(split, root=root)
        if action == "predict":
            return predict_split(split, root=root)
        if action == "compile-split":
            manifest_path = Path(root) / "manifests" / f"{split}.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"ingest {split!r} before compiling questions")
            with manifest_path.open("r", encoding="utf-8") as stream:
                compiled, compiler_audit = compile_questions(json.load(stream), split=split)
            destination = Path(root) / "route-features" / f"{split}_questions.json"
            audit_path = Path(root) / "logs" / f"{split}_compiler_audit.json"
            _atomic_json(destination, {"schema_version": 1, "split": split, "questions": compiled})
            _atomic_json(audit_path, compiler_audit)
            return {"split": split, "questions": str(destination), "audit": str(audit_path), **compiler_audit}
        return index_split_images(split, root=root)
    if action in {"build-frames", "build-frame-manifests", "index-frames"}:
        return build_frame_manifests(source_root=source_root, root=root, sample_fps=sample_fps, max_frames=max_frames, frame_width=frame_width, video_ids=video_ids)
    if action in {"build-geometry", "geometry-contract"}:
        return build_geometry_contracts(root=root, video_ids=video_ids)
    if action in {"build-route-signature", "route-signature"}:
        return build_route_signatures(root=root, video_ids=video_ids)
    if action in {"build-visual-index", "visual-index"}:
        return build_visual_indexes(root=root, video_ids=video_ids)
    if action in {"build-routes", "route-features"}:
        return build_route_features(root=root, video_ids=video_ids)
    raise ValueError("--action must be bootstrap, ingest, audit, predict, index-split, compile-split, build-frames, build-geometry, build-route-signature, build-visual-index, or build-routes")


# Keep the import path/test suite dependency-light while exposing real Modal actions
# whenever the Modal SDK is present in the execution image.
DATA_VOLUME = None
app = None
if modal is not None:  # pragma: no cover - Modal decorators are tested remotely
    DATA_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    FRAME_IMAGE = (modal.Image.debian_slim(python_version="3.11")
                   .apt_install("ffmpeg")
                   .pip_install("pillow", "pytesseract", "opencv-python-headless", "numpy")
                   .add_local_python_source("submission", "question_planner", "route_solver", "specialists", "landmark_matcher", "geometry_backend", "geometry_contract", "motion_solver", "map_solver"))
    app = modal.App("kilometervision-pathfinder")

    @app.function(image=FRAME_IMAGE, volumes={str(DATA_ROOT.parent): DATA_VOLUME}, timeout=60 * 60)
    def modal_action(
        action: str = "bootstrap", split: str = "", source_root: str = str(SHARED_VIDEO_ROOT),
        sample_fps: float = DEFAULT_FRAME_FPS, max_frames: int = DEFAULT_MAX_FRAMES,
        frame_width: int = DEFAULT_FRAME_WIDTH, video_id: str = "", straight_line: bool = False, dino_landmark: bool = False,
        gemini_direct: bool = False,
    ) -> dict[str, Any]:
        if straight_line:
            os.environ["KILOMETERVISION_ENABLE_STRAIGHT_LINE"] = "1"
        if dino_landmark:
            os.environ["KILOMETERVISION_ENABLE_DINO_LANDMARK"] = "1"
        if gemini_direct:
            os.environ["KILOMETERVISION_ENABLE_GEMINI_DIRECT"] = "1"
        return run_action(action, split or None, root=DATA_ROOT, source_root=source_root, sample_fps=sample_fps, max_frames=max_frames, frame_width=frame_width, video_ids=[video_id] if video_id else None)

    @app.local_entrypoint()
    def main(
        action: str = "bootstrap", split: str = "", source_root: str = str(SHARED_VIDEO_ROOT),
        sample_fps: float = DEFAULT_FRAME_FPS, max_frames: int = DEFAULT_MAX_FRAMES,
        frame_width: int = DEFAULT_FRAME_WIDTH, video_id: str = "", straight_line: bool = False, dino_landmark: bool = False,
        gemini_direct: bool = False,
    ) -> None:
        result = modal_action.remote(action, split, source_root, sample_fps, max_frames, frame_width, video_id, straight_line, dino_landmark, gemini_direct)
        print(json.dumps(result, indent=2, sort_keys=True))
def _local_cli() -> None:  # pragma: no cover - CLI wrapper only
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("bootstrap", "ingest", "audit", "predict", "index-split", "compile-split", "build-frames", "build-frame-manifests", "index-frames", "build-geometry", "geometry-contract", "build-route-signature", "route-signature", "build-visual-index", "visual-index", "build-routes", "route-features"), default="bootstrap")
    parser.add_argument("--split", choices=tuple(SOURCES), default=None)
    parser.add_argument("--root", default=str(DATA_ROOT))
    parser.add_argument("--source-root", default=str(SHARED_VIDEO_ROOT))
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_FRAME_FPS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--video-id", action="append", dest="video_ids", default=None)
    parser.add_argument("--straight-line", action="store_true", help="enable the opt-in straight-line distance probe")
    parser.add_argument("--dino-landmark", action="store_true", help="enable frozen DINO landmark evidence")
    args = parser.parse_args()
    if args.straight_line:
        os.environ["KILOMETERVISION_ENABLE_STRAIGHT_LINE"] = "1"
    if args.dino_landmark:
        os.environ["KILOMETERVISION_ENABLE_DINO_LANDMARK"] = "1"
    print(json.dumps(run_action(args.action, args.split, root=args.root, source_root=args.source_root, sample_fps=args.sample_fps, max_frames=args.max_frames, frame_width=args.frame_width, video_ids=args.video_ids), indent=2, sort_keys=True))


if __name__ == "__main__":
    # ``modal run`` invokes the decorated local entrypoint itself. A direct Python
    # invocation remains useful for local schema/bootstrap checks, even if Modal
    # happens to be installed in the developer environment.
    _local_cli()
