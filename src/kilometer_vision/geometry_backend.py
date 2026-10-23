"""Backend-neutral contracts and geometry helpers for PATHFINDER.

Heavy geometry models are optional. This module contains only standard-library
code at import time so metadata/schema jobs can run without torch, CUDA, or
model checkpoints. Adapters are discovered by package name and imported by the
caller only when explicitly selected.
"""
from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
Vec3 = tuple[float, float, float]
Mat3 = tuple[float, ...]
IDENTITY_ROTATION: Mat3 = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _vec(values: Sequence[float], n: int, name: str) -> tuple[float, ...]:
    if len(values) != n or not all(math.isfinite(float(v)) for v in values):
        raise ValueError(f"{name} must contain {n} finite values")
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class CameraPose:
    timestamp_seconds: float
    rotation: Mat3 = IDENTITY_ROTATION
    translation: Vec3 = (0.0, 0.0, 0.0)
    confidence: float = 0.0
    available: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("pose timestamp must be finite and non-negative")
        object.__setattr__(self, "rotation", _vec(self.rotation, 9, "rotation"))
        object.__setattr__(self, "translation", _vec(self.translation, 3, "translation"))
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("pose confidence must be in [0, 1]")


@dataclass(frozen=True)
class DepthSummary:
    minimum: float | None = None
    median: float | None = None
    maximum: float | None = None
    confidence: float = 0.0
    metric: bool = False
    available: bool = False

    def __post_init__(self) -> None:
        values = (self.minimum, self.median, self.maximum)
        if any(v is not None and (not math.isfinite(float(v)) or float(v) < 0) for v in values):
            raise ValueError("depth values must be finite and non-negative")
        present = [float(v) for v in values if v is not None]
        if present and present != sorted(present):
            raise ValueError("depth summary must be ordered minimum <= median <= maximum")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("depth confidence must be in [0, 1]")


@dataclass(frozen=True)
class RelativeTransform:
    from_index: int
    to_index: int
    rotation: Mat3 = IDENTITY_ROTATION
    translation: Vec3 = (0.0, 0.0, 0.0)
    scale: float = 1.0
    confidence: float = 0.0
    available: bool = False

    def __post_init__(self) -> None:
        if self.from_index < 0 or self.to_index < 0 or self.to_index <= self.from_index:
            raise ValueError("relative transform indexes must be increasing and non-negative")
        object.__setattr__(self, "rotation", _vec(self.rotation, 9, "rotation"))
        object.__setattr__(self, "translation", _vec(self.translation, 3, "translation"))
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("relative transform scale must be positive")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("transform confidence must be in [0, 1]")


@dataclass(frozen=True)
class RouteSignature:
    headings_degrees: tuple[float, ...] = ()
    turn_angles_degrees: tuple[float, ...] = ()
    segment_lengths: tuple[float, ...] = ()
    topology: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: float = 1.0
    available: bool = False

    def __post_init__(self) -> None:
        for value in (*self.headings_degrees, *self.turn_angles_degrees, *self.segment_lengths):
            if not math.isfinite(float(value)):
                raise ValueError("route signature values must be finite")
        if any(float(v) < 0 for v in self.segment_lengths):
            raise ValueError("segment lengths must be non-negative")
        if not 0 <= float(self.uncertainty) <= 1:
            raise ValueError("route uncertainty must be in [0, 1]")


@dataclass(frozen=True)
class GeometryFrame:
    timestamp_seconds: float
    pose: CameraPose
    depth: DepthSummary = field(default_factory=DepthSummary)
    confidence: float = 0.0
    source: str = "unknown"
    status: str = "unavailable"


@dataclass(frozen=True)
class SequenceGeometry:
    video_id: str
    backend: str
    status: str
    frames: tuple[GeometryFrame, ...] = ()
    relative_transforms: tuple[RelativeTransform, ...] = ()
    route_signature: RouteSignature = field(default_factory=RouteSignature)
    errors: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def fallback_geometry(video_id: str, timestamps: Iterable[float], reason: str = "geometry_backend_unavailable") -> SequenceGeometry:
    """Build a deterministic, explicitly unavailable result for safe fallback jobs."""
    times = [float(t) for t in timestamps]
    if any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("timestamps must be finite and non-negative")
    if times != sorted(times) or len(set(times)) != len(times):
        raise ValueError("timestamps must be sorted and unique")
    frames = tuple(GeometryFrame(t, CameraPose(t), source="fallback", status="geometry_unavailable") for t in times)
    transforms = tuple(RelativeTransform(i, i + 1) for i in range(max(0, len(times) - 1)))
    return SequenceGeometry(video_id, "fallback", "geometry_unavailable", frames, transforms, RouteSignature(), (reason,))


def discover_backends() -> dict[str, dict[str, Any]]:
    """Discover optional adapters without importing their heavy modules."""
    packages = {
        "da3": ("depth_anything_3", "Depth Anything 3"),
        "vggt": ("vggt", "VGGT/VGGT-Omega adapter"),
        "mapanything": ("mapanything", "MapAnything"),
    }
    return {name: {"package": package, "label": label, "available": importlib.util.find_spec(package) is not None}
            for name, (package, label) in packages.items()}


def _matvec(rotation: Mat3, vector: Vec3) -> Vec3:
    return tuple(sum(rotation[row * 3 + col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def _add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def _scale(value: float, vector: Vec3) -> Vec3:
    return tuple(value * x for x in vector)  # type: ignore[return-value]


def _centroid(points: Sequence[Vec3]) -> Vec3:
    n = len(points)
    return _scale(1.0 / n, tuple(sum(p[i] for p in points) for i in range(3)))  # type: ignore[arg-type]


def _horn_rotation(source: Sequence[Vec3], target: Sequence[Vec3]) -> Mat3:
    covariance = [[sum(a[i] * b[j] for a, b in zip(source, target)) for j in range(3)] for i in range(3)]
    sxx, sxy, sxz = covariance[0]
    syx, syy, syz = covariance[1]
    szx, szy, szz = covariance[2]
    trace = sxx + syy + szz
    matrix = [[trace, syz - szy, szx - sxz, sxy - syx],
              [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
              [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
              [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz]]
    # Jacobi eigensolver avoids the slow/unstable power iteration when the
    # dominant eigenvalue is negative or nearly repeated.
    eigen = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    for _ in range(80):
        p, r = max(((i, j) for i in range(4) for j in range(i + 1, 4)), key=lambda ij: abs(eigen[ij[0]][ij[1]]))
        if abs(eigen[p][r]) < 1e-12:
            break
        angle = 0.5 * math.atan2(2.0 * eigen[p][r], eigen[r][r] - eigen[p][p])
        c, s = math.cos(angle), math.sin(angle)
        for k in range(4):
            epk, erk = eigen[p][k], eigen[r][k]
            eigen[p][k], eigen[r][k] = c * epk - s * erk, s * epk + c * erk
        for k in range(4):
            ekp, ekr = eigen[k][p], eigen[k][r]
            eigen[k][p], eigen[k][r] = c * ekp - s * ekr, s * ekp + c * ekr
            vkp, vkr = vectors[k][p], vectors[k][r]
            vectors[k][p], vectors[k][r] = c * vkp - s * vkr, s * vkp + c * vkr
    column = max(range(4), key=lambda i: eigen[i][i])
    q = [vectors[i][column] for i in range(4)]
    norm = math.sqrt(sum(value * value for value in q)) or 1.0
    w, x, y, z = [value / norm for value in q]
    return (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))


@dataclass(frozen=True)
class Sim3Alignment:
    rotation: Mat3
    translation: Vec3
    scale: float
    inlier_count: int
    residuals: tuple[float, ...]
    confidence: float


def align_trajectories(source: Sequence[Sequence[float]], target: Sequence[Sequence[float]], *, robust: bool = True) -> Sim3Alignment:
    """Estimate target ~= scale * R * source + translation."""
    src = [_vec(p, 3, "source point") for p in source]
    dst = [_vec(p, 3, "target point") for p in target]
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("at least three corresponding points are required")

    def fit(indices: Sequence[int]) -> tuple[Mat3, Vec3, float, list[float]]:
        s, d = [src[i] for i in indices], [dst[i] for i in indices]
        if len(s) < 3:
            raise ValueError("three points are required for alignment")
        # Build an orthonormal frame from the first non-collinear triple. This
        # is deterministic, pure Python, and avoids a fragile hand-written SVD.
        chosen = None
        for a in range(len(s) - 2):
            for b in range(a + 1, len(s) - 1):
                for c in range(b + 1, len(s)):
                    u = _sub(s[b], s[a]); un = math.sqrt(_dot(u, u))
                    v0 = _sub(s[c], s[a])
                    if un > 1e-9:
                        vproj = _sub(v0, _scale(_dot(v0, u) / _dot(u, u), u))
                        vn = math.sqrt(_dot(vproj, vproj))
                        if vn > 1e-9:
                            chosen = (a, b, c, u, vproj)
                            break
                if chosen: break
            if chosen: break
        if not chosen:
            raise ValueError("source trajectory is degenerate")
        a, b, c, su, sv = chosen
        du = _sub(d[b], d[a]); dv0 = _sub(d[c], d[a])
        def frame(first: Vec3, second: Vec3) -> tuple[Vec3, Vec3, Vec3]:
            fn = math.sqrt(_dot(first, first)); e1 = _scale(1.0 / fn, first)
            ortho = _sub(second, _scale(_dot(second, e1), e1)); on = math.sqrt(_dot(ortho, ortho)); e2 = _scale(1.0 / on, ortho)
            e3 = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
            return e1, e2, e3
        sb, db = frame(su, sv), frame(du, dv0)
        # Basis vectors are stored as columns; R = B_target * B_source^T.
        rotation = tuple(sum(db[k][row] * sb[k][col] for k in range(3)) for row in range(3) for col in range(3))
        ratios = []
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                ds = math.sqrt(_dot(_sub(s[i], s[j]), _sub(s[i], s[j])))
                dd = math.sqrt(_dot(_sub(d[i], d[j]), _sub(d[i], d[j])))
                if ds > 1e-9: ratios.append(dd / ds)
        ordered_ratios = sorted(ratios)
        scale = ordered_ratios[len(ordered_ratios) // 2] if ordered_ratios else 1.0
        cs, cd = _centroid(s), _centroid(d)
        translation = _sub(cd, _scale(scale, _matvec(rotation, cs)))
        residuals = []
        for k in range(len(src)):
            transformed = _add(_scale(scale, _matvec(rotation, src[k])), translation)
            residuals.append(math.sqrt(sum((transformed[j] - dst[k][j]) ** 2 for j in range(3))))
        return rotation, translation, scale, residuals

    indices = list(range(len(src)))
    rotation, translation, scale, residuals = fit(indices)
    if robust and len(indices) >= 5:
        ordered = sorted(residuals)
        median = ordered[len(ordered) // 2]
        deviations = sorted(abs(value - median) for value in residuals)
        mad = deviations[len(deviations) // 2] or 1e-9
        threshold = median + 3.0 * 1.4826 * mad
        inliers = [i for i, value in enumerate(residuals) if value <= threshold]
        if len(inliers) >= 3 and len(inliers) < len(indices):
            rotation, translation, scale, residuals = fit(inliers)
            indices = inliers
    mean_residual = sum(residuals[i] for i in indices) / len(indices)
    confidence = max(0.0, min(1.0, (len(indices) / len(src)) * math.exp(-mean_residual)))
    return Sim3Alignment(rotation, translation, scale, len(indices), tuple(residuals), confidence)


def apply_alignment(points: Sequence[Sequence[float]], alignment: Sim3Alignment) -> list[Vec3]:
    return [_add(_scale(alignment.scale, _matvec(alignment.rotation, _vec(point, 3, "point"))), alignment.translation) for point in points]


def extract_route_signature(poses: Sequence[CameraPose], *, turn_threshold_degrees: float = 20.0) -> RouteSignature:
    """Extract a topology-neutral turn signature; unavailable poses stay uncertain."""
    available = [pose for pose in poses if pose.available]
    if len(available) < 2:
        return RouteSignature(uncertainty=1.0, available=False)
    headings = []
    for pose in available:
        forward = _matvec(pose.rotation, (0.0, 0.0, 1.0))
        headings.append(math.degrees(math.atan2(forward[0], forward[2])))
    turns, lengths = [], []
    for index, (left, right) in enumerate(zip(available, available[1:])):
        delta = (headings[index + 1] - headings[index] + 180) % 360 - 180
        lengths.append(math.sqrt(sum((right.translation[i] - left.translation[i]) ** 2 for i in range(3))))
        if abs(delta) >= turn_threshold_degrees:
            turns.append(delta)
    return RouteSignature(tuple(headings), tuple(turns), tuple(lengths), {"turn_count": len(turns), "segment_count": len(lengths)}, uncertainty=max(0.0, 1.0 - sum(p.confidence for p in available) / len(available)), available=True)


__all__ = ["CameraPose", "DepthSummary", "RelativeTransform", "RouteSignature", "GeometryFrame", "SequenceGeometry", "Sim3Alignment", "fallback_geometry", "discover_backends", "align_trajectories", "apply_alignment", "extract_route_signature"]
