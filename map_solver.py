"""Structural candidate-map comparison using question-window motion evidence."""
from __future__ import annotations

import io
import math
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence


def motion_polyline(steps: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """Integrate accepted relative yaw into an unscaled 2-D walking trace."""
    heading = 0.0
    x = y = 0.0
    points = [(x, y)]
    for step in steps:
        if not step.get("accepted") or step.get("yaw_delta_degrees") is None:
            continue
        delta = max(-30.0, min(30.0, float(step["yaw_delta_degrees"])))
        heading += math.radians(delta)
        x += math.sin(heading)
        y -= math.cos(heading)
        points.append((x, y))
    return points


def _skeleton(mask):
    import cv2
    import numpy as np
    source = (mask > 0).astype("uint8") * 255
    result = np.zeros_like(source)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(source):
        opened = cv2.morphologyEx(source, cv2.MORPH_OPEN, element)
        result = cv2.bitwise_or(result, cv2.subtract(source, opened))
        source = cv2.erode(source, element)
    return result


def route_mask(image_bytes: bytes):
    """Extract the saturated red route/pins from a Google-map candidate."""
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("candidate image could not be decoded")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 90, 110), (12, 255, 255)),
                        cv2.inRange(hsv, (168, 90, 110), (180, 255, 255)))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return _skeleton(red)


def polyline_mask(points: Sequence[tuple[float, float]], size: int = 256):
    import cv2
    import numpy as np
    canvas = np.zeros((size, size), dtype="uint8")
    if len(points) < 2:
        return canvas
    values = np.asarray(points, dtype="float32")
    values -= values.mean(axis=0)
    extent = max(float(values[:, 0].max()-values[:, 0].min()), float(values[:, 1].max()-values[:, 1].min()), 1e-6)
    values = values * ((size * .78) / extent) + size / 2
    cv2.polylines(canvas, [values.round().astype("int32")], False, 255, 3, cv2.LINE_AA)
    return _skeleton(canvas)


def _hu(mask) -> list[float]:
    import cv2
    moments = cv2.HuMoments(cv2.moments(mask, binaryImage=True)).flatten()
    return [-math.copysign(math.log10(abs(float(value)) + 1e-30), float(value)) for value in moments]


def shape_distance(left, right) -> float:
    a, b = _hu(left), _hu(right)
    return sum(min(20.0, abs(x-y)) for x, y in zip(a, b))


def skeleton_diameter_path(mask) -> list[tuple[float, float]]:
    """Recover the longest ordered path from a route skeleton.

    Small pin/marker branches are naturally ignored because the graph diameter
    follows the longest shortest path through the largest connected component.
    """
    import cv2
    import numpy as np

    binary = (mask > 0).astype("uint8")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return []
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    coordinates = [tuple(map(int, row)) for row in np.argwhere(labels == component)]
    nodes = set(coordinates)
    if len(nodes) < 2:
        return [(float(col), float(row)) for row, col in coordinates]

    def neighbours(node):
        row, col = node
        return [
            (row + dr, col + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (row + dr, col + dc) in nodes
        ]

    def farthest(start, keep_parent=False):
        queue = deque([start]); distance = {start: 0}; parent = {}
        while queue:
            node = queue.popleft()
            for other in neighbours(node):
                if other in distance:
                    continue
                distance[other] = distance[node] + 1
                parent[other] = node
                queue.append(other)
        end = max(distance, key=distance.get)
        return end, parent if keep_parent else None

    endpoints = [node for node in nodes if len(neighbours(node)) == 1]
    start = endpoints[0] if endpoints else next(iter(nodes))
    first, _ = farthest(start)
    second, parent = farthest(first, keep_parent=True)
    ordered = [second]
    while ordered[-1] != first:
        if ordered[-1] not in parent:
            return []
        ordered.append(parent[ordered[-1]])
    ordered.reverse()
    return [(float(col), float(row)) for row, col in ordered]


def skeleton_path_between_markers(mask, markers: Mapping[str, tuple[float, float]]) -> list[tuple[float, float]]:
    """Recover the shortest skeleton path between S and E marker-nearest pixels."""
    import numpy as np
    nodes = {tuple(map(int, row)) for row in np.argwhere(mask > 0)}
    if len(nodes) < 2 or not {"S", "E"}.issubset(markers):
        return []
    def nearest(marker):
        x, y = marker
        return min(nodes, key=lambda node: (node[1] - x) ** 2 + (node[0] - y) ** 2)
    start, target = nearest(markers["S"]), nearest(markers["E"])
    queue = deque([start]); parent = {start: None}
    while queue:
        node = queue.popleft()
        if node == target:
            break
        row, col = node
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                other = (row + dr, col + dc)
                if (dr or dc) and other in nodes and other not in parent:
                    parent[other] = node
                    queue.append(other)
    if target not in parent:
        return []
    path = [target]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return [(float(col), float(row)) for row, col in path]


def resample_polyline(points: Sequence[tuple[float, float]], count: int = 64):
    import numpy as np

    values = np.asarray(points, dtype="float64")
    if len(values) < 2 or count < 2:
        return np.empty((0, 2), dtype="float64")
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-9:
        return np.repeat(values[:1], count, axis=0)
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(targets, cumulative, values[:, axis]) for axis in range(2)]
    )


def _normalise(points):
    import numpy as np
    values = np.asarray(points, dtype="float64")
    values = values - values.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(values))
    return values / max(scale, 1e-9)


def procrustes_distance(left, right) -> float:
    """Translation/rotation/scale invariant ordered point distance."""
    import numpy as np
    a, b = _normalise(left), _normalise(right)
    u, _, vt = np.linalg.svd(b.T @ a)
    rotation = u @ vt
    return float(np.sqrt(np.mean(np.sum((a - b @ rotation) ** 2, axis=1))))


def discrete_frechet_distance(left, right) -> float:
    """Discrete Fréchet distance between normalized ordered polylines."""
    import numpy as np
    a, b = _normalise(left), _normalise(right)
    if not len(a) or not len(b):
        return float("inf")
    table = np.full((len(a), len(b)), float("inf"), dtype="float64")
    for i in range(len(a)):
        for j in range(len(b)):
            distance = float(np.linalg.norm(a[i] - b[j]))
            if i == 0 and j == 0:
                table[i, j] = distance
            elif i == 0:
                table[i, j] = max(table[i, j - 1], distance)
            elif j == 0:
                table[i, j] = max(table[i - 1, j], distance)
            else:
                table[i, j] = max(min(table[i - 1, j], table[i - 1, j - 1], table[i, j - 1]), distance)
    return float(table[-1, -1])


def symmetric_chamfer_distance(left, right) -> float:
    """Bidirectional nearest-point distance for normalized polylines."""
    import numpy as np
    a, b = _normalise(left), _normalise(right)
    if not len(a) or not len(b):
        return float("inf")
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float((distances.min(axis=1).mean() + distances.min(axis=0).mean()) / 2.0)


def curvature_sequence(points, stride: int = 4) -> list[float]:
    import numpy as np
    values = np.asarray(points, dtype="float64")
    if len(values) < 2 * stride + 1:
        return []
    before = values[stride:-stride] - values[:-2 * stride]
    after = values[2 * stride:] - values[stride:-stride]
    angles = np.arctan2(after[:, 1], after[:, 0]) - np.arctan2(before[:, 1], before[:, 0])
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    return angles.tolist()


def dtw_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return float("inf")
    previous = [float("inf")] * (len(right) + 1); previous[0] = 0.0
    for x in left:
        current = [float("inf")] * (len(right) + 1)
        for index, y in enumerate(right, 1):
            cost = abs(math.atan2(math.sin(x-y), math.cos(x-y))) / math.pi
            current[index] = cost + min(current[index-1], previous[index], previous[index-1])
        previous = current
    return previous[-1] / max(len(left), len(right))


def ordered_shape_distance(left_points, right_points, allow_reverse: bool = True,
                           reflection_penalty: float = 0.04,
                           frechet_weight: float = 0.0,
                           chamfer_weight: float = 0.0) -> dict[str, float | str]:
    """Compare ordered routes while enumerating traversal and axis conventions."""
    import numpy as np
    left = resample_polyline(left_points)
    right = resample_polyline(right_points)
    if not len(left) or not len(right):
        return {"score": float("inf"), "procrustes": float("inf"), "turn_dtw": float("inf"), "hypothesis": "missing"}
    hypotheses = []
    for reverse in ((False, True) if allow_reverse else (False,)):
        candidate = right[::-1].copy() if reverse else right.copy()
        for reflect in (False, True):
            transformed = candidate.copy()
            if reflect:
                transformed[:, 0] *= -1
            point_score = procrustes_distance(left, transformed)
            turn_scores = [
                dtw_distance(curvature_sequence(left, stride), curvature_sequence(transformed, stride))
                for stride in (2, 4, 8)
            ]
            turn_score = float(np.mean(turn_scores))
            frechet = discrete_frechet_distance(left, transformed)
            chamfer = symmetric_chamfer_distance(left, transformed)
            score = point_score + 0.35 * turn_score + frechet_weight * frechet + chamfer_weight * chamfer + (reflection_penalty if reflect else 0.0)
            hypotheses.append((score, point_score, turn_score, frechet, chamfer, f"{'reverse' if reverse else 'forward'}-{'reflect' if reflect else 'direct'}"))
    score, point_score, turn_score, frechet, chamfer, hypothesis = min(hypotheses)
    return {"score": float(score), "procrustes": float(point_score), "turn_dtw": float(turn_score), "frechet": float(frechet), "chamfer": float(chamfer), "hypothesis": hypothesis}


def route_marker_positions(image_bytes: bytes) -> dict[str, tuple[float, float]]:
    """Locate Google-style red S/E pins using their white glyph structure.

    The pin tip is consistently about 11 px below the glyph centre in the
    400px challenge renderings. The block E has a ten-pixel vertical stem;
    the curved S has no column longer than six pixels. Distance-transform
    peaks isolate the two large route pins from the narrow route stroke.
    """
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 100, 130), (12, 255, 255)),
                        cv2.inRange(hsv, (168, 100, 130), (180, 255, 255)))
    distance = cv2.distanceTransform(red, cv2.DIST_L2, 5)
    positions: dict[str, tuple[float, float]] = {}
    for _ in range(2):
        _, peak, _, (x, tip_y) = cv2.minMaxLoc(distance)
        if peak < 5.0:
            break
        center_y = tip_y - 11
        y0, y1 = max(0, center_y - 8), min(image.shape[0], center_y + 9)
        x0, x1 = max(0, x - 8), min(image.shape[1], x + 9)
        glyph = np.all(image[y0:y1, x0:x1] > 210, axis=2)
        label = "E" if glyph.sum(axis=0).max(initial=0) >= 8 else "S"
        positions[label] = (float(x), float(tip_y))
        cv2.circle(distance, (x, tip_y), 25, 0.0, -1)
    return positions if set(positions) == {"S", "E"} else {}


def route_path(image_bytes: bytes, pin_radius: int = 15, marker_guided: bool = False) -> list[tuple[float, float]]:
    """Extract the ordered route centreline after removing S/E pin bodies."""
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 90, 110), (12, 255, 255)),
                        cv2.inRange(hsv, (168, 90, 110), (180, 255, 255)))
    markers = route_marker_positions(image_bytes)
    for x, tip_y in markers.values():
        cv2.circle(red, (int(round(x)), int(round(tip_y - 11))), pin_radius, 0, -1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    skeleton = _skeleton(red)
    guided = skeleton_path_between_markers(skeleton, markers) if marker_guided else []
    return guided or orient_path_from_start(skeleton_diameter_path(skeleton), markers)


def orient_path_from_start(path: Sequence[tuple[float, float]], markers: Mapping[str, tuple[float, float]]):
    if len(path) < 2 or "S" not in markers:
        return list(path)
    start = markers["S"]
    left = math.hypot(path[0][0] - start[0], path[0][1] - start[1])
    right = math.hypot(path[-1][0] - start[0], path[-1][1] - start[1])
    return list(path) if left <= right else list(reversed(path))


def ordered_candidate_scores(question: Mapping[str, Any], motion: Mapping[str, Any], archive_path: Path) -> list[dict[str, float | str]]:
    return ordered_trace_candidate_scores(question, motion_polyline(motion.get("steps", [])), archive_path)


def ordered_trace_candidate_scores(question: Mapping[str, Any], trace, archive_path: Path,
                                   reflection_penalty: float = 0.04,
                                   frechet_weight: float = 0.0,
                                   marker_guided: bool = False,
                                   chamfer_weight: float = 0.0) -> list[dict[str, Any]]:
    """Score five map candidates against any time-ordered 2-D trajectory."""
    references = question.get("image_paths")
    if not isinstance(references, list) or len(references) != 5:
        return []
    scores = []
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        for reference in references:
            matches = [name for name in members if Path(name).name == Path(str(reference)).name]
            image_bytes = archive.read(matches[0]) if len(matches) == 1 else b""
            markers = route_marker_positions(image_bytes) if image_bytes else {}
            path = route_path(image_bytes, marker_guided=marker_guided) if image_bytes else []
            if markers:
                path = orient_path_from_start(path, markers)
            result = ordered_shape_distance(
                trace, path, allow_reverse=not bool(markers), reflection_penalty=reflection_penalty,
                frechet_weight=frechet_weight, chamfer_weight=chamfer_weight
            )
            result["candidate_path_points"] = len(path)
            result["marker_positions"] = markers
            scores.append(result)
    return scores


def endpoint_marker_features(image_bytes: bytes, path: Sequence[tuple[float, float]], radius: int = 14) -> list[dict[str, Any]]:
    """Summarize saturated colors around both recovered route endpoints."""
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or len(path) < 2:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]
    features = []
    for x, y in (path[0], path[-1]):
        col, row = int(round(x)), int(round(y))
        y0, y1 = max(0, row-radius), min(height, row+radius+1)
        x0, x1 = max(0, col-radius), min(width, col+radius+1)
        patch = hsv[y0:y1, x0:x1]
        saturated = patch[(patch[..., 1] >= 80) & (patch[..., 2] >= 70)]
        hue_hist = np.histogram(saturated[:, 0] if len(saturated) else [], bins=18, range=(0, 180))[0]
        features.append({
            "x": col, "y": row,
            "saturated_pixels": int(len(saturated)),
            "red_pixels": int(np.sum((saturated[:, 0] <= 12) | (saturated[:, 0] >= 168))) if len(saturated) else 0,
            "green_pixels": int(np.sum((saturated[:, 0] >= 35) & (saturated[:, 0] <= 90))) if len(saturated) else 0,
            "blue_pixels": int(np.sum((saturated[:, 0] >= 95) & (saturated[:, 0] <= 135))) if len(saturated) else 0,
            "hue_histogram": hue_hist.tolist(),
        })
    return features


def score_candidates(question: Mapping[str, Any], motion: Mapping[str, Any], archive_path: Path) -> list[float]:
    references = question.get("image_paths")
    if not isinstance(references, list) or len(references) != 5:
        return []
    trace = polyline_mask(motion_polyline(motion.get("steps", [])))
    scores: list[float] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        for reference in references:
            matches = [name for name in members if Path(name).name == Path(str(reference)).name]
            scores.append(shape_distance(trace, route_mask(archive.read(matches[0]))) if len(matches) == 1 else float("inf"))
    return scores


__all__ = ["motion_polyline", "route_mask", "route_path", "polyline_mask", "shape_distance", "score_candidates", "skeleton_diameter_path", "resample_polyline", "procrustes_distance", "curvature_sequence", "dtw_distance", "ordered_shape_distance", "ordered_candidate_scores", "ordered_trace_candidate_scores", "endpoint_marker_features", "route_marker_positions", "orient_path_from_start"]
