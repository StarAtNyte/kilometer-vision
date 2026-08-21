"""Schema-safe prediction generation and auditing for KilometerVision.

The default selector is deliberately a deterministic placeholder until PATHFINDER's
route specialists are wired in. It uses only question/options and never reads
organizer ``answer`` or ``answer_id`` fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from question_planner import classify_question, classify_question_facets, compile_options
from specialists import choose, score_compass, score_distance
from landmark_matcher import match_reference
from motion_solver import compass_option_index


def _experiment_enabled(name: str) -> bool:
    """Apply promoted defaults while retaining explicit per-lane overrides."""
    default = "1" if name.lower() == "motion" else "0"
    return os.environ.get(f"KILOMETERVISION_ENABLE_{name.upper()}", default).strip().lower() in {
        "1", "true", "yes", "on",
    }


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


def _fallback_index(video_id: str, question_id: str, option_count: int) -> int:
    """Stable balanced fallback, useful only for schema/smoke validation."""
    digest = hashlib.sha256(f"{video_id}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % option_count


def _geometry_summary(video_id: str, root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"available": False, "status": "no_volume_root"}
    path = root / "geometry" / f"{video_id}.json"
    if not path.exists():
        path = root / "geometry" / f"{video_id}.da3.json"
    if not path.exists():
        return {"available": False, "status": "missing_geometry"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload.get("frames", [])
        pose_count = sum(1 for frame in frames if frame.get("extrinsic") is not None)
        timestamps = [float(frame["timestamp_seconds"]) for frame in frames if frame.get("timestamp_seconds") is not None]
        positions = []
        yaws = []
        confidence = []
        for frame in frames:
            extrinsic = frame.get("extrinsic")
            if isinstance(extrinsic, list) and len(extrinsic) >= 3 and all(isinstance(row, list) and len(row) >= 4 for row in extrinsic[:3]):
                rotation = [[float(extrinsic[row][col]) for col in range(3)] for row in range(3)]
                positions.append((float(extrinsic[0][3]), float(extrinsic[1][3]), float(extrinsic[2][3])))
                # DA3 exports camera poses with a forward axis in the third
                # column. Keep this as relative evidence only; no north claim.
                yaws.append(math.degrees(math.atan2(rotation[0][2], rotation[2][2])))
            if frame.get("confidence") is not None:
                confidence.append(max(0.0, min(1.0, float(frame["confidence"]))))
        path_length = sum(sum((b[i] - a[i]) ** 2 for i in range(3)) ** 0.5 for a, b in zip(positions, positions[1:]))
        displacement = sum((positions[-1][i] - positions[0][i]) ** 2 for i in range(3)) ** 0.5 if len(positions) >= 2 else 0.0
        yaw_delta = None
        if len(yaws) >= 2:
            yaw_delta = (yaws[-1] - yaws[0] + 180.0) % 360.0 - 180.0
        return {"available": bool(frames), "backend": payload.get("backend", "unknown"), "frames": len(frames), "pose_frames": pose_count, "scale": payload.get("contract", {}).get("scale", "relative"), "first_timestamp": min(timestamps) if timestamps else None, "last_timestamp": max(timestamps) if timestamps else None, "path_length_relative": path_length, "displacement_relative": displacement, "closure_ratio": displacement / path_length if path_length > 1e-9 else 1.0, "mean_confidence": sum(confidence) / len(confidence) if confidence else 0.0, "relative_yaw_delta_degrees": yaw_delta}
    except (OSError, ValueError, TypeError, IndexError, ZeroDivisionError):
        return {"available": False, "status": "invalid_geometry"}


def _geometry_fallback_index(video_id: str, question_id: str, option_count: int, geometry: Mapping[str, Any]) -> int:
    """Geometry-aware deterministic fallback for non-metric predicates.

    It never invents absolute heading or map identity: route-state values only
    perturb the stable fallback when the semantic specialist has no predicate.
    """
    signature = "|".join(str(geometry.get(key, "unknown")) for key in ("backend", "frames", "pose_frames", "closure_ratio", "path_length_relative", "mean_confidence"))
    digest = hashlib.sha256(f"{video_id}:{question_id}:{signature}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % option_count


def _vlm_record(video_id: str, question_id: str, root: Path | None, split: str | None = None) -> Mapping[str, Any] | None:
    """Read one question's VLM evidence-card record, if the lane has run for it.

    Records are written by ``modal_vlm.py`` to ``vlm/<video_id>.json`` as
    ``{question_id: {selected_index, raw_text, status, checkpoint}}``. Missing files,
    missing records, and malformed JSON all return ``None`` rather than raising, so
    the VLM lane is purely additive to the existing fallback chain.
    """
    if root is None:
        return None
    paths = ([root / "vlm" / split / f"{video_id}.json"] if split else []) + [root / "vlm" / f"{video_id}.json"]
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        record = payload.get(question_id) if isinstance(payload, Mapping) else None
        if isinstance(record, Mapping):
            return record
    return None


def _vlm_choice(record: Mapping[str, Any] | None) -> int | None:
    """Return the VLM's selected index only when it parsed a clean, unambiguous answer."""
    if record is None or record.get("status") != "ok":
        return None
    index = record.get("selected_index")
    return index if isinstance(index, int) else None


def _dino_landmark_record(video_id: str, question_id: str, root: Path | None, split: str | None) -> Mapping[str, Any] | None:
    """Read a frozen DINO patch-retrieval record, if a run has materialized one."""
    if root is None or not split:
        return None
    path = root / "experiments" / "landmark-dinov2" / split / f"{video_id}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload.get(str(question_id)) if isinstance(payload, Mapping) else None
        return record if isinstance(record, Mapping) and record.get("status") == "ok" else None
    except (OSError, ValueError, TypeError):
        return None


def _dino_landmark_match(record: Mapping[str, Any] | None) -> tuple[bool, float | None]:
    """Apply an external-only, conservative present/absent gate."""
    if record is None:
        return False, None
    matches = record.get("top_matches")
    if not isinstance(matches, list) or not matches or not isinstance(matches[0], Mapping):
        return False, None
    top = matches[0]
    patch = float(top.get("patch_local_similarity", 0.0) or 0.0)
    mutual = float(top.get("mutual_patch_ratio", 0.0) or 0.0)
    present = patch >= 0.90 and mutual >= 0.40
    timestamp = top.get("timestamp_seconds")
    return present, float(timestamp) if timestamp is not None else None


def _motion_record(video_id: str, question_id: str, root: Path | None, split: str | None,
                   facet: str | None = None) -> Mapping[str, Any] | None:
    if root is None or not split:
        return None
    paths = ([root / "motion" / split / facet / f"{video_id}.json"] if facet else []) + [
        root / "motion" / split / f"{video_id}.json"
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        record = payload.get(question_id) if isinstance(payload, Mapping) else None
        if isinstance(record, Mapping):
            return record
    return None


def _dpvo_map_choice(video_id: str, question_id: str, question: Mapping[str, Any],
                     root: Path | None, split: str | None) -> int | None:
    """Match a DPVO ground-plane trajectory to the five candidate route maps."""
    if root is None or not split:
        return None
    variant = "fps1-focal0.9-w256-b256-p16"
    path = root / "experiments" / "dpvo-motion" / split / variant / f"{video_id}.json"
    archive = root / "manifests" / f"{split}_images.zip"
    if not path.exists() or not archive.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload.get(question_id) if isinstance(payload, Mapping) else None
        trace = record.get("trajectory_2d") if isinstance(record, Mapping) else None
        if not isinstance(trace, list) or len(trace) < 2:
            return None
        from map_solver import ordered_trace_candidate_scores
        # SVD/PCA ground-plane axes have arbitrary signs. A reflected trace is
        # therefore the same VO geometry, not weaker evidence.
        scores = ordered_trace_candidate_scores(
            question, trace, archive, reflection_penalty=0.0
        )
        if len(scores) != 5 or not all(math.isfinite(float(item["score"])) for item in scores):
            return None
        return min(range(5), key=lambda index: float(scores[index]["score"]))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _direction_from_angle(angle: float) -> str:
    directions = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
    return directions[int(((angle % 360.0) + 22.5) // 45.0) % 8]


def _heading_posterior(question: Mapping[str, Any], geometry: Mapping[str, Any]) -> dict[str, float] | None:
    """Convert relative yaw to a weak compass posterior when an anchor exists."""
    delta = geometry.get("relative_yaw_delta_degrees")
    if delta is None or geometry.get("pose_frames", 0) < 2:
        return None
    text = str(question.get("question", "")).lower()
    anchors = {"north": 0.0, "northeast": 45.0, "east": 90.0, "southeast": 135.0,
               "south": 180.0, "southwest": 225.0, "west": 270.0, "northwest": 315.0}
    match = re.search(r"(?:start(?:ing)?|initial(?:ly)?|begin(?:ning)?)\W+(?:facing|heading|toward)\W+(northwest|northeast|southwest|southeast|north|south|east|west)\b", text)
    if not match:
        match = re.search(r"\b(facing|heading)\W+(northwest|northeast|southwest|southeast|north|south|east|west)\b", text)
        initial = match.group(2) if match else None
    else:
        initial = match.group(1)
    if initial is None:
        return None
    target = _direction_from_angle(anchors[initial] + float(delta))
    posterior = {direction: 0.0 for direction in anchors}
    posterior[target] = max(0.0, min(1.0, float(geometry.get("mean_confidence", 0.0))))
    # Relative pose is noisy; distribute the remaining mass to neighbors.
    remaining = 1.0 - posterior[target]
    ordered = list(anchors)
    target_index = ordered.index(target)
    for offset in (-1, 1):
        posterior[ordered[(target_index + offset) % 8]] += remaining / 2.0
    return posterior


def _landmark_presence_index(options: list[Any], present: bool) -> int | None:
    positive = ("yes", "appears", "appeared", "seen", "shown", "visited", "present", "does")
    negative = ("no", "not", "never", "absent", "didn't", "did not", "doesn't", "does not")
    tokens = positive if present else negative
    scores = []
    for option in options:
        text = str(option).lower()
        scores.append(sum(1 for token in tokens if token in text))
    if max(scores, default=0) == 0:
        return None
    return max(range(len(options)), key=lambda index: (scores[index], -index))


def _geometry_distance_estimate(estimate_m: float, geometry: Mapping[str, Any]) -> tuple[float, float]:
    """Use relative motion only as a bounded prior around the walking-time estimate."""
    if not geometry.get("available") or geometry.get("path_length_relative", 0.0) <= 0:
        return estimate_m, max(10.0, estimate_m * 0.5)
    closure = max(0.0, min(1.0, float(geometry.get("closure_ratio", 1.0))))
    confidence = max(0.0, min(1.0, float(geometry.get("mean_confidence", 0.0))))
    multiplier = 0.9 + 0.2 * max(0.0, min(1.0, confidence * (1.0 - 0.5 * closure)))
    return estimate_m * multiplier, max(8.0, estimate_m * (0.25 + 0.2 * (1.0 - confidence)))


def _is_straight_line_distance(question: Mapping[str, Any]) -> bool:
    """Detect the metric requested by the question, without using its answer."""
    text = str(question.get("question", "")).lower()
    return any(cue in text for cue in ("straight-line", "straight line", "line-of-sight", "line of sight"))


def _straight_line_prior(estimate_m: float, geometry: Mapping[str, Any]) -> tuple[float, float]:
    """Convert a route-length prior into a conservative endpoint-distance prior.

    Elapsed walking time is a route-length proxy.  It is systematically too large
    for the explicit straight-line/line-of-sight questions.  Relative VO can provide
    a weak directness cue, but cannot provide metric scale, so the correction is
    deliberately bounded and uncertainty is widened.  This is an experiment lane,
    not a claim that monocular geometry is metric.
    """
    directness = 0.70
    if geometry.get("available") and geometry.get("path_length_relative", 0.0) > 0:
        ratio = float(geometry.get("closure_ratio", 1.0))
        if math.isfinite(ratio):
            # Keep noisy full-video closure from collapsing a local question.
            directness = max(0.55, min(0.85, 0.55 + 0.20 * max(0.0, min(1.0, ratio))))
    corrected = estimate_m * directness
    return corrected, max(15.0, estimate_m * 0.40)


def select_option(video_id: str, question_id: str, question: Mapping[str, Any], *, root: Path | None = None, split: str | None = None) -> int:
    """Return a safe fallback option; specialist hooks can replace this function."""
    options = question.get("options")
    if not isinstance(options, list) or len(options) != 5:
        raise ValueError(f"{video_id}/{question_id}: expected five options")
    # Compile for a validated planner trace. The selected index remains independent
    # of organizer labels and is intentionally not validation-tuned.
    compiled = compile_options(question)
    category = compiled["category"]
    if category == "compass" and _experiment_enabled("motion"):
        motion = _motion_record(video_id, question_id, root, split, "compass")
        yaw = motion.get("summary", {}).get("net_yaw_degrees") if motion and motion.get("status") == "ok" else None
        if yaw is not None:
            motion_index = compass_option_index(question, float(yaw))
            if motion_index is not None:
                return motion_index
    if category == "map" and _experiment_enabled("dpvo"):
        dpvo_index = _dpvo_map_choice(video_id, question_id, question, root, split)
        if dpvo_index is not None:
            return dpvo_index
    # The VLM lane shows a strong, category-isolated bias toward the last option on
    # distance questions (it cannot judge metric distance from frames and defaults to
    # a habitual answer instead of admitting uncertainty), while compass/map/landmark
    # answers look genuinely varied. The walking-time/geometry distance specialist
    # below is a real numeric estimate, not a guess, so distance keeps using it.
    if category != "distance":
        vlm_index = _vlm_choice(_vlm_record(video_id, question_id, root, split))
        if vlm_index is not None and 0 <= vlm_index < len(options):
            return vlm_index
    geometry = _geometry_summary(video_id, root)
    if category == "distance":
        start = float(question.get("start_seconds", 0.0) or 0.0)
        end = float(question.get("end_seconds", start) or start)
        estimate_m = max(0.0, end - start) * 1.25
        dino_record = None
        dino_present = False
        dino_timestamp = None
        if (_experiment_enabled("dino_landmark") and "landmark_presence" in classify_question_facets(question)
                and question.get("image_paths")):
            dino_record = _dino_landmark_record(video_id, question_id, root, split)
            dino_present, dino_timestamp = _dino_landmark_match(dino_record)
            if dino_record is not None and not dino_present:
                selected = _landmark_presence_index(options, False)
                if selected is not None:
                    return selected
            if dino_present and dino_timestamp is not None:
                estimate_m = abs(float(question.get("end_seconds", 0.0) or 0.0) - dino_timestamp) * 1.25
        if root is not None and split and question.get("image_paths") and not dino_present:
            manifest = root / "frames" / f"{video_id}.json"
            archive = root / "manifests" / f"{split}_images.zip"
            if manifest.exists() and archive.exists():
                match = match_reference(manifest, archive, str(question["image_paths"][0]))
                if match.get("timestamp_seconds") is not None:
                    duration = float(question.get("end_seconds", 0.0) or 0.0)
                    estimate_m = abs(duration - float(match["timestamp_seconds"])) * 1.25
        adjusted_estimate, uncertainty = _geometry_distance_estimate(estimate_m, geometry)
        if _experiment_enabled("straight_line") and _is_straight_line_distance(question):
            adjusted_estimate, uncertainty = _straight_line_prior(adjusted_estimate, geometry)
        return choose(score_distance(options, adjusted_estimate, uncertainty))
    if category == "compass":
        posterior = _heading_posterior(question, geometry)
        if posterior is not None:
            return choose(score_compass(options, posterior))
        text = str(question.get("question", "")).lower()
        posterior = {direction: 0.0 for direction in ("north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest")}
        for direction in posterior:
            if direction in text:
                posterior[direction] = 1.0
                break
        if any(posterior.values()):
            return choose(score_compass(options, posterior))
        # Do not perturb opaque/image options without a route predicate.
        text = " ".join([str(question.get("question", "")), *(str(option) for option in options)]).lower()
        return _geometry_fallback_index(video_id, question_id, len(options), geometry) if any(token in text for token in ("left", "right", "turn", "straight", "visited", "before", "again")) else _fallback_index(video_id, question_id, len(options))
    if category == "landmark" and root is not None and split and question.get("image_paths"):
        manifest = root / "frames" / f"{video_id}.json"
        archive = root / "manifests" / f"{split}_images.zip"
        if manifest.exists() and archive.exists():
            match = match_reference(manifest, archive, str(question["image_paths"][0]))
            if match.get("timestamp_seconds") is not None:
                selected = _landmark_presence_index(options, float(match.get("similarity", 0.0)) >= 0.55)
                if selected is not None:
                    return selected
    text = " ".join([str(question.get("question", "")), *(str(option) for option in options)]).lower()
    return _geometry_fallback_index(video_id, question_id, len(options), geometry) if any(token in text for token in ("left", "right", "turn", "straight", "visited", "before", "again")) else _fallback_index(video_id, question_id, len(options))


def generate_predictions(
    payload: Mapping[str, Any],
    selector: Callable[[str, str, Mapping[str, Any]], int] = select_option,
    *, root: Path | None = None, split: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    evidence: dict[str, Any] = {"schema_version": 1, "videos": {}}
    for video_id, questions in payload.items():
        if not isinstance(questions, Mapping):
            raise ValueError(f"{video_id}: questions must be an object")
        rows: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        geometry = _geometry_summary(str(video_id), root)
        for key, question in questions.items():
            if not isinstance(question, Mapping):
                raise ValueError(f"{video_id}/{key}: question must be an object")
            try:
                index = selector(str(video_id), str(key), question, root=root, split=split)
            except TypeError:
                index = selector(str(video_id), str(key), question)
            options = question.get("options")
            if not isinstance(options, list) or not 0 <= index < len(options):
                raise ValueError(f"{video_id}/{key}: selector returned invalid option")
            rows.append({"id": question.get("id", int(key) if str(key).isdigit() else key), "answer": str(options[index]), "answer_id": index})
            category = classify_question(question)
            certificate = "global_audit" if category in {"distance", "map"} else "local_visual_certificate"
            question_timestamp = question.get("start_seconds", geometry.get("first_timestamp"))
            vlm_record = _vlm_record(str(video_id), str(key), root, split)
            vlm_status = vlm_record.get("status") if vlm_record is not None else "unavailable"
            vlm_checkpoint = vlm_record.get("checkpoint") if vlm_record is not None else None
            motion_record = _motion_record(str(video_id), str(key), root, split, category)
            traces.append({"id": question.get("id", int(key) if str(key).isdigit() else key), "category": category, "selected_index": index, "retrieval_mode": certificate, "evidence": [{"status": "deterministic_specialist_or_fallback", "source": "question_compiler", "timestamp_seconds": question_timestamp}, {"status": "geometry_retrieval_context", "source": geometry.get("backend", geometry.get("status", "unknown")), "frames": geometry.get("frames", 0), "pose_frames": geometry.get("pose_frames", 0), "scale": geometry.get("scale", "unknown"), "timestamp_range": [geometry.get("first_timestamp"), geometry.get("last_timestamp")] }, {"status": vlm_status, "source": "vlm_evidence_card", "checkpoint": vlm_checkpoint}, {"status": motion_record.get("status") if motion_record else "unavailable", "source": "question_window_motion", "net_yaw_degrees": motion_record.get("summary", {}).get("net_yaw_degrees") if motion_record else None}]})
        predictions[str(video_id)] = rows
        evidence["videos"][str(video_id)] = traces
    return predictions, evidence


def audit_predictions(payload: Mapping[str, Any], predictions: Mapping[str, Any]) -> dict[str, Any]:
    expected: set[tuple[str, str]] = set()
    for video_id, questions in payload.items():
        if not isinstance(questions, Mapping):
            raise ValueError(f"{video_id}: malformed question object")
        for key in questions:
            expected.add((str(video_id), str(key)))
    actual: set[tuple[str, str]] = set()
    errors: list[str] = []
    for video_id, rows in predictions.items():
        if not isinstance(rows, list):
            errors.append(f"{video_id}: prediction value is not a list")
            continue
        for row in rows:
            if not isinstance(row, Mapping) or not {"id", "answer", "answer_id"}.issubset(row):
                errors.append(f"{video_id}: malformed prediction row")
                continue
            key = str(row["id"])
            actual.add((str(video_id), key))
            if not isinstance(row["answer"], str) or not isinstance(row["answer_id"], int) or not 0 <= row["answer_id"] < 5:
                errors.append(f"{video_id}/{key}: invalid answer fields")
                continue
            source = payload.get(video_id, {})
            question = source.get(key) if isinstance(source, Mapping) else None
            if question is None and isinstance(source, Mapping):
                question = next((item for item in source.values() if isinstance(item, Mapping) and str(item.get("id")) == key), None)
            options = question.get("options") if isinstance(question, Mapping) else None
            if not isinstance(options, list) or len(options) != 5:
                errors.append(f"{video_id}/{key}: source question lacks five options")
            elif str(options[row["answer_id"]]) != row["answer"]:
                errors.append(f"{video_id}/{key}: answer does not equal option addressed by answer_id")
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"missing questions: {missing[:5]}")
    if extra:
        errors.append(f"extra questions: {extra[:5]}")
    return {"valid": not errors, "expected": len(expected), "actual": len(actual), "errors": errors}


def generate_split(manifest: Path, output: Path, evidence_output: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    predictions, evidence = generate_predictions(payload, root=manifest.parent.parent, split=manifest.stem)
    audit = audit_predictions(payload, predictions)
    if not audit["valid"]:
        raise ValueError(json.dumps(audit))
    _atomic_json(output, predictions)
    _atomic_json(evidence_output, evidence)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate_split(args.manifest, args.output, args.evidence), indent=2))
