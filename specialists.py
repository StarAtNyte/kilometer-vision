"""Uncertainty-aware specialist scoring contracts for PATHFINDER.

Specialists never invent evidence: missing geometry/retrieval evidence yields a
uniform score and an explicit ``insufficient_evidence`` status. This keeps the
submission layer auditable while later Modal jobs add calibrated measurements.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence


def _uniform(n: int, status: str) -> dict[str, Any]:
    if n <= 0:
        raise ValueError("at least one option is required")
    return {"scores": [1.0 / n] * n, "status": status, "evidence": []}


def _normalise(values: Sequence[float]) -> list[float]:
    values = [max(0.0, float(value)) for value in values]
    total = sum(values)
    return [value / total for value in values] if total else [1.0 / len(values)] * len(values)


def score_compass(options: Sequence[str], heading_posterior: Mapping[str, float] | None = None) -> dict[str, Any]:
    if not heading_posterior:
        return _uniform(len(options), "insufficient_heading_evidence")
    def option_direction(option: Any) -> str:
        text = str(option).lower()
        match = re.search(r"\b(northwest|northeast|southwest|southeast|north|south|east|west)\b", text)
        return match.group(1) if match else text.strip()
    scores = [float(heading_posterior.get(option_direction(option), 0.0)) for option in options]
    return {"scores": _normalise(scores), "status": "heading_posterior", "evidence": []}


def score_distance(options: Sequence[str], estimated_m: float | None, uncertainty_m: float | None = None) -> dict[str, Any]:
    if estimated_m is None or not math.isfinite(estimated_m):
        return _uniform(len(options), "insufficient_metric_scale")
    sigma = max(float(uncertainty_m or max(estimated_m * 0.35, 5.0)), 1e-6)
    values: list[float] = []
    for option in options:
        digits = "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in str(option)).split()
        candidate = float(digits[0]) if digits else float("nan")
        values.append(math.exp(-0.5 * ((candidate - estimated_m) / sigma) ** 2) if math.isfinite(candidate) else 0.0)
    return {"scores": _normalise(values), "status": "metric_posterior", "evidence": []}


def score_similarity(options: Sequence[str], similarities: Sequence[float] | None, status: str) -> dict[str, Any]:
    if similarities is None or len(similarities) != len(options):
        return _uniform(len(options), "insufficient_visual_evidence")
    return {"scores": _normalise(similarities), "status": status, "evidence": []}


def choose(scores: Mapping[str, Any]) -> int:
    values = scores.get("scores", [])
    if not values:
        raise ValueError("specialist result has no scores")
    return max(range(len(values)), key=lambda index: (float(values[index]), -index))


__all__ = ["score_compass", "score_distance", "score_similarity", "choose"]
