"""Question-window camera-motion helpers for compass and route experiments."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def direction_from_degrees(value: float) -> str:
    names = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
    return names[int(((float(value) % 360.0) + 22.5) // 45.0) % 8]


def initial_heading(question: Mapping[str, Any]) -> str | None:
    import re
    text = str(question.get("question", "")).lower()
    match = re.search(r"(?:pointing|facing|heading)\s+(?:towards?\s+)?(?:the\s+)?(northeast|northwest|southeast|southwest|north|south|east|west)\b", text)
    return match.group(1) if match else None


def compass_prediction(question: Mapping[str, Any], yaw_delta_degrees: float) -> str | None:
    anchor = initial_heading(question)
    if anchor is None:
        return None
    angles = {"north": 0.0, "northeast": 45.0, "east": 90.0, "southeast": 135.0,
              "south": 180.0, "southwest": 225.0, "west": 270.0, "northwest": 315.0}
    return direction_from_degrees(angles[anchor] + float(yaw_delta_degrees))


def compass_option_index(question: Mapping[str, Any], yaw_delta_degrees: float) -> int | None:
    """Select the offered direction nearest to the continuous predicted heading."""
    anchor = initial_heading(question)
    options = question.get("options")
    if anchor is None or not isinstance(options, Sequence):
        return None
    angles = {"north": 0.0, "northeast": 45.0, "east": 90.0, "southeast": 135.0,
              "south": 180.0, "southwest": 225.0, "west": 270.0, "northwest": 315.0}
    target = angles[anchor] + float(yaw_delta_degrees)
    candidates = [(abs(wrap_degrees(angles[str(option).strip().lower()] - target)), index)
                  for index, option in enumerate(options) if str(option).strip().lower() in angles]
    return min(candidates)[1] if candidates else None


def summarize_steps(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = [step for step in steps if step.get("accepted") and step.get("yaw_delta_degrees") is not None]
    yaw = sum(float(step["yaw_delta_degrees"]) for step in accepted)
    confidences = [float(step.get("confidence", 0.0)) for step in accepted]
    return {
        "step_count": len(steps),
        "accepted_step_count": len(accepted),
        "net_yaw_degrees": wrap_degrees(yaw),
        "accumulated_absolute_yaw_degrees": sum(abs(float(step["yaw_delta_degrees"])) for step in accepted),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
    }


__all__ = ["wrap_degrees", "direction_from_degrees", "initial_heading", "compass_prediction", "compass_option_index", "summarize_steps"]
