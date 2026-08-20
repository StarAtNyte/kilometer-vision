"""Dependency-free question and option planning for KilometerVision.

The planner intentionally uses only question text and option text.  Organizer
answer fields are never consulted, making it safe to run on held-out metadata.
The returned structures are JSON-serialisable and suitable for retrieval
planners or specialist routing.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


_CATEGORIES = ("compass", "distance", "map", "landmark")
_COMPASS_ALIASES = {
    "n": "north", "north": "north",
    "ne": "northeast", "northeast": "northeast",
    "e": "east", "east": "east",
    "se": "southeast", "southeast": "southeast",
    "s": "south", "south": "south",
    "sw": "southwest", "southwest": "southwest",
    "w": "west", "west": "west",
    "nw": "northwest", "northwest": "northwest",
}
_DIRECTION_RE = re.compile(
    r"\b(?:north(?:east|west)?|south(?:east|west)?|east|west|n|ne|e|se|s|sw|w|nw)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z])")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_DISTANCE_UNIT_RE = re.compile(
    r"(?P<number>\d+(?:\.\d+)?|\.\d+)\s*(?P<unit>kilometers?|km|meters?|metres?|miles?|mi|feet|ft|m)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a", "an", "and", "around", "at", "be", "by", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "path", "the", "this",
    "to", "was", "what", "which", "where", "with",
}


def _question_text(question: str | Mapping[str, Any]) -> str:
    if isinstance(question, str):
        return question
    if isinstance(question, Mapping):
        value = question.get("question", "")
        return value if isinstance(value, str) else str(value)
    raise TypeError("question must be a string or mapping")


def _has_reference_image(question: str | Mapping[str, Any]) -> bool:
    if not isinstance(question, Mapping):
        return False
    paths = question.get("image_paths")
    return isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)) and bool(paths)


def classify_question(question: str | Mapping[str, Any]) -> str:
    """Return one of ``compass``, ``distance``, ``map``, or ``landmark``.

    Classification is deliberately conservative and deterministic.  More
    specific spatial cues take precedence over generic image/landmark cues:
    map questions first, then explicit metric-distance questions, then compass,
    with landmark as the image/place fallback.
    """
    text = _question_text(question).lower()
    has_image = _has_reference_image(question)
    map_cues = ("map", "route trace", "path correctly", "represent the path", "trajectory")
    distance_cues = (
        "distance", "how far", "meters", "metres", "kilometer", "kilometre", " km",
        "miles", "line-of-sight", "line of sight", "straight-line", "straight line",
    )
    compass_cues = (
        "compass", "direction", "facing", "heading", "orientation", "pointing",
        "pointing towards", "pointing toward", "turn left", "turn right", "north", "south", "east", "west",
    )
    landmark_cues = ("landmark", "location", "reference image", "does the image", "appeared")

    if any(cue in text for cue in map_cues):
        return "map"
    if any(cue in text for cue in distance_cues):
        return "distance"
    if any(cue in text for cue in compass_cues):
        return "compass"
    if has_image or any(cue in text for cue in landmark_cues):
        return "landmark"
    # Unrecognised spatial prompts are safest in the landmark/VLM lane.
    return "landmark"


def classify_question_facets(question: str | Mapping[str, Any]) -> tuple[str, ...]:
    """Return every reasoning operation required by a question, in solve order.

    ``classify_question`` is intentionally a single-label compatibility API.  That
    loses important structure for prompts such as "does this image show a location,
    and if so, how far is it from the end?": presence retrieval must happen before
    distance estimation.  Facets preserve that composition without consulting any
    organizer answer fields.
    """
    text = _question_text(question).lower()
    has_image = _has_reference_image(question)
    facets: list[str] = []
    is_map = any(cue in text for cue in ("map", "route trace", "represent the path", "trajectory"))
    if (has_image and not is_map) or any(cue in text for cue in ("landmark", "reference image", "does the image", "image show a location")):
        facets.append("landmark_presence")
    if is_map:
        facets.append("map")
        if "sets of directions" in text or "directions correctly represent" in text:
            facets.append("route_instructions")
    if not is_map and any(cue in text for cue in ("direction", "facing", "heading", "orientation", "pointing", "compass")):
        facets.append("compass")
    if any(cue in text for cue in ("distance", "how far", "line-of-sight", "line of sight", "straight-line", "straight line")):
        facets.append("distance")
    if not facets:
        facets.append(classify_question(question))
    return tuple(dict.fromkeys(facets))


def _normalise_directions(text: str) -> list[str]:
    found: list[str] = []
    for match in _DIRECTION_RE.finditer(text):
        direction = _COMPASS_ALIASES[match.group(0).lower()]
        if direction not in found:
            found.append(direction)
    return found


def _distance_values(text: str) -> list[float]:
    values: list[float] = []
    covered: list[tuple[int, int]] = []
    for match in _DISTANCE_UNIT_RE.finditer(text):
        value = float(match.group("number"))
        unit = match.group("unit").lower()
        if unit.startswith("km") or unit.startswith("kilometer") or unit.startswith("kilometre"):
            value *= 1000.0
        elif unit.startswith("mile") or unit == "mi":
            value *= 1609.344
        elif unit.startswith("feet") or unit == "ft":
            value *= 0.3048
        values.append(value)
        covered.append(match.span())
    # Bare numbers are retained when an option is clearly a distance option;
    # this handles organizer answers such as "66" or "Around 40:00" without
    # mistaking ordinary prose numbers for metric values.
    if not values:
        for match in _NUMBER_RE.finditer(text):
            prefix = text[max(0, match.start() - 12):match.start()].lower()
            if any(cue in prefix for cue in ("around", "about", "meter", "metre", "distance", "line")):
                values.append(float(match.group(0)))
    return values


def _atomic_text(text: str) -> list[str]:
    atoms = [part.strip(" ;,/") for part in re.split(r"\s+(?:and\s+then|followed by|and|then|or)\s+|[;|]", text, flags=re.IGNORECASE)]
    atoms = [part for part in atoms if part]
    return atoms or [text.strip()]


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def compile_options(question: str | Mapping[str, Any]) -> dict[str, Any]:
    """Compile five answer options into retrieval-friendly atomic predicates.

    The input accepts either the organizer question mapping or a mapping with
    ``question`` and ``options`` fields.  The answer/answer_id fields, if
    present, are ignored.  Distances are normalised to metres; compass labels
    are normalised to lowercase full direction names.
    """
    text = _question_text(question)
    if not isinstance(question, Mapping):
        raise TypeError("compile_options requires a mapping containing options")
    raw_options = question.get("options", [])
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, (str, bytes)):
        raise TypeError("question options must be a sequence")
    options = [str(option) for option in raw_options]
    token_lists = [_tokens(option) for option in options]
    counts = Counter(token for tokens in token_lists for token in set(tokens))
    compiled: list[dict[str, Any]] = []
    for index, option in enumerate(options):
        tokens = token_lists[index]
        discriminating = [token for token in tokens if counts[token] == 1]
        compiled.append({
            "index": index,
            "text": option,
            "atomic_text": _atomic_text(option),
            "normalized_compass_directions": _normalise_directions(option),
            "numeric_distance_values_m": _distance_values(option),
            "discriminating_tokens": list(dict.fromkeys(discriminating)),
        })
    return {
        "question": text,
        "category": classify_question(question),
        "options": compiled,
    }


__all__ = ["classify_question", "classify_question_facets", "compile_options"]
