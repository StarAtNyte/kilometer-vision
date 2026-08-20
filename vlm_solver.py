"""Dependency-light frame selection, prompting, and answer parsing for the VLM lane.

This module intentionally has no torch/PIL/transformers import at module load time,
so it is unit-testable without GPU or Modal (matching ``route_solver.py``'s and
``landmark_matcher.py``'s dependency-light style). ``modal_vlm.py`` imports it inside
the GPU function to build prompts, pick frames, and parse the model's answer; the
actual model call and image decoding live only in ``modal_vlm.py``.

The evidence-card contract stays strict on purpose: ``parse_answer`` returns ``None``
rather than guessing on anything ambiguous, so ``submission.py`` can fall through to
the existing deterministic specialists instead of trusting an unparseable response.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

_OPTION_LETTERS = "ABCDE"
_BARE_ANSWER_RE = re.compile(r"^[\s\(\[\{\.:\-]*(?:option\s+)?([A-Ea-e]|[1-5])[\s\)\]\}\.:,]*$")
_LABELED_ANSWER_RE = re.compile(r"\b(?:answer|option|choice)\b\s*(?:is|:)?\s*[\(\[]?([A-Ea-e]|[1-5])\b", re.IGNORECASE)


def select_frames(question: Mapping[str, Any], frame_manifest: Mapping[str, Any], *, max_frames: int = 6) -> list[dict[str, Any]]:
    """Pick up to ``max_frames`` frame records, windowed to the question when possible.

    Falls back to an even sample across the whole manifest when the question has no
    ``start_seconds``/``end_seconds`` window, or when the window contains no frames.
    """
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    frames = frame_manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        return []
    start = question.get("start_seconds")
    end = question.get("end_seconds")
    pool = frames
    if start is not None and end is not None:
        start_value, end_value = float(start), float(end)
        if end_value > start_value:
            windowed = [
                frame for frame in frames
                if isinstance(frame, Mapping) and start_value <= float(frame.get("timestamp_seconds", -1.0)) <= end_value
            ]
            if windowed:
                pool = windowed
    if len(pool) <= max_frames:
        return list(pool)
    step = (len(pool) - 1) / (max_frames - 1) if max_frames > 1 else 0.0
    indices = sorted({round(position * step) for position in range(max_frames)})
    return [pool[index] for index in indices]


def build_prompt(
    question: Mapping[str, Any],
    category: str,
    *,
    route_signature_narrative: str | None = None,
    ocr_text: Sequence[str] | None = None,
) -> str:
    """Render the question/options/evidence into a strict single-letter-answer prompt."""
    options = question.get("options", [])
    lines = [
        f"You are answering a {category} spatial-reasoning question about a walking-tour video.",
        f"Question: {question.get('question', '')}",
        "Options:",
    ]
    for index, option in enumerate(options):
        if index >= len(_OPTION_LETTERS):
            break
        lines.append(f"{_OPTION_LETTERS[index]}. {option}")
    if route_signature_narrative:
        lines.append(f"Geometry evidence (relative-scale camera pose, not ground truth): {route_signature_narrative}")
    if ocr_text:
        joined = "; ".join(text for text in ocr_text if text)
        if joined:
            lines.append(f"Text visible in the frames (OCR, may be noisy): {joined}")
    lines.append(
        "Look carefully at the attached frames and any reference/candidate images. "
        "Reply with exactly one letter from A to E and nothing else."
    )
    return "\n".join(lines)


def parse_answer(raw_text: str | None, option_count: int = 5) -> int | None:
    """Return a strict 0-based option index, or ``None`` when the reply is ambiguous."""
    if option_count <= 0:
        raise ValueError("option_count must be positive")
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text.strip()
    match = _BARE_ANSWER_RE.match(text) or _LABELED_ANSWER_RE.search(text)
    if not match:
        return None
    token = match.group(1).upper()
    index = int(token) - 1 if token.isdigit() else _OPTION_LETTERS.index(token)
    return index if 0 <= index < option_count else None


__all__ = ["select_frames", "build_prompt", "parse_answer"]
