"""VLM evidence-card solver: self-hosted Qwen2.5-VL on Modal GPU.

For each question this reads: the compiled question/options, up to
``MAX_QUESTION_FRAMES`` frames extracted on demand with ffmpeg directly from the raw
source video at the question's actual ``start_seconds``/``end_seconds`` window (or
evenly across the whole video when there is no window), up to
``MAX_REFERENCE_IMAGES`` reference/candidate images from the split's image archive,
the real geometry route-signature narrative from ``src/kilometer_vision/geometry_contract.py`` (Part A),
and any OCR text already indexed near that window. It asks the model for exactly one
letter A-E and writes ``vlm/<video_id>.json`` records that ``src/kilometer_vision/submission.py:_vlm_choice``
reads as the primary decision lane.

Frames are extracted on demand rather than read from the pre-built
``frames/<video_id>.json`` cache: that cache was built with ``sample_fps=0.1,
max_frames=60`` and so only ever covers the first ~590 seconds of every video, while
the source videos actually run 60-110+ minutes and question windows span the full
duration. Reading from the cache silently starved most questions of any relevant
visual evidence. This module goes straight to ``/data/shared/walking-tours`` instead.

This module is never unit tested locally (no GPU/torch/transformers/ffmpeg in the dev
sandbox); its only real verification is a live ``modal run`` invocation, which is an
explicitly gated, costed action -- see the project plan. Every per-question failure is
caught and recorded as ``status: "error"``/``"unparsed"``/``"no_images"`` rather than
raising, so one bad question cannot lose an entire video's batch, and
``src/kilometer_vision/submission.py`` never treats a non-"ok" record as a real answer.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import modal

from kilometer_vision.question_planner import classify_question
from kilometer_vision.vlm_solver import build_prompt, parse_answer, select_frames

ROOT = Path("/data/2707-kilometer-vision")
SHARED_VIDEO_ROOT = Path("/data/shared/walking-tours")
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch", "torchvision", "transformers>=4.49", "accelerate", "pillow", "qwen-vl-utils[decord]")
    .add_local_python_source('kilometer_vision')
)
app = modal.App("kilometervision-vlm")

DEFAULT_CHECKPOINT = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_QUESTION_FRAMES = 6
MAX_REFERENCE_IMAGES = 5
FRAME_WIDTH = 640
OPTION_LETTERS = "ABCDE"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _geometry_narrative(video_id: str, root: Path) -> str | None:
    path = root / "geometry" / f"{video_id}.route_signature.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    narrative = payload.get("narrative")
    return narrative if isinstance(narrative, str) else None


def _find_source_video(video_id: str, source_root: Path) -> Path | None:
    directory = source_root / video_id
    if not directory.is_dir():
        return None
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and ".partial" not in path.stem:
            return path
    return None


def _probe_duration(source: Path) -> float | None:
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(source)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, timeout=30, text=True)
        value = float(result.stdout.strip())
        return value if value > 0 else None
    except Exception:
        return None


def _select_timestamps(question: dict[str, Any], duration: float | None, max_frames: int) -> list[float]:
    """Evenly sample up to max_frames timestamps within the question's actual window.

    Falls back to spanning the whole video when the question has no valid window, so
    map/landmark questions without an explicit start/end still get broad coverage
    instead of only the first few minutes.
    """
    if not duration or duration <= 0 or max_frames <= 0:
        return []
    window_start, window_end = 0.0, duration
    start, end = question.get("start_seconds"), question.get("end_seconds")
    if start is not None and end is not None:
        try:
            start_value, end_value = float(start), float(end)
        except (TypeError, ValueError):
            start_value = end_value = None
        if start_value is not None and end_value is not None and end_value > start_value:
            window_start = max(0.0, min(start_value, duration))
            window_end = max(window_start, min(end_value, duration))
    if window_end <= window_start:
        window_start, window_end = 0.0, duration
    if max_frames == 1:
        return [window_start]
    step = (window_end - window_start) / (max_frames - 1)
    return [window_start + step * index for index in range(max_frames)]


def _select_full_route_timestamps(duration: float | None, max_frames: int) -> list[float]:
    """Sample the complete source route for map identity, not local QA."""
    if not duration or duration <= 0 or max_frames <= 0:
        return []
    if max_frames == 1:
        return [0.0]
    return [duration * index / (max_frames - 1) for index in range(max_frames)]


def _extract_frame(source: Path, timestamp: float, width: int):
    from PIL import Image

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "frame.jpg"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, timestamp):.6f}",
            "-i", str(source),
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "3",
            "-y", str(output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=30)
            if not output.exists() or output.stat().st_size == 0:
                return None
            return Image.open(output).convert("RGB").copy()
        except Exception:
            return None


def _question_frame_images(question: dict[str, Any], source_video: Path | None, duration: float | None, max_frames: int):
    if source_video is None:
        return []
    images = []
    for timestamp in _select_timestamps(question, duration, max_frames):
        image = _extract_frame(source_video, timestamp, FRAME_WIDTH)
        if image is not None:
            images.append((timestamp, image))
    return images


def _reference_images(question: dict[str, Any], root: Path, split: str):
    from PIL import Image

    references = question.get("image_paths")
    if not isinstance(references, list) or not references:
        return []
    archive_path = root / "manifests" / f"{split}_images.zip"
    if not archive_path.exists():
        return []
    images = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.namelist()
            for reference in references[:MAX_REFERENCE_IMAGES]:
                basename = Path(str(reference)).name
                matches = [name for name in members if Path(name).name == basename]
                if len(matches) != 1:
                    images.append(None)
                    continue
                try:
                    images.append(Image.open(io.BytesIO(archive.read(matches[0]))).convert("RGB"))
                except Exception:
                    images.append(None)
    except (OSError, zipfile.BadZipFile):
        return []
    return images


def _ocr_snippets(video_id: str, records: list[dict[str, Any]], root: Path) -> list[str]:
    path = root / "ocr" / f"{video_id}.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    indices = {record.get("index") for record in records if isinstance(record, dict)}
    return [row.get("text", "") for row in payload.get("frames", []) if row.get("index") in indices and row.get("text")]


def _message_content(category: str, options: list[Any], reference_images: list[Any], frame_images: list[Any], prompt: str) -> list[dict[str, Any]]:
    """Build the multimodal message content, labeling map candidate images by letter."""
    content: list[dict[str, Any]] = []
    if category == "map" and len(reference_images) == len(options):
        for letter, image in zip(OPTION_LETTERS, reference_images):
            if image is None:
                continue
            content.append({"type": "text", "text": f"Candidate map {letter}:"})
            content.append({"type": "image", "image": image})
    else:
        for image in reference_images:
            if image is None:
                continue
            content.append({"type": "text", "text": "Reference image:"})
            content.append({"type": "image", "image": image})
    for index, (timestamp, image) in enumerate(frame_images):
        role = "START" if index == 0 else "END" if index == len(frame_images) - 1 else f"step {index + 1}"
        content.append({"type": "text", "text": f"Ordered video frame {role}, timestamp {timestamp:.1f}s:"})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return content


@app.function(image=IMAGE, gpu="A100-80GB", volumes={"/data": VOLUME}, timeout=6 * 3600)
def run_vlm_solver(
    video_id: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    split: str = "test",
    max_frames: int = MAX_QUESTION_FRAMES,
    category_filter: str = "",
    output_namespace: str = "vlm",
    full_context_maps: bool = False,
) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    manifest_path = ROOT / "manifests" / f"{split}.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = (sorted(payload.keys()) if video_id == "__all__"
               else [target for target in video_id.split(",") if target])

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(checkpoint)

    summary: list[dict[str, Any]] = []
    for target in targets:
        questions = payload.get(target)
        if not isinstance(questions, dict):
            summary.append({"video_id": target, "status": "missing_questions"})
            continue
        frame_manifest_path = ROOT / "frames" / f"{target}.json"
        frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8")) if frame_manifest_path.exists() else {}
        probe = frame_manifest.get("probe") if isinstance(frame_manifest.get("probe"), dict) else {}
        duration = probe.get("duration_seconds")
        source_video = _find_source_video(target, SHARED_VIDEO_ROOT)
        if duration is None and source_video is not None:
            duration = _probe_duration(source_video)
        # Existing DA3 artifacts were generated from the obsolete globally capped
        # 0-590s frame cache. They have no poses in nearly every current question
        # window, so feeding their whole-video narrative is actively misleading.
        narrative = None
        records: dict[str, Any] = {}
        for key, question in questions.items():
            if not isinstance(question, dict):
                records[str(key)] = {"selected_index": None, "raw_text": None, "status": "malformed_question", "checkpoint": checkpoint}
                continue
            options = question.get("options") if isinstance(question.get("options"), list) else []
            category = classify_question(question)
            if category_filter and category != category_filter:
                continue
            if full_context_maps and category == "map":
                frame_images = []
                for timestamp in _select_full_route_timestamps(duration, max_frames):
                    image = _extract_frame(source_video, timestamp, FRAME_WIDTH)
                    if image is not None:
                        frame_images.append((timestamp, image))
            else:
                frame_images = _question_frame_images(question, source_video, duration, max_frames)
            reference_images = _reference_images(question, ROOT, split)
            # The pre-built frame manifest only ever covers the first ~590 seconds
            # (see module docstring); reuse it purely as a best-effort OCR text source
            # for questions that happen to overlap that window, not for images.
            old_frame_records = select_frames(question, frame_manifest, max_frames=max_frames)
            ocr_text = _ocr_snippets(target, old_frame_records, ROOT)
            if not frame_images and not any(image is not None for image in reference_images):
                records[str(key)] = {"selected_index": None, "raw_text": None, "status": "no_images", "checkpoint": checkpoint}
                continue
            prompt = build_prompt(question, category, route_signature_narrative=narrative, ocr_text=ocr_text)
            content = _message_content(category, options, reference_images, frame_images, prompt)
            try:
                messages = [{"role": "user", "content": content}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated)]
                raw_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                index = parse_answer(raw_text, len(options) or 5)
                status = "ok" if index is not None else "unparsed"
            except Exception as error:  # a single bad question must not lose the batch
                raw_text, index, status = None, None, f"error:{type(error).__name__}"
            records[str(key)] = {"selected_index": index, "raw_text": raw_text, "status": status, "checkpoint": checkpoint}
            del frame_images, reference_images
            torch.cuda.empty_cache()
        output = ROOT / output_namespace / split / f"{target}.json"
        _atomic_json(output, records)
        VOLUME.commit()
        ok_count = sum(1 for record in records.values() if record["status"] == "ok")
        summary.append({"video_id": target, "output": str(output), "questions": len(records), "ok": ok_count})
    return summary[0] if video_id != "__all__" else {"backend": "qwen-vl", "checkpoint": checkpoint, "videos": summary}


@app.local_entrypoint()
def main(video_id: str, checkpoint: str = DEFAULT_CHECKPOINT, split: str = "test", max_frames: int = MAX_QUESTION_FRAMES, category_filter: str = "", output_namespace: str = "vlm", full_context_maps: bool = False) -> None:
    print(json.dumps(run_vlm_solver.remote(video_id, checkpoint, split, max_frames, category_filter, output_namespace, full_context_maps), indent=2))
