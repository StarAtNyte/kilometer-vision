"""Qwen3.8 route-instruction audit on Modal (never used for labels/training)."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import modal

from question_planner import classify_question_facets

ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
CHECKPOINT = "Qwen/Qwen3.8-27B"
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE = (modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
         .apt_install("ffmpeg", "git")
         .pip_install("torch==2.7.1", "torchvision==0.22.1", "accelerate",
                      "qwen-vl-utils==0.0.14", "pillow")
         .pip_install("git+https://github.com/huggingface/transformers.git")
         .add_local_python_source("question_planner"))
app = modal.App("kilometervision-qwen38-route")


def _source(video_id: str) -> Path:
    directory = SOURCE_ROOT / video_id
    source = next((p for p in sorted(directory.iterdir()) if p.is_file()
                   and p.suffix.lower() in VIDEO_SUFFIXES and ".partial" not in p.stem), None)
    if source is None:
        raise FileNotFoundError(video_id)
    return source


def _frames(source: Path, start: float, end: float, count: int):
    from PIL import Image
    with tempfile.TemporaryDirectory() as temporary:
        pattern = str(Path(temporary) / "frame_%03d.jpg")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                        "-t", str(max(0.1, end - start)), "-i", str(source),
                        "-vf", f"fps={count / max(0.1, end-start):.8f},scale=640:-2",
                        "-frames:v", str(count), "-q:v", "4", pattern],
                       check=True, timeout=600)
        return [Image.open(path).convert("RGB").copy()
                for path in sorted(Path(temporary).glob("frame_*.jpg"))]


def _parse(text: str, n: int) -> int | None:
    match = re.search(r"(?:answer|choice|option|index)\s*[:=]?\s*([A-E]|[0-4])", text, re.I)
    if not match:
        match = re.search(r"\b([A-E])\b", text, re.I)
    if not match:
        match = re.search(r"\b([0-4])\b", text)
    if not match:
        return None
    token = match.group(1).upper()
    index = ord(token) - ord("A") if token.isalpha() else int(token)
    return index if 0 <= index < n else None


@app.function(image=IMAGE, gpu="H200", memory=65536, volumes={"/data": VOLUME}, timeout=6 * 3600)
def run(video_id: str, split: str = "test", frame_count: int = 16) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    VOLUME.reload()
    model = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
        cache_dir=str(ROOT / "checkpoints" / "huggingface"), attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(CHECKPOINT, cache_dir=str(ROOT / "checkpoints" / "huggingface"))
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    records = {}
    source = _source(video_id)
    for question_id, question in manifest[video_id].items():
        if "map" not in classify_question_facets(question) or question.get("image_paths"):
            continue
        frames = _frames(source, float(question["start_seconds"]), float(question["end_seconds"]), frame_count)
        options = question.get("options", [])
        content = []
        for i, frame in enumerate(frames):
            content += [{"type": "text", "text": f"Ordered frame {i + 1}/{len(frames)}:"},
                        {"type": "image", "image": frame}]
        option_text = "\n".join(f"{chr(65+i)}. {value}" for i, value in enumerate(options))
        content.append({"type": "text", "text": (
            "Choose the one route-instruction option that exactly matches the ordered walking video. "
            "Track start, endpoint, street sequence, sustained turns, and direction; ignore head pans. "
            "Return exactly one line: ANSWER: A/B/C/D/E.\n\nOPTIONS:\n" + option_text)})
        messages = [{"role": "user", "content": content}]
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False, preserve_thinking=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[rendered], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        records[str(question_id)] = {"selected_index": _parse(raw, len(options)), "raw_text": raw,
                                     "status": "ok" if _parse(raw, len(options)) is not None else "unparsed",
                                     "checkpoint": CHECKPOINT, "frame_count": frame_count}
        del inputs, generated
        torch.cuda.empty_cache()
    output = ROOT / "experiments" / "qwen38-route" / split / f"{video_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    VOLUME.commit()
    return {"video_id": video_id, "records": len(records),
            "ok": sum(x["status"] == "ok" for x in records.values()), "output": str(output)}


@app.local_entrypoint()
def main(video_id: str, split: str = "test", frame_count: int = 16) -> None:
    print(json.dumps(run.remote(video_id, split, frame_count), indent=2))
