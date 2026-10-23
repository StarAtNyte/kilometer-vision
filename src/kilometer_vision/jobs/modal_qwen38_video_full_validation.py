"""Corrected Qwen3.8-27B full-validation probe using real native video input.

The earlier ``src/kilometer_vision/jobs/modal_qwen38_full_validation.py`` run (5/18, see progress.md
2026-08-23) had two harness bugs, not a genuine model weakness:

1. It loaded the checkpoint with ``AutoModelForImageTextToText``. Qwen3.8-27B's
   own model card specifies ``AutoModelForMultimodalLM``.
2. It never actually built a ``{"type": "video", ...}`` message part. Frames
   were passed as separate ``{"type": "image"}`` entries captioned "Ordered
   video frame START/step/END", so the model saw a slideshow of stills with
   text labels rather than a real video tensor -- discarding exactly the
   native temporal encoding that is supposed to make this model strong at
   motion/compass/map reasoning.

This probe fixes both: correct Auto class, and a genuine ``video`` content
part built the same way as the already-working ``src/kilometer_vision/jobs/modal_qwen38_native_video_map.py``
map-only probe, generalized to every question category. Answer-blind only:
organizer ``answer_id`` is copied into the record after generation, purely
for audit scoring, and never enters the prompt or control flow.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from kilometer_vision.question_planner import classify_question
from kilometer_vision.vlm_solver import build_prompt, parse_answer

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "accelerate", "pillow")
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .add_local_python_source('kilometer_vision')
)
APP = modal.App("kilometervision-qwen38-video-full-validation")
CHECKPOINT = "Qwen/Qwen3.8-27B"


def _video_message_content(category: str, options: list, reference_images: list, frame_images: list, prompt: str) -> list[dict]:
    """Like modal_vlm._message_content, but frames go in as one real video part."""
    content: list[dict] = []
    if category == "map" and len(reference_images) == len(options):
        for letter, image in zip("ABCDE", reference_images):
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
    if frame_images:
        content.append({"type": "video", "video": [image for _, image in frame_images], "fps": 1.0})
    content.append({"type": "text", "text": prompt})
    return content


@APP.function(image=IMAGE, gpu="A100-80GB", memory=65536, volumes={"/data": VOLUME}, timeout=2 * 3600)
def run(split: str = "validation", video_id: str = "__all__", max_frames: int = 16):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from kilometer_vision.jobs.modal_vlm import _find_source_video, _probe_duration, _question_frame_images, _reference_images

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    targets = sorted(manifest) if video_id == "__all__" else [x for x in video_id.split(",") if x]
    model = AutoModelForMultimodalLM.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=str(ROOT / "checkpoints" / "huggingface"),
    )
    processor = AutoProcessor.from_pretrained(CHECKPOINT, cache_dir=str(ROOT / "checkpoints" / "huggingface"))
    all_rows = []
    for target in targets:
        source = _find_source_video(target, Path("/data/shared/walking-tours"))
        duration = _probe_duration(source) if source is not None else None
        for question_id, question in manifest.get(target, {}).items():
            options = question.get("options", []) if isinstance(question, dict) else []
            category = classify_question(question)
            frames = _question_frame_images(question, source, duration, max_frames)
            references = _reference_images(question, ROOT, split)
            if not frames and not any(x is not None for x in references):
                raw, selected, status = None, None, "no_images"
            else:
                prompt = build_prompt(question, category)
                try:
                    content = _video_message_content(category, options, references, frames, prompt)
                    messages = [{"role": "user", "content": content}]
                    inputs = processor.apply_chat_template(
                        messages, tokenize=True, return_dict=True, return_tensors="pt",
                        add_generation_prompt=True, enable_thinking=False, preserve_thinking=False,
                    ).to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                    raw = processor.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
                    selected = parse_answer(raw, len(options) or 5)
                    status = "ok" if selected is not None else "unparsed"
                    del inputs, generated
                    torch.cuda.empty_cache()
                except Exception as error:
                    raw, selected, status = None, None, f"error:{type(error).__name__}"
            all_rows.append({
                "video_id": target,
                "question_id": str(question_id),
                "category": category,
                "selected_index": selected,
                "raw_text": raw,
                "status": status,
                "answer_id": question.get("answer_id"),
                "native_video": True,
                "ground_truth_used_for_inference": False,
            })
    output = ROOT / "experiments" / "qwen38-video-full" / f"{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint": CHECKPOINT, "max_frames": max_frames, "rows": all_rows, "ground_truth_used_for_inference": False}, indent=2) + "\n")
    VOLUME.commit()
    by_category = {}
    for category in sorted({row["category"] for row in all_rows}):
        subset = [row for row in all_rows if row["category"] == category]
        by_category[category] = {"questions": len(subset), "ok": sum(row["status"] == "ok" for row in subset), "correct": sum(row["selected_index"] == row["answer_id"] for row in subset)}
    return {"rows": len(all_rows), "by_category": by_category, "artifact": str(output)}


@APP.local_entrypoint()
def main(split: str = "validation", video_id: str = "__all__", max_frames: int = 16):
    print(json.dumps(run.remote(split, video_id, max_frames), indent=2))
