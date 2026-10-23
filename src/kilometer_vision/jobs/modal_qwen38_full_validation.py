"""Bounded Qwen3.8 full-family validation probe.

This is an answer-blind inference pass: organizer answers are only copied into
the post-hoc audit record after generation and never enter the prompt or control
flow used to choose an answer.
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
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "accelerate", "qwen-vl-utils==0.0.14", "pillow")
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .add_local_python_source('kilometer_vision')
)
APP = modal.App("kilometervision-qwen38-full-validation")
CHECKPOINT = "Qwen/Qwen3.8-27B"


@APP.function(image=IMAGE, gpu="H200", memory=65536, volumes={"/data": VOLUME}, timeout=6 * 3600)
def run(split: str = "validation", video_id: str = "__all__", max_frames: int = 16):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from kilometer_vision.jobs.modal_vlm import FRAME_WIDTH, _find_source_video, _message_content, _probe_duration, _question_frame_images, _reference_images

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    targets = sorted(manifest) if video_id == "__all__" else [x for x in video_id.split(",") if x]
    model = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=str(ROOT / "checkpoints" / "huggingface"),
        attn_implementation="sdpa",
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
                    messages = [{"role": "user", "content": _message_content(category, options, references, frames, prompt)}]
                    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[rendered], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
                    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
                    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
                "ground_truth_used_for_inference": False,
            })
    output = ROOT / "experiments" / "qwen38-full" / f"{split}.json"
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
