"""Qwen3.8 official native-video/map-panel probe on validation maps."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import modal

from question_planner import classify_question_facets

ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "accelerate", "qwen-vl-utils==0.0.14", "pillow", "opencv-python-headless", "numpy==1.26.4")
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .add_local_python_source("map_solver", "question_planner", "modal_bounded_geometry_qwen")
)
APP = modal.App("kilometervision-qwen38-native-video-map")
CHECKPOINT = "Qwen/Qwen3.8-27B"


@APP.function(image=IMAGE, gpu="H200", memory=65536, volumes={"/data": VOLUME}, timeout=60 * 60)
def run(split: str = "validation", checkpoint: str = CHECKPOINT):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from modal_bounded_geometry_qwen import _candidate_paths, _frame, _motion_aware_indices, _render_panel, _source

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    archive_path = ROOT / "manifests" / f"{split}_images.zip"
    model = AutoModelForMultimodalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
        cache_dir=str(ROOT / "checkpoints" / "huggingface"),
    )
    processor = AutoProcessor.from_pretrained(CHECKPOINT, cache_dir=str(ROOT / "checkpoints" / "huggingface"))
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        for video_id, questions in manifest.items():
            motion_path = ROOT / "experiments" / "dpvo-motion" / split / "fps1-focal0.9-w256-b256-p16" / f"{video_id}.json"
            if not motion_path.exists():
                continue
            motions = json.loads(motion_path.read_text())
            for question_id, question in questions.items():
                if "map" not in classify_question_facets(question) or len(question.get("image_paths", [])) != 5:
                    continue
                motion = motions.get(str(question_id), {})
                trace, timestamps = motion.get("trajectory_2d", []), motion.get("timestamps", [])
                if len(trace) < 4 or len(trace) != len(timestamps):
                    continue
                paths = _candidate_paths(question, archive)
                panel = _render_panel(trace, list(zip("ABCDE", paths)))
                indices = _motion_aware_indices(trace, count=8)
                start = float(motion.get("start_seconds", question.get("start_seconds", 0.0)))
                fps = max(float(motion.get("sample_fps", 1.0)), 1e-6)
                frames = [_frame(_source(video_id), start + float(timestamps[i]) / fps) for i in indices]
                prompt = (
                    "Solve this five-choice walking-route map question. The first image is a normalized "
                    "top-down comparison: observed trajectory is on the left, candidates A-E are on the right. "
                    "Green means start and red means end. Absolute scale, rotation, and mirror are irrelevant. "
                    "Use ordered turns, topology, bends, and endpoints in the diagram and the native ordered video. "
                    "Return exactly one letter A, B, C, D, or E.\n"
                )
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": panel},
                    {"type": "video", "video": frames, "fps": 1.0},
                    {"type": "text", "text": prompt},
                ]}]
                try:
                    inputs = processor.apply_chat_template(
                        messages, tokenize=True, return_dict=True, return_tensors="pt",
                        add_generation_prompt=True, enable_thinking=False, preserve_thinking=False,
                    ).to(model.device)
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                    raw = processor.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
                    letter = next((x for x in raw.upper() if x in "ABCDE"), None)
                    prediction = "ABCDE".index(letter) if letter else None
                    status = "ok" if prediction is not None else "unparsed"
                except Exception as error:
                    raw, prediction, status = None, None, f"error:{type(error).__name__}"
                rows.append({"video_id": video_id, "question_id": str(question_id), "prediction": prediction, "raw_text": raw, "status": status, "answer_id": question.get("answer_id"), "checkpoint": checkpoint, "native_video": True, "ground_truth_used_for_inference": False})
                if "inputs" in locals():
                    del inputs, generated
                torch.cuda.empty_cache()
    output = ROOT / "experiments" / "qwen38-native-video-map" / f"{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows, "ground_truth_used_for_inference": False}, indent=2) + "\n")
    VOLUME.commit()
    return {"rows": len(rows), "ok": sum(row["status"] == "ok" for row in rows), "correct": sum(row["prediction"] == row["answer_id"] for row in rows), "predictions": [row["prediction"] for row in rows], "artifact": str(output)}


@APP.local_entrypoint()
def main(split: str = "validation", checkpoint: str = CHECKPOINT):
    print(json.dumps(run.remote(split, checkpoint), indent=2))
