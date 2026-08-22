"""Qwen3.8 official native-video/map-panel probe on validation maps."""
from __future__ import annotations

import json
import io
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


# A100-80GB is sufficient for the 27B bf16 checkpoint and works on profiles
# without H200 billing enabled.  Keep this validation probe affordable.
@APP.function(image=IMAGE, gpu="A100-80GB", memory=65536, volumes={"/data": VOLUME}, timeout=60 * 60)
def run(split: str = "validation", checkpoint: str = CHECKPOINT, direct: bool = False):
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
            # Profiles may have a different cached DPVO variant (for example the
            # provisioned baptinjean volume contains fps2/p8). Use the first
            # available trajectory rather than silently returning zero rows.
            if not direct and not motion_path.exists():
                candidates = sorted((ROOT / "experiments" / "dpvo-motion" / split).glob("*/" + f"{video_id}.json"))
                motion_path = candidates[0] if candidates else None
            if not direct and (motion_path is None or not motion_path.exists()):
                continue
            motions = {} if direct else json.loads(motion_path.read_text())
            for question_id, question in questions.items():
                if "map" not in classify_question_facets(question) or len(question.get("image_paths", [])) != 5:
                    continue
                try:
                    source = _source(video_id)
                except FileNotFoundError:
                    # The baptinjean volume currently contains only a small
                    # validation video subset; leave missing videos for a later
                    # data-copy step instead of aborting the whole probe.
                    continue
                paths = _candidate_paths(question, archive)
                if direct:
                    # Clean ablation: native video plus the five original maps,
                    # with no rendered trajectory or DPVO-derived evidence.
                    start = float(question.get("start_seconds", 0.0))
                    end = float(question.get("end_seconds", start + 60.0))
                    if end <= start:
                        end = start + 60.0
                    frames = [_frame(source, start + (end - start) * i / 7.0) for i in range(8)]
                    candidate_images = []
                    for reference in question.get("image_paths", []):
                        matches = [name for name in archive.namelist() if Path(name).name == Path(str(reference)).name]
                        if len(matches) != 1:
                            candidate_images = []
                            break
                        from PIL import Image
                        candidate_images.append(Image.open(io.BytesIO(archive.read(matches[0]))).convert("RGB"))
                    prompt = ("Solve the five-choice walking-tour map question from the video. "
                              "The next five images are candidates A, B, C, D, E in that order. "
                              "Determine which map matches the route and return exactly one letter A, B, C, D, or E.")
                    messages = [{"role": "user", "content": (
                        [{"type": "video", "video": frames, "fps": 1.0}]
                        + [{"type": "image", "image": image} for image in candidate_images]
                        + [{"type": "text", "text": prompt}]
                    )}]
                else:
                    motion = motions.get(str(question_id), {})
                    trace, timestamps = motion.get("trajectory_2d", []), motion.get("timestamps", [])
                    if len(trace) < 4 or len(trace) != len(timestamps):
                        continue
                    panel = _render_panel(trace, list(zip("ABCDE", paths)))
                    indices = _motion_aware_indices(trace, count=8)
                    start = float(motion.get("start_seconds", question.get("start_seconds", 0.0)))
                    fps = max(float(motion.get("sample_fps", 1.0)), 1e-6)
                    frames = [_frame(source, start + float(timestamps[i]) / fps) for i in indices]
                    prompt = ("Solve this five-choice walking-route map question. The first image is a normalized "
                              "top-down comparison: observed trajectory is on the left, candidates A-E are on the right. "
                              "Green means start and red means end. Absolute scale, rotation, and mirror are irrelevant. "
                              "Use ordered turns, topology, bends, and endpoints in the diagram and the native ordered video. "
                              "Return exactly one letter A, B, C, D, or E.\n")
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
                rows.append({"video_id": video_id, "question_id": str(question_id), "prediction": prediction, "raw_text": raw, "status": status, "answer_id": question.get("answer_id"), "checkpoint": checkpoint, "native_video": True, "direct_ablation": direct, "ground_truth_used_for_inference": False})
                if "inputs" in locals():
                    del inputs, generated
                torch.cuda.empty_cache()
    output = ROOT / "experiments" / "qwen38-native-video-map" / f"{split}{'-direct' if direct else ''}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows, "ground_truth_used_for_inference": False}, indent=2) + "\n")
    VOLUME.commit()
    return {"rows": len(rows), "ok": sum(row["status"] == "ok" for row in rows), "correct": sum(row["prediction"] == row["answer_id"] for row in rows), "predictions": [row["prediction"] for row in rows], "artifact": str(output)}


@APP.local_entrypoint()
def main(split: str = "validation", checkpoint: str = CHECKPOINT, direct: bool = False):
    print(json.dumps(run.remote(split, checkpoint, direct), indent=2))
