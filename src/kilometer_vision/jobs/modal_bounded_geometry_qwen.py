"""Label-safe geometry evidence cards for a stronger Qwen map verifier.

The inference contract is frozen in code: exact question window, an existing
scale-free DPVO trajectory, motion-aware timestamps selected from that trajectory,
five symmetrically decoded S->E candidate paths, RGB/depth evidence, and one
Qwen3-VL decision. Validation ``answer_id`` is copied only after prediction for
reporting and is never included in a prompt or selection rule.
"""
from __future__ import annotations

import io
import json
import math
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

from kilometer_vision.map_solver import orient_path_from_start, resample_polyline, route_marker_positions, route_path
from kilometer_vision.question_planner import classify_question_facets


ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
CHECKPOINT = "Qwen/Qwen3-VL-32B-Instruct"
DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-Large-hf"
DPVO_VARIANT = "fps1-focal0.9-w256-b256-p16"

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1",
        "accelerate", "qwen-vl-utils==0.0.14", "pillow", "opencv-python-headless",
        "numpy==1.26.4",
    )
    .pip_install("git+https://github.com/huggingface/transformers.git")
    .add_local_python_source('kilometer_vision')
)
app = modal.App("kilometervision-bounded-geometry-qwen")


def _source(video_id: str) -> Path:
    directory = SOURCE_ROOT / video_id
    source = next((p for p in sorted(directory.iterdir()) if p.is_file()
                   and p.suffix.lower() in VIDEO_SUFFIXES and ".partial" not in p.stem), None)
    if source is None:
        raise FileNotFoundError(video_id)
    return source


def _motion_aware_indices(trace, count: int = 8) -> list[int]:
    """Select endpoints, progress anchors, and separated high-curvature poses."""
    import numpy as np
    values = np.asarray(trace, dtype="float64")
    if len(values) <= count:
        return list(range(len(values)))
    steps = np.diff(values, axis=0)
    headings = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    curvature = np.zeros(len(values)); curvature[1:-1] = np.abs(np.diff(headings))
    chosen = {0, len(values) - 1}
    # Preserve broad route coverage independently of local motion magnitude.
    for fraction in (0.2, 0.4, 0.6, 0.8):
        chosen.add(round(fraction * (len(values) - 1)))
    separation = max(2, len(values) // 20)
    for index in np.argsort(curvature)[::-1]:
        if len(chosen) >= count:
            break
        if all(abs(int(index) - old) >= separation for old in chosen):
            chosen.add(int(index))
    # Fill deterministically if curvature peaks collided with coverage anchors.
    for index in np.linspace(0, len(values) - 1, count, dtype=int):
        if len(chosen) >= count:
            break
        chosen.add(int(index))
    return sorted(chosen)[:count]


def _normalised(points, box):
    import numpy as np
    values = np.asarray(points, dtype="float64")
    if len(values) < 2:
        return []
    values -= values.min(0, keepdims=True)
    extent = np.maximum(values.max(0), 1e-6)
    scale = min((box[2] - box[0] - 28) / extent[0], (box[3] - box[1] - 28) / extent[1])
    values *= scale
    values += [(box[0] + box[2] - values[:, 0].max()) / 2,
               (box[1] + box[3] - values[:, 1].max()) / 2]
    return [tuple(row) for row in values]


def _render_panel(trace, candidates):
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (1500, 760), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "OBSERVED SCALE-FREE TRAJECTORY (time: blue -> red)", fill="black")
    observed_box = (20, 48, 500, 740); observed = _normalised(trace, observed_box)
    for index in range(1, len(observed)):
        ratio = index / max(1, len(observed) - 1)
        color = (round(230 * ratio), 60, round(230 * (1-ratio)))
        draw.line((observed[index-1], observed[index]), fill=color, width=7)
    if observed:
        draw.ellipse((observed[0][0]-10, observed[0][1]-10, observed[0][0]+10, observed[0][1]+10), fill="green")
        draw.ellipse((observed[-1][0]-10, observed[-1][1]-10, observed[-1][0]+10, observed[-1][1]+10), fill="red")
    boxes = [(520, 48, 830, 365), (845, 48, 1155, 365), (1170, 48, 1480, 365),
             (680, 410, 990, 727), (1005, 410, 1315, 727)]
    for (letter, path), box in zip(candidates, boxes):
        draw.rectangle(box, outline=(180, 180, 180), width=2)
        draw.text((box[0]+10, box[1]+8), f"CANDIDATE {letter}  S -> E", fill="black")
        line = _normalised(path, (box[0]+4, box[1]+30, box[2]-4, box[3]-4))
        if len(line) >= 2:
            draw.line(line, fill=(210, 30, 30), width=6)
            draw.ellipse((line[0][0]-8, line[0][1]-8, line[0][0]+8, line[0][1]+8), fill="green")
            draw.ellipse((line[-1][0]-8, line[-1][1]-8, line[-1][0]+8, line[-1][1]+8), fill="red")
    return canvas


def _frame(source: Path, timestamp: float):
    from PIL import Image
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "frame.jpg"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(timestamp),
                        "-i", str(source), "-frames:v", "1", "-vf", "scale=640:-2",
                        "-q:v", "3", str(target)], check=True, timeout=90)
        return Image.open(target).convert("RGB").copy()


def _depth_image(pipe, image):
    import numpy as np
    from PIL import Image
    depth = np.asarray(pipe(image)["depth"], dtype="float32")
    low, high = np.percentile(depth, (2, 98)); depth = np.clip((depth-low) / max(high-low, 1e-6), 0, 1)
    # Relative inverse-depth visualization: brighter means nearer. Never metric.
    return Image.fromarray(((1-depth) * 255).astype("uint8")).convert("RGB")


def _candidate_paths(question, archive):
    paths = []
    members = archive.namelist()
    for reference in question.get("image_paths", []):
        matches = [name for name in members if Path(name).name == Path(str(reference)).name]
        raw = archive.read(matches[0]) if len(matches) == 1 else b""
        markers = route_marker_positions(raw) if raw else {}
        path = route_path(raw) if raw else []
        paths.append(orient_path_from_start(path, markers))
    return paths


@app.function(image=IMAGE, gpu="H200", memory=65536, volumes={"/data": VOLUME}, timeout=6*3600)
def run(split: str = "validation", video_id: str = "__all__", perturbation: str = "base", checkpoint: str = CHECKPOINT):
    os.environ["HF_HOME"] = str(ROOT / "checkpoints" / "huggingface")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline
    from qwen_vl_utils import process_vision_info

    VOLUME.reload()
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
        cache_dir=os.environ["HF_HOME"],
    )
    processor = AutoProcessor.from_pretrained(CHECKPOINT, cache_dir=os.environ["HF_HOME"])
    depth_pipe = pipeline("depth-estimation", model=DEPTH_CHECKPOINT, device=0,
                          model_kwargs={"cache_dir": os.environ["HF_HOME"]})
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    archive_path = ROOT / "manifests" / f"{split}_images.zip"
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        for target, questions in manifest.items():
            if video_id != "__all__" and target != video_id:
                continue
            motion_path = ROOT / "experiments" / "dpvo-motion" / split / DPVO_VARIANT / f"{target}.json"
            if not motion_path.exists():
                continue
            motions = json.loads(motion_path.read_text())
            for question_id, question in questions.items():
                if "map" not in classify_question_facets(question) or len(question.get("image_paths", [])) != 5:
                    continue
                motion = motions.get(str(question_id), {}); trace = motion.get("trajectory_2d", [])
                timestamps = motion.get("timestamps", [])
                if len(trace) < 4 or len(timestamps) != len(trace):
                    continue
                indices = _motion_aware_indices(trace, count=8)
                if perturbation == "drop-odd":
                    indices = indices[::2] + ([indices[-1]] if indices[-1] not in indices[::2] else [])
                elif perturbation == "trim":
                    indices = [index for index in indices if index == indices[0] or index == indices[-1]
                               or .05*len(trace) <= index <= .95*len(trace)]
                candidate_paths = _candidate_paths(question, archive)
                candidates = list(zip("ABCDE", candidate_paths))
                if perturbation == "layout-rotate":
                    # Same candidate identities, different screen positions. A
                    # geometry-aware answer must be invariant to this rotation.
                    candidates = candidates[2:] + candidates[:2]
                panel = _render_panel(trace, candidates)
                sample_fps = max(float(motion.get("sample_fps", 1.0)), 1e-6)
                start_seconds = float(motion.get("start_seconds", question.get("start_seconds", 0.0)))
                absolute_timestamps = [start_seconds + float(timestamps[index]) / sample_fps for index in indices]
                rgb = [_frame(_source(target), timestamp) for timestamp in absolute_timestamps]
                # Depth is expensive and redundant; use start, strongest interior turn, end.
                depth_indices = sorted({0, len(rgb)//2, len(rgb)-1})
                depths = [_depth_image(depth_pipe, rgb[index]) for index in depth_indices]
                prompt = (
                    "You are solving a walking-route map question. The first diagram compares the observed "
                    "scale-free camera trajectory with five independently normalized candidate routes. Green is "
                    "start and red is end; absolute rotation, mirror convention, and scale are not reliable. Use "
                    "topology, ordered turns, loops, endpoint displacement, and the ordered RGB frames. The grayscale "
                    "images are RELATIVE inverse-depth only (bright=near), never metric distance. Choose the candidate "
                    "whose S-to-E route best matches the walk. Reply with exactly one letter A, B, C, D, or E."
                )
                content = [{"type":"text","text":"Normalized geometry comparison:"}, {"type":"image","image":panel}]
                for order, image in enumerate(rgb):
                    content += [{"type":"text","text":f"Motion-aware ordered RGB keyframe {order+1}/{len(rgb)}:"},
                                {"type":"image","image":image}]
                for order, image in enumerate(depths):
                    content += [{"type":"text","text":f"Relative inverse-depth view {order+1}/{len(depths)}:"},
                                {"type":"image","image":image}]
                content.append({"type":"text","text":prompt})
                messages = [{"role":"user","content":content}]
                rendered = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False, preserve_thinking=False,
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[rendered], images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
                letter = next((value for value in answer.upper() if value in "ABCDE"), None)
                prediction = "ABCDE".index(letter) if letter else None
                records.append({"video_id":target, "question_id":str(question_id), "prediction":prediction,
                                "raw_text":answer, "selected_timestamps":absolute_timestamps,
                                "perturbation":perturbation, "checkpoint":checkpoint,
                                "depth_checkpoint":DEPTH_CHECKPOINT, "ground_truth_used_for_inference":False,
                                "answer_id":question.get("answer_id")})
                del inputs, generated, rgb, depths; torch.cuda.empty_cache()
    result = {"split":split, "perturbation":perturbation, "records":records,
              "ground_truth_used_for_inference":False}
    output = ROOT / "experiments" / "bounded-geometry-qwen" / split / f"{video_id}-{perturbation}.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2)+"\n")
    VOLUME.commit(); return result


@app.local_entrypoint()
def main(split: str = "validation", video_id: str = "__all__", perturbation: str = "base", checkpoint: str = CHECKPOINT):
    result = run.remote(split, video_id, perturbation, checkpoint)
    print(json.dumps({"split":split, "perturbation":perturbation, "records":len(result["records"]),
                      "predictions":[row["prediction"] for row in result["records"]]}, indent=2))
