"""LoGeR long-context camera trajectory experiment for map-question windows."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import modal

from question_planner import classify_question_facets


ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git")
    .run_commands(
        "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126",
        "pip install numpy==1.26.4 pillow opencv-python-headless plyfile huggingface_hub safetensors natsort einops pyyaml scipy roma evo accelerate",
        "git clone --depth 1 https://github.com/Junyi42/LoGeR.git /opt/LoGeR",
    )
    .add_local_python_source("question_planner")
)
app = modal.App("kilometervision-loger-motion")


def _source(video_id: str) -> Path:
    directory = SOURCE_ROOT / video_id
    source = next((path for path in sorted(directory.iterdir())
                   if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
                   and ".partial" not in path.stem), None)
    if source is None:
        raise FileNotFoundError(video_id)
    return source


def _ground_projection(positions) -> list[list[float]]:
    import numpy as np
    values = np.asarray(positions, dtype="float64")
    values -= values.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(values, full_matrices=False)
    return (values @ axes[:2].T).tolist()


@app.function(image=IMAGE, volumes={"/data": VOLUME}, gpu="H200", memory=65536,
              timeout=3 * 60 * 60)
def run_loger(split: str = "validation", only_video_id: str = "G0fWvIxy88A",
              frame_count: int = 60) -> dict:
    import sys
    import time
    import cv2
    import numpy as np
    import torch
    import yaml
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, "/opt/LoGeR")
    from loger.models.pi3 import Pi3

    VOLUME.reload()
    checkpoint = Path(hf_hub_download(
        "Junyi42/LoGeR", filename="LoGeR/latest.pt",
        cache_dir=str(ROOT / "checkpoints" / "huggingface"),
    ))
    config_path = Path("/opt/LoGeR/ckpts/LoGeR/original_config.yaml")
    model_config = yaml.safe_load(config_path.read_text())["model"]
    model = Pi3(**model_config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.eval().cuda()

    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    records = []
    for video_id, questions in manifest.items():
        if only_video_id and video_id != only_video_id:
            continue
        for question_id, question in questions.items():
            if len(question.get("image_paths", [])) != 5 or "map" not in classify_question_facets(question):
                continue
            start = float(question["start_seconds"]); end = float(question["end_seconds"])
            with tempfile.TemporaryDirectory() as tmp:
                pattern = Path(tmp) / "%06d.jpg"
                fps = max(1e-6, (frame_count - 1) / max(end - start, 1e-6))
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                    "-t", str(end-start), "-i", str(_source(video_id)),
                    "-vf", f"fps={fps},scale=504:280", "-frames:v", str(frame_count),
                    "-q:v", "3", str(pattern),
                ], check=True, timeout=max(300, int(end-start)*2))
                paths = sorted(Path(tmp).glob("*.jpg"))
                images = np.stack([
                    cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                    for path in paths
                ])
            images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255)
            started = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(
                    images_tensor[None], window_size=32, overlap_size=3,
                    reset_every=0, num_iterations=1, sim3=False, se3=False,
                    turn_off_ttt=False, turn_off_swa=False,
                )
            poses = prediction["camera_poses"][0].detach().float().cpu().numpy()
            positions = poses[:, :3, 3]
            timestamps = np.linspace(start, end, len(positions)).tolist()
            records.append({
                "video_id": video_id, "question_id": str(question_id),
                "start_seconds": start, "end_seconds": end,
                "timestamps": timestamps, "positions": positions.tolist(),
                "ground_trace": _ground_projection(positions),
                "frame_count": len(positions), "runtime_seconds": time.time() - started,
                "model": "Junyi42/LoGeR", "checkpoint_bytes": checkpoint.stat().st_size,
                "answer_id": question.get("answer_id"),
            })
    result = {"split": split, "records": records, "ground_truth_used_for_inference": False}
    output = ROOT / "experiments" / "loger-motion" / split / f"{only_video_id}-n{frame_count}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main(split: str = "validation", video_id: str = "G0fWvIxy88A", frame_count: int = 60) -> None:
    result = run_loger.remote(split, video_id, frame_count)
    print(json.dumps({"split": split, "records": len(result["records"]),
                      "frames": [row["frame_count"] for row in result["records"]],
                      "runtime": [row["runtime_seconds"] for row in result["records"]]}, indent=2))
