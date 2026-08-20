"""Official DA3-Streaming trajectory experiment for exact map windows.

This lane intentionally calls ByteDance's released streaming implementation;
it does not reproduce chunk stitching locally.  Challenge labels are copied to
the report only after inference and are never passed to the model or matcher.
"""
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
DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git", "curl", "libgl1", "libglib2.0-0", "build-essential")
    .run_commands(
        f"git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git /opt/da3 && cd /opt/da3 && git checkout {DA3_COMMIT} && git submodule update --init --recursive",
        "pip install /opt/da3",
        "pip install faiss-cpu pandas prettytable numba pypose triton",
    )
    .add_local_python_source("question_planner")
)
app = modal.App("kilometervision-da3-streaming-motion")


def _source(video_id: str) -> Path | None:
    directory = SOURCE_ROOT / video_id
    if not directory.is_dir():
        return None
    return next((path for path in sorted(directory.iterdir()) if path.is_file()
                 and path.suffix.lower() in VIDEO_SUFFIXES and ".partial" not in path.stem), None)


def _ground_projection(positions) -> list[list[float]]:
    """Best-fit 2-D plane, retaining temporal order and no answer information."""
    import numpy as np
    values = np.asarray(positions, dtype="float64")
    values -= values.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(values, full_matrices=False)
    return (values @ axes[:2].T).tolist()


def _pedestrian_path(positions, radius: int = 4) -> list[list[float]]:
    """Suppress high-frequency gait/camera-centre wobble before projection."""
    import numpy as np
    values = np.asarray(positions, dtype="float64")
    if len(values) < 3:
        return _ground_projection(values)
    width = max(1, 2 * radius + 1)
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(width, dtype="float64") / width
    smooth = np.column_stack([np.convolve(padded[:, axis], kernel, mode="valid")
                              for axis in range(3)])
    return _ground_projection(smooth)


def _write_config(path: Path, weights: Path, salad: Path, chunk_size: int) -> None:
    # Keep the official algorithmic defaults; disable bulky depth artifacts.
    text = f"""Weights:
  DA3: '{weights}'
  DA3_CONFIG: '{weights.parent / 'config.json'}'
  SALAD: '{salad}'
Model:
  chunk_size: {chunk_size}
  overlap: {chunk_size // 2}
  loop_chunk_size: 20
  loop_enable: True
  useDBoW: False
  delete_temp_files: True
  align_lib: 'triton'
  align_method: 'sim3'
  scale_compute_method: 'auto'
  align_type: 'dense'
  ref_view_strategy: 'saddle_balanced'
  ref_view_strategy_loop: 'saddle_balanced'
  depth_threshold: 15.0
  save_depth_conf_result: False
  save_debug_info: False
  Sparse_Align:
    keypoint_select: 'orb'
    keypoint_num: 5000
  IRLS:
    delta: 0.1
    max_iters: 5
    tol: 1e-9
  Pointcloud_Save:
    sample_ratio: 0.00001
    conf_threshold_coef: 0.75
Loop:
  SALAD:
    image_size: [336, 336]
    batch_size: 32
    similarity_threshold: 0.85
    top_k: 5
    use_nms: True
    nms_threshold: 25
  SIM3_Optimizer:
    lang_version: 'python'
    max_iterations: 30
    lambda_init: '1e-6'
"""
    path.write_text(text)


@app.function(image=IMAGE, volumes={"/data": VOLUME}, gpu="H200", memory=65536,
              timeout=6 * 60 * 60)
def run(split: str = "validation", video_id: str = "G0fWvIxy88A", sample_fps: float = 1.0,
        chunk_size: int = 60) -> dict:
    import os
    import sys
    import numpy as np
    from huggingface_hub import hf_hub_download

    VOLUME.reload()
    checkpoint_dir = ROOT / "checkpoints" / "da3-streaming"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(hf_hub_download("depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                                   "model.safetensors", cache_dir=checkpoint_dir))
    config_json = Path(hf_hub_download("depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                                       "config.json", cache_dir=checkpoint_dir))
    # The released runner expects model and config beside one another.
    if not (weights.parent / "config.json").exists():
        (weights.parent / "config.json").write_bytes(config_json.read_bytes())
    salad = checkpoint_dir / "dino_salad.ckpt"
    if not salad.exists():
        subprocess.run(["curl", "-L", "--fail", "-o", str(salad),
                        "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"],
                       check=True)

    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    source = _source(video_id)
    records = []
    for question_id, question in manifest.get(video_id, {}).items():
        if "map" not in classify_question_facets(question):
            continue
        start = float(question.get("start_seconds", 0.0)); end = float(question.get("end_seconds", start))
        if source is None or end <= start:
            continue
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            frames = temporary_path / "frames"; output = temporary_path / "output"
            frames.mkdir(); output.mkdir()
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                            "-t", str(end-start), "-i", str(source), "-vf",
                            f"fps={sample_fps},scale=504:-2", "-q:v", "3", str(frames / "%06d.jpg")],
                           check=True, timeout=max(180, int(end-start)*2))
            config = temporary_path / "config.yaml"; _write_config(config, weights, salad, chunk_size)
            os.chdir("/opt/da3/da3_streaming")
            sys.path.insert(0, "/opt/da3/da3_streaming")
            from da3_streaming import DA3_Streaming, load_config
            runner = DA3_Streaming(str(frames), str(output), load_config(str(config)))
            runner.run(); runner.close()
            poses = np.loadtxt(output / "camera_poses.txt", dtype="float64").reshape(-1, 4, 4)
            positions = poses[:, :3, 3]
        records.append({
            "video_id": video_id, "question_id": str(question_id), "status": "ok",
            "method": "official-da3-streaming", "upstream_commit": DA3_COMMIT,
            "checkpoint": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            "sample_fps": sample_fps, "chunk_size": chunk_size, "loop_closure": True,
            "start_seconds": start, "end_seconds": end, "pose_frames": len(positions),
            "positions_3d": positions.tolist(), "trajectory_2d": _ground_projection(positions),
            "pedestrian_trajectory_2d": _pedestrian_path(positions),
            "ground_truth_used_for_inference": False, "answer_id": question.get("answer_id"),
        })
    result = {"split": split, "video_id": video_id, "sample_fps": sample_fps,
              "chunk_size": chunk_size, "ground_truth_used_for_inference": False, "records": records}
    target = ROOT / "experiments" / "da3-streaming-motion" / split / f"fps{sample_fps:g}-chunk{chunk_size}" / f"{video_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main(split: str = "validation", video_id: str = "G0fWvIxy88A", sample_fps: float = 1.0,
         chunk_size: int = 60) -> None:
    result = run.remote(split, video_id, sample_fps, chunk_size)
    print(json.dumps({"video_id": video_id, "sample_fps": sample_fps, "records": [
        {"question_id": row["question_id"], "status": row["status"], "pose_frames": row["pose_frames"]}
        for row in result["records"]]}, indent=2))
