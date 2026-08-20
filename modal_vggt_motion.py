"""Camera-only VGGT trajectory experiment for exact map-question windows."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import modal

from question_planner import classify_question_facets


ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch", "torchvision", "numpy", "pillow", "huggingface_hub",
        "einops", "safetensors", "git+https://github.com/facebookresearch/vggt.git",
    )
    .add_local_python_source("question_planner")
)
app = modal.App("kilometervision-vggt-motion")


def _source(video_id: str) -> Path | None:
    directory = SOURCE_ROOT / video_id
    if not directory.is_dir():
        return None
    return next((path for path in sorted(directory.iterdir())
                 if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
                 and ".partial" not in path.stem), None)


def _extract(source: Path, start: float, end: float, count: int, directory: Path) -> list[Path]:
    duration = end - start
    fps = max(1e-6, (count - 1) / duration)
    pattern = directory / "%06d.jpg"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
        "-t", str(duration), "-i", str(source), "-vf", f"fps={fps},scale=640:-2",
        "-frames:v", str(count), "-q:v", "3", str(pattern),
    ], check=True, timeout=max(180, int(duration) * 2))
    return sorted(directory.glob("*.jpg"))


def _camera_centres(extrinsics) -> list[list[float]]:
    """Convert OpenCV world-to-camera 3x4 matrices to camera centres."""
    import numpy as np
    values = np.asarray(extrinsics, dtype="float64")
    centres = []
    for matrix in values:
        rotation, translation = matrix[:3, :3], matrix[:3, 3]
        centres.append((-rotation.T @ translation).tolist())
    return centres


def _project_ground(centres: list[list[float]]) -> list[list[float]]:
    """Label-free best-fit 2-D projection of a monocular camera trajectory."""
    import numpy as np
    values = np.asarray(centres, dtype="float64")
    values -= values.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(values, full_matrices=False)
    return (values @ axes[:2].T).tolist()


def _similarity_align(source, target):
    """Umeyama Sim(3) mapping source overlap points onto target points."""
    import numpy as np
    source = np.asarray(source, dtype="float64"); target = np.asarray(target, dtype="float64")
    source_mean = source.mean(axis=0); target_mean = target.mean(axis=0)
    left = source - source_mean; right = target - target_mean
    covariance = left.T @ right / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1
    rotation = vt.T @ correction @ u.T
    scale = float((singular * np.diag(correction)).sum() / max((left * left).sum() / len(source), 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _merge_chunks(chunks: list[list[list[float]]], overlap: int) -> list[list[float]]:
    import numpy as np
    if not chunks:
        return []
    merged = np.asarray(chunks[0], dtype="float64")
    for chunk in chunks[1:]:
        values = np.asarray(chunk, dtype="float64")
        shared = min(overlap, len(values), len(merged))
        scale, rotation, translation = _similarity_align(values[:shared], merged[-shared:])
        aligned = scale * (values @ rotation.T) + translation
        # Average duplicate overlap estimates, then append unseen frames.
        merged[-shared:] = 0.5 * (merged[-shared:] + aligned[:shared])
        merged = np.vstack((merged, aligned[shared:]))
    return merged.tolist()


@app.function(image=IMAGE, volumes={"/data": VOLUME}, gpu="A100-80GB", memory=32768,
              timeout=6 * 60 * 60)
def run_vggt(split: str = "validation", max_frames: int = 60,
             chunk_size: int = 0, overlap: int = 8, target_video: str = "__all__") -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", str(ROOT / "checkpoints" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    device = "cuda"
    dtype = torch.bfloat16
    model = VGGT.from_pretrained(
        "facebook/VGGT-1B", cache_dir=os.environ["HF_HOME"]
    ).to(device).eval()
    summaries = []
    for video_id, questions in manifest.items():
        if target_video != "__all__" and video_id != target_video:
            continue
        source = _source(video_id)
        records = {}
        for question_id, question in questions.items():
            if "map" not in classify_question_facets(question):
                continue
            start = float(question.get("start_seconds", 0.0))
            end = float(question.get("end_seconds", start))
            if source is None or end <= start:
                records[str(question_id)] = {"status": "missing_video_or_window"}
                continue
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    paths = _extract(source, start, end, max_frames, Path(temporary))
                    images = load_and_preprocess_images([str(path) for path in paths]).to(device)
                    if chunk_size > overlap and len(images) > chunk_size:
                        chunks = []
                        stride = chunk_size - overlap
                        starts = list(range(0, len(images) - overlap, stride))
                        if starts[-1] + chunk_size < len(images):
                            starts.append(len(images) - chunk_size)
                        starts = sorted(set(starts))
                        for chunk_start in starts:
                            batch = images[chunk_start:chunk_start + chunk_size]
                            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
                                tokens, _ = model.aggregator(batch[None])
                                pose_encoding = model.camera_head(tokens)[-1]
                            extrinsic, _ = pose_encoding_to_extri_intri(pose_encoding, batch.shape[-2:])
                            chunks.append(_camera_centres(extrinsic[0].float().cpu().numpy()))
                        centres = _merge_chunks(chunks, overlap)
                    else:
                        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
                            tokens, _ = model.aggregator(images[None])
                            pose_encoding = model.camera_head(tokens)[-1]
                        extrinsic, _ = pose_encoding_to_extri_intri(pose_encoding, images.shape[-2:])
                        centres = _camera_centres(extrinsic[0].float().cpu().numpy())
                records[str(question_id)] = {
                    "status": "ok", "method": "VGGT-1B overlapping chunks" if chunk_size else "VGGT-1B camera-only", "start_seconds": start,
                    "end_seconds": end, "frames": len(paths), "camera_centres_3d": centres,
                    "trajectory_2d": _project_ground(centres), "chunk_size": chunk_size or None,
                    "overlap": overlap if chunk_size else None,
                }
            except Exception as error:
                records[str(question_id)] = {
                    "status": f"error:{type(error).__name__}", "error": str(error),
                    "start_seconds": start, "end_seconds": end,
                }
            variant = f"chunk{chunk_size}-overlap{overlap}-frames{max_frames}" if chunk_size else f"global-frames{max_frames}"
            output = ROOT / "experiments" / "vggt-motion" / split / variant / f"{video_id}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(records, indent=2) + "\n")
            VOLUME.commit()
        if records:
            summaries.append({"video_id": video_id, "records": records})
    return {"split": split, "max_frames": max_frames, "chunk_size": chunk_size or None,
            "overlap": overlap if chunk_size else None, "videos": summaries}


@app.local_entrypoint()
def main(split: str = "validation", max_frames: int = 60,
         chunk_size: int = 0, overlap: int = 8, target_video: str = "__all__") -> None:
    result = run_vggt.remote(split, max_frames, chunk_size, overlap, target_video)
    print(json.dumps(result, indent=2))
