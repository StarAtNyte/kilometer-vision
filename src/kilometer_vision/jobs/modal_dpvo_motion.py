"""Native-continuity DPVO trajectories for map-question windows."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import modal

from kilometer_vision.question_planner import classify_question_facets


ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0"})
    .apt_install("ffmpeg", "git", "wget", "unzip", "ninja-build", "clang")
    .run_commands(
        "pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121",
        "pip install wheel tensorboard numba tqdm einops pypose kornia numpy==1.26.4 plyfile evo opencv-python-headless yacs",
        "git clone --depth 1 https://github.com/princeton-vl/DPVO.git /opt/DPVO",
        "mkdir -p /opt/DPVO/thirdparty && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip -O /tmp/eigen.zip && unzip -q /tmp/eigen.zip -d /opt/DPVO/thirdparty",
        "cd /opt/DPVO && pip install --no-build-isolation .",
        "wget -q https://huggingface.co/pablovela5620/dpvo/resolve/main/dpvo.pth -O /opt/DPVO/dpvo.pth && test $(stat -c%s /opt/DPVO/dpvo.pth) -gt 10000000",
    )
    .run_commands(
        "pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html",
    )
    .add_local_python_source('kilometer_vision')
)
app = modal.App("kilometervision-dpvo-motion")


def _source(video_id: str) -> Path | None:
    directory = SOURCE_ROOT / video_id
    if not directory.is_dir():
        return None
    return next((path for path in sorted(directory.iterdir())
                 if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
                 and ".partial" not in path.stem), None)


def _project_ground(positions) -> list[list[float]]:
    import numpy as np
    values = np.asarray(positions, dtype="float64")
    values -= values.mean(axis=0, keepdims=True)
    _, _, axes = np.linalg.svd(values, full_matrices=False)
    return (values @ axes[:2].T).tolist()


@app.function(image=IMAGE, volumes={"/data": VOLUME}, gpu="H200", memory=32768,
              timeout=6 * 60 * 60)
def run_dpvo(split: str = "validation", sample_fps: float = 5.0,
             focal_scale: float = 0.9, patches_per_frame: int = 24,
             image_width: int = 256, only_video_id: str = "") -> dict:
    import sys
    sys.path.insert(0, "/opt/DPVO")
    import cv2
    import numpy as np
    import torch
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO

    cfg.merge_from_file("/opt/DPVO/config/default.yaml")
    # The default 2048-keyframe allocation exceeds even 80 GB at 640 px.
    # Walking-tour windows retain far fewer active keyframes than input frames.
    cfg.BUFFER_SIZE = 256
    cfg.PATCHES_PER_FRAME = patches_per_frame
    properties = torch.cuda.get_device_properties(0)
    print(
        f"GPU={properties.name} VRAM_GiB={properties.total_memory / 2**30:.1f} "
        f"CUDA={torch.version.cuda} buffer={cfg.BUFFER_SIZE} "
        f"patches={cfg.PATCHES_PER_FRAME} width={image_width}",
        flush=True,
    )
    manifest = json.loads((ROOT / "manifests" / f"{split}.json").read_text())
    summaries = []
    for video_id, questions in manifest.items():
        if only_video_id and video_id != only_video_id:
            continue
        variant = (
            f"fps{sample_fps:g}-focal{focal_scale:g}-w{image_width}"
            f"-b{cfg.BUFFER_SIZE}-p{patches_per_frame}"
        )
        output = ROOT / "experiments" / "dpvo-motion" / split / variant / f"{video_id}.json"
        try:
            records = json.loads(output.read_text()) if output.exists() else {}
        except (OSError, ValueError):
            records = {}
        window_cache = {
            (float(record["start_seconds"]), float(record["end_seconds"])): dict(record)
            for record in records.values()
            if record.get("status") == "ok"
            and record.get("start_seconds") is not None
            and record.get("end_seconds") is not None
        }
        source = _source(video_id)
        for question_id, question in questions.items():
            if "map" not in classify_question_facets(question):
                continue
            if records.get(str(question_id), {}).get("status") == "ok":
                continue
            start = float(question.get("start_seconds", 0)); end = float(question.get("end_seconds", start))
            if source is None or end <= start:
                records[str(question_id)] = {"status": "missing_video_or_window"}; continue
            window_key = (start, end)
            if window_key in window_cache:
                records[str(question_id)] = dict(window_cache[window_key])
                records[str(question_id)]["reused_window_trajectory"] = True
                print(
                    f"route_reuse video={video_id} question={question_id} "
                    f"seconds={end-start:.1f}", flush=True
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(records, indent=2) + "\n"); VOLUME.commit()
                continue
            slam = None
            try:
                print(f"route_start video={video_id} question={question_id} seconds={end-start:.1f}", flush=True)
                with tempfile.TemporaryDirectory() as temporary:
                    pattern = Path(temporary) / "%06d.jpg"
                    subprocess.run([
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                        "-t", str(end-start), "-i", str(source),
                        "-vf", f"fps={sample_fps},scale={image_width}:-2", "-q:v", "3", str(pattern),
                    ], check=True, timeout=max(180, int(end-start)*2))
                    paths = sorted(Path(temporary).glob("*.jpg"))
                    print(f"frames_extracted video={video_id} count={len(paths)}", flush=True)
                    for index, path in enumerate(paths):
                        image = cv2.imread(str(path)); height, width = image.shape[:2]
                        image = image[:height-height%16, :width-width%16]
                        height, width = image.shape[:2]
                        intrinsics = torch.tensor(
                            [focal_scale*width, focal_scale*width, width/2, height/2],
                            dtype=torch.float32, device="cuda",
                        )
                        tensor = torch.from_numpy(image).permute(2, 0, 1).cuda()
                        if slam is None:
                            slam = DPVO(cfg, "/opt/DPVO/dpvo.pth", ht=height, wd=width, viz=False)
                        slam(index, tensor, intrinsics)
                        if index and index % 100 == 0:
                            allocated = torch.cuda.memory_allocated() / 2**30
                            reserved = torch.cuda.memory_reserved() / 2**30
                            print(
                                f"route_progress video={video_id} frame={index}/{len(paths)} "
                                f"allocated_GiB={allocated:.1f} reserved_GiB={reserved:.1f}",
                                flush=True,
                            )
                    if slam is None:
                        raise RuntimeError("no frames extracted")
                    poses, timestamps = slam.terminate()
                    positions = np.asarray(poses[:, :3], dtype="float64")
                    del slam; torch.cuda.empty_cache()
                records[str(question_id)] = {
                    "status": "ok", "method": "DPVO", "sample_fps": sample_fps,
                    "focal_scale": focal_scale, "start_seconds": start, "end_seconds": end,
                    "input_frames": len(paths), "pose_frames": len(positions),
                    "positions_3d": positions.tolist(), "trajectory_2d": _project_ground(positions),
                    "timestamps": np.asarray(timestamps).tolist(),
                }
                window_cache[window_key] = dict(records[str(question_id)])
                print(f"route_done video={video_id} poses={len(positions)}", flush=True)
            except Exception as error:
                records[str(question_id)] = {"status": f"error:{type(error).__name__}", "error": str(error)}
                window_cache[window_key] = dict(records[str(question_id)])
                print(f"route_error video={video_id} type={type(error).__name__} error={error}", flush=True)
                if "slam" in locals() and slam is not None:
                    del slam
                torch.cuda.empty_cache()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(records, indent=2) + "\n"); VOLUME.commit()
        if records:
            summaries.append({"video_id": video_id, "records": records})
    return {
        "split": split, "sample_fps": sample_fps, "focal_scale": focal_scale,
        "patches_per_frame": patches_per_frame, "image_width": image_width,
        "only_video_id": only_video_id, "videos": summaries,
    }


@app.local_entrypoint()
def main(split: str = "validation", sample_fps: float = 5.0, focal_scale: float = 0.9,
         patches_per_frame: int = 24, image_width: int = 256,
         only_video_id: str = ""):
    result = run_dpvo.remote(
        split, sample_fps, focal_scale, patches_per_frame, image_width, only_video_id
    )
    compact = dict(result)
    compact["videos"] = [
        {
            "video_id": video["video_id"],
            "questions": {
                question_id: {
                    "status": record.get("status"),
                    "input_frames": record.get("input_frames"),
                    "pose_frames": record.get("pose_frames"),
                    "reused_window_trajectory": record.get("reused_window_trajectory", False),
                }
                for question_id, record in video["records"].items()
            },
        }
        for video in result["videos"]
    ]
    print(json.dumps(compact, indent=2))
