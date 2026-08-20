"""Answer-blind VGGT-SLAM 2.0 cross-density route experiment."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import modal

from question_planner import classify_question_facets


ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
SLAM_COMMIT = "35327ac28b7d193df9ccc39ba6346052bb6f1207"
VGGT_COMMIT = "6e6e16107b88e8e76c751826af10d4295d87ecd2"
SALAD_COMMIT = "33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d"
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121",
        "pip install numpy pillow open3d huggingface_hub einops safetensors pytorch-metric-learning pytorch-lightning termcolor viser==0.2.23 tqdm omegaconf opencv-python-headless scipy requests trimesh matplotlib lz4 gtsam-develop ftfy regex",
        f"git clone https://github.com/MIT-SPARK/VGGT-SLAM.git /opt/vggt-slam && cd /opt/vggt-slam && git checkout {SLAM_COMMIT}",
        f"git clone https://github.com/Dominic101/salad.git /opt/vggt-slam/third_party/salad && cd /opt/vggt-slam/third_party/salad && git checkout {SALAD_COMMIT} && pip install -e .",
        f"git clone https://github.com/MIT-SPARK/VGGT_SPARK.git /opt/vggt-slam/third_party/vggt && cd /opt/vggt-slam/third_party/vggt && git checkout {VGGT_COMMIT} && pip install -e .",
        "pip install -e /opt/vggt-slam",
        "pip install gdown && mkdir -p /root/.cache/torch/hub/checkpoints && gdown 1u83Dmqmm1-uikOPr58IIhfIzDYwFxCy1 -O /root/.cache/torch/hub/checkpoints/dino_salad.ckpt",
    )
    .add_local_python_source("question_planner")
)
app = modal.App("kilometervision-vggt-slam2")


def _source(video_id: str) -> Path | None:
    directory = SOURCE_ROOT / video_id
    if not directory.is_dir():
        return None
    return next((path for path in sorted(directory.iterdir()) if path.is_file()
                 and path.suffix.lower() in VIDEO_SUFFIXES and ".partial" not in path.stem), None)


def _map_window(manifest: dict, video_id: str) -> tuple[float, float]:
    for question in manifest.get(video_id, {}).values():
        if "map" in classify_question_facets(question):
            start = float(question.get("start_seconds", 0.0))
            end = float(question.get("end_seconds", start))
            if end > start:
                return start, end
    raise ValueError(f"no map window for {video_id}")


def _extract(source: Path, start: float, end: float, fps: float, directory: Path) -> list[Path]:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
        "-t", str(end - start), "-i", str(source), "-vf", f"fps={fps},scale=640:-2",
        "-q:v", "3", str(directory / "frame_%06d.jpg"),
    ], check=True, timeout=max(240, int(end - start) * 2))
    paths = sorted(directory.glob("*.jpg"))
    if len(paths) < 17:
        raise RuntimeError(f"only {len(paths)} frames extracted")
    return paths


def _read_poses(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        values = [float(value) for value in line.split()]
        rows.append({"frame": int(values[0]), "translation": values[1:4],
                     "quaternion_xyzw": values[4:8]})
    return rows


@app.function(image=IMAGE, volumes={"/data": VOLUME}, gpu="A100-80GB", memory=65536,
              timeout=6 * 60 * 60)
def run(video_id: str = "G0fWvIxy88A", densities: str = "1,2") -> dict:
    VOLUME.reload()
    manifest = json.loads((ROOT / "manifests" / "validation.json").read_text())
    source = _source(video_id)
    if source is None:
        raise FileNotFoundError(video_id)
    start, end = _map_window(manifest, video_id)
    variants = {}
    for fps in [float(value) for value in densities.split(",")]:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            images = work / "images"; images.mkdir()
            paths = _extract(source, start, end, fps, images)
            pose_path = work / "poses.txt"
            command = [
                "python", "/opt/vggt-slam/main.py", "--image_folder", str(images),
                "--submap_size", "16", "--max_loops", "1", "--log_results",
                "--skip_dense_log", "--log_path", str(pose_path),
            ]
            completed = subprocess.run(command, cwd="/opt/vggt-slam", check=False,
                                       text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=5 * 60 * 60)
            if completed.returncode:
                raise RuntimeError("VGGT-SLAM failed:\n" +
                                   "\n".join(completed.stdout.splitlines()[-80:]))
            poses = _read_poses(pose_path)
            variants[f"{fps:g}fps"] = {
                "input_frames": len(paths), "selected_keyframes": len(poses),
                "poses": poses, "loop_log_mentions": completed.stdout.count("Loop closure"),
                "stdout_tail": completed.stdout.splitlines()[-30:],
            }
    result = {
        "method": "official VGGT-SLAM 2.0 cross-density",
        "upstream_commits": {"vggt_slam": SLAM_COMMIT, "vggt_spark": VGGT_COMMIT,
                             "salad": SALAD_COMMIT},
        "video_id": video_id, "start_seconds": start, "end_seconds": end,
        "ground_truth_used": False, "variants": variants,
    }
    output = ROOT / "experiments" / "vggt-slam2" / f"{video_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main(video_id: str = "G0fWvIxy88A", densities: str = "1,2") -> None:
    result = run.remote(video_id, densities)
    print(json.dumps({name: {"input_frames": row["input_frames"],
                             "selected_keyframes": row["selected_keyframes"],
                             "loop_log_mentions": row["loop_log_mentions"]}
                      for name, row in result["variants"].items()}, indent=2))
