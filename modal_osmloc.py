"""Bounded OSMLoc feasibility experiments on KilometerVision route regions."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import modal


ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
SOURCE_ROOT = Path("/data/shared/walking-tours")
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git", "libspatialindex-dev", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121",
        "pip install numpy==1.26.4 opencv-python-headless pillow scipy omegaconf hydra-core==1.3.2 pytorch-lightning==1.9.5 torchmetrics==0.11.4 matplotlib plotly tqdm lxml rtree scikit-learn geopy exifread urllib3 fvcore",
        "git clone --depth 1 https://github.com/WHU-USI3DV/OSMLoc.git /opt/OSMLoc",
    )
    .pip_install("requests")
)
app = modal.App("kilometervision-osmloc")


def _source(video_id: str) -> Path:
    directory = SOURCE_ROOT / video_id
    source = next((path for path in sorted(directory.iterdir())
                   if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
                   and ".partial" not in path.stem), None)
    if source is None:
        raise FileNotFoundError(video_id)
    return source


def _load_model():
    import sys
    os.chdir("/opt/OSMLoc")
    sys.path.insert(0, "/opt/OSMLoc")
    from maploc.module import GenericModule

    checkpoint = ROOT / "checkpoints" / "osmloc" / "loca_polar_small.ckpt"
    module = GenericModule.load_from_checkpoint(
        checkpoint, map_location="cpu",
        cfg={"model": {"name": "osmloc", "image_encoder": {"val": True}}},
    )
    return module.eval().cuda(), checkpoint


@app.function(image=IMAGE, volumes={"/data": VOLUME}, gpu="A100-80GB", timeout=30 * 60)
def smoke_load() -> dict:
    import sys
    import torch

    module, checkpoint = _load_model()
    parameters = sum(parameter.numel() for parameter in module.parameters())
    result = {
        "checkpoint": str(checkpoint), "parameters": parameters,
        "device": torch.cuda.get_device_name(0), "loaded": True,
    }
    output = ROOT / "experiments" / "osmloc-smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.function(image=IMAGE, volumes={"/data": VOLUME}, gpu="A100-80GB", timeout=30 * 60)
def smoke_infer(video_id: str, timestamp: float, latitude: float, longitude: float) -> dict:
    """Run one frame against one 128 m OSM crop to validate the input adapter."""
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms as tvf
    from lightning_fabric.utilities.apply_func import move_data_to_device
    module, checkpoint = _load_model()
    from maploc.data.dataset import MapLocDataset
    from maploc.data.torch import collate
    from maploc.osm.tiling import TileManager
    from maploc.utils.geo import BoundaryBox, Projection
    from maploc.utils.wrappers import Camera
    from omegaconf import OmegaConf

    with tempfile.TemporaryDirectory() as tmp:
        frame_path = Path(tmp) / "frame.jpg"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(timestamp),
            "-i", str(_source(video_id)), "-frames:v", "1", "-q:v", "2", str(frame_path),
        ], check=True, timeout=180)
        image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    focal = 0.9 * width
    camera = Camera.from_dict({
        "model": "PINHOLE", "width": width, "height": height,
        "params": np.array([focal, focal, width / 2, height / 2]),
    }).float()
    dataset = MapLocDataset.__new__(MapLocDataset)
    dataset.cfg = OmegaConf.create({
        "rectify_pitch": True, "target_focal_length": None, "resize_image": 512,
        "pad_to_square": True, "pad_to_multiple": 32, "reduce_fov": None,
    })
    dataset.tfs = tvf.Compose([])
    image_tensor, valid, camera, _, _ = dataset.process_image(
        image, camera, 0.0, 0.0, seed=0, jitter=False,
    )

    projection = Projection(latitude, longitude)
    center = projection.project(np.array([latitude, longitude]))
    bbox = BoundaryBox(center - 64.0, center + 64.0)
    osm_cache = ROOT / "experiments" / "osmloc-osm" / f"{video_id}-{latitude:.5f}-{longitude:.5f}.json"
    osm_cache.parent.mkdir(parents=True, exist_ok=True)
    manager = TileManager.from_bbox(projection, bbox, ppm=2, path=osm_cache, tile_size=128)
    canvas = manager.query(bbox)
    sample = {
        "image": image_tensor, "valid": valid, "camera": camera,
        "map": torch.from_numpy(np.ascontiguousarray(canvas.raster)).long(),
    }
    batch = move_data_to_device(collate([sample]), "cuda")
    with torch.inference_mode():
        prediction = module(batch)
    uv = prediction["uv_max"][0].detach().cpu().numpy()
    xy = canvas.to_xy(uv)
    latlon = projection.unproject(xy)
    scores = prediction["scores"][0].detach().float().cpu()
    flat = scores.flatten()
    top = torch.topk(flat, 2).values
    result = {
        "video_id": video_id, "timestamp": timestamp,
        "crop_center_latlon": [latitude, longitude],
        "prediction_latlon": latlon.tolist(), "uv": uv.tolist(),
        "score_max": float(top[0]), "score_margin": float(top[0] - top[1]),
        "checkpoint": str(checkpoint), "adapter": "approx-focal-0.9w-zero-roll-pitch",
    }
    output = ROOT / "experiments" / "osmloc-smoke-infer.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.function(image=IMAGE, volumes={"/data": VOLUME}, gpu="A100-80GB", timeout=60 * 60)
def corridor_search(video_id: str, start: float, end: float,
                    anchor_latlons: list[list[float]], frame_count: int = 3,
                    focal_scale: float = 0.9) -> dict:
    """Search one metadata-bounded OSM corridor for several ordered video frames."""
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms as tvf
    from lightning_fabric.utilities.apply_func import move_data_to_device
    module, checkpoint = _load_model()
    from maploc.data.dataset import MapLocDataset
    from maploc.data.torch import collate
    from maploc.osm.reader import OSMData
    from maploc.osm.tiling import TileManager
    from maploc.utils.geo import BoundaryBox, Projection
    from maploc.utils.wrappers import Camera
    from omegaconf import OmegaConf

    points = np.asarray(anchor_latlons, dtype="float64")
    projection = Projection.from_points(points)
    xy = projection.project(points)
    bbox = BoundaryBox(xy.min(0) - 96.0, xy.max(0) + 96.0)
    osm_cache = ROOT / "experiments" / "osmloc-osm" / f"{video_id}-corridor.json"
    osm_cache.parent.mkdir(parents=True, exist_ok=True)
    if osm_cache.exists():
        cached_box = OSMData.from_file(osm_cache).box
        if cached_box is not None:
            bbox = bbox & projection.project(cached_box)
    manager = TileManager.from_bbox(projection, bbox, ppm=2, path=osm_cache, tile_size=128)
    canvas = manager.query(bbox)
    map_tensor = torch.from_numpy(np.ascontiguousarray(canvas.raster)).long()
    timestamps = np.linspace(start, end, frame_count).tolist()
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, timestamp in enumerate(timestamps):
            frame_path = Path(tmp) / f"{index:03d}.jpg"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(timestamp),
                "-i", str(_source(video_id)), "-frames:v", "1", "-q:v", "2", str(frame_path),
            ], check=True, timeout=180)
            image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            focal = focal_scale * width
            camera = Camera.from_dict({
                "model": "PINHOLE", "width": width, "height": height,
                "params": np.array([focal, focal, width / 2, height / 2]),
            }).float()
            dataset = MapLocDataset.__new__(MapLocDataset)
            dataset.cfg = OmegaConf.create({
                "rectify_pitch": True, "target_focal_length": None, "resize_image": 512,
                "pad_to_square": True, "pad_to_multiple": 32, "reduce_fov": None,
            })
            dataset.tfs = tvf.Compose([])
            image_tensor, valid, camera, _, _ = dataset.process_image(
                image, camera, 0.0, 0.0, seed=0, jitter=False,
            )
            batch = move_data_to_device(collate([{
                "image": image_tensor, "valid": valid, "camera": camera, "map": map_tensor,
            }]), "cuda")
            with torch.inference_mode():
                prediction = module(batch)
            scores = prediction["scores"][0].detach().float().cpu()
            spatial = scores.max(-1).values
            first_flat = int(spatial.argmax())
            first_v, first_u = np.unravel_index(first_flat, spatial.shape)
            suppressed = spatial.clone()
            radius = 32
            suppressed[max(0, first_v-radius):first_v+radius+1,
                       max(0, first_u-radius):first_u+radius+1] = -torch.inf
            second = float(suppressed.max())
            first = float(spatial[first_v, first_u])
            uv = np.array([first_u, first_v], dtype="float64")
            predicted_latlon = projection.unproject(canvas.to_xy(uv))
            records.append({
                "timestamp": timestamp, "prediction_latlon": predicted_latlon.tolist(),
                "spatial_score": first, "spatial_margin_16m": first - second,
            })
            del batch, prediction
            torch.cuda.empty_cache()

    result = {
        "video_id": video_id, "start": start, "end": end,
        "anchor_latlons": anchor_latlons, "corridor_bbox_latlon": [
            projection.unproject(bbox.min_).tolist(), projection.unproject(bbox.max_).tolist(),
        ], "map_shape": list(canvas.raster.shape), "records": records,
        "checkpoint": str(checkpoint), "focal_scale": focal_scale,
        "ground_truth_used": False,
    }
    output = ROOT / "experiments" / f"osmloc-corridor-{video_id}-f{focal_scale:.2f}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit()
    return result


@app.local_entrypoint()
def main(mode: str = "load", video_id: str = "G0fWvIxy88A", timestamp: float = 2680.0,
         latitude: float = 53.96230, longitude: float = -1.08190) -> None:
    if mode == "load":
        result = smoke_load.remote()
    elif mode == "infer":
        result = smoke_infer.remote(video_id, timestamp, latitude, longitude)
    elif mode == "corridor":
        # York reporting route: city center plus three public-description landmarks.
        anchors = [[53.9656579, -1.0743052], [53.9628119, -1.0856005],
                   [53.9623561, -1.0820834], [53.9558294, -1.0801796]]
        calls = [corridor_search.spawn(video_id, 2574.0, 2789.0, anchors, 3, scale)
                 for scale in (0.75, 0.9, 1.1)]
        result = [call.get() for call in calls]
    else:
        raise ValueError(mode)
    print(json.dumps(result, indent=2))
