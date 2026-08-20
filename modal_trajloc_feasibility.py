"""Label-free TrajLoc video-to-satellite feasibility experiment.

Builds a bounded public-imagery gallery around a metadata-derived center and
compares two temporal samplings plus a different-city negative.  It never
reads answer fields and does not generate challenge choices.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import modal

from question_planner import classify_question_facets


ROOT = Path("/data/2707-kilometer-vision")
SOURCE_ROOT = Path("/data/shared/walking-tours")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
TRAJLOC_COMMIT = "6ee45f1b157fc4de815d9790113745721b062c60"
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.10")
    .apt_install("ffmpeg", "git")
    .run_commands(
        "pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121",
        f"git clone https://github.com/mvrl/TrajLoc.git /opt/trajloc && cd /opt/trajloc && git checkout {TRAJLOC_COMMIT}",
        "pip install 'setuptools<81' wheel huggingface_hub pyyaml pillow tqdm matplotlib ftfy regex",
        "pip install --no-build-isolation git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1",
    )
    .add_local_python_source("question_planner")
)
app = modal.App("kilometervision-trajloc-feasibility")


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


def _tile_xy(latitude: float, longitude: float, zoom: int) -> tuple[int, int]:
    scale = 2 ** zoom
    x = int((longitude + 180.0) / 360.0 * scale)
    lat_rad = math.radians(max(-85.0511, min(85.0511, latitude)))
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * scale)
    return x, y


def _tile_center(x: int, y: int, zoom: int) -> tuple[float, float]:
    scale = 2 ** zoom
    lon = (x + 0.5) / scale * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / scale))))
    return lat, lon


def _extract(source: Path, start: float, end: float, directory: Path) -> list[Path]:
    # Fifteen uniform frames provide two interleaved seven-frame queries.
    fps = 14.0 / max(end - start, 1e-6)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                    "-t", str(end-start), "-i", str(source), "-vf", f"fps={fps},scale=640:-2",
                    "-frames:v", "15", "-q:v", "3", str(directory / "%03d.jpg")],
                   check=True, timeout=max(180, int(end-start)*2))
    paths = sorted(directory.glob("*.jpg"))
    if len(paths) < 14:
        raise RuntimeError(f"only {len(paths)} frames extracted")
    return paths


def _download_gallery(center: tuple[float, float], directory: Path, zoom: int = 16,
                      radius: int = 4) -> list[dict]:
    """Create overlapping 3x3 Esri World Imagery mosaics around tile centers."""
    import urllib.request
    from PIL import Image

    center_x, center_y = _tile_xy(*center, zoom)
    tile_dir = directory / "tiles"; mosaic_dir = directory / "mosaics"
    tile_dir.mkdir(parents=True); mosaic_dir.mkdir(parents=True)
    required = range(-radius-1, radius+2)
    for dy in required:
        for dx in required:
            x, y = center_x + dx, center_y + dy
            target = tile_dir / f"{x}-{y}.jpg"
            request = urllib.request.Request(
                f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}",
                headers={"User-Agent": "KilometerVision research feasibility/1.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                target.write_bytes(response.read())
    items = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            x, y = center_x + dx, center_y + dy
            canvas = Image.new("RGB", (768, 768))
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    with Image.open(tile_dir / f"{x+ox}-{y+oy}.jpg") as tile:
                        canvas.paste(tile.convert("RGB"), ((ox+1)*256, (oy+1)*256))
            path = mosaic_dir / f"{x}-{y}.jpg"; canvas.save(path, quality=92)
            latitude, longitude = _tile_center(x, y, zoom)
            items.append({"x": x, "y": y, "latitude": latitude,
                          "longitude": longitude, "path": path})
    return items


@app.function(image=IMAGE, volumes={"/data": VOLUME}, gpu="A100-80GB", memory=32768,
              timeout=60 * 60)
def run(target_video_id: str = "G0fWvIxy88A", negative_video_id: str = "4MYTCny_FZs",
        center_lat: float = 53.9656579, center_lon: float = -1.0743052) -> dict:
    import sys
    import torch
    import torch.nn.functional as functional
    from huggingface_hub import hf_hub_download
    from PIL import Image

    VOLUME.reload()
    sys.path.insert(0, "/opt/trajloc")
    from model_loader import load_model

    checkpoint = hf_hub_download("MVRL/TrajLoc", "trajloc_full.pth",
                                 cache_dir=ROOT / "checkpoints" / "trajloc")
    model, preprocess, _, _ = load_model("CLIP-L/14", expand_text=True, use_trajmod=True,
                                         use_sinusoidal=True)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    state = state.get("state_dict", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    # The released checkpoint predates an upstream attribute rename.  Preserve
    # strict loading while translating only that known module prefix.
    state = {key.replace("traj_aggregator.", "video_film.", 1)
             if key.startswith("traj_aggregator.") else key: value
             for key, value in state.items()}
    model.load_state_dict(state, strict=True, assign=True); model.cuda().eval()

    manifest = json.loads((ROOT / "manifests" / "validation.json").read_text())
    queries = {}
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        gallery = _download_gallery((center_lat, center_lon), work / "gallery")
        satellite_tensors = []
        for item in gallery:
            with Image.open(item["path"]) as image:
                tensor, _ = preprocess(image.convert("RGB"), None)
            satellite_tensors.append(tensor)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            satellite_features = model.encode_image(torch.stack(satellite_tensors).cuda())
        satellite_features = functional.normalize(satellite_features.float(), dim=-1)

        for video_id in (target_video_id, negative_video_id):
            source = _source(video_id)
            if source is None:
                raise FileNotFoundError(video_id)
            start, end = _map_window(manifest, video_id)
            frame_dir = work / video_id; frame_dir.mkdir()
            frames = _extract(source, start, end, frame_dir)
            variants = {"even": frames[0:14:2], "odd": frames[1:15:2]}
            for variant, paths in variants.items():
                images = []
                for path in paths:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                video, _ = preprocess(images, None)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    feature = model.encode_video(video.unsqueeze(0).cuda())
                feature = functional.normalize(feature.float(), dim=-1)
                scores = (feature @ satellite_features.T).squeeze(0)
                values, indices = scores.topk(10)
                matches = []
                for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), 1):
                    item = gallery[index]
                    matches.append({"rank": rank, "score": score, "x": item["x"], "y": item["y"],
                                    "latitude": item["latitude"], "longitude": item["longitude"]})
                queries[f"{video_id}-{variant}"] = {"start_seconds": start, "end_seconds": end,
                    "frame_count": len(paths), "matches": matches,
                    "score_mean": float(scores.mean()), "score_std": float(scores.std())}

    result = {"method": "TrajLoc video-only cross-view retrieval", "upstream_commit": TRAJLOC_COMMIT,
              "checkpoint": "MVRL/TrajLoc/trajloc_full.pth", "gallery_source": "Esri World Imagery",
              "gallery_center": [center_lat, center_lon], "gallery_zoom": 16, "gallery_size": 81,
              "target_video_id": target_video_id, "negative_video_id": negative_video_id,
              "ground_truth_used": False, "queries": queries}
    output = ROOT / "experiments" / "trajloc-feasibility" / f"{target_video_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n")
    VOLUME.commit(); return result


@app.local_entrypoint()
def main(target_video_id: str = "G0fWvIxy88A", negative_video_id: str = "4MYTCny_FZs",
         center_lat: float = 53.9656579, center_lon: float = -1.0743052) -> None:
    result = run.remote(target_video_id, negative_video_id, center_lat, center_lon)
    print(json.dumps({name: {"top1": row["matches"][0], "top10": [
        [match["x"], match["y"]] for match in row["matches"]]}
        for name, row in result["queries"].items()}, indent=2))
