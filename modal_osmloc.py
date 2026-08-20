"""Bounded OSMLoc feasibility experiments on KilometerVision route regions."""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal


ROOT = Path("/data/2707-kilometer-vision")
VOLUME = modal.Volume.from_name("perception-test-2026-data", create_if_missing=True)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libspatialindex-dev", "libgl1", "libglib2.0-0")
    .run_commands(
        "pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121",
        "pip install numpy==1.26.4 opencv-python-headless pillow scipy omegaconf hydra-core==1.3.2 pytorch-lightning==1.9.5 torchmetrics==0.11.4 matplotlib plotly tqdm lxml rtree scikit-learn geopy exifread urllib3 fvcore",
        "git clone --depth 1 https://github.com/WHU-USI3DV/OSMLoc.git /opt/OSMLoc",
    )
)
app = modal.App("kilometervision-osmloc")


@app.function(image=IMAGE, volumes={"/data": VOLUME}, gpu="A100-80GB", timeout=30 * 60)
def smoke_load() -> dict:
    import sys
    import torch

    os.chdir("/opt/OSMLoc")
    sys.path.insert(0, "/opt/OSMLoc")
    from maploc.module import GenericModule

    checkpoint = ROOT / "checkpoints" / "osmloc" / "loca_polar_small.ckpt"
    module = GenericModule.load_from_checkpoint(
        checkpoint, map_location="cpu",
        cfg={"model": {"name": "osmloc", "image_encoder": {"val": True}}},
    )
    module.eval().cuda()
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


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(smoke_load.remote(), indent=2))
