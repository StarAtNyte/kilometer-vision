"""Optional learned geometry lane.

Runs Depth Anything 3 on bounded frame windows and persists only geometry summaries
and camera poses to the Modal Volume. It never copies source videos locally.
"""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from typing import Any
import modal

ROOT = Path('/data/2707-kilometer-vision')
VOLUME = modal.Volume.from_name('perception-test-2026-data', create_if_missing=True)
IMAGE = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('git', 'ffmpeg', 'libgl1', 'libglib2.0-0')
         .pip_install('git+https://github.com/ByteDance-Seed/Depth-Anything-3.git')
         .add_local_python_source('geometry_backend'))
app = modal.App('kilometervision-geometry')


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


@app.function(image=IMAGE, gpu='A100-80GB', volumes={'/data': VOLUME}, timeout=3600)
def run_da3(video_id: str, checkpoint: str = 'depth-anything/DA3-LARGE', window: int = 16) -> dict[str, Any]:
    import torch
    from depth_anything_3.api import DepthAnything3
    targets = sorted(p.stem for p in (ROOT / 'frames').glob('*.json') if not p.name.endswith('.features.json') and not p.name.endswith('.partial')) if video_id == '__all__' else [video_id]
    device = torch.device('cuda')
    model = DepthAnything3.from_pretrained(checkpoint).to(device=device)
    results = []
    for target in targets:
        manifest_path = ROOT / 'frames' / f'{target}.json'
        if not manifest_path.exists():
            results.append({'video_id': target, 'status': 'missing_manifest'})
            continue
        manifest = json.loads(manifest_path.read_text())
        frames = [frame for frame in manifest.get('frames', []) if frame.get('path')]
        records = []
        for start in range(0, len(frames), window):
            chunk = frames[start:start + window]
            images = [str((manifest_path.parent / frame['path']).resolve()) for frame in chunk]
            prediction = model.inference(images)
            extrinsics = prediction.extrinsics.tolist() if prediction.extrinsics is not None else [None] * len(chunk)
            intrinsics = prediction.intrinsics.tolist() if prediction.intrinsics is not None else [None] * len(chunk)
            depth = prediction.depth
            confidence = getattr(prediction, 'conf', None)
            for i, frame in enumerate(chunk):
                values = depth[i]
                if hasattr(values, 'detach'):
                    values = values.detach().float().cpu().numpy()
                values = values.reshape(-1)
                conf_value = None
                if confidence is not None:
                    conf = confidence[i]
                    if hasattr(conf, 'detach'):
                        conf = conf.detach().float().cpu().numpy()
                    conf_value = float(conf.mean())
                records.append({'index': frame['index'], 'timestamp_seconds': frame['timestamp_seconds'], 'extrinsic': extrinsics[i], 'intrinsic': intrinsics[i], 'depth': {'min': float(values.min()), 'median': float(__import__('numpy').median(values)), 'max': float(values.max()), 'metric': 'METRIC' in checkpoint.upper()}, 'confidence': conf_value, 'source': 'DA3'})
            del prediction
            torch.cuda.empty_cache()
        output = ROOT / 'geometry' / f'{target}.da3.json'
        atomic_json(output, {'schema_version': 1, 'video_id': target, 'backend': 'da3', 'checkpoint': checkpoint, 'status': 'available', 'frames': records})
        results.append({'video_id': target, 'output': str(output), 'frames': len(records), 'backend': 'da3'})
        VOLUME.commit()
    return results[0] if video_id != '__all__' else {'backend': 'da3', 'videos': results}


@app.local_entrypoint()
def main(video_id: str, checkpoint: str = 'depth-anything/DA3-LARGE', window: int = 16):
    print(json.dumps(run_da3.remote(video_id, checkpoint, window), indent=2))
