import json
from pathlib import Path
import modal
V=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
A=modal.App('kilometervision-read-spatialaxiom')
I=modal.Image.debian_slim(python_version='3.11')
@A.function(image=I,volumes={'/data':V})
def read():
 V.reload(); root=Path('/data/2707-kilometer-vision/experiments/spatialaxiom-map-probe')
 return [json.loads(p.read_text()) for p in sorted(root.glob('*.json'))]
@A.local_entrypoint()
def main(): print(json.dumps(read.remote(),indent=2))
