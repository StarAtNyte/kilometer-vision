import json
from pathlib import Path
import modal
V=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
A=modal.App('kilometervision-read-map-validation')
I=modal.Image.debian_slim(python_version='3.11')
@A.function(image=I,volumes={'/data':V})
def read():
 V.reload(); p=Path('/data/2707-kilometer-vision/experiments/map-ordered-geometry/validation.json')
 d=json.loads(p.read_text()); return [{'video_id':r['video_id'],'question_id':r['question_id'],'p':r.get('dpvo_predictions'), 'g':r.get('gemini_gated_prediction'),'gm':r.get('gemini_duel_prediction'),'conf':r.get('gemini_duel_confidence')} for r in d['records']]
@A.local_entrypoint()
def main(): print(json.dumps(read.remote(),indent=2))
