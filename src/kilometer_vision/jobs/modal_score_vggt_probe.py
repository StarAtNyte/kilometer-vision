from __future__ import annotations
import json, zipfile
from pathlib import Path
import modal
from kilometer_vision.map_solver import ordered_trace_candidate_scores

ROOT=Path('/data/2707-kilometer-vision')
V=modal.Volume.from_name('perception-test-2026-data', create_if_missing=True)
I=modal.Image.debian_slim(python_version='3.11').pip_install('opencv-python-headless','numpy').add_local_python_source('kilometer_vision')
A=modal.App('kilometervision-score-vggt-probe')

@A.function(image=I, volumes={'/data':V}, timeout=20*60)
def run(video_id='fGX0Te6pFvk', variant='global-frames12', split='test'):
 V.reload(); manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); archive=ROOT/'manifests'/f'{split}_images.zip'
 rows=[]
 for qid,q in manifest[video_id].items():
  if 'image_paths' not in q or 'map' not in q.get('question','').lower(): continue
  p=ROOT/'experiments'/'vggt-motion'/split/variant/f'{video_id}.json'
  payload=json.loads(p.read_text()); trace=payload.get(str(qid),{}).get('trajectory_2d',[])
  scores=ordered_trace_candidate_scores(q,trace,archive,reflection_penalty=0.0) if trace else []
  rows.append({'question_id':qid,'prediction':min(range(len(scores)),key=lambda i:float(scores[i]['score'])) if scores else None,'answer_id':q.get('answer_id'),'top2':sorted([(i,round(float(s['score']),5)) for i,s in enumerate(scores)],key=lambda x:x[1])[:2]})
 return rows

@A.local_entrypoint()
def main(video_id='fGX0Te6pFvk', variant='global-frames12', split='test'):
 print(json.dumps(run.remote(video_id,variant,split),indent=2))
