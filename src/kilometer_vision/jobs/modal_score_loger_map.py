from __future__ import annotations
import json
from pathlib import Path
import modal
from kilometer_vision.map_solver import ordered_trace_candidate_scores
ROOT=Path('/data/2707-kilometer-vision'); V=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
I=modal.Image.debian_slim(python_version='3.11').pip_install('opencv-python-headless','numpy').add_local_python_source('kilometer_vision')
A=modal.App('kilometervision-score-loger-map')
@A.function(image=I,volumes={'/data':V})
def run(video_id='__all__',split='validation',frame_count=60):
 V.reload(); m=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); arch=ROOT/'manifests'/f'{split}_images.zip'; rows=[]
 targets=sorted(m) if video_id=='__all__' else [video_id]
 shared=ROOT/'experiments'/'loger-motion'/split/f'-n{frame_count}.json'
 if video_id=='__all__' and shared.exists():
  shared_rows=json.loads(shared.read_text()).get('records',[])
  grouped={vid:[] for vid in targets}
  for item in shared_rows: grouped.setdefault(item['video_id'],[]).append(item)
 else: grouped={vid: json.loads((ROOT/'experiments'/'loger-motion'/split/f'{vid}-n{frame_count}.json').read_text()).get('records',[]) for vid in targets if (ROOT/'experiments'/'loger-motion'/split/f'{vid}-n{frame_count}.json').exists()}
 for vid in targets:
  d={'records':grouped.get(vid,[])}
  for row in d['records']:
   q=m[vid][str(row['question_id'])]; scores=ordered_trace_candidate_scores(q,row['ground_trace'],arch,reflection_penalty=0.0); order=sorted(range(len(scores)),key=lambda i:float(scores[i]['score'])); rows.append({'video_id':vid,'question_id':row['question_id'],'prediction':order[0],'ranking':order,'answer_id':q.get('answer_id'),'margin':float(scores[order[1]]['score'])-float(scores[order[0]]['score'])})
 out=ROOT/'experiments'/'loger-motion'/split/f'map-score-n{frame_count}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'split':split,'records':rows,'ground_truth_used_for_inference':False},indent=2)+'\n'); V.commit(); return rows
@A.local_entrypoint()
def main(video_id: str='__all__',split: str='validation',frame_count: int=60): print(json.dumps(run.remote(video_id,split,frame_count),indent=2))
