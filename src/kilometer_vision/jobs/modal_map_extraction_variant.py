from __future__ import annotations
import json, zipfile
from pathlib import Path
import modal
from kilometer_vision.map_solver import motion_polyline, route_marker_positions, route_path, route_path_full_guided, route_path_connected_component, ordered_shape_distance
from kilometer_vision.question_planner import classify_question_facets
ROOT=Path('/data/2707-kilometer-vision'); V=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
I=modal.Image.debian_slim(python_version='3.11').pip_install('opencv-python-headless','numpy').add_local_python_source('kilometer_vision')
A=modal.App('kilometervision-map-extraction-variant')
@A.function(image=I,volumes={'/data':V},timeout=30*60)
def run(split='validation'):
 V.reload(); m=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); arch=ROOT/'manifests'/f'{split}_images.zip'; rows=[]
 with zipfile.ZipFile(arch) as z:
  names=z.namelist()
  for vid,qs in m.items():
   mp=ROOT/'experiments'/'dpvo-motion'/split/'fps1-focal0.9-w256-b256-p16'/f'{vid}.json'
   if not mp.exists(): continue
   motions=json.loads(mp.read_text())
   for qid,q in qs.items():
    if 'map' not in classify_question_facets(q): continue
    trace=motions.get(str(qid),{}).get('trajectory_2d',[])
    vals=[]
    for ref in q.get('image_paths',[]):
     hit=[n for n in names if Path(n).name==Path(str(ref)).name]
     raw=z.read(hit[0]) if len(hit)==1 else b''; markers=route_marker_positions(raw); path=route_path_connected_component(raw)
     vals.append(ordered_shape_distance(trace,path,allow_reverse=not bool(markers),reflection_penalty=.04,turn_weight=.35))
    pred=min(range(len(vals)),key=lambda i:float(vals[i]['score'])) if vals else None
    rows.append({'video_id':vid,'question_id':str(qid),'prediction':pred,'answer_id':q.get('answer_id'),'scores':vals,'variant':'connected-component'})
 out=ROOT/'experiments'/'map-extraction-variant'/f'{split}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'split':split,'records':rows,'ground_truth_used_for_inference':False},indent=2)+'\n'); V.commit(); return {'records':len(rows),'correct':sum(r['prediction']==r['answer_id'] for r in rows if r['answer_id'] is not None),'predictions':[(r['video_id'],r['prediction'],r['answer_id']) for r in rows],'artifact':str(out)}
@A.local_entrypoint()
def main(split='validation'): print(json.dumps(run.remote(split),indent=2))
