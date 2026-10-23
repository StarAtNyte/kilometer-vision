"""Test endpoint-anchored trajectory alignment against centered Procrustes."""
from __future__ import annotations
import json, math, zipfile
from pathlib import Path
import modal
from kilometer_vision.map_solver import route_marker_positions, route_path, orient_path_from_start, resample_polyline, curvature_sequence, dtw_distance
from kilometer_vision.question_planner import classify_question_facets

ROOT=Path('/data/2707-kilometer-vision'); VOLUME=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
IMAGE=modal.Image.debian_slim(python_version='3.11').pip_install('opencv-python-headless','numpy').add_local_python_source('kilometer_vision')
APP=modal.App('kilometervision-map-endpoint-alignment'); VARIANT='fps1-focal0.9-w256-b256-p16'

def anchored(points,reflect=False):
 import numpy as np
 v=resample_polyline(points); v=v-v[0]; end=v[-1]; scale=float(np.linalg.norm(end))
 if scale<1e-9:return np.zeros_like(v)
 angle=math.atan2(float(end[1]),float(end[0])); c,s=math.cos(-angle),math.sin(-angle); r=np.array([[c,-s],[s,c]]); v=v@r.T/scale
 if reflect:v[:,1]*=-1
 return v
def score(left,right,reflect=False):
 import numpy as np
 a=anchored(left); b=anchored(right,reflect); point=float(np.sqrt(np.mean(np.sum((a-b)**2,axis=1))))
 turns=[]
 for stride in (2,4,8): turns.append(dtw_distance(curvature_sequence(left,stride),curvature_sequence(right,stride)))
 return point+0.35*float(np.mean(turns))

@APP.function(image=IMAGE,volumes={'/data':VOLUME},timeout=60*60)
def run(split='validation'):
 VOLUME.reload(); manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); archive_path=ROOT/'manifests'/f'{split}_images.zip'; rows=[]
 with zipfile.ZipFile(archive_path) as a:
  for vid,qs in manifest.items():
   mp=ROOT/'experiments'/'dpvo-motion'/split/VARIANT/f'{vid}.json'
   if not mp.exists():continue
   motion=json.loads(mp.read_text()); motion=motion.get(vid,motion)
   for qid,q in qs.items():
    if 'map' not in classify_question_facets(q) or len(q.get('image_paths',[]))!=5:continue
    trace=motion.get(str(qid),{}).get('trajectory_2d',[]); 
    if len(trace)<4:continue
    paths=[]
    for ref in q['image_paths']:
     hit=[n for n in a.namelist() if Path(n).name==Path(str(ref)).name]; raw=a.read(hit[0]); marks=route_marker_positions(raw); paths.append(orient_path_from_start(route_path(raw),marks))
    for name,mirror in (('anchored-direct',False),('anchored-mirror',True)):
     ss=[score(trace,p,mirror) for p in paths]; order=sorted(range(5),key=ss.__getitem__); rows.append({'video_id':vid,'question_id':str(qid),'variant':name,'prediction':order[0],'ranking':order,'scores':ss,'answer_id':q.get('answer_id'),'ground_truth_used_for_inference':False})
 summary={n:{'questions':len([r for r in rows if r['variant']==n]),'correct':sum(r['prediction']==r['answer_id'] for r in rows if r['variant']==n)} for n in ('anchored-direct','anchored-mirror')}; out=ROOT/'experiments'/'map-endpoint-alignment'/f'{split}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'split':split,'summary':summary,'rows':rows,'ground_truth_used_for_inference':False},indent=2)+'\n'); VOLUME.commit(); return {'summary':summary,'artifact':str(out)}

@APP.local_entrypoint()
def main(split='validation'):print(run.remote(split))
