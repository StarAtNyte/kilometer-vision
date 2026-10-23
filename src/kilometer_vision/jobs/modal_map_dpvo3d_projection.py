"""Compare yaw-integrated 2-D DPVO against raw 3-D position projections."""
from __future__ import annotations
import json, zipfile
from pathlib import Path
import modal
from kilometer_vision.map_solver import ordered_trace_candidate_scores
from kilometer_vision.question_planner import classify_question_facets

ROOT=Path('/data/2707-kilometer-vision'); VOLUME=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
IMAGE=modal.Image.debian_slim(python_version='3.11').pip_install('opencv-python-headless','numpy').add_local_python_source('kilometer_vision')
APP=modal.App('kilometervision-map-dpvo3d-projection'); VARIANT='fps1-focal0.9-w256-b256-p16'

@APP.function(image=IMAGE,volumes={'/data':VOLUME},timeout=60*60)
def run(split='validation'):
 VOLUME.reload(); manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); archive=ROOT/'manifests'/f'{split}_images.zip'; rows=[]; projections={'yaw':None,'xz':(0,2),'xy':(0,1),'yz':(1,2)}
 with zipfile.ZipFile(archive) as a:
  for vid,qs in manifest.items():
   mp=ROOT/'experiments'/'dpvo-motion'/split/VARIANT/f'{vid}.json'
   if not mp.exists():continue
   motion=json.loads(mp.read_text()); motion=motion.get(vid,motion)
   for qid,q in qs.items():
    if 'map' not in classify_question_facets(q) or len(q.get('image_paths',[]))!=5:continue
    m=motion.get(str(qid),{}); base=m.get('trajectory_2d',[]); pos=m.get('positions_3d',[])
    for name,axes in projections.items():
     trace=base if axes is None else [[float(p[axes[0]]),float(p[axes[1]])] for p in pos if len(p)>=3]
     if len(trace)<4:continue
     scores=ordered_trace_candidate_scores(q,trace,archive,reflection_penalty=0.0); order=sorted(range(5),key=lambda i:float(scores[i]['score'])); rows.append({'video_id':vid,'question_id':str(qid),'variant':name,'prediction':order[0],'ranking':order,'margin':float(scores[order[1]]['score']-scores[order[0]]['score']),'answer_id':q.get('answer_id'),'ground_truth_used_for_inference':False})
 summary={n:{'questions':len([r for r in rows if r['variant']==n]),'correct':sum(r['prediction']==r['answer_id'] for r in rows if r['variant']==n)} for n in projections}; out=ROOT/'experiments'/'map-dpvo3d-projection'/f'{split}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'split':split,'summary':summary,'rows':rows,'ground_truth_used_for_inference':False},indent=2)+'\n'); VOLUME.commit(); return {'summary':summary,'artifact':str(out)}

@APP.local_entrypoint()
def main(split='validation'):print(run.remote(split))
