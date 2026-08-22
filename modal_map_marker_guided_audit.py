"""Audit marker-guided S-to-E skeleton extraction on map questions."""
from __future__ import annotations
import json, zipfile
from pathlib import Path
import modal
from map_solver import ordered_trace_candidate_scores
from question_planner import classify_question_facets

ROOT=Path('/data/2707-kilometer-vision'); VOLUME=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
IMAGE=modal.Image.debian_slim(python_version='3.11').pip_install('opencv-python-headless','numpy').add_local_python_source('map_solver','question_planner')
APP=modal.App('kilometervision-map-marker-guided-audit-v2'); VARIANT='fps1-focal0.9-w256-b256-p16'

@APP.function(image=IMAGE,volumes={'/data':VOLUME},timeout=60*60)
def run(split='validation'):
 VOLUME.reload(); manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); archive=ROOT/'manifests'/f'{split}_images.zip'; rows=[]
 with zipfile.ZipFile(archive) as a:
  for vid,qs in manifest.items():
   mp=ROOT/'experiments'/'dpvo-motion'/split/VARIANT/f'{vid}.json'
   if not mp.exists():continue
   motion=json.loads(mp.read_text()); motion=motion.get(vid,motion)
   for qid,q in qs.items():
    if 'map' not in classify_question_facets(q) or len(q.get('image_paths',[]))!=5:continue
    trace=motion.get(str(qid),{}).get('trajectory_2d',[])
    if len(trace)<4:continue
    specs=(('guided',{'marker_guided':True,'reflection_penalty':0.0}),('guided_frechet',{'marker_guided':True,'reflection_penalty':0.0,'frechet_weight':0.25}),('guided_chamfer',{'marker_guided':True,'reflection_penalty':0.0,'chamfer_weight':0.25})) if split=='validation' else (('guided',{'marker_guided':True,'reflection_penalty':0.0}),)
    for name,kwargs in specs:
     scores=ordered_trace_candidate_scores(q,trace,archive,**kwargs); order=sorted(range(5),key=lambda i:float(scores[i]['score'])); rows.append({'video_id':vid,'question_id':str(qid),'variant':name,'prediction':order[0],'ranking':order,'scores':scores,'answer_id':q.get('answer_id'),'ground_truth_used_for_inference':False})
 summary={v:{'questions':len([r for r in rows if r['variant']==v]),'correct':sum(r['prediction']==r['answer_id'] for r in rows if r['variant']==v)} for v in sorted({r['variant'] for r in rows})}; out=ROOT/'experiments'/'map-marker-guided-audit'/f'{split}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'split':split,'summary':summary,'rows':rows,'ground_truth_used_for_inference':False},indent=2)+'\n'); VOLUME.commit(); return {'summary':summary,'artifact':str(out)}

@APP.local_entrypoint()
def main(split='validation'):print(run.remote(split))
