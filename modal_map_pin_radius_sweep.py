"""Sweep candidate-map pin removal radii and path extraction modes."""
from __future__ import annotations
import json, zipfile
from pathlib import Path
import modal
from map_solver import route_marker_positions, route_path, orient_path_from_start, ordered_shape_distance
from question_planner import classify_question_facets

ROOT=Path('/data/2707-kilometer-vision'); VOLUME=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
IMAGE=modal.Image.debian_slim(python_version='3.11').pip_install('opencv-python-headless','numpy').add_local_python_source('map_solver','question_planner')
APP=modal.App('kilometervision-map-pin-radius-sweep'); VARIANT='fps1-focal0.9-w256-b256-p16'; RADII=(8,10,12,15,18,22,26)

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
    specs=[(r,g) for r in RADII for g in (False,True)] if split=='validation' else [(8,False),(12,False),(15,False)]
    for radius,guided in specs:
      paths=[]
      for ref in q['image_paths']:
       hit=[n for n in a.namelist() if Path(n).name==Path(str(ref)).name]; raw=a.read(hit[0]); marks=route_marker_positions(raw); paths.append(orient_path_from_start(route_path(raw,pin_radius=radius,marker_guided=guided),marks))
      scores=[ordered_shape_distance(trace,p,allow_reverse=not bool(route_marker_positions(a.read([n for n in a.namelist() if Path(n).name==Path(str(ref)).name][0]))),reflection_penalty=0.0) for ref,p in zip(q['image_paths'],paths)]
      order=sorted(range(5),key=lambda i:float(scores[i]['score'])); rows.append({'video_id':vid,'question_id':str(qid),'radius':radius,'guided':guided,'prediction':order[0],'ranking':order,'margin':float(scores[order[1]]['score']-scores[order[0]]['score']),'answer_id':q.get('answer_id'),'ground_truth_used_for_inference':False})
 summary={f"r{r}-g{int(g)}":{'questions':len([x for x in rows if x['radius']==r and x['guided']==g]),'correct':sum(x['prediction']==x['answer_id'] for x in rows if x['radius']==r and x['guided']==g)} for r,g in sorted({(x['radius'],x['guided']) for x in rows})}; out=ROOT/'experiments'/'map-pin-radius-sweep'/f'{split}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'split':split,'summary':summary,'rows':rows,'ground_truth_used_for_inference':False},indent=2)+'\n'); VOLUME.commit(); return {'summary':summary,'artifact':str(out)}

@APP.local_entrypoint()
def main(split='validation'):print(run.remote(split))
