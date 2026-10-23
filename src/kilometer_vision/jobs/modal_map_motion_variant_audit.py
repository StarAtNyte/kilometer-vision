"""Audit available DPVO motion variants on validation map questions."""
from __future__ import annotations
import json
from pathlib import Path
import modal
from kilometer_vision.map_solver import ordered_trace_candidate_scores
from kilometer_vision.question_planner import classify_question_facets

ROOT = Path('/data/2707-kilometer-vision')
VOLUME = modal.Volume.from_name('perception-test-2026-data', create_if_missing=True)
IMAGE = modal.Image.debian_slim(python_version='3.11').pip_install('opencv-python-headless', 'numpy').add_local_python_source('kilometer_vision')
APP = modal.App('kilometervision-map-motion-variant-audit')
VARIANTS = ['fps5-focal0.9', 'fps5-focal0.9-w320-b512', 'fps5-focal0.9-w320-b256', 'fps5-focal0.9-w320-b64', 'fps2-focal0.9-w256-b256-p24', 'fps2-focal0.9-w256-b256-p8', 'fps1-focal0.9-w256-b256-p16', 'fps0.5-focal0.9-w256-b256-p16', 'fps1.5-focal0.9-w256-b256-p16', 'fps1-focal0.75-w256-b256-p32', 'fps1-focal0.75-w256-b256-p16']

@APP.function(image=IMAGE, volumes={'/data': VOLUME}, timeout=60*60)
def run(split='validation'):
    VOLUME.reload(); manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); archive=ROOT/'manifests'/f'{split}_images.zip'
    rows=[]
    for vid, questions in manifest.items():
        for qid,q in questions.items():
            if 'map' not in classify_question_facets(q): continue
            for variant in VARIANTS:
                path=ROOT/'experiments'/'dpvo-motion'/split/variant/f'{vid}.json'
                if not path.exists(): continue
                motions=json.loads(path.read_text()); trace=motions.get(str(qid),{}).get('trajectory_2d',[])
                if len(trace)<4: continue
                scores=ordered_trace_candidate_scores(q,trace,archive,reflection_penalty=0.0)
                order=sorted(range(len(scores)),key=lambda i:float(scores[i]['score']))
                rows.append({'video_id':vid,'question_id':str(qid),'variant':variant,'prediction':order[0],'ranking':order,'margin':float(scores[order[1]]['score']-scores[order[0]]['score']),'answer_id':q.get('answer_id')})
    summary={}
    for v in VARIANTS:
        x=[r for r in rows if r['variant']==v]; summary[v]={'questions':len(x),'correct':sum(r['prediction']==r['answer_id'] for r in x)}
    out=ROOT/'experiments'/'map-motion-variant-audit'/f'{split}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'split':split,'summary':summary,'rows':rows,'ground_truth_used_for_inference':False},indent=2)+'\n'); VOLUME.commit(); return {'summary':summary,'artifact':str(out)}

@APP.local_entrypoint()
def main(split='validation'): print(run.remote(split))
