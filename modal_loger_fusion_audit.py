from __future__ import annotations
import json
from pathlib import Path
import modal
ROOT=Path('/data/2707-kilometer-vision'); V=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
I=modal.Image.debian_slim(python_version='3.11')
A=modal.App('kilometervision-loger-fusion-audit')
@A.function(image=I,volumes={'/data':V})
def run(split='validation',frame_count=30):
 V.reload(); base=json.loads((ROOT/'experiments'/'map-ordered-geometry'/f'{split}.json').read_text())['records']; loger=json.loads((ROOT/'experiments'/'loger-motion'/split/f'map-score-n{frame_count}.json').read_text())['records']; by={(r['video_id'],str(r['question_id'])):r for r in loger}; rows=[]
 for r in base:
  key=(r['video_id'],str(r['question_id'])); l=by.get(key,{}).get('prediction'); p=r.get('dpvo_predictions',{}).get('fps1-focal0.9-w256-b256-p16'); g=r.get('gemini_gated_prediction'); fused=l if l is not None and l!=p and l==g else p
  rows.append({'video_id':r['video_id'],'question_id':r['question_id'],'p16':p,'gemini_gate':g,'loger':l,'fused':fused,'answer_id':r['answer_id']})
 return {'rows':rows,'p16_correct':sum(x['p16']==x['answer_id'] for x in rows),'loger_correct':sum(x['loger']==x['answer_id'] for x in rows),'fused_correct':sum(x['fused']==x['answer_id'] for x in rows),'ground_truth_used_for_inference':False}
@A.local_entrypoint()
def main(split='validation',frame_count=30): print(json.dumps(run.remote(split,frame_count),indent=2))
