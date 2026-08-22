"""Merge the answer-blind marker-guided map lane into a fixed baseline."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--rows',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    base=json.loads(a.base.read_text()); rows=json.loads(a.rows.read_text())['rows']; manifest=json.loads(a.manifest.read_text())
    changed=0
    for r in rows:
        if r.get('variant')!='guided' or not isinstance(r.get('prediction'),int): continue
        q=next((x for x in base.get(r['video_id'],[]) if str(x['id'])==str(r['question_id'])),None)
        m=manifest.get(r['video_id'],{}).get(str(r['question_id']),{})
        if q is None or not m.get('image_paths'): continue
        if q['answer_id'] != r['prediction']:
            q['answer_id']=r['prediction']; changed+=1
        q['answer']=m['options'][r['prediction']]
    a.output.write_text(json.dumps(base,indent=2)+'\n')
    print(json.dumps({'changed':changed,'questions':sum(len(x) for x in base.values()),'output':str(a.output)}))
if __name__=='__main__': main()
