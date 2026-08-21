from pathlib import Path
import json, modal
ROOT=Path('/data/2707-kilometer-vision'); V=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
I=modal.Image.debian_slim(python_version='3.11'); A=modal.App('kilometervision-list-windows')
@A.function(image=I,volumes={'/data':V})
def run(video_id='G0fWvIxy88A',split='test'):
 V.reload(); m=json.loads((ROOT/'manifests'/f'{split}.json').read_text())
 return [{'id':str(qid),'start':q.get('start_seconds'),'end':q.get('end_seconds'),'images':len(q.get('image_paths',[]))} for qid,q in m.get(video_id,{}).items() if 'image_paths' in q]
@A.local_entrypoint()
def main(video_id='G0fWvIxy88A',split='test'): print(json.dumps(run.remote(video_id,split),indent=2))
