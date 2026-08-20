"""Bounded SpatialAxiom-9B map-question probe."""
from __future__ import annotations
import io, json, zipfile
from pathlib import Path
import modal
ROOT=Path('/data/2707-kilometer-vision'); SRC=Path('/data/shared/walking-tours')
V=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
I=(modal.Image.debian_slim(python_version='3.11').pip_install('torch','torchvision','transformers','accelerate','pillow','opencv-python-headless')
   .add_local_python_source('question_planner'))
A=modal.App('kilometervision-spatialaxiom-probe')
CHECKPOINT='SpatialAxiom/SpatialAxiom-9B'
def _source(vid):
 for p in sorted((SRC/vid).iterdir()):
  if p.suffix.lower() in {'.mp4','.mkv','.mov','.avi','.webm'} and '.partial' not in p.stem:return p
 raise FileNotFoundError(vid)
@A.function(image=I,gpu='H100',memory=32768,volumes={'/data':V},timeout=60*60)
def run(video_id='buBU7NrfyGM',question_id='0',split='validation',frame_count=8,candidate_blind=False):
 import cv2, torch
 from PIL import Image
 try:
  from transformers import AutoModelForMultimodalLM as Model, AutoProcessor
 except ImportError:
  from transformers import AutoModelForImageTextToText as Model, AutoProcessor
 V.reload(); osroot=ROOT/'checkpoints'/'huggingface'; osroot.mkdir(parents=True,exist_ok=True)
 model=Model.from_pretrained(CHECKPOINT,dtype=torch.bfloat16,device_map='auto',low_cpu_mem_usage=True,cache_dir=str(osroot))
 proc=AutoProcessor.from_pretrained(CHECKPOINT,cache_dir=str(osroot))
 q=json.loads((ROOT/'manifests'/f'{split}.json').read_text())[video_id][str(question_id)]
 cap=cv2.VideoCapture(str(_source(video_id))); n=max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),1); frames=[]
 for idx in [round(i*(n-1)/max(frame_count-1,1)) for i in range(frame_count)]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,arr=cap.read()
  if ok: frames.append(Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)).resize((448,252)))
 cap.release()
 parts=[]
 for i,im in enumerate(frames): parts += [{'type':'text','text':f'Ordered walking frame {i+1}/{len(frames)}:'},{'type':'image','image':im}]
 if not candidate_blind:
  with zipfile.ZipFile(ROOT/'manifests'/f'{split}_images.zip') as z:
   names=z.namelist()
   for i,ref in enumerate(q['image_paths']):
    hit=[x for x in names if Path(x).name==Path(str(ref)).name]
    im=Image.open(io.BytesIO(z.read(hit[0]))).convert('RGB').resize((448,448))
    parts += [{'type':'text','text':f'Candidate map {"ABCDE"[i]}:'},{'type':'image','image':im}]
  prompt=('Which candidate map (A-E) matches the walking route? The red route runs from S to E. '
          'Use the ordered video frames, route topology, endpoint direction, turns, and bends. '
          'Ignore map scale and absolute orientation. Your FIRST line must be exactly ANSWER: A, '
          'ANSWER: B, ANSWER: C, ANSWER: D, or ANSWER: E. Then give at most two short sentences.')
 else:
  prompt=('Extract a candidate-blind route record from these ordered walking frames. Report only visible '
          'city/area, readable street or landmark names, start and end surroundings, route sequence, '
          'turns, and compass displacement. Do not guess. Use compact bullet points and stop after 200 tokens.')
 parts.append({'type':'text','text':prompt})
 messages=[{'role':'user','content':parts}]
 inputs=proc.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt',enable_thinking=False).to(model.device)
 with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=1024,do_sample=False)
 text=proc.batch_decode(out[:,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)[0]
 result={'video_id':video_id,'question_id':str(question_id),'model':CHECKPOINT,'frame_count':frame_count,'raw_text':text,'ground_truth_used_for_inference':False}
 outp=ROOT/'experiments'/'spatialaxiom-map-probe'/f'{split}-{video_id}-{question_id}-f{frame_count}.json'; outp.parent.mkdir(parents=True,exist_ok=True); outp.write_text(json.dumps(result,indent=2)+'\n'); V.commit(); return result
@A.local_entrypoint()
def main(video_id: str='buBU7NrfyGM',question_id: str='0',split: str='validation',frame_count: int=8,candidate_blind: bool=False): print(json.dumps(run.remote(video_id,question_id,split,frame_count,candidate_blind),indent=2))
