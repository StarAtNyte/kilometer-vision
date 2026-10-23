"""Focused Qwen3-VL geometry-panel probe; no video/depth processing."""
from __future__ import annotations
import json, zipfile
from pathlib import Path
import modal
from kilometer_vision.map_solver import route_marker_positions, route_path, orient_path_from_start
from kilometer_vision.question_planner import classify_question_facets

ROOT=Path('/data/2707-kilometer-vision'); VOLUME=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11').pip_install('torch==2.7.1','torchvision==0.22.1','transformers>=4.57,<5','accelerate','qwen-vl-utils==0.0.14','pillow','opencv-python-headless','numpy==1.26.4').add_local_python_source('kilometer_vision')
APP=modal.App('kilometervision-qwen-map-panel-probe'); MODEL='Qwen/Qwen3-VL-32B-Instruct'; VARIANT='fps1-focal0.9-w256-b256-p16'

def norm(points,box):
 import numpy as np
 v=np.asarray(points,dtype='float64');
 if len(v)<2:return []
 v-=v.min(0,keepdims=True); ext=np.maximum(v.max(0),1e-6); scale=min((box[2]-box[0]-28)/ext[0],(box[3]-box[1]-28)/ext[1]); v*=scale; v += [(box[0]+box[2]-v[:,0].max())/2,(box[1]+box[3]-v[:,1].max())/2]; return [tuple(x) for x in v]
def panel(trace,candidates):
 from PIL import Image,ImageDraw
 c=Image.new('RGB',(1500,760),'white'); d=ImageDraw.Draw(c); d.text((20,15),'OBSERVED TRAJECTORY (blue to red)',fill='black'); o=norm(trace,(20,48,500,740))
 for i in range(1,len(o)):
  t=i/max(1,len(o)-1); d.line((o[i-1],o[i]),fill=(round(230*t),60,round(230*(1-t))),width=7)
 boxes=[(520,48,830,365),(845,48,1155,365),(1170,48,1480,365),(680,410,990,727),(1005,410,1315,727)]
 for (letter,path),b in zip(candidates,boxes):
  d.rectangle(b,outline=(180,180,180),width=2); d.text((b[0]+8,b[1]+8),f'CANDIDATE {letter} S -> E',fill='black'); line=norm(path,(b[0]+4,b[1]+30,b[2]-4,b[3]-4));
  if len(line)>=2:d.line(line,fill=(210,30,30),width=6)
 return c

@APP.function(image=IMAGE,gpu='H200',memory=65536,volumes={'/data':VOLUME},timeout=2*3600)
def run(split='validation'):
 import torch
 from transformers import AutoModelForImageTextToText,AutoProcessor
 from qwen_vl_utils import process_vision_info
 from PIL import Image
 os_cache=str(ROOT/'checkpoints'/'huggingface'); import os; os.environ['HF_HOME']=os_cache; VOLUME.reload()
 model=AutoModelForImageTextToText.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='auto',low_cpu_mem_usage=True,cache_dir=os_cache); proc=AutoProcessor.from_pretrained(MODEL,cache_dir=os_cache)
 manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); archive_path=ROOT/'manifests'/f'{split}_images.zip'; rec=[]
 with zipfile.ZipFile(archive_path) as a:
  for vid,qs in manifest.items():
   motion_path=ROOT/'experiments'/'dpvo-motion'/split/VARIANT/f'{vid}.json';
   if not motion_path.exists():continue
   motion=json.loads(motion_path.read_text())
   for qid,q in qs.items():
    if 'map' not in classify_question_facets(q) or len(q.get('image_paths',[]))!=5:continue
    trace=motion.get(str(qid),{}).get('trajectory_2d',[]); paths=[]
    for ref in q['image_paths']:
     hit=[n for n in a.namelist() if Path(n).name==Path(str(ref)).name]; raw=a.read(hit[0]); marks=route_marker_positions(raw); paths.append(orient_path_from_start(route_path(raw),marks))
    p=panel(trace,list(zip('ABCDE',paths))); content=[{'type':'text','text':'Choose the candidate map matching the ordered observed scale-free trajectory. Green=start, red=end; ignore absolute rotation, mirror, and scale. Compare turn order, topology, bends, and endpoint displacement. Reply with exactly one letter A-E.'},{'type':'image','image':p}]; msgs=[{'role':'user','content':content}]; rendered=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); imgs,vids=process_vision_info(msgs); inputs=proc(text=[rendered],images=imgs,videos=vids,padding=True,return_tensors='pt').to(model.device)
    with torch.inference_mode(): out=model.generate(**inputs,max_new_tokens=8,do_sample=False)
    raw_answer=proc.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip(); letter=next((x for x in raw_answer.upper() if x in 'ABCDE'),None); pred='ABCDE'.index(letter) if letter else None; rec.append({'video_id':vid,'question_id':str(qid),'prediction':pred,'raw_text':raw_answer,'answer_id':q.get('answer_id'),'ground_truth_used_for_inference':False}); del inputs,out; torch.cuda.empty_cache()
 result={'split':split,'model':MODEL,'variant':VARIANT,'records':rec,'ground_truth_used_for_inference':False}; out=ROOT/'experiments'/'qwen-map-panel-probe'/f'{split}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); VOLUME.commit(); return {'questions':len(rec),'correct':sum(x['prediction']==x['answer_id'] for x in rec),'artifact':str(out)}

@APP.local_entrypoint()
def main(split='validation'):print(run.remote(split))
