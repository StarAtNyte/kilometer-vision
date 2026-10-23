"""Qwen3-VL RGB+geometry-panel control on validation maps (no depth)."""
from __future__ import annotations
import json, subprocess, tempfile, zipfile
from pathlib import Path
import modal
from kilometer_vision.map_solver import route_marker_positions, route_path, orient_path_from_start
from kilometer_vision.question_planner import classify_question_facets

ROOT=Path('/data/2707-kilometer-vision'); SRC=Path('/data/shared/walking-tours'); VOLUME=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
MODEL='Qwen/Qwen3-VL-32B-Instruct'; VARIANT='fps1-focal0.9-w256-b256-p16'; SUFFIXES={'.mkv','.mp4','.mov','.webm','.avi','.m4v'}
IMAGE=modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11').apt_install('ffmpeg').pip_install('torch==2.7.1','torchvision==0.22.1','transformers>=4.57,<5','accelerate','qwen-vl-utils==0.0.14','pillow','opencv-python-headless','numpy==1.26.4').add_local_python_source('kilometer_vision')
APP=modal.App('kilometervision-qwen-rgb-panel-probe')

def norm(points,b):
 import numpy as np
 v=np.asarray(points,dtype='float64')
 if len(v)<2:return []
 v-=v.min(0,keepdims=True); e=np.maximum(v.max(0),1e-6); s=min((b[2]-b[0]-28)/e[0],(b[3]-b[1]-28)/e[1]); v*=s; v += [(b[0]+b[2]-v[:,0].max())/2,(b[1]+b[3]-v[:,1].max())/2]; return [tuple(x) for x in v]
def panel(trace,paths):
 from PIL import Image,ImageDraw
 c=Image.new('RGB',(1500,760),'white'); d=ImageDraw.Draw(c); d.text((20,15),'OBSERVED TRAJECTORY blue to red',fill='black'); o=norm(trace,(20,48,500,740))
 for i in range(1,len(o)):
  t=i/max(1,len(o)-1); d.line((o[i-1],o[i]),fill=(round(230*t),60,round(230*(1-t))),width=7)
 boxes=[(520,48,830,365),(845,48,1155,365),(1170,48,1480,365),(680,410,990,727),(1005,410,1315,727)]
 for i,path in enumerate(paths):
  b=boxes[i]; d.rectangle(b,outline=(180,180,180),width=2); d.text((b[0]+8,b[1]+8),f'CANDIDATE {"ABCDE"[i]} S -> E',fill='black'); l=norm(path,(b[0]+4,b[1]+30,b[2]-4,b[3]-4))
  if len(l)>=2:d.line(l,fill=(210,30,30),width=6)
 return c
def source(video_id):
 directory=SRC/video_id
 return next(x for x in sorted(directory.iterdir()) if x.suffix.lower() in SUFFIXES and '.partial' not in x.stem)
def frame(video,t):
 from PIL import Image
 with tempfile.TemporaryDirectory() as td:
  out=Path(td)/'x.jpg'; subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(t),'-i',str(video),'-frames:v','1','-vf','scale=640:-2','-q:v','4',str(out)],check=True,timeout=60); return Image.open(out).convert('RGB').copy()

@APP.function(image=IMAGE,gpu='H200',memory=65536,volumes={'/data':VOLUME},timeout=2*3600)
def run(split='validation'):
 import os,torch
 from transformers import AutoModelForImageTextToText,AutoProcessor
 from qwen_vl_utils import process_vision_info
 os.environ['HF_HOME']=str(ROOT/'checkpoints'/'huggingface'); VOLUME.reload(); model=AutoModelForImageTextToText.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map='auto',low_cpu_mem_usage=True,cache_dir=os.environ['HF_HOME']); proc=AutoProcessor.from_pretrained(MODEL,cache_dir=os.environ['HF_HOME'])
 manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); archive_path=ROOT/'manifests'/f'{split}_images.zip'; rec=[]
 with zipfile.ZipFile(archive_path) as a:
  for vid,qs in manifest.items():
   mp=ROOT/'experiments'/'dpvo-motion'/split/VARIANT/f'{vid}.json'
   if not mp.exists():continue
   motion=json.loads(mp.read_text()); motion=motion.get(vid,motion); video=source(vid)
   for qid,q in qs.items():
    if 'map' not in classify_question_facets(q) or len(q.get('image_paths',[]))!=5:continue
    m=motion.get(str(qid),{}); trace=m.get('trajectory_2d',[]); ts=m.get('timestamps',[])
    if len(trace)<4 or len(ts)!=len(trace):continue
    paths=[]
    for ref in q['image_paths']:
     hit=[n for n in a.namelist() if Path(n).name==Path(str(ref)).name]; raw=a.read(hit[0]); marks=route_marker_positions(raw); paths.append(orient_path_from_start(route_path(raw),marks))
    p=panel(trace,paths); inds=[0,len(ts)//3,2*len(ts)//3,len(ts)-1]; fps=max(float(m.get('sample_fps',1)),1e-6); start=float(m.get('start_seconds',0)); rgb=[frame(video,start+float(ts[i])/fps) for i in inds]
    content=[{'type':'text','text':'Solve the walking-route map question. The panel shows an ordered scale-free observed trajectory (blue=start, red=end) and five normalized candidate routes. Ignore absolute scale/rotation/mirror; compare turn order, topology, bends, and endpoints. Use the ordered RGB walking frames as visual evidence. Reply with exactly one letter A-E.'},{'type':'image','image':p}]
    for i,im in enumerate(rgb):content += [{'type':'text','text':f'ordered walking frame {i+1}/4'},{'type':'image','image':im}]
    msgs=[{'role':'user','content':content}]; rendered=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); imgs,vids=process_vision_info(msgs); inputs=proc(text=[rendered],images=imgs,videos=vids,padding=True,return_tensors='pt').to(model.device)
    with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=8,do_sample=False)
    raw_answer=proc.batch_decode(out[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip(); letter=next((x for x in raw_answer.upper() if x in 'ABCDE'),None); rec.append({'video_id':vid,'question_id':str(qid),'prediction':'ABCDE'.index(letter) if letter else None,'raw_text':raw_answer,'answer_id':q.get('answer_id'),'ground_truth_used_for_inference':False}); del inputs,out,rgb; torch.cuda.empty_cache()
 result={'split':split,'model':MODEL,'variant':VARIANT,'frame_count':4,'records':rec,'ground_truth_used_for_inference':False}; out=ROOT/'experiments'/'qwen-rgb-panel-probe'/f'{split}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); VOLUME.commit(); return {'questions':len(rec),'correct':sum(x['prediction']==x['answer_id'] for x in rec),'artifact':str(out)}

@APP.local_entrypoint()
def main(split='validation'):print(run.remote(split))
