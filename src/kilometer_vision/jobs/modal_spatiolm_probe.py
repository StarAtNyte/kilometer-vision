"""Bounded SpatioLM spatial-reasoning probe."""
from __future__ import annotations
import io, json, zipfile
from pathlib import Path
import modal

ROOT=Path('/data/2707-kilometer-vision'); SRC=Path('/data/shared/walking-tours')
VOL=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
IMAGE=(modal.Image.debian_slim(python_version='3.11').apt_install('git','ffmpeg','libgl1','libglib2.0-0').pip_install('torch==2.6.0','torchvision==0.21.0','transformers==4.56.1','accelerate>=1.10,<2','timm','einops','sentencepiece','pillow','opencv-python-headless','addict').run_commands('git clone --depth 1 https://github.com/xiaomi-research/spatio-lm.git /opt/spatio-lm','pip install -e /opt/spatio-lm'))
APP=modal.App('kilometervision-spatiolm-probe'); CHECKPOINT='xiaomi-research/SpatioLM-Understanding-InternVL3.5'

def _source(vid):
    for p in sorted((SRC/vid).iterdir()):
        if p.suffix.lower() in {'.mp4','.mkv','.mov','.avi','.webm'} and '.partial' not in p.stem:return p
    raise FileNotFoundError(vid)

@APP.function(image=IMAGE,gpu='A100',memory=48000,volumes={'/data':VOL},timeout=60*60)
def run(video_id='4MYTCny_FZs',question_id='0',split='validation',frame_count=8):
    import cv2,torch
    frame_count=int(frame_count)
    from PIL import Image,ImageDraw
    from torchvision import transforms
    from transformers import AutoTokenizer
    from spatiolm.models import InternVL3RChatModel
    VOL.reload(); manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); q=manifest[video_id][str(question_id)]
    model=InternVL3RChatModel.from_pretrained(CHECKPOINT,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True,cache_dir=str(ROOT/'checkpoints'/'huggingface')).eval().cuda()
    tokenizer=AutoTokenizer.from_pretrained(CHECKPOINT,trust_remote_code=True,use_fast=False,cache_dir=str(ROOT/'checkpoints'/'huggingface'))
    cap=cv2.VideoCapture(str(_source(video_id))); start=float(q.get('start_seconds',0)); end=float(q.get('end_seconds',start+60)); frames=[]
    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_MSEC,(start+(end-start)*i/max(frame_count-1,1))*1000); ok,a=cap.read()
        if ok: frames.append(Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)).resize((448,252)))
    cap.release(); panel=Image.new('RGB',(896,756),'white'); draw=ImageDraw.Draw(panel)
    for i,im in enumerate(frames[:8]):
        x,y=(i%2)*448,(i//2)*189; panel.paste(im.resize((448,189)),(x,y)); draw.text((x+5,y+5),f'VIDEO {i+1}',fill='red')
    if q.get('image_paths'):
        with zipfile.ZipFile(ROOT/'manifests'/f'{split}_images.zip') as z:
            for i,ref in enumerate(q['image_paths'][:5]):
                hit=[n for n in z.namelist() if Path(n).name==Path(str(ref)).name]
                if hit:
                    with Image.open(io.BytesIO(z.read(hit[0]))) as im: im=im.convert('RGB').resize((179,179)); panel.paste(im,(i*179,567)); draw.text((i*179+5,572),'ABCDE'[i],fill='red')
    tf=transforms.Compose([transforms.Resize((448,448)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))]); pixels=tf(panel).unsqueeze(0).to('cuda',dtype=torch.bfloat16)
    prompt=q['question']+'\nOptions:\n'+'\n'.join(f"{'ABCDE'[i]}. {x}" for i,x in enumerate(q['options']))+'\nReturn exactly one final option letter A, B, C, D, or E.'
    with torch.inference_mode(): answer=model.chat(tokenizer,pixels,prompt,{'max_new_tokens':128,'do_sample':False})
    result={'video_id':video_id,'question_id':str(question_id),'model':CHECKPOINT,'raw_response':answer,'ground_truth_used_for_inference':False}; out=ROOT/'experiments'/'spatiolm-probe'/f'{split}-{video_id}-{question_id}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); VOL.commit(); return result

@APP.local_entrypoint()
def main(video_id='4MYTCny_FZs',question_id='0',split='validation',frame_count=8): print(json.dumps(run.remote(video_id,question_id,split,frame_count),indent=2))
