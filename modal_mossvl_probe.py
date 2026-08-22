"""Bounded MOSS-VL temporal-reasoning probe."""
from __future__ import annotations
import json, queue, threading, tempfile, zipfile
from pathlib import Path
import modal

ROOT=Path('/data/2707-kilometer-vision'); SRC=Path('/data/shared/walking-tours')
VOL=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
IMAGE=(modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg','libgl1','libglib2.0-0').pip_install('torch==2.8.0','torchvision==0.23.0','transformers==4.57.1','accelerate','einops','pillow','torchcodec==0.7.0','joblib'))
APP=modal.App('kilometervision-mossvl-probe'); CHECKPOINT='OpenMOSS-Team/MOSS-VL-Instruct-0708'

def _source(vid):
    for p in sorted((SRC/vid).iterdir()):
        if p.suffix.lower() in {'.mp4','.mkv','.mov','.avi','.webm'} and '.partial' not in p.stem:return p
    raise FileNotFoundError(vid)

@APP.function(image=IMAGE,gpu='A100',memory=48000,volumes={'/data':VOL},timeout=60*60)
def run(video_id='4MYTCny_FZs',question_id='0',split='validation',fps=1.0,max_frames=256):
    import torch
    from transformers import AutoModelForCausalLM,AutoProcessor
    VOL.reload(); q=json.loads((ROOT/'manifests'/f'{split}.json').read_text())[video_id][str(question_id)]
    processor=AutoProcessor.from_pretrained(CHECKPOINT,trust_remote_code=True,frame_extract_num_threads=1,cache_dir=str(ROOT/'checkpoints'/'huggingface'))
    model=AutoModelForCausalLM.from_pretrained(CHECKPOINT,trust_remote_code=True,device_map='auto',torch_dtype=torch.bfloat16,attn_implementation='sdpa',cache_dir=str(ROOT/'checkpoints'/'huggingface')).eval()
    prompt=('Answer the walking-tour question. Track the route and return only one final option letter '
            'A, B, C, D, or E. For a map, compare the ordered route shape, start/end, and turns against '
            'all supplied candidate images.\nQuestion: '+q['question']+'\nOptions:\n'+
            '\n'.join(f"{'ABCDE'[i]}. {x}" for i,x in enumerate(q['options'])))
    source=_source(video_id); start=float(q.get('start_seconds',0)); end=float(q.get('end_seconds',start+60))
    clip=Path(tempfile.gettempdir())/f'moss-{video_id}-{question_id}.mp4'
    import subprocess
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(start),'-i',str(source),'-t',str(max(1,end-start)),'-an','-vf','scale=512:-2','-c:v','libx264','-preset','ultrafast','-pix_fmt','yuv420p','-y',str(clip)],check=True,timeout=300)
    media=[{'type':'video','video':str(clip)}]
    if q.get('image_paths'):
        with zipfile.ZipFile(ROOT/'manifests'/f'{split}_images.zip') as z, tempfile.TemporaryDirectory() as td:
            members=z.namelist()
            for i,ref in enumerate(q['image_paths']):
                hit=[n for n in members if Path(n).name==Path(str(ref)).name]
                if hit:
                    p=Path(td)/f'{i}.png'; p.write_bytes(z.read(hit[0])); media.append({'type':'image','image':str(p)})
            media.append({'type':'text','text':prompt})
            messages=[{'role':'user','content':media}]
            iq,oq=queue.Queue(),queue.Queue()
            worker=threading.Thread(target=model.offline_generate,args=(processor,iq,oq),kwargs={'vision_chunked_length':64},daemon=True); worker.start()
            iq.put({'messages':messages,'media_kwargs':{'video_fps':float(fps),'min_frames':8,'max_frames':int(max_frames),'num_extract_threads':4},'generate_kwargs':{'max_new_tokens':128,'do_sample':False,'temperature':1.0}})
            chunks=[]; errors=[]
            while True:
                item=oq.get()
                if item=='<|round_end|>': break
                if item.startswith('[ERROR] '): errors.append(item)
                elif item not in ('<|round_start|>',): chunks.append(item)
            iq.put({'stop_offline_generate':True}); worker.join(timeout=30); text=''.join(chunks) if chunks else ('ERROR: '+' | '.join(errors))
    else:
        with torch.inference_mode(): text=model.offline_video_generate(processor,prompt=prompt,video=str(clip),video_fps=float(fps),min_frames=8,max_frames=int(max_frames),max_new_tokens=128,do_sample=False,vision_chunked_length=64)
    result={'video_id':video_id,'question_id':str(question_id),'model':CHECKPOINT,'raw_response':text,'ground_truth_used_for_inference':False}
    out=ROOT/'experiments'/'mossvl-probe'/f'{split}-{video_id}-{question_id}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); VOL.commit(); return result

@APP.local_entrypoint()
def main(video_id='4MYTCny_FZs',question_id='0',split='validation',fps=1.0,max_frames=256): print(json.dumps(run.remote(video_id,question_id,split,fps,max_frames),indent=2))
