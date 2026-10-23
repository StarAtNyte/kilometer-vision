"""Dense pretrained landmark retrieval inside exact question windows."""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import modal

ROOT=Path('/data/2707-kilometer-vision'); SOURCE_ROOT=Path('/data/shared/walking-tours')
VOLUME=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
IMAGE=(modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg')
       .pip_install('torch','transformers>=4.49','pillow','accelerate')
       .add_local_python_source('kilometer_vision'))
app=modal.App('kilometervision-landmark')
VIDEO_SUFFIXES={'.avi','.m4v','.mkv','.mov','.mp4','.webm'}


def _source(video_id: str):
    directory=SOURCE_ROOT/video_id
    return next((p for p in sorted(directory.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES and '.partial' not in p.stem),None) if directory.is_dir() else None


def _reference(question: dict[str,Any],split: str):
    from PIL import Image
    refs=question.get('image_paths',[])
    if not refs:return None
    with zipfile.ZipFile(ROOT/'manifests'/f'{split}_images.zip') as archive:
        matches=[n for n in archive.namelist() if Path(n).name==Path(str(refs[0])).name]
        return Image.open(io.BytesIO(archive.read(matches[0]))).convert('RGB') if len(matches)==1 else None


def _frames(source: Path,start: float,end: float,step_seconds: float,directory: Path):
    from PIL import Image
    pattern=directory/'%06d.jpg'
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(start),'-t',str(end-start),'-i',str(source),'-vf',f'fps=1/{step_seconds},scale=448:-2','-q:v','4',str(pattern)],check=True,timeout=max(120,int(end-start)*2))
    paths=sorted(directory.glob('*.jpg'))
    return [(start+i*step_seconds,Image.open(path).convert('RGB').copy()) for i,path in enumerate(paths)]


def _feature_tensor(value):
    if hasattr(value,'pooler_output'): return value.pooler_output
    if hasattr(value,'image_embeds'): return value.image_embeds
    return value


@app.function(image=IMAGE,gpu='A100-80GB',volumes={'/data':VOLUME},timeout=3*3600)
def run_landmark(split: str='validation',video_id: str='__all__',step_seconds: float=2.0,checkpoint: str='google/siglip2-base-patch16-224'):
    import torch
    from transformers import AutoModel,AutoProcessor
    from kilometer_vision.question_planner import classify_question_facets
    manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text());targets=sorted(manifest) if video_id=='__all__' else [video_id]
    model=AutoModel.from_pretrained(checkpoint,torch_dtype=torch.bfloat16).to('cuda').eval();processor=AutoProcessor.from_pretrained(checkpoint)
    result=[]
    for target in targets:
        source=_source(target);records={}
        for key,q in manifest.get(target,{}).items():
            if 'landmark_presence' not in classify_question_facets(q):continue
            ref=_reference(q,split);start=float(q.get('start_seconds',0));end=float(q.get('end_seconds',start))
            if ref is None or source is None or end<=start:continue
            with tempfile.TemporaryDirectory() as temporary:
                frames=_frames(source,start,end,step_seconds,Path(temporary))
                with torch.no_grad():
                    ref_inputs=processor(images=[ref],return_tensors='pt').to('cuda');ref_feature=_feature_tensor(model.get_image_features(**ref_inputs));ref_feature=ref_feature/ref_feature.norm(dim=-1,keepdim=True)
                    scored=[]
                    for offset in range(0,len(frames),64):
                        batch=frames[offset:offset+64];inputs=processor(images=[im for _,im in batch],return_tensors='pt').to('cuda');features=_feature_tensor(model.get_image_features(**inputs));features=features/features.norm(dim=-1,keepdim=True);similarities=(features@ref_feature.T).squeeze(1).float().cpu().tolist();scored.extend((t,float(s)) for (t,_),s in zip(batch,similarities))
            ranked=sorted(scored,key=lambda row:row[1],reverse=True)
            records[str(key)]={'status':'ok','checkpoint':checkpoint,'step_seconds':step_seconds,'frames':len(scored),'top_matches':[{'timestamp_seconds':t,'similarity':s} for t,s in ranked[:10]],'similarity_median':sorted(s for _,s in scored)[len(scored)//2] if scored else None}
        out=ROOT/'experiments'/'landmark'/split/f'{target}.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(records,indent=2));VOLUME.commit();result.append({'video_id':target,'questions':len(records)})
    return result


@app.local_entrypoint()
def main(split: str='validation',video_id: str='__all__',step_seconds: float=2.0,checkpoint: str='google/siglip2-base-patch16-224'):
    print(json.dumps(run_landmark.remote(split,video_id,step_seconds,checkpoint),indent=2))
