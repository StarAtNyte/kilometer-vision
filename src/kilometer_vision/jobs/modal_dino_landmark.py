"""Frozen DINOv2 patch-retrieval experiment for landmark questions.

This deliberately uses fixed GeM pooling rather than fitting a VLAD vocabulary on
challenge data. It is an AnyLoc-inspired, training-free gate before adopting the
heavier official SegVLAD pipeline.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

ROOT=Path('/data/2707-kilometer-vision'); SOURCE_ROOT=Path('/data/shared/walking-tours')
VOLUME=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
IMAGE=(modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg')
       .pip_install('torch','torchvision','transformers>=4.49','pillow','accelerate')
       .add_local_python_source('kilometer_vision'))
app=modal.App('kilometervision-dino-landmark')
VIDEO_SUFFIXES={'.avi','.m4v','.mkv','.mov','.mp4','.webm'}


def _source(video_id):
    directory=SOURCE_ROOT/video_id
    return next((p for p in sorted(directory.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES and '.partial' not in p.stem),None) if directory.is_dir() else None


def _reference(question,split):
    from PIL import Image
    refs=question.get('image_paths',[])
    if not refs:return None
    with zipfile.ZipFile(ROOT/'manifests'/f'{split}_images.zip') as archive:
        matches=[name for name in archive.namelist() if Path(name).name==Path(str(refs[0])).name]
        return Image.open(io.BytesIO(archive.read(matches[0]))).convert('RGB') if len(matches)==1 else None


def _frames(source,start,end,step,directory):
    from PIL import Image
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-ss',str(start),'-t',str(end-start),'-i',str(source),'-vf',f'fps=1/{step},scale=448:-2','-q:v','4',str(directory/'%06d.jpg')],check=True,timeout=max(120,int(end-start)*2))
    return [(start+i*step,Image.open(path).convert('RGB').copy()) for i,path in enumerate(sorted(directory.glob('*.jpg')))]


@app.function(image=IMAGE,gpu='A100-80GB',volumes={'/data':VOLUME},timeout=3*3600)
def run_dino(split: str='validation',video_id: str='__all__',step_seconds: float=2.0,checkpoint: str='facebook/dinov2-base'):
    import torch
    from transformers import AutoImageProcessor,AutoModel
    from kilometer_vision.question_planner import classify_question_facets
    manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text());targets=sorted(manifest) if video_id=='__all__' else [video_id]
    processor=AutoImageProcessor.from_pretrained(checkpoint);model=AutoModel.from_pretrained(checkpoint,torch_dtype=torch.bfloat16).to('cuda').eval()
    # The supervised MMS-VPR pair head is optional and is deliberately not
    # required for the frozen, label-free patch retrieval gate.  Its checkpoint
    # is absent on fresh volumes and its challenge-transfer audit was negative.
    verifier = None
    verifier_path = ROOT/'models'/'landmark-verifier'/'mms-vpr-dinov2-pair-verifier.pt'
    if verifier_path.exists():
        verifier=torch.nn.Sequential(torch.nn.Linear(3072,512),torch.nn.GELU(),torch.nn.Dropout(0.1),torch.nn.Linear(512,1)).to('cuda')
        verifier.load_state_dict(torch.load(verifier_path,map_location='cuda',weights_only=True));verifier.eval()
    def encode(images):
        inputs=processor(images=images,return_tensors='pt').to('cuda')
        with torch.no_grad(): patches=model(**inputs).last_hidden_state[:,1:].float()
        # Fixed positive/negative GeM over patch tokens preserves more local-region
        # evidence than a single CLS token without learning from challenge media.
        positive=patches.clamp_min(0).pow(3).mean(dim=1).pow(1/3)
        negative=-(-patches).clamp_min(0).pow(3).mean(dim=1).pow(1/3)
        features=torch.cat((positive,negative),dim=-1)
        return features/features.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    def patch_features(images):
        inputs=processor(images=images,return_tensors='pt').to('cuda')
        with torch.no_grad(): patches=model(**inputs).last_hidden_state[:,1:].float()
        return patches/patches.norm(dim=-1,keepdim=True).clamp_min(1e-9)
    def local_score(reference_patches, candidate_patches):
        similarity=reference_patches@candidate_patches.T
        ref_best=similarity.max(dim=1).values;candidate_best=similarity.max(dim=0).values
        ref_k=max(1,ref_best.numel()//10);candidate_k=max(1,candidate_best.numel()//10)
        score=(ref_best.topk(ref_k).values.mean()+candidate_best.topk(candidate_k).values.mean())/2
        ref_to_candidate=similarity.argmax(dim=1);candidate_to_ref=similarity.argmax(dim=0)
        mutual=sum(int(candidate_to_ref[j])==i for i,j in enumerate(ref_to_candidate.tolist()))/max(1,min(similarity.shape))
        return float(score),float(mutual)
    def verifier_probability(reference_feature,candidate_feature):
        features=torch.cat((torch.abs(reference_feature-candidate_feature),reference_feature*candidate_feature),dim=-1)
        with torch.no_grad():return float(torch.sigmoid(verifier(features)).item())
    summaries=[]
    for target in targets:
        source=_source(target);records={}
        for key,q in manifest.get(target,{}).items():
            if 'landmark_presence' not in classify_question_facets(q):continue
            ref=_reference(q,split);start=float(q.get('start_seconds',0));end=float(q.get('end_seconds',start))
            if ref is None or source is None or end<=start:continue
            with tempfile.TemporaryDirectory() as temporary:
                frames=_frames(source,start,end,step_seconds,Path(temporary));ref_feature=encode([ref]);scored=[]
                for offset in range(0,len(frames),64):
                    batch=frames[offset:offset+64];similarities=(encode([image for _,image in batch])@ref_feature.T).squeeze(1).cpu().tolist();scored.extend((timestamp,float(score)) for (timestamp,_),score in zip(batch,similarities))
                ranked=sorted(scored,key=lambda row:row[1],reverse=True);by_time={timestamp:image for timestamp,image in frames};reference_patches=patch_features([ref])[0]
                reranked=[]
                for timestamp,global_similarity in ranked[:10]:
                    candidate_image=by_time[timestamp];candidate_patches=patch_features([candidate_image])[0];patch_score,mutual_ratio=local_score(reference_patches,candidate_patches)
                    pair_probability=verifier_probability(ref_feature,encode([candidate_image])) if verifier is not None else None
                    reranked.append({'timestamp_seconds':timestamp,'similarity':global_similarity,'patch_local_similarity':patch_score,'mutual_patch_ratio':mutual_ratio,'external_pair_probability':pair_probability})
            values=sorted(score for _,score in scored);reranked.sort(key=lambda row:(row['patch_local_similarity'],row['mutual_patch_ratio']),reverse=True)
            records[str(key)]={'status':'ok','checkpoint':checkpoint,'pooling':'fixed_signed_gem_p3','local_reranking':'bidirectional_top10pct_patch_similarity','external_verifier':'MMS-VPR class-disjoint pair head; frozen threshold 0.5','step_seconds':step_seconds,'frames':len(scored),'top_matches':reranked,'global_top_matches':[{'timestamp_seconds':t,'similarity':s} for t,s in ranked[:10]],'similarity_median':values[len(values)//2] if values else None}
        output=ROOT/'experiments'/'landmark-dinov2'/split/f'{target}.json';output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(records,indent=2));VOLUME.commit();summaries.append({'video_id':target,'questions':len(records)})
    return summaries


@app.function(image=IMAGE,volumes={'/data':VOLUME},timeout=10*60)
def summarize_verifier(split: str='validation'):
    rows=[]
    for path in sorted((ROOT/'experiments'/'landmark-dinov2'/split).glob('*.json')):
        for question_id,record in json.loads(path.read_text()).items():
            matches=record.get('top_matches',[])
            probabilities=[item.get('external_pair_probability') for item in matches
                           if item.get('external_pair_probability') is not None]
            rows.append({'video_id':path.stem,'question_id':question_id,'top_timestamp_seconds':matches[0].get('timestamp_seconds') if matches else None,'top_patch_local_similarity':matches[0].get('patch_local_similarity') if matches else None,'top_mutual_patch_ratio':matches[0].get('mutual_patch_ratio') if matches else None,'top_external_pair_probability':matches[0].get('external_pair_probability') if matches else None,'max_external_pair_probability':max(probabilities,default=None)})
    return rows


@app.local_entrypoint()
def main(action: str='run',split: str='validation',video_id: str='__all__',step_seconds: float=2.0,checkpoint: str='facebook/dinov2-base'):
    result=run_dino.remote(split,video_id,step_seconds,checkpoint) if action=='run' else summarize_verifier.remote(split)
    print(json.dumps(result,indent=2))
