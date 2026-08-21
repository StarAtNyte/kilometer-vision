"""Remote, GPU-free preflight for resumable VGGT map jobs."""
from pathlib import Path
import json, subprocess, modal

ROOT = Path('/data/2707-kilometer-vision')
SOURCE = Path('/data/shared/walking-tours')
V = modal.Volume.from_name('perception-test-2026-data', create_if_missing=True)
I = modal.Image.debian_slim(python_version='3.11').apt_install('ffmpeg')
A = modal.App('kilometervision-vggt-preflight')

@A.function(image=I, volumes={'/data': V}, timeout=20*60)
def run(video_id='G0fWvIxy88A', split='test', frame_count=12):
    V.reload()
    frame_count = int(frame_count)
    manifest = json.loads((ROOT/'manifests'/f'{split}.json').read_text())
    directory = SOURCE/video_id
    videos = [p for p in directory.iterdir()] if directory.is_dir() else []
    source = next((p for p in videos if p.suffix.lower() in {'.mp4','.mkv','.mov','.webm','.avi','.m4v'} and '.partial' not in p.stem), None)
    rows=[]
    for qid,q in manifest.get(video_id,{}).items():
        if 'map' not in str(q.get('question','')).lower() and 'image_paths' not in q:
            continue
        start=float(q.get('start_seconds',0) or 0); end=float(q.get('end_seconds',start) or start)
        row={'question_id':qid,'start':start,'end':end,'source':str(source) if source else None}
        if source and end>start:
            out=Path('/tmp/vggt-preflight-%06d.jpg')
            cmd=['ffmpeg','-hide_banner','-loglevel','error','-ss',str(start),'-t',str(end-start),'-i',str(source),'-vf',f'fps={max(1e-6,(frame_count-1)/(end-start))},scale=640:-2','-frames:v',str(frame_count),'-q:v','5',str(out)]
            try:
                subprocess.run(cmd,check=True,timeout=max(180,int(end-start)*2))
                row['frames']=len(list(Path('/tmp').glob('vggt-preflight-*.jpg')))
            except Exception as e: row['error']=f'{type(e).__name__}: {e}'
            for p in Path('/tmp').glob('vggt-preflight-*.jpg'):
                try: p.unlink()
                except OSError: pass
        rows.append(row)
    return {'video_id':video_id,'source_exists':source is not None,'rows':rows}

@A.local_entrypoint()
def main(video_id='G0fWvIxy88A',split='test',frame_count=12):
    print(json.dumps(run.remote(video_id,split,frame_count),indent=2))
