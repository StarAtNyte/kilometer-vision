from pathlib import Path
import os, modal
V=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
I=modal.Image.debian_slim(python_version='3.11')
A=modal.App('kilometervision-inspect-vggt-cache')
@A.function(image=I,volumes={'/data':V},timeout=10*60)
def run():
 V.reload(); root=Path('/data/2707-kilometer-vision/checkpoints/huggingface/models--facebook--VGGT-1B'); rows=[]
 for p in sorted(root.rglob('*')):
  try: rows.append({'path':str(p),'is_file':p.is_file(),'is_dir':p.is_dir(),'size':p.stat().st_size if p.is_file() else None,'target':os.readlink(p) if p.is_symlink() else None})
  except OSError as e: rows.append({'path':str(p),'error':str(e)})
 out=Path('/data/2707-kilometer-vision/experiments/vggt-motion/test')
 rows.append({'output_root':str(out),'entries':[str(p) for p in sorted(out.rglob('*')) if p.is_file()] if out.exists() else []})
 return rows
@A.local_entrypoint()
def main():
 import json; print(json.dumps(run.remote(),indent=2))
