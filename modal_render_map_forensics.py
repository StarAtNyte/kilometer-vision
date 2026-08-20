from pathlib import Path
import json, zipfile, modal
from map_solver import route_path, route_marker_positions, orient_path_from_start
ROOT=Path('/data/2707-kilometer-vision'); V=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True)
I=modal.Image.debian_slim(python_version='3.11').pip_install('opencv-python-headless','numpy','pillow').add_local_python_source('map_solver')
A=modal.App('kilometervision-map-forensics')
@A.function(image=I,volumes={'/data':V},timeout=20*60)
def run():
 import numpy as np
 from PIL import Image,ImageDraw
 V.reload(); vid='buBU7NrfyGM'; motion=json.loads((ROOT/'experiments/dpvo-motion/validation/fps1-focal0.9-w256-b256-p16/buBU7NrfyGM.json').read_text()); tr=[[p[0],p[2]] for p in motion['0']['positions_3d']]; z=ROOT/'manifests/validation_images.zip'; files=[]
 with zipfile.ZipFile(z) as a:
  for n in a.namelist():
   if 'buBU7NrfyGM' in n and n.endswith('.png'):files.append((n,a.read(n)))
 def norm(p,b):
  v=np.asarray(p,float);v-=v.min(0);e=np.maximum(v.max(0),1e-6);s=min((b[2]-b[0]-20)/e[0],(b[3]-b[1]-20)/e[1]);v*=s;v += [(b[0]+b[2]-v[:,0].max())/2,(b[1]+b[3]-v[:,1].max())/2];return [tuple(x) for x in v]
 c=Image.new('RGB',(1500,760),'white');d=ImageDraw.Draw(c);o=norm(tr,(20,40,500,740));d.text((20,12),'DPVO trace (blue)',fill='black');d.line(o,fill='blue',width=6)
 boxes=[(520,40,830,350),(850,40,1160,350),(1180,40,1490,350),(680,420,990,730),(1010,420,1320,730)]
 for i,(n,raw) in enumerate(sorted(files)):
  marks=route_marker_positions(raw);p=orient_path_from_start(route_path(raw),marks);b=boxes[i];d.text((b[0],b[1]-22),Path(n).stem[-5:],fill='black');d.line(norm(p,b),fill='red',width=5)
 out=ROOT/'experiments/map-forensics/bu-panel-xz.png';out.parent.mkdir(parents=True,exist_ok=True);c.save(out);V.commit();return str(out)
@A.local_entrypoint()
def main():print(run.remote())
