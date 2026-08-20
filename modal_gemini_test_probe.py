"""One-call, answer-blind Gemini probe for a low-margin test map."""
from __future__ import annotations
import io, json, os, zipfile
from pathlib import Path
import modal
from map_solver import route_marker_positions, route_path, orient_path_from_start, ordered_trace_candidate_scores
from question_planner import classify_question_facets

ROOT=Path('/data/2707-kilometer-vision'); VOLUME=modal.Volume.from_name('perception-test-2026-data',create_if_missing=True); SECRET=modal.Secret.from_name('gemini-api-keys')
IMAGE=modal.Image.debian_slim(python_version='3.11').pip_install('google-genai','pillow','opencv-python-headless','numpy==1.26.4').add_local_python_source('map_solver','question_planner')
APP=modal.App('kilometervision-gemini-test-probe'); MODEL='gemini-3.6-flash'; TARGET_VIDEO='fGX0Te6pFvk'; TARGET_Q='39'; VARIANT='fps1-focal0.9-w256-b256-p16'

def key():
 vals=[x.strip() for x in os.environ.get('GEMINI_API_KEYS','').split(',') if x.strip()]
 for n in ('GOOGLE_API_KEY','GEMINI_API_KEY','API_KEY'):
  if os.environ.get(n): return os.environ[n]
 if vals:return vals[0]
 raise RuntimeError('no key')
def sample(points,n=24):
 if len(points)<=n:return [[round(float(x),4),round(float(y),4)] for x,y in points]
 return [[round(float(points[round(i*(len(points)-1)/(n-1))][0]),4),round(float(points[round(i*(len(points)-1)/(n-1))][1]),4)] for i in range(n)]

@APP.function(image=IMAGE,secrets=[SECRET],volumes={'/data':VOLUME},timeout=10*60)
def run(video_id=TARGET_VIDEO,question_id=TARGET_Q,model=MODEL,split='test'):
 from google import genai
 from google.genai import types
 from PIL import Image
 VOLUME.reload(); manifest=json.loads((ROOT/'manifests'/f'{split}.json').read_text()); q=manifest[video_id][str(question_id)]; motion=json.loads((ROOT/'experiments'/'dpvo-motion'/split/VARIANT/f'{video_id}.json').read_text()); motion=motion.get(video_id,motion); trace=motion[str(question_id)]['trajectory_2d']; archive_path=ROOT/'manifests'/f'{split}_images.zip'; parts=[]
 scores=ordered_trace_candidate_scores(q,trace,archive_path,reflection_penalty=0.0); order=sorted(range(5),key=lambda i:float(scores[i]['score'])); parts.append(types.Part.from_text(text='Choose the map matching this ordered, scale-free walking trajectory. Compare turn order, topology, bends, and S/E endpoint displacement. Candidate identity is the letter printed with each route. Return JSON only with answer (A-E), confidence, and reason. Observed trajectory: '+json.dumps(sample(trace))))
 with zipfile.ZipFile(archive_path) as a:
  for i,ref in enumerate(q['image_paths']):
   hit=[n for n in a.namelist() if Path(n).name==Path(str(ref)).name]; raw=a.read(hit[0]); marks=route_marker_positions(raw); path=orient_path_from_start(route_path(raw),marks); parts.append(types.Part.from_text(text=f'Candidate {"ABCDE"[i]} ordered route samples: {json.dumps(sample(path))}; pins: {json.dumps(marks)}'))
   with Image.open(io.BytesIO(raw)) as im:
    b=io.BytesIO(); im.convert('RGB').resize((400,400)).save(b,format='JPEG',quality=82); parts.append(types.Part.from_bytes(data=b.getvalue(),mime_type='image/jpeg'))
 client=genai.Client(api_key=key(),http_options=types.HttpOptions(timeout=180_000)); result={'split':split,'video_id':video_id,'question_id':str(question_id),'model':model,'p16_ranking':order,'p16_margin':float(scores[order[1]]['score']-scores[order[0]]['score']),'ground_truth_used_for_inference':False}
 try:
  response=client.models.generate_content(model=model,contents=[types.Content(role='user',parts=parts)],config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level='low'),media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,max_output_tokens=1000,response_mime_type='application/json',response_schema={'type':'object','properties':{'answer':{'type':'string','enum':list('ABCDE')},'confidence':{'type':'number'},'reason':{'type':'string'}},'required':['answer','confidence','reason']}))
  result['raw_response']=response.text
 except Exception as exc: result['error']=f'{type(exc).__name__}: {exc}'
 out=ROOT/'experiments'/'gemini-test-probe'/f'{split}-{video_id}-{question_id}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); VOLUME.commit(); return result

@APP.local_entrypoint()
def main(video_id=TARGET_VIDEO,question_id=TARGET_Q,model=MODEL,split='test'): print(run.remote(video_id,question_id,model,split))
