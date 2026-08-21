import modal
A=modal.App('kilometervision-gpu-smoke')
I=modal.Image.debian_slim(python_version='3.11').pip_install('torch')
@A.function(image=I,gpu='T4',timeout=5*60)
def run():
 import torch
 return {'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
@A.local_entrypoint()
def main(): print(run.remote())
