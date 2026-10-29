"""Cache independent Torch FP32 probabilities and single-view logits for ORT checks."""
import argparse, json, sys, time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'deploy')
p.add_argument('--weights',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--limit',type=int,default=0)
a=p.parse_args();sys.path.insert(0,str(a.source.resolve()))
import torch, numpy as np
from PIL import Image
from predict_torch import Engine
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True)
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
engine=Engine(a.weights,a.device,False)
paths=sorted(p for p in a.input_dir.rglob('*') if p.suffix.lower() in {'.jpg','.png','.jpeg'})
if a.limit:paths=paths[:a.limit]
start=time.monotonic()
with torch.inference_mode():
 for path in paths:
  raw=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32).transpose(2,0,1)[None].copy()
  np.save(a.output/(path.stem+'.npy'),engine.probability(raw))
  for entry in engine.entries:
   family,seed=entry['family'],entry['seed']
   x=raw if family=='mit_b2' else raw/np.float32(255)
   y=engine.logits(family,seed,torch.from_numpy(x).to(a.device)).cpu().numpy()
   np.save(a.output/f'{path.stem}__{family}_{seed}.npy',y)
(a.output/'reference.json').write_text(json.dumps({'images':[p.name for p in paths],'device':a.device,'torch':torch.__version__,'TF32':False,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
