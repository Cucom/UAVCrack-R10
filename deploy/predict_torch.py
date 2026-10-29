"""Portable inference, no Lightning/MMCV/MMSegmentation/training environment."""
import argparse,json,time,zipfile,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import build
# Shared input/format helpers do not import ONNX Runtime in this module.
VIEWS=[(),(-1,),(-2,),(-2,-1)]
class Engine:
    def __init__(self,model_dir,device='cpu',amp=False):
        self.device=device;self.amp=amp;self.models={};self.entries=json.loads((Path(model_dir)/'WEIGHTS.json').read_text())['files']
        if not amp:
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.matmul.allow_tf32=False
        for e in self.entries:
            path=Path(model_dir)/e['file']
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
            if h.hexdigest()!=e['sha256']:raise ValueError(f'Weight SHA mismatch: {path.name}')
            m=build(e['family']);m.load_state_dict(torch.load(Path(model_dir)/e['file'],map_location='cpu',weights_only=True),strict=True);self.models[e['family'],e['seed']]=m.to(device).eval()
    def logits(self,family,seed,x):
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.amp and family!='mit_b2'):
            return self.models[family,seed](x).float()
    @torch.inference_mode()
    def probability(self,image):
        raw=torch.from_numpy(np.ascontiguousarray(image)).to(self.device)
        scaled=torch.from_numpy(np.ascontiguousarray(image/np.float32(255))).to(self.device);families={}
        for family in ('resnet34','mit_b2','convnext_ema'):
            x=raw if family=='mit_b2' else scaled
            total=None;seed_probs=[]
            for seed in (42,43):
                accum=None
                for axes in ([()] if family=='mit_b2' else VIEWS):
                    logits=self.logits(family,seed,x.flip(axes) if axes else x)
                    if axes:logits=logits.flip(axes)
                    q=logits.softmax(1)[:,1:2] if family=='mit_b2' else logits.sigmoid()
                    accum=q if accum is None else accum+q
                    if family=='resnet34':total=q if total is None else total+q
                seed_probs.append(accum/(1 if family=='mit_b2' else 4))
            families[family]=(total/8 if family=='resnet34' else (seed_probs[0]+seed_probs[1])/2)[0,0].cpu().numpy()
        return .65*families['resnet34']+.25*families['mit_b2']+.10*families['convnext_ema']
def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--amp',action='store_true');p.add_argument('--expected-count',type=int,default=300);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    paths=sorted(f for f in a.input_dir.rglob('*') if f.suffix.lower() in ('.jpg','.jpeg','.png'));assert len(paths)==a.expected_count
    assert len({f.stem for f in paths})==len(paths), 'Duplicate output stems'
    if a.amp and a.device!='cuda':raise ValueError('AMP requires CUDA')
    engine=Engine(a.model_dir,a.device,a.amp);out=a.output/'result';out.mkdir(parents=True);start=time.monotonic()
    for i,f in enumerate(paths,1):
        with Image.open(f) as im:
            assert im.size==(672,378);x=np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)[None]
        q=engine.probability(x);assert q.shape==(378,672) and np.isfinite(q).all() and q.min()>=0 and q.max()<=1
        Image.fromarray((q>=np.float32(.4)).astype('uint8')).save(out/f.with_suffix('.png').name)
        if i%30==0:print(f'{i}/{len(paths)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    with zipfile.ZipFile(a.output/'result.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.glob('*.png')):z.write(f,'result/'+f.name)
    print(json.dumps({'images':len(paths),'elapsed_seconds':time.monotonic()-start,'device':a.device,'amp':a.amp}),flush=True)
if __name__=='__main__':main()
