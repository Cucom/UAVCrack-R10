"""Complete six-model predictor; NumPy/Pillow/ONNX Runtime only."""
import argparse,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
VIEWS=[(),(-1,),(-2,),(-2,-1)]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def read_image(path):
    with Image.open(path) as im:
        if im.size!=(672,378):raise ValueError(f'Expected 672x378 input: {path}')
        a=np.asarray(im.convert('RGB'),dtype=np.float32)
    return np.ascontiguousarray(a.transpose(2,0,1)[None])
def sigmoid(x):
    z=np.exp(-np.abs(x),dtype=np.float32);z/=1+z
    return np.where(x>=0,1-z,z)
def softmax_crack(x):
    z=np.exp(x-x.max(axis=1,keepdims=True),dtype=np.float32)
    return z[:,1:2]/z.sum(axis=1,keepdims=True)
def paths_in(folder,count):
    paths=sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if len(paths)!=count or len({p.stem for p in paths})!=count:raise ValueError('Input count or filename-stem mismatch')
    return paths
class Engine:
    def __init__(self,model_dir,provider='cpu',threads=2):
        model_dir=Path(model_dir);manifest=json.loads((model_dir/'ONNX_FILES.json').read_text());self.entries=manifest['files'];self.sessions={}
        if [(e['family'],e['seed']) for e in self.entries]!=[(f,s) for f in ('resnet34','mit_b2','convnext_ema') for s in (42,43)]:raise ValueError('Unexpected family/seed order')
        if provider=='cuda' and hasattr(ort,'preload_dlls'):ort.preload_dlls()
        if provider=='cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('CUDA provider unavailable; refusing silent CPU fallback')
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        providers=[('CUDAExecutionProvider',{'use_tf32':0,'cudnn_conv_algo_search':'DEFAULT'}),'CPUExecutionProvider'] if provider=='cuda' else ['CPUExecutionProvider']
        for e in self.entries:
            p=model_dir/e['file']
            if sha(p)!=e['sha256']:raise ValueError(f'ONNX SHA mismatch: {p.name}')
            session=ort.InferenceSession(str(p),sess_options=options,providers=providers)
            if provider=='cuda' and session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('CUDA session initialization failed')
            session.disable_fallback()
            channels=2 if e['family']=='mit_b2' else 1
            if session.get_inputs()[0].shape!=[1,3,378,672] or session.get_outputs()[0].shape!=[1,channels,378,672]:raise ValueError('Graph shape mismatch')
            self.sessions[e['family'],e['seed']]=session
    def logits(self,family,seed,image):
        y=self.sessions[family,seed].run(['logits'],{'image':np.ascontiguousarray(image)})[0]
        if not np.isfinite(y).all():raise ValueError('Nonfinite logits')
        return y
    def probability(self,image):
        if image.dtype!=np.float32 or image.shape!=(1,3,378,672):raise ValueError('Expected FP32 RGB NCHW raw pixels')
        families={}
        for family in ('resnet34','mit_b2','convnext_ema'):
            family_input=image if family=='mit_b2' else image/np.float32(255)
            total=None;seed_probs=[]
            for seed in (42,43):
                accum=None
                for axes in ([()] if family=='mit_b2' else VIEWS):
                    logits=self.logits(family,seed,np.flip(family_input,axes) if axes else family_input)
                    if axes:logits=np.flip(logits,axes)
                    q=softmax_crack(logits) if family=='mit_b2' else sigmoid(logits)
                    accum=q if accum is None else accum+q
                    if family=='resnet34':total=q if total is None else total+q
                seed_probs.append(accum/(1 if family=='mit_b2' else np.float32(4)))
            families[family]=(total/np.float32(8) if family=='resnet34' else (seed_probs[0]+seed_probs[1])/np.float32(2))[0,0]
        return .65*families['resnet34']+.25*families['mit_b2']+.10*families['convnext_ema']
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--provider',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=2);p.add_argument('--expected-count',type=int,default=300);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    paths=paths_in(a.input_dir,a.expected_count);engine=Engine(a.model_dir,a.provider,a.threads);out=a.output/'result';out.mkdir(parents=True);start=time.monotonic();records=[]
    for i,path in enumerate(paths,1):
        q=engine.probability(read_image(path));assert q.shape==(378,672) and np.isfinite(q).all() and q.min()>=0 and q.max()<=1
        dest=out/path.with_suffix('.png').name;Image.fromarray((q>=np.float32(.4)).astype('uint8')).save(dest);records.append({'input':path.name,'input_sha256':sha(path),'output':dest.name,'sha256':sha(dest)})
        if i%30==0:print(f'{i}/{len(paths)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    with zipfile.ZipFile(a.output/'result.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.glob('*.png')):z.write(f,'result/'+f.name)
    result={'count':len(paths),'provider':a.provider,'onnxruntime':ort.__version__,'elapsed_seconds':time.monotonic()-start,'zip_sha256':sha(a.output/'result.zip'),'threshold':.4,'original_submission_replaced':False}
    (a.output/'metadata.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'predictions.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
