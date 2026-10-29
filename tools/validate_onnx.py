"""Validate pure ORT logits/fusion and masks against an independent FP32 cache."""
import argparse,sys,json,zipfile,io,importlib.util
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'deploy')
p.add_argument('--models',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--original-zip',type=Path)
p.add_argument('--provider',choices=['cpu','cuda'],default='cpu');p.add_argument('--output',type=Path,required=True)
a=p.parse_args();sys.path.insert(0,str(a.source.resolve()))
assert importlib.util.find_spec('torch') is None,'Use an independent pure ORT environment'
import numpy as np
from PIL import Image
from predict_onnx import Engine,read_image
engine=Engine(a.models,a.provider,2);meta=json.loads((a.reference/'reference.json').read_text());rows=[];logits=[]
paths={p.name:p for p in a.input_dir.rglob('*') if p.is_file()}
original=zipfile.ZipFile(a.original_zip) if a.original_zip else None
try:
 for name in meta['images']:
  path=paths[name];raw=read_image(path)
  for entry in engine.entries:
   family,seed=entry['family'],entry['seed'];x=raw if family=='mit_b2' else raw/np.float32(255)
   actual=engine.logits(family,seed,x);expected=np.load(a.reference/f'{path.stem}__{family}_{seed}.npy')
   passed=bool(np.allclose(actual,expected,atol=2e-4,rtol=1e-3))
   logits.append({'family':family,'seed':seed,'image':name,'max_abs':float(np.abs(actual-expected).max()),'passed':passed});assert passed,logits[-1]
  actual=engine.probability(raw);expected=np.load(a.reference/(path.stem+'.npy'))
  assert np.allclose(actual,expected,atol=2e-4,rtol=1e-3),name
  mask=actual>=np.float32(.4)
  row={'image':name,'max_abs':float(np.abs(actual-expected).max()),'mask_difference_vs_fp32':int((mask!=(expected>=np.float32(.4))).sum())}
  if original:
   with Image.open(io.BytesIO(original.read('result/'+path.stem+'.png'))) as im:old=np.asarray(im)
   row['mask_difference_vs_original']=int((mask!=old).sum())
  rows.append(row)
finally:
 if original:original.close()
if a.output.exists():raise FileExistsError(a.output)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'passed':True,'torch_installed':False,'provider':a.provider,'per_image':rows,'logits':logits,'official_score_inferred':False},indent=2)+'\n')
print(json.dumps({'passed':True,'images':len(rows),'max_abs':max(r['max_abs'] for r in rows)}))
