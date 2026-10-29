"""Check runtime-model SHA, original ZIP format, and sharing-archive CRC/SHA."""
import argparse,hashlib,json,zipfile,io
from pathlib import Path
import numpy as np
from PIL import Image

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True)
 p.add_argument('--submission',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True)
 p.add_argument('--packages',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=[]
 for sub,name in [('weights','WEIGHTS.json'),('onnx','ONNX_FILES.json')]:
  files=json.loads((a.models/sub/name).read_text())['files'];assert len(files)==6
  for e in files:assert sha(a.models/sub/e['file'])==e['sha256'];rows.append(sub+'/'+e['file'])
 inputs={p.stem:p for p in a.input_dir.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}};assert len(inputs)==300
 with zipfile.ZipFile(a.submission) as z:
  assert z.testzip() is None
  assert len(z.namelist())==300 and set(z.namelist())=={'result/'+n+'.png' for n in inputs}
  for n in z.namelist():
   with Image.open(io.BytesIO(z.read(n))) as im:
    assert im.format=='PNG' and im.mode=='L' and im.size==(672,378) and set(np.unique(np.asarray(im)))<={0,1}
 archives=[]
 if a.packages:
  for path in sorted(a.packages.glob('*.zip')):
   with zipfile.ZipFile(path) as z:assert z.testzip() is None
   archives.append({'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size})
 report={'passed':True,'models':rows,'original_zip_sha256':sha(a.submission),'original_images':300,'archives':archives,'retrained':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
