"""Extract pure state_dict files using an explicit family/seed/checkpoint manifest.

Only load trusted training checkpoints: Lightning files may contain pickle objects.
The input JSON contains files: [{family, seed, checkpoint, sha256}].
"""
import argparse,json,hashlib,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'deploy');a=p.parse_args()
sys.path.insert(0,str(a.source.resolve()))
import torch
from models import build
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True);rows=[]
for entry in json.loads(a.manifest.read_text())['files']:
 path=Path(entry['checkpoint']);digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==entry['sha256']
 ck=torch.load(path,map_location='cpu',weights_only=False);state=ck['state_dict']
 model=build(entry['family']);model.load_state_dict(state,strict=True)
 assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
 dest=a.output/f"{entry['family']}_seed{entry['seed']}.pth";torch.save(state,dest)
 rows.append({'family':entry['family'],'seed':entry['seed'],'file':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'source_checkpoint_sha256':digest})
(a.output/'WEIGHTS.json').write_text(json.dumps({'files':rows},indent=2)+'\n')
