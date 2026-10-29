"""Explicit secondary-manifest adapter; original model and Trainer unchanged."""
import hashlib,json
from pathlib import Path
from projects.uav_crack.resunet_description import CrackDataset

RECEIPTS=[]
def dataset_factory(root,manifest,split,augment=False,limit=0,*,recipe,protocol):
    active=protocol['secondary_fold'];primary=Path(manifest)
    assert primary==Path('manifests/split.csv')
    assert hashlib.sha256(primary.read_bytes()).hexdigest()==protocol['split_sha256']
    path=Path(active['manifest']);assert hashlib.sha256(path.read_bytes()).hexdigest()==active['sha256']
    assert split in ('train','val'),'secondary diagnostics disallow full-data'
    assert recipe=='r4', 'Only retained ResNet34 recipe is supported'
    cls=CrackDataset
    data=cls(root,path,split,augment,limit)
    receipt={'recipe':recipe,'fold_id':active['id'],'active_manifest':str(path),'active_manifest_sha256':active['sha256'],'split':split,'count':len(data),'augment':augment,'limit':limit,'source_groups':sorted({r['source_group'] for r in data.rows}),'rows_sha256':hashlib.sha256(json.dumps(data.rows,sort_keys=True).encode()).hexdigest(),'images':[r['image'] for r in data.rows]}
    RECEIPTS.append(receipt);print(json.dumps({'actual_dataset':{k:v for k,v in receipt.items() if k not in ('images','source_groups')}}),flush=True)
    return data
