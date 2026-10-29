"""Training bootstrap and strict checks for the retained final recipes."""
import csv, hashlib, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Local legacy environment only. Clean installations use the pinned training requirements.
compat = ROOT / 'private/runtime_compat'
if compat.exists():
    sys.path.insert(0, str(compat))
if (ROOT / 'upstream/mmsegmentation/mmseg').exists():
    sys.path.insert(0, str(ROOT / 'upstream/mmsegmentation'))
os.chdir(ROOT)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

MANIFESTS = {
    'full': ('manifests/full_training_p59.csv', 'cfe7c73049c0338977357caeb7a35b536093c5e325d88743e7bf87496bf585a2'),
    'primary': ('manifests/split.csv', '4a8edf368bbc4768f00ff0c1d39e64e918a671a9311523ea7f50f1b511eb8ef2'),
    'secondary': ('manifests/secondary_group_split_v2.csv', 'c2c94da23284257c6b2a3353ab9372dfe2ada75c03401135e83073c0fd836b5d'),
}

def manifest_for(fold, data_root):
    name, expected = MANIFESTS[fold]
    path = ROOT / name
    if sha(path) != expected:
        raise ValueError('Manifest identity mismatch: ' + name)
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    tr = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    assert (len(tr), len(va)) == ((1200, 0) if fold == 'full' else (984, 216))
    assert {r['source_group'] for r in tr}.isdisjoint({r['source_group'] for r in va})
    assert len({r['image'] for r in rows}) == 1200
    for row in rows:
        for key in ('image', 'mask'):
            if not (Path(data_root) / row[key]).is_file():
                raise FileNotFoundError(row[key])
    return path
