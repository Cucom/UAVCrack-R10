"""Package only explicitly allowlisted files; verify every member SHA and CRC."""
import argparse,json,zipfile,hashlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
p.add_argument('--allowlist',type=Path,required=True,help='JSON array of relative file paths')
p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix',default='uavcrack_r10');a=p.parse_args()
files=json.loads(a.allowlist.read_text());assert files and len(files)==len(set(files))
root=a.root.resolve();rows=[]
for name in files:
 rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
 path=root/rel;assert path.is_file() and not path.is_symlink() and root in path.resolve().parents
 rows.append((path,(Path(a.prefix)/rel).as_posix(),hashlib.sha256(path.read_bytes()).hexdigest()))
a.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.output,'x',zipfile.ZIP_DEFLATED) as z:
 for path,name,digest in rows:z.write(path,name)
with zipfile.ZipFile(a.output) as z:
 assert z.testzip() is None
 for path,name,digest in rows:assert hashlib.sha256(z.read(name)).hexdigest()==digest
print(json.dumps({'file':str(a.output),'files':len(rows),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'CRC_and_member_SHA_passed':True}))
