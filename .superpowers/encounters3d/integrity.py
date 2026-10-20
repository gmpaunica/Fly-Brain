"""Preserve historical code, data and experiments with a streamed hash ledger."""
import hashlib
import json
from pathlib import Path
import sys

root=Path.cwd()
ledger=root/'.superpowers/encounters3d/protected.json'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
if sys.argv[1]=='record':
    paths=[p for folder in ('src','tests','data','experiments') for p in (root/folder).rglob('*')
           if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    ledger.write_text(json.dumps({p.relative_to(root).as_posix():sha(p) for p in paths},indent=2))
    print('protected',len(paths))
else:
    records=json.loads(ledger.read_text())
    changed=[p for p,h in records.items() if not (root/p).exists() or sha(root/p)!=h]
    print('protected',len(records),'changed',changed)
    assert not changed
