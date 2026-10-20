"""Protect the existing science/cache/experiment bytes during a display repair."""
import hashlib
import json
from pathlib import Path

root=Path.cwd();out=Path(__file__).resolve().parent
allowed={'src/flybrain/flight3d/activity.py','src/flybrain/flight3d/brain_controls.py'}
before=out/'preservation-before.json'
if not before.exists():
    records={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ('src','data','experiments') for p in (root/folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    before.write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(f'Snapshotted {len(records)} existing files')
else:
    records=json.loads(before.read_text(encoding='utf-8'))
    changed=[name for name,digest in records.items() if not (root/name).exists()
             or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
    unexpected=[name for name in changed if name not in allowed]
    result={'existing_files':len(records),'display_changes':changed,
            'protected_unchanged':len(records)-len(allowed),'unexpected_changes':unexpected}
    (out/'preservation-final.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));assert not unexpected
