"""Snapshot protected source/cache/experiment bytes for this display task."""
import hashlib
import json
from pathlib import Path

root = Path.cwd()
out = root / '.superpowers/activity-classes-2026-10-04'
allowed = {'src/flybrain/flight3d/anatomical_viewer.py',
           'src/flybrain/flight3d/brain_controls.py',
           'src/flybrain/flight3d/branch_activity.py'}
before = out / 'preservation-before.json'
if not before.exists():
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ('src', 'data', 'experiments')
              for p in (root / folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    before.write_text(json.dumps(hashes, indent=2))
    print(f'Snapshotted {len(hashes)} existing files')
else:
    hashes = json.loads(before.read_text())
    changed = [name for name, digest in hashes.items()
               if not (root / name).exists()
               or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    unexpected = [name for name in changed if name not in allowed]
    record = {'existing_files': len(hashes), 'allowed_display_changes': changed,
              'protected_unchanged': len(hashes) - len(allowed),
              'unexpected_changes': unexpected}
    (out / 'preservation-final.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    assert not unexpected
