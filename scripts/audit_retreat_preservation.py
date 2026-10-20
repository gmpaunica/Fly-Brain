"""Hash original source/data/experiments before additive retreat development."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'artifacts/retreat3d-2026-10-06/preservation.json'

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__ == '__main__':
    if sys.argv[1] == 'record':
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        if RECORD.exists():
            raise FileExistsError(RECORD)
        files = [p for folder in ('src', 'data', 'experiments')
                 for p in (ROOT / folder).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.suffix != '.pyc']
        RECORD.write_text(json.dumps({p.relative_to(ROOT).as_posix(): digest(p)
                                      for p in files}, indent=2), encoding='utf-8')
        print(f'Recorded {len(files)} protected files')
    else:
        files = json.loads(RECORD.read_text(encoding='utf-8'))
        changed = [name for name, value in files.items()
                   if not (ROOT / name).exists() or digest(ROOT / name) != value]
        print(json.dumps(dict(protected=len(files), changed=changed)))
        sys.exit(bool(changed))
