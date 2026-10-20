"""Read-only hash check of every original source/data/experiment file."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RECORD=ROOT/'artifacts/turnflee-2026-10-06/preservation.json'


def check() -> dict:
    """Report changed/missing originals; additions are intentionally separate."""
    records=json.loads(RECORD.read_text(encoding='utf-8'))
    changed=[name for name,value in records.items() if not (ROOT/name).is_file()
             or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value]
    return dict(protected=len(records),changed=changed)


if __name__=='__main__':
    result=check(); print(json.dumps(result)); raise SystemExit(bool(result['changed']))
