"""Additive saved-only CSV representation adapter; frozen sources stay intact.

The sealed CSV uses NumPy 2.x scalar repr inside three-element vectors. JSON
stores the same finite float64 numbers without wrappers. Validate every number
bit-for-bit, then run the original independent auditor on a temporary CSV view.
No scientific record, receipt, policy or gate is edited or monkeypatched.
"""
import ast
import csv
import gzip
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

VECTORS={'position','orientation','velocity','predator_position'}


def csv_vector(value:str) -> list[float]:
    """Parse finite literal triples only; no Python expression is executed."""
    plain=re.sub(r'np\.float64\(([^()]*)\)',r'(\1)',value)
    try:vector=ast.literal_eval(plain)
    except (ValueError,SyntaxError) as exc:raise ValueError('invalid CSV vector literal') from exc
    if (not isinstance(vector,list) or len(vector)!=3
        or any(type(v) not in (int,float) or not math.isfinite(v) for v in vector)):
        raise ValueError('finite three-element CSV vector required')
    return [float(v) for v in vector]


def normalize_table(trace_path:Path,csv_path:Path,output:Path) -> int:
    """Compare raw CSV with JSON before writing a disposable canonical view."""
    intervals=0
    with gzip.open(trace_path,'rt') as trace,csv_path.open(newline='') as table,output.open('x',newline='') as target:
        reader=csv.DictReader(table);writer=csv.DictWriter(target,fieldnames=reader.fieldnames);writer.writeheader()
        for line in trace:
            record=json.loads(line);row=next(reader,None)
            if row is None:raise ValueError('missing CSV row')
            for field in reader.fieldnames:
                expected=record.get(field)
                if field in VECTORS:
                    actual=csv_vector(row[field])
                    if (not isinstance(expected,list) or len(expected)!=3
                        or [v.hex() for v in actual]!=[float(v).hex() for v in expected]):
                        raise ValueError('CSV vector value mismatch')
                    row[field]=str(actual)
                elif row[field]!=('' if expected is None else str(expected)):
                    raise ValueError('CSV scalar value mismatch')
            writer.writerow(row);intervals+=1
        if next(reader,None) is not None:raise ValueError('extra CSV row')
    return intervals


def audit(directory:Path) -> dict:
    """Retain raw seal/source checks and all independent numerical/gate checks."""
    from flybrain.escape_flight_validation.audit import audit as original,audit_completed
    from flybrain.flight3d.anatomy import digest
    # The original auditor verifies all raw hashes, inventory, circuit and
    # candidate sources before entering its CSV/JSON reconciliation loop.
    try:return original(directory)
    except ValueError as error:
        if str(error)!='CSV/JSON mismatch':raise
    config=json.loads((directory/'config.json').read_text());result=json.loads((directory/'results.json').read_text())
    root=Path(__file__).parent.resolve()
    with tempfile.TemporaryDirectory(prefix='saved-audit-view-',dir=root) as folder:
        view=Path(folder).resolve()
        if not view.is_relative_to(root):raise ValueError('temporary audit path outside artifact directory')
        count=normalize_table(directory/'traces.jsonl.gz',directory/'traces.csv',view/'traces.csv')
        for name in ('traces.jsonl.gz','progress.json','progress.reservations.jsonl'):
            shutil.copy2(directory/name,view/name)
        shutil.copytree(directory/'actions',view/'actions')
        checked=audit_completed(view,config,result)
    checked['serialization_adapter']=dict(source_sha256=digest(Path(__file__)),
        raw_csv_sha256=digest(directory/'traces.csv'),json_sha256=digest(directory/'traces.jsonl.gz'),
        compared_intervals=count,comparison='exact float64 hexadecimal values; scalar text unchanged',neural_steps=0)
    return checked


if __name__=='__main__':
    from flybrain.escape_flight_validation.runner import RUN
    print(json.dumps(audit(RUN),indent=2))
