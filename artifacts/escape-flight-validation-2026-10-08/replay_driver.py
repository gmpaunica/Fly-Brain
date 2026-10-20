"""Exact frozen replay after the additive saved CSV format audit; no new cases."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
import torch
from audit_serialized import audit
from flybrain.escape_flight_validation.replay import verify_case
from flybrain.escape_flight_validation.runner import RUN,case_order
from flybrain.escape_flight_validation.acceptance import ART
from flybrain.escape_flight_validation.worker import supervise
from flybrain.escape_diagnostic_journal import ReservationJournal
from flybrain.escape_diagnostic_recovery.runner import write
from flybrain.flight3d.anatomy import digest


def run() -> dict:
    """Use the retained replay function with separately journaled verification."""
    torch.set_num_threads(1);checked=audit(RUN)
    if checked['status']!='audit-passed':raise ValueError('saved audit required')
    write(ART/'replay-once.json',dict(receipt_sha256=digest(RUN/'receipt.json'),
        purpose='exact saved-action reproducibility; no new scientific cases',
        driver_sha256=digest(Path(__file__)),audit_adapter_sha256=digest(Path(__file__).with_name('audit_serialized.py'))))
    groups=defaultdict(list)
    with gzip.open(RUN/'traces.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);groups[(r['seed'],r['state'],r['intervention'])].append(r)
    budget=ReservationJournal(ART/'replay-progress.json',576000);cases=0;status='failed';error=None
    try:
        for case in case_order():
            path=RUN/'actions'/f'{case["seed"]}-{case["state"]}-{case["intervention"]}.json'
            verify_case(path,groups[(case['seed'],case['state'],case['intervention'])],budget)
            cases+=1;print(f'exact replay {cases}/48',flush=True)
        status='exact-replay-passed'
    except BaseException as exc:error=f'{type(exc).__name__}: {exc}';raise
    finally:
        budget.close()
        result=dict(status=status,error=error,cases=cases,replay_neural_steps=budget.steps,
            new_scientific_cases=0,receipt_sha256=digest(RUN/'receipt.json'),driver_sha256=digest(Path(__file__)))
        write(ART/'replay.json',result)
    return result


if __name__=='__main__':
    if '--supervise' in sys.argv:
        raise SystemExit(supervise([sys.executable,str(Path(__file__).resolve())],RUN,ART/'replay.log',ART/'replay-worker.json'))
    print(json.dumps(run(),indent=2))
