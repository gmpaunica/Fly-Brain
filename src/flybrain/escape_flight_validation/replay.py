"""Exact saved-action replay verification, separately counted from science."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import torch
from flybrain.escape_diagnostic_journal import ReservationJournal
from flybrain.escape_diagnostic_recovery.runner import write
from flybrain.flight3d.anatomy import digest
from .records import load_actions,setup_case,advance_record
from .runner import RUN,case_order
from .acceptance import ART
from .audit import audit


def verify_case(path:Path,records:list[dict],budget=None) -> int:
    """Compare every saved interval field, including counts and intervention."""
    case,candidate,actions=load_actions(path)
    if len(actions)!=len(records):raise ValueError('exact replay duration mismatch')
    session=setup_case(case,candidate)
    for i,(action,expected) in enumerate(zip(actions,records)):
        if budget is not None:budget.reserve_interval()
        actual=advance_record(session,case,i,action)
        canonical=lambda r:json.loads(json.dumps(r,allow_nan=False))
        if canonical(actual)!=canonical(expected):
            changed=[k for k in actual if canonical(actual[k])!=canonical(expected.get(k))]
            raise ValueError(f'exact replay mismatch interval {i}: {changed}')
    return len(records)*40


def run(directory:Path=RUN,art:Path=ART) -> dict:
    """Replay existing cases only; no stimulus search or scientific retry."""
    torch.set_num_threads(1);checked=audit(directory)
    if checked['status']!='audit-passed':raise ValueError('complete saved audit required before replay')
    write(art/'replay-once.json',dict(receipt_sha256=digest(directory/'receipt.json'),purpose='exact action reproducibility; no new scientific cases'))
    groups=defaultdict(list)
    with gzip.open(directory/'traces.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);groups[(r['seed'],r['state'],r['intervention'])].append(r)
    budget=ReservationJournal(art/'replay-progress.json',576000);cases=0;status='failed';error=None
    try:
        for case in case_order():
            path=directory/'actions'/f'{case["seed"]}-{case["state"]}-{case["intervention"]}.json'
            verify_case(path,groups[(case['seed'],case['state'],case['intervention'])],budget)
            cases+=1;print(f'exact replay {cases}/48',flush=True)
        status='exact-replay-passed'
    except BaseException as exc:error=f'{type(exc).__name__}: {exc}';raise
    finally:
        budget.close()
        result=dict(status=status,error=error,cases=cases,replay_neural_steps=budget.steps,
            new_scientific_cases=0,receipt_sha256=digest(directory/'receipt.json'))
        write(art/'replay.json',result)
    return result


if __name__=='__main__':print(json.dumps(run(),indent=2))
