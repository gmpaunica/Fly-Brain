"""Four predetermined exact neural/sensory/physical replays; no new science."""
import gzip
import json
from pathlib import Path
import torch
from flybrain.escape_diagnostic_journal import ReservationJournal
from flybrain.escape_diagnostic_recovery.runner import write
from flybrain.flight3d.anatomy import digest
from .session import EdgeSession
from .validation import ART, BASE, cases, record_interval

SELECTED = (('development',0),('development',1),('development',4),('robustness',28))


def verify_case(path: Path, case: dict, expected: list[dict], budget) -> None:
    """Check every interval field, including eye hashes, spikes and servo state."""
    session = EdgeSession.from_log(path,replay=False)
    if len(session.log.steps)!=len(expected):
        raise ValueError('replay duration mismatch')
    session.begin_replay()
    event = 0
    for index,(action,row) in enumerate(zip(session.log.steps,expected)):
        while event<len(session.log.spawns) and session.log.spawns[event].step==index:
            spawn = session.log.spawns[event]
            session.spawn(spawn.kind,spawn.position)
            event += 1
        budget.reserve_interval()
        if record_interval(session,case,index,action)!=row:
            raise ValueError(f'exact replay mismatch at interval {index}')
    if event!=len(session.log.spawns):
        raise ValueError('unconsumed replay spawn')


def run(base: Path = BASE, art: Path = ART) -> dict:
    """Consume the declared 96,000-step verification budget once."""
    torch.set_num_threads(1)
    receipts = {}
    for phase in ('development','robustness'):
        checked = json.loads((art/f'{phase}-audit.json').read_text())
        receipts[phase] = digest(base/phase/'receipt.json')
        if checked['status']!='audit-passed' or checked['experiment_status']!='passed' or checked['receipt_sha256']!=receipts[phase]:
            raise ValueError('passing saved audit required')
    write(art/'replay-once.json',dict(selected=SELECTED,limit=96000,new_scientific_cases=0))
    budget = ReservationJournal(art/'replay-progress.json',96000)
    count = 0
    status,error = 'failed',None
    try:
        for phase,index in SELECTED:
            config = json.loads((base/phase/'config.json').read_text())
            with gzip.open(base/phase/'traces'/f'{index:03}.jsonl.gz','rt') as stream:
                records = [json.loads(line) for line in stream]
            verify_case(base/phase/'actions'/f'{index:03}.json',cases(phase,config['seed_offset'])[index],records,budget)
            count += 1
            print(f'exact replay {count}/4 | {phase} case {index}',flush=True)
        status = 'exact-replay-passed'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        budget.close()
        value = dict(status=status,error=error,cases=count,neural_steps=budget.steps,
            new_scientific_cases=0,receipts=receipts)
        write(art/'replay.json',value)
    return value


if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
