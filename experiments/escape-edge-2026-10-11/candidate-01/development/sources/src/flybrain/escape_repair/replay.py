"""Four predetermined full-interval exact replays, separately budgeted."""
import gzip
import json
from pathlib import Path
import torch
from flybrain.escape_diagnostic_journal import ReservationJournal
from flybrain.escape_diagnostic_recovery.runner import write
from flybrain.flight3d.anatomy import digest
from .session import RepairSession
from .validation import RUN, ART, cases, record_interval

INDICES = (0, 1, 4, 76)


def verify_case(path: Path, case: dict, expected: list[dict], budget) -> None:
    """Compare every interval value, including sensory hashes and neural counts."""
    session = RepairSession.from_log(path, replay=False)
    if len(session.log.steps) != len(expected):
        raise ValueError('replay duration mismatch')
    session.begin_replay()
    event = 0
    for index, (action, row) in enumerate(zip(session.log.steps, expected)):
        while event < len(session.log.spawns) and session.log.spawns[event].step == index:
            spawn = session.log.spawns[event]
            session.spawn(spawn.kind, spawn.position)
            event += 1
        budget.reserve_interval()
        if record_interval(session, case, index, action) != row:
            raise ValueError(f'exact replay mismatch at interval {index}')
    if event != len(session.log.spawns):
        raise ValueError('unconsumed replay spawn')


def run() -> dict:
    torch.set_num_threads(1)
    checked = json.loads((ART / 'audit.json').read_text())
    if checked['status'] != 'audit-passed' or checked['receipt_sha256'] != digest(RUN / 'receipt.json'):
        raise ValueError('saved audit required')
    write(ART / 'replay-once.json', dict(indices=INDICES, limit=48000, retry=False))
    budget = ReservationJournal(ART / 'replay-progress.json', 48000)
    status, error, count = 'failed', None, 0
    try:
        for index in INDICES:
            with gzip.open(RUN / 'traces' / f'{index:03}.jsonl.gz', 'rt') as stream:
                records = [json.loads(line) for line in stream]
            verify_case(RUN / 'actions' / f'{index:03}.json', cases()[index], records, budget)
            count += 1
            print(f'exact replay {count}/4', flush=True)
        status = 'exact-replay-passed'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        budget.close()
        result = dict(status=status, error=error, cases=count, replay_neural_steps=budget.steps,
                      new_scientific_cases=0, receipt_sha256=digest(RUN / 'receipt.json'))
        write(ART / 'replay.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
