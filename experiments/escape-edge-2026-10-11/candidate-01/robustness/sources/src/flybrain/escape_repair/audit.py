"""Independent saved arithmetic, physical metrics and source/budget audit."""
from collections import deque
import csv
import gzip
import json
import math
from pathlib import Path
from flybrain.escape_diagnostic_recovery.runner import verify_hashes
from flybrain.escape_flight_validation.audit import audit_temporal, reconstruct_physical, close_enough, require
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.flight3d.anatomy import digest, load_manifest
from flybrain.turnflee.audit import audit_counts
from .session import source_hashes
from .validation import cases, LIMIT


def check_case(case: dict, records: list[dict], saved: dict, manifest: dict) -> None:
    """Reconstruct counts/readout, sustained image clearance and actual flight."""
    require(len(records) == 300, 'case duration mismatch')
    require(all(all(r.get(k) == v for k, v in case.items()) for r in records), 'case identity mismatch')
    counts = audit_counts(records, manifest)
    trigger, side = audit_temporal(records, manifest)
    require(counts['finite'] and counts['max_downstream_hz'] <= 240, 'neural health mismatch')
    small = deque(maxlen=15)
    for r in records:
        size = r['threat']['size_degrees']
        require(math.isfinite(size), 'nonfinite threat size')
        if size > 10:
            small.clear()
        else:
            small.append(size)
        growth = (small[-1] - small[0]) / .28 if len(small) == 15 else 0.
        weak = len(small) == 15 and growth <= 5
        require(r['threat']['weak_continuous'] == weak, 'clearance timer mismatch')
        require(close_enough(growth, r['clearance_growth_degrees_s']), 'clearance growth mismatch')
    if case['intervention'] == 'visual-silent':
        vision = [i for i, node in enumerate(manifest['nodes']) if 'vision' in node['roles']]
        require(all(r[k][i] == 0 for r in records for k in ('actual_counts', 'reference_counts') for i in vision),
                'silent intervention mismatch')
    onset = next((r['end_seconds'] for r in records if r['threat']['strong']), None)
    angle = case['azimuth'] % 360
    correct = side in ('left', 'right') and (angle in (0, 180) or side == ('left' if 0 < angle < 180 else 'right'))
    timely = bool(trigger is not None and onset is not None and 0 <= trigger - onset <= .5 + 1e-12 and correct)
    physical = reconstruct_physical(case, trigger, side, records)
    expected = dict(**case, trigger_seconds=trigger, side=side, first_strong_seconds=onset,
                    timely_correct_side=timely, **physical, success=bool(timely and physical['physical_success']),
                    triggers=records[-1]['triggers'], finite=counts['finite'], max_downstream_hz=counts['max_downstream_hz'])
    require(all(k in saved and close_enough(v, saved[k]) for k, v in expected.items()), 'case summary mismatch')


def audit(directory: Path) -> dict:
    """No brain construction or simulation: reconcile all recorded evidence."""
    receipt = json.loads((directory / 'receipt.json').read_text())
    config = json.loads((directory / 'config.json').read_text())
    result = json.loads((directory / 'results.json').read_text())
    seal = json.loads((directory.parent / 'seal.json').read_text())
    require(seal['receipt_sha256'] == digest(directory / 'receipt.json'), 'seal mismatch')
    require(receipt['status'] == result['status'] == seal['status'], 'terminal status mismatch')
    verify_hashes({str(directory / name): h for name, h in receipt['outputs'].items()})
    require({p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() and p != directory / 'receipt.json'} == set(receipt['outputs']), 'output inventory mismatch')
    verify_hashes(config['bound'])
    verify_hashes({str(directory / 'sources' / name): h for name, h in config['bound'].items()})
    verify_hashes(config['protected'])
    require(config['repair_sources'] == source_hashes(), 'repair source inventory mismatch')
    candidate = CandidateConfig.from_record(config['candidate'])
    require(candidate.identity == config['candidate_identity'] == receipt['candidate_identity'], 'candidate binding mismatch')
    require(config['cases'] == cases() and config['max_kernel_steps'] == LIMIT, 'declared protocol mismatch')
    rows = result['closed_loop']
    require(len(rows) == 96 and result['cases'] == 96, 'missing cases')
    manifest = load_manifest()
    with (directory / 'results.csv').open(newline='') as stream:
        table = list(csv.DictReader(stream))
    require(len(table) == 96, 'CSV coverage mismatch')
    for index, (case, row) in enumerate(zip(cases(), rows)):
        require({k: json.loads(v) for k, v in table[index].items()} == row, 'CSV summary mismatch')
        with gzip.open(directory / 'traces' / f'{index:03}.jsonl.gz', 'rt') as stream:
            records = [json.loads(line) for line in stream]
        check_case(case, records, row, manifest)
        action = json.loads((directory / 'actions' / f'{index:03}.json').read_text())
        require(action['seed'] == case['seed'] and action['initial_state'] == case['state']
                and action['candidate'] == config['candidate'] and action['repair_sources'] == config['repair_sources']
                and action['intervention'] == case['intervention'] and action['dt_seconds'] == .02
                and action['steps'] == [[0., 0., 0.]] * 300, 'action configuration mismatch')
        az = math.radians(case['azimuth'])
        expected_spawn = [dict(step=0, kind='predator', position=[.3 + .075 * math.sin(az), .3 + .075 * math.cos(az), .021 if case['state'] == 'resting' else .1])]
        require(action['spawns'] == expected_spawn, 'action spawn mismatch')
    progress = json.loads((directory / 'progress.json').read_text())
    require(progress['limit'] == LIMIT, 'reservation limit mismatch')
    with (directory / progress['journal']).open() as stream:
        reservations = [json.loads(line) for line in stream]
    require(len(reservations) == 28800 and all(r == dict(reserved_steps=(i + 1) * 40, limit=LIMIT) for i, r in enumerate(reservations)), 'reservation accounting mismatch')
    successes = sum(r['success'] for r in rows if r['intervention'] == 'intact')
    silent = sum(r['triggers'] for r in rows if r['intervention'] == 'visual-silent')
    benchmarks = {label: sum(r['success'] for r in rows if r['intervention'] == 'intact' and r['benchmark'] == label) for label in ('historical', 'seed-robustness')}
    status = 'passed' if successes == 48 and silent == 0 else 'closed-physical-failed'
    require(result['status'] == status and result['intact_successes'] == successes
            and result['silent_triggers'] == silent and result['per_benchmark'] == benchmarks
            and result['reserved_kernel_steps'] == LIMIT and result['optimizer_updates'] == 0, 'terminal gate mismatch')
    return dict(status='audit-passed', experiment_status=status, cases=96, intact_successes=successes,
                silent_triggers=silent, per_benchmark=benchmarks, audit_neural_steps=0,
                receipt_sha256=digest(directory / 'receipt.json'))


if __name__ == '__main__':
    from .validation import RUN, ART
    from flybrain.escape_diagnostic_recovery.runner import write
    result = audit(RUN)
    write(ART / 'audit.json', result)
    print(json.dumps(result, indent=2))
