"""Bounded, source-frozen development and seed-robustness physical evaluation."""
import argparse
import csv
from dataclasses import asdict
import gzip
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import torch

from flybrain.escape_diagnostic_journal import ReservationJournal
from flybrain.escape_diagnostic_recovery.runner import write, verify_hashes
from flybrain.escape_flight_validation.records import advance_record
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.flight3d.anatomy import digest
from flybrain.turnflee.protocol import closed_cases, correct_side
from .session import EdgeSession, source_hashes
from .metrics import edge_metrics

ART = Path('artifacts/escape-edge-2026-10-11')
BASE = Path('experiments/escape-edge-2026-10-11/candidate-01')
LIMIT = 1152000
INTERVALS = 600


def cases(phase: str, offset: int = 400000) -> list[dict]:
    """Historical development; seed robustness reuses historical geometry."""
    if phase not in ('development', 'robustness'):
        raise ValueError('development or robustness required')
    if type(offset) is not int or offset <= 0:
        raise ValueError('positive seed offset required')
    return [dict(**{**c, 'seed': c['seed']+(offset if phase == 'robustness' else 0)},
                 benchmark=phase) for c in closed_cases()]


def gate(expected: list[dict], rows: list[dict]) -> bool:
    """Require every named intact case and zero silent triggers, without duplicates."""
    fields = ('azimuth', 'state', 'intervention', 'seed', 'benchmark')
    key = lambda r: tuple(r[k] for k in fields)
    return (len(rows) == len(expected) == 48 and [key(r) for r in rows] == [key(c) for c in expected]
        and all(r['success'] for r in rows if r['intervention'] == 'intact')
        and all(r['triggers'] == 0 for r in rows if r['intervention'] == 'visual-silent'))


def setup_case(case: dict, candidate: CandidateConfig) -> EdgeSession:
    """Evaluator geometry stays outside the separate named-start calibration."""
    session = EdgeSession(case['seed'], candidate, case['state'])
    origin = session.world.position.copy()
    angle = math.radians(case['azimuth'])
    predator = (float(origin[0]+.075*math.sin(angle)), float(origin[1]+.075*math.cos(angle)),
                max(.021, float(origin[2])))
    session.spawn('predator', predator)
    session.set_intervention(case['intervention'])
    return session


def record_interval(session: EdgeSession, case: dict, interval: int, action=None) -> dict:
    """Record actual movement and passive estimated control state canonically."""
    r = advance_record(session, case, interval, action)
    p = session.controller.escape
    r.update(endpoint_estimate=None if p.endpoint is None else p.endpoint.tolist(),
        estimated_displacement=session.controller.idle.estimate.displacement.tolist(),
        desired_velocity=p.desired_velocity.tolist(), braking_reason=p.braking_reason,
        clearance_growth_degrees_s=session.threat.clearance_growth)
    return json.loads(json.dumps(r, allow_nan=False))


def summary(case: dict, rows: list[dict]) -> dict:
    """Combine unchanged neural response criteria with the new edge motion contract."""
    onset = next((r['end_seconds'] for r in rows if r['threat']['strong']), None)
    first = next((r for r in rows if r['triggers']), None)
    trigger = first['end_seconds'] if first else None
    side = first['selected_side'] if first else None
    timely = bool(trigger is not None and onset is not None
        and 0 <= trigger-onset <= .5+1e-12 and correct_side(case['azimuth'], side))
    metrics = edge_metrics(case, trigger, side, rows)
    return dict(**case, trigger_seconds=trigger, side=side, first_strong_seconds=onset,
        timely_correct_side=timely, **metrics, success=bool(timely and metrics['physical_success']),
        triggers=rows[-1]['triggers'], finite=all(r['finite'] for r in rows),
        max_downstream_hz=max(r['max_downstream_hz'] for r in rows))


def run(phase: str, base: Path = BASE, art: Path = ART, offset: int = 400000) -> dict:
    """Consume a fixed batch once; always seal terminal results, never auto-retry."""
    torch.set_num_threads(1)
    preflight = json.loads((art/'preflight.json').read_text())
    verify_hashes(preflight['bound'])
    verify_hashes(preflight['proofs'])
    verify_hashes(preflight['protected'])
    if preflight['edge_sources'] != source_hashes() or preflight['status'] != 'passed':
        raise ValueError('edge preflight/source mismatch')
    candidate = CandidateConfig.from_record(preflight['candidate'])
    if phase == 'robustness':
        prior = json.loads((base/'development/results.json').read_text())
        prior_config = json.loads((base/'development/config.json').read_text())
        if prior['status'] != 'passed' or prior_config['bound'] != preflight['bound']:
            raise ValueError('passing source-frozen development required')
    expected = cases(phase, offset)
    base.mkdir(parents=True, exist_ok=True)
    write(base/f'{phase}-once.json', dict(phase=phase, consumed=True, automatic_retry=False))
    directory = base/phase
    directory.mkdir()
    budget = None
    results = []
    status, error = 'failed', None
    try:
        config = dict(schema='escape-edge-validation-v1', phase=phase, cases=expected,
            seconds_per_case=12, max_kernel_steps=LIMIT,
            hypothesis='Latched IMU edge flight reaches and holds a safe map edge without premature recovery.',
            criteria=dict(intact_successes=24, silent_triggers=0, arrival_seconds=10,
                hover_seconds=2, wall_distance_m=[.005,.025], hover_speed_m_s=.01,
                hover_altitude_m=[.085,.115], side_deadline_seconds=.5, turn_tolerance_degrees=3,
                takeoff_height_m=.008, takeoff_intervals=3, separation_gain_m=.02,
                premature_speed_m_s=.04, cruise_wall_distance_m=.06, startup_grace_seconds=.4,
                reverse_step_m=.002),
            candidate=asdict(candidate), candidate_identity=candidate.identity,
            bound=preflight['bound'], edge_sources=source_hashes(), protected=preflight['protected'],
            proofs=preflight['proofs'], preflight_sha256=digest(art/'preflight.json'),
            seed_offset=offset, python=platform.python_version(), platform=platform.platform(),
            software={n:importlib.metadata.version(n) for n in ('torch','numpy','panda3d','pygame')},
            device='cpu', threads=1, commit=None, cells=589, anatomical_edges=4139,
            gains=[4,2], optimizer_updates=0, learning_rate=None, checkpoint=None,
            stop='health/source/measurement/I-O/budget fault; otherwise all 48 cases; seal, no auto-retry')
        write(directory/'config.json', config)
        for name, value in config['bound'].items():
            archived = directory/'sources'/name
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(Path(name).read_bytes())
            if digest(archived) != value:
                raise ValueError('source archive mismatch')
        (directory/'traces').mkdir()
        (directory/'actions').mkdir()
        budget = ReservationJournal(directory/'progress.json', LIMIT)
        for index, case in enumerate(expected):
            verify_hashes(config['bound'])
            session = setup_case(case, candidate)
            records = []
            with gzip.open(directory/'traces'/f'{index:03}.jsonl.gz', 'xt', encoding='utf-8') as trace:
                for interval in range(INTERVALS):
                    if interval % 50 == 0:
                        verify_hashes(config['bound'])
                    budget.reserve_interval()
                    r = record_interval(session, case, interval)
                    trace.write(json.dumps(r, allow_nan=False, separators=(',',':'))+'\n')
                    trace.flush()
                    records.append(r)
            session.log.save(directory/'actions'/f'{index:03}.json')
            row = summary(case, records)
            results.append(row)
            write(directory/f'case-{index:03}.json', row)
            print(f'{phase} {index+1}/48 | {case["azimuth"]} {case["state"]} {case["intervention"]} | success={row["success"]} | arrival={row["arrival_seconds"]}', flush=True)
        verify_hashes(config['bound'])
        verify_hashes(config['protected'])
        status = 'passed' if gate(expected, results) else 'closed-physical-failed'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if budget:
            budget.close()
        result = dict(status=status, error=error, cases=len(results), closed_loop=results,
            intact_successes=sum(r['success'] for r in results if r['intervention']=='intact'),
            silent_triggers=sum(r['triggers'] for r in results if r['intervention']=='visual-silent'),
            reserved_kernel_steps=budget.steps if budget else 0, optimizer_updates=0)
        write(directory/'results.json', result)
        if results:
            with (directory/'results.csv').open('x', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows({k:json.dumps(v) for k,v in r.items()} for r in results)
        outputs = {p.relative_to(directory).as_posix():digest(p) for p in directory.rglob('*') if p.is_file()}
        write(directory/'receipt.json', dict(status=status, outputs=outputs, candidate_identity=candidate.identity))
        write(base/f'{phase}-seal.json', dict(status=status, receipt_sha256=digest(directory/'receipt.json')))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('development','robustness'))
    args = parser.parse_args()
    result = run(args.phase)
    print(json.dumps({k:v for k,v in result.items() if k != 'closed_loop'}, indent=2))
