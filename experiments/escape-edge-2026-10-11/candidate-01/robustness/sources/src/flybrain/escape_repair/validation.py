"""One declared full-neural repair evaluation, with bounded durable stepping."""
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
from flybrain.escape_flight_validation.runner import summary
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.flight3d.anatomy import digest
from flybrain.turnflee.protocol import closed_cases
from .session import RepairSession, source_hashes

ART = Path('artifacts/escape-repair-2026-10-08')
RUN = Path('experiments/escape-repair-2026-10-08/run')
LIMIT = 1152000


def cases() -> list[dict]:
    """Historical geometry and a prospectively fixed additional seed set."""
    return [dict(c, seed=c['seed'] + offset, benchmark=label)
            for label, offset in (('historical', 0), ('seed-robustness', 200000)) for c in closed_cases()]


def gate(rows: list[dict]) -> bool:
    """Every intact escape must pass; zero silent triggers; no missing cases."""
    expected = cases()
    return (len(rows) == len(expected)
            and all(all(row.get(k) == v for k, v in case.items()) for row, case in zip(rows, expected))
            and all(r['success'] if r['intervention'] == 'intact' else r['triggers'] == 0 for r in rows))


def setup_case(case: dict, candidate: CandidateConfig) -> RepairSession:
    """Initialize evaluator geometry independently of controller calibration."""
    session = RepairSession(case['seed'], candidate, case['state'])
    origin = session.world.position
    angle = math.radians(case['azimuth'])
    predator = (float(origin[0] + .075 * math.sin(angle)),
                float(origin[1] + .075 * math.cos(angle)), max(.021, float(origin[2])))
    session.spawn('predator', predator)
    session.set_intervention(case['intervention'])
    return session


def record_interval(session: RepairSession, case: dict, interval: int, action=None) -> dict:
    """Canonical numerical serialization plus auditable clearance measurement."""
    record = advance_record(session, case, interval, action)
    record['clearance_growth_degrees_s'] = session.threat.clearance_growth
    return json.loads(json.dumps(record, allow_nan=False))


def totals(rows: list[dict]) -> dict:
    return dict(intact_successes=sum(r['success'] for r in rows if r['intervention'] == 'intact'),
        silent_triggers=sum(r['triggers'] for r in rows if r['intervention'] == 'visual-silent'),
        per_benchmark={label: sum(r['success'] for r in rows if r['intervention'] == 'intact' and r['benchmark'] == label)
                       for label in ('historical', 'seed-robustness')})


def run(directory: Path = RUN) -> dict:
    """Consume once, reserve before stepping, seal complete or failed execution."""
    torch.set_num_threads(1)
    preflight = json.loads((ART / 'preflight.json').read_text())
    verify_hashes(preflight['bound'])
    verify_hashes(preflight['proofs'])
    verify_hashes(preflight['protected'])
    if preflight['repair_sources'] != source_hashes() or preflight['status'] != 'passed':
        raise ValueError('repair preflight/source mismatch')
    candidate = CandidateConfig.from_record(preflight['candidate'])
    directory.parent.mkdir(parents=True, exist_ok=True)
    write(directory.parent / 'once.json', dict(directory=str(directory.resolve()), retry=False))
    directory.mkdir()
    rows = []
    budget = None
    error = None
    status = 'failed'
    try:
        config = dict(schema='escape-repair-validation-v1', cases=cases(), seconds_per_case=6,
            max_kernel_steps=LIMIT, hypothesis='Visual clearance and bounded idle recovery complete every declared intact escape.',
            criteria=dict(intact=48, silent_triggers=0, turn_degrees=90, turn_tolerance_degrees=3,
                          separation_m=.02, braking_speed_m_s=.02, stable_recovery_ticks=15),
            candidate=asdict(candidate), candidate_identity=candidate.identity,
            bound=preflight['bound'], repair_sources=source_hashes(), protected=preflight['protected'],
            preflight_sha256=digest(ART / 'preflight.json'), python=platform.python_version(),
            software={n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'panda3d', 'pygame')},
            device='cpu', threads=1, commit=None, cells=589, anatomical_edges=4139,
            gains=[4, 2], learning_rate=None, checkpoint=None, optimizer_updates=0,
            stop='health/source/measurement/I-O/budget fault, otherwise all 96; no automatic retry')
        write(directory / 'config.json', config)
        for name, value in config['bound'].items():
            archive = directory / 'sources' / name
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(Path(name).read_bytes())
            if digest(archive) != value:
                raise ValueError('source archive mismatch')
        (directory / 'traces').mkdir()
        (directory / 'actions').mkdir()
        budget = ReservationJournal(directory / 'progress.json', LIMIT)
        for index, case in enumerate(config['cases']):
            candidate.verify()
            session = setup_case(case, candidate)
            records = []
            with gzip.open(directory / 'traces' / f'{index:03}.jsonl.gz', 'xt', encoding='utf-8') as trace:
                for interval in range(300):
                    verify_hashes(config['repair_sources'])
                    budget.reserve_interval()
                    r = record_interval(session, case, interval)
                    trace.write(json.dumps(r, allow_nan=False, separators=(',', ':')) + '\n')
                    trace.flush()
                    records.append(r)
            session.log.save(directory / 'actions' / f'{index:03}.json')
            row = summary(case, records)
            rows.append(row)
            write(directory / f'case-{index:03}.json', row)
            print(f'{index + 1}/96 | {case["benchmark"]} {case["azimuth"]} {case["state"]} {case["intervention"]} | success={row["success"]} | intact={totals(rows)["intact_successes"]}', flush=True)
        verify_hashes(config['bound'])
        verify_hashes(config['protected'])
        status = 'passed' if gate(rows) else 'closed-physical-failed'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if budget:
            budget.close()
        result = dict(status=status, error=error, cases=len(rows), closed_loop=rows,
                      reserved_kernel_steps=budget.steps if budget else 0,
                      optimizer_updates=0, retry_authorized=False, **totals(rows))
        write(directory / 'results.json', result)
        if rows:
            with (directory / 'results.csv').open('x', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows({k: json.dumps(v) for k, v in row.items()} for row in rows)
        outputs = {p.relative_to(directory).as_posix(): digest(p) for p in directory.rglob('*') if p.is_file()}
        write(directory / 'receipt.json', dict(status=status, outputs=outputs, candidate_identity=candidate.identity))
        write(directory.parent / 'seal.json', dict(status=status, receipt_sha256=digest(directory / 'receipt.json')))
    return result


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k != 'closed_loop'}, indent=2))
