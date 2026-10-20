"""One fixed source-bound calibration/validation; fail closed, never tune/retry."""
import argparse
from dataclasses import asdict
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import numpy as np
import torch
from flybrain.flight3d.anatomy import CACHE, digest, load_manifest
from flybrain.flight3d.contracts import FlightObservation
from flybrain.encounters3d.scene import SceneSpec, encounter_eyes
from .neural import BrainPair, FrozenDecoder

MAX_SECONDS = 504
MAX_NEURAL_STEPS = 1008000


def cases(phase: str) -> list[dict]:
    """Exactly the requested azimuth/elevation/approach combinations."""
    if phase not in ('calibration', 'validation'):
        raise ValueError('calibration or validation required')
    azimuths = range(0, 360, 45) if phase == 'calibration' else [11.25+22.5*i for i in range(16)]
    return [dict(phase=phase, azimuth=azimuth, elevation=elevation,
                 profile=profile, seed=1000000+i*6+j*2+profile)
            for i, azimuth in enumerate(azimuths) for j, elevation in enumerate((-20, 0, 20))
            for profile in range(2)]


def closed_cases() -> list[dict]:
    return [dict(azimuth=45*i, state=state, intervention=condition, seed=1010000+i)
            for i in range(8) for state in ('resting', 'hovering', 'cruising')
            for condition in ('intact', 'visual-silent')]


def score_windows(windows: list[np.ndarray], decoder: FrozenDecoder, azimuth: float) -> dict:
    """Require usable direction and no wrong confident post-onset classification."""
    decoded = [decoder.decode(w) for w in windows]
    errors = [abs((math.degrees(r.heading_radians)-azimuth+180) % 360-180)
              for r in decoded if r.confident]
    return dict(success=bool(errors) and max(errors) <= 22.5,
                usable_windows=len(errors), errors_degrees=errors,
                decoded=[asdict(r) for r in decoded])


def validation_gate(rows: list[dict]) -> bool:
    expected = cases('validation')
    return (len(rows) == 96 and sum(r['success'] for r in rows) >= 90 and
            all(sum(r['success'] for r in rows if r['azimuth'] == az) >= 5
                for az in {c['azimuth'] for c in expected}))


def presentation(case: dict, pair: BrainPair, trace, writer) -> list[np.ndarray]:
    """Evaluator geometry becomes RGB; only observations enter paired brains."""
    pair.reset(); windows = []
    az = math.radians(case['azimuth']); el = math.radians(case['elevation'])
    for interval in range(75):
        eyes = np.zeros((2, 24, 48, 3), dtype=np.float32)
        if interval >= 25:
            start, end = ((.17, .08), (.12, .06))[case['profile']]
            distance = start+(end-start)*(interval-25)/49
            position = (.3+distance*math.sin(az)*math.cos(el),
                        .3+distance*math.cos(az)*math.cos(el), .1+distance*math.sin(el))
            eyes = encounter_eyes(np.array((.3, .3, .1)), np.zeros(3),
                                  SceneSpec(case['seed'], 'predator', position))
        observation = FlightObservation(0., 0., eyes, (0., 0., 0.), (0., 0., 9.81), False, False)
        # Align windows at stimulus onset (.5 s), dropping the half-baseline window.
        if interval == 25:
            pair.intervals = 0; pair.accumulated[:] = 0
        window = pair.advance(observation)
        if interval >= 25 and window is not None:
            windows.append(window)
        record = dict(**case, interval=interval, end_seconds=(interval+1)*.02,
                      eyes_sha256=hashlib.sha256(eyes.tobytes()).hexdigest(),
                      actual_counts=pair.actual.last_counts.tolist(),
                      reference_counts=pair.reference.last_counts.tolist(),
                      pathway=asdict(pair.actual.diagnostic),
                      wing_vector=None if window is None else window.tolist())
        trace.write(json.dumps(record, separators=(',', ':'))+'\n')
        writer.writerow(dict(phase=case['phase'], azimuth=case['azimuth'], elevation=case['elevation'],
            profile=case['profile'], interval=interval, actual_wings=int(pair.actual.last_counts[list(pair.indices)].sum()),
            reference_wings=int(pair.reference.last_counts[list(pair.indices)].sum()),
            vector_norm='' if window is None else float(np.linalg.norm(window))))
    return windows


def run_directional(directory: Path) -> dict:
    """Fresh-only locked run; failed validation closes all scientific execution."""
    directory.mkdir(parents=True, exist_ok=False)
    # One protocol receipt outside run directory prevents renamed-directory retry.
    authorization = directory.parent/'retreat3d-motion-repair-once.json'
    with authorization.open('x', encoding='utf-8') as stream:
        json.dump(dict(directory=str(directory.resolve()), status='consumed', retry=False), stream)
    torch.set_num_threads(1)
    pair = BrainPair(load_manifest())
    config = dict(hypothesis='Restoring motion input produces directional wing-spike responses sufficient for a frozen decoder to support escape from an approaching predator.',
        max_cases=192, max_seconds=MAX_SECONDS, max_neural_steps=MAX_NEURAL_STEPS,
        stop='first failed gate; no tuning/retry/alternate decoder/neural changes',
        calibration=cases('calibration'), validation=cases('validation'), closed_loop=closed_cases(),
        seconds_per_presentation=1.5, baseline_seconds=.5, window_seconds=.2,
        template_pool='mean of five post-onset windows, across six cases per direction',
        validation_rule='at least one confident window; every confident window error <=22.5 degrees',
        norm_min=1., similarity_min=.8, margin_min=.1, validation_min=90, group_min=5,
        imu=dict(angular_velocity=[0., 0., 0.], specific_force=[0., 0., 9.81]),
        seed=1000000, python=platform.python_version(), platform=platform.platform(), device='cpu',
        software={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'panda3d', 'pygame')},
        network_cells=589, retained_edges=len(pair.actual.edge_order), commit=None,
        optimizer_updates=0, learning_rate=None, checkpoint=None,
        circuit_sha256=digest(CACHE/'circuit.json'),
        source_sha256={p.as_posix(): digest(p) for folder in ('src/flybrain/retreat3d', 'src/flybrain/encounters3d', 'src/flybrain/flight3d') for p in Path(folder).glob('*.py')},
        wing_ids=[pair.actual.ids[i] for i in pair.indices])
    (directory/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    pooled = [[] for _ in range(8)]; rows = []
    with gzip.open(directory/'traces.jsonl.gz', 'wt', encoding='utf-8') as trace, \
         (directory/'traces.csv').open('w', newline='', encoding='utf-8') as table:
        writer = csv.DictWriter(table, fieldnames=['phase', 'azimuth', 'elevation', 'profile', 'interval', 'actual_wings', 'reference_wings', 'vector_norm'])
        writer.writeheader()
        for case in cases('calibration'):
            windows = presentation(case, pair, trace, writer)
            pooled[int(case['azimuth']/45)].extend(windows)
            rows.append(dict(**case, windows=[w.tolist() for w in windows]))
            print(f"calibration {len(rows)}/48", flush=True)
        decoder = FrozenDecoder(np.array([np.mean(v, axis=0) for v in pooled]), tuple(config['wing_ids']))
        decoder.save(directory/'templates.json')
        (directory/'calibration.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        validated = []
        for case in cases('validation'):
            windows = presentation(case, pair, trace, writer)
            validated.append(dict(**case, **score_windows(windows, decoder, case['azimuth']),
                                  windows=[w.tolist() for w in windows]))
            print(f"validation {len(validated)}/96 | successes {sum(r['success'] for r in validated)}", flush=True)
    passed = validation_gate(validated)
    result = dict(status='directional-passed' if passed else 'closed-validation-failed',
        validation_passed=passed, validation_successes=sum(r['success'] for r in validated),
        groups={str(az): sum(r['success'] for r in validated if r['azimuth'] == az)
                for az in sorted({r['azimuth'] for r in validated})},
        validation=validated, cases=144, simulated_seconds=216, neural_steps=432000,
        closed_loop_cases=0, retreat_delivered=False, optimizer_updates=0, retry_authorized=False)
    (directory/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (directory/'receipt.json').write_text(json.dumps(dict(status=result['status'],
        sha256={p.name: digest(p) for p in directory.iterdir() if p.is_file()}), indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    result = run_directional(parser.parse_args().directory)
    print(json.dumps({k: result[k] for k in ('status', 'validation_successes', 'neural_steps')}))
