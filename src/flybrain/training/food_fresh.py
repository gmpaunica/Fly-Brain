"""Single-use 20-second fresh assay, conditional on the recorded train gate."""

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from collections.abc import Sequence

import torch

from flybrain.neural.banc_motor import make_banc_controller
from flybrain.neural.banc_network import load_verified_banc_config
from flybrain.neural.food_readout import (FoodDecoder, load_food_checkpoint,
                                          make_food_controller)
from flybrain.world.food import FoodObservation

from .food_evaluation import run_controller_case
from .food_fit import DEFAULT_OUTPUT as FIT
from .food_preflight import PROJECT_ROOT, generate_cases


DEFAULT_OUTPUT = PROJECT_ROOT / 'experiments/food-stop-fresh-2026-09-28'


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_exact_p(trained: Sequence[bool], control: Sequence[bool]) -> float:
    """One-sided exact sign test on discordant paired success outcomes."""
    if len(trained) != len(control) or not trained:
        raise ValueError('equal nonempty paired outcomes required')
    if any(type(value) is not bool for value in (*trained, *control)):
        raise ValueError('boolean outcomes required')
    wins = sum(a and not b for a, b in zip(trained, control))
    losses = sum(b and not a for a, b in zip(trained, control))
    n = wins + losses
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def validation_gate(trained: Sequence[dict]) -> bool:
    """Exactly 16 untouched validation cases, at least 14 full successes."""
    return len(trained) == 16 and sum(row['success'] is True for row in trained) >= 14


def final_gates(trained: Sequence[dict], untrained: Sequence[dict],
                fixed: Sequence[dict]) -> dict:
    """Check absolute, sector, paired-gain and exact one-sided gates."""
    if any(len(rows) != 64 for rows in (trained, untrained, fixed)):
        raise ValueError('64 complete paired final cases per controller required')
    sectors = {sector: [row for row in trained if row['heading_sector'] == sector]
               for sector in range(8)}
    sector_successes = {sector: sum(row['success'] is True for row in rows)
                        for sector, rows in sectors.items()}
    count = sum(row['success'] is True for row in trained)
    paired = {}
    for name, control in (('untrained', untrained), ('fixed', fixed)):
        gain = count - sum(row['success'] is True for row in control)
        p = paired_exact_p([row['success'] for row in trained],
                           [row['success'] for row in control])
        paired[name] = {'successes': sum(row['success'] is True for row in control),
                        'gain': gain, 'one_sided_exact_p': p,
                        'passed': gain >= 8 and p <= .025}
    passed = (count >= 58 and all(len(sectors[s]) == 8 and sector_successes[s] >= 7
                                  for s in range(8))
              and all(pair['passed'] for pair in paired.values()))
    return {'trained_successes': count, 'sector_successes': sector_successes,
            'paired': paired, 'passed': passed}


def _fit_evidence() -> tuple[dict, str, dict]:
    protocol_path = FIT / 'protocol.json'
    fit_path = FIT / 'results.json'
    if not protocol_path.is_file() or not fit_path.is_file():
        raise ValueError('completed fit evidence required')
    fit = json.loads(fit_path.read_text(encoding='utf-8'))
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    digest = _digest(protocol_path)
    if (fit.get('training_gate_passed') is not True
            or fit.get('numerical_gate_passed') is not True
            or fit.get('protocol_sha256') != digest
            or fit.get('fresh_evaluated') is not False
            or fit.get('checkpoint_sha256') != _digest(FIT / 'trained.pt')
            or fit.get('untrained_sha256') != _digest(FIT / 'untrained.pt')
            or fit.get('dataset_sha256') != _digest(FIT / 'data.pt')
            or fit.get('fit', {}).get('optimizer_steps') != 1500
            or fit.get('training_cases', {}).get('successes', {}).get('trained', 0) < 35
            or any(_digest(PROJECT_ROOT / name) != value
                   for name, value in protocol['source_sha256'].items())):
        raise ValueError('passing, unchanged training gate required')
    return protocol, digest, fit


def prepare_fresh(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    """Lock the previously reserved splits and assay gates before outcomes."""
    if output_dir.exists():
        raise FileExistsError('fresh assay output already exists')
    fit_protocol, fit_digest, fit = _fit_evidence()
    sources = ('src/flybrain/neural/food_contact.py',
               'src/flybrain/neural/food_readout.py',
               'src/flybrain/training/food_evaluation.py',
               'src/flybrain/training/food_fresh.py')
    record = {'schema': 'food_stop_fresh_v1',
              'hypothesis': 'The frozen 85-cell neural-history readout reaches, slows and remains stopped at food on fresh full-circle starts more often than identical untrained and fixed controls.',
              'cases': {split: [asdict(case) for case in group]
                        for split, group in generate_cases().items()
                        if split in ('validation', 'final')},
              'fit_sha256': {name: _digest(FIT / name) for name in
                             ('protocol.json', 'results.json', 'data.pt',
                              'trained.pt', 'untrained.pt')},
              'fit_protocol_sha256': fit_digest,
              'source_sha256': {name: _digest(PROJECT_ROOT / name) for name in sources},
              'normalization': fit['normalization'],
              'seed': fit_protocol['seed'], 'network_size': 85,
              'trainable_parameters': 419, 'optimizer_steps': 0,
              'learning_rate': None,
              'objective': '20-second paired full-contact behavior, approach brake, exact stop and uninterrupted dwell',
              'gates': {'validation_successes_min': 14, 'final_successes_min': 58,
                        'final_sector_successes_min': 7,
                        'final_paired_gain_min': 8,
                        'final_one_sided_exact_p_max': .025,
                        'contact_block_changes_stop': True},
              'budget': {'validation_episodes': 48, 'final_episodes_if_validation_passes': 192,
                         'world_ticks_max': 240000, 'neural_steps_max': 4800400,
                         'intervention_neural_steps_max': 400,
                         'optimizer_updates': 0},
              'stop_condition': 'One use. Stop before final if validation <14/16. Do not tune, retrain or replace the checkpoint after any failure.',
              'software_device': {'python': platform.python_version(),
                                  'torch': torch.__version__, 'device': 'CPU'},
              'commit': None, 'prepared_utc': datetime.now(timezone.utc).isoformat()}
    output_dir.mkdir(parents=True)
    path = output_dir / 'protocol.json'
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return path


def _models(protocol: dict):
    normal = protocol['normalization']
    mean = torch.tensor(normal['mean'])
    scale = torch.tensor(normal['scale'])
    models = {name: load_food_checkpoint(FIT / f'{name}.pt',
                                         protocol_sha256=protocol['fit_protocol_sha256'],
                                         mean=mean, scale=scale)
              for name in ('trained', 'untrained')}
    for model in models.values():
        model.requires_grad_(False)
    return models


def _evaluate(split: str, models: dict) -> dict[str, list[dict]]:
    config = load_verified_banc_config()
    cases = generate_cases()[split]
    episodes = {}
    with torch.no_grad():
        for name, factory in (
            ('trained', lambda: make_food_controller(models['trained'])),
            ('untrained', lambda: make_food_controller(models['untrained'])),
            ('fixed', lambda: make_banc_controller(config)),
        ):
            episodes[name] = [run_controller_case(case, factory()) for case in cases]
    return episodes


def _save_split(output_dir: Path, split: str, episodes: dict) -> None:
    (output_dir / f'{split}.json').write_text(
        json.dumps(episodes, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output_dir / f'{split}.csv').open('x', encoding='utf-8', newline='') as stream:
        first = next(iter(episodes.values()))[0]
        writer = csv.DictWriter(stream, fieldnames=['controller', *first.keys()])
        writer.writeheader()
        for name, rows in episodes.items():
            writer.writerows({'controller': name, **row} for row in rows)


def _contact_intervention(model) -> dict:
    """Compare the same frozen head with and without contact-cell drive."""
    commands = {}
    with torch.no_grad():
        for blocked in (False, True):
            controller = make_food_controller(model)
            controller.network.contact_blocked = blocked
            observation = FoodObservation(1.5, 1.5, True)
            series = [controller.decide(observation).command for _ in range(10)]
            commands['blocked' if blocked else 'intact'] = [
                {'forward': command.forward, 'turn': command.turn} for command in series]
    intact = sum(row['forward'] == 0 and row['turn'] == 0 for row in commands['intact'][-5:])
    blocked = sum(row['forward'] == 0 and row['turn'] == 0 for row in commands['blocked'][-5:])
    return {'commands': commands, 'last_five_stops_intact': intact,
            'last_five_stops_blocked': blocked,
            'changes_stop': intact > blocked,
            'neural_steps': 400}


def run_fresh(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    """Consume once, evaluate validation first and final only after its gate."""
    path = output_dir / 'protocol.json'
    protocol = json.loads(path.read_text(encoding='utf-8'))
    _, fit_digest, _ = _fit_evidence()
    expected_cases = {split: [asdict(case) for case in group]
                      for split, group in generate_cases().items()
                      if split in ('validation', 'final')}
    if (protocol.get('schema') != 'food_stop_fresh_v1'
            or protocol.get('fit_protocol_sha256') != fit_digest
            or protocol.get('cases') != expected_cases
            or any(_digest(FIT / name) != digest
                   for name, digest in protocol['fit_sha256'].items())
            or any(_digest(PROJECT_ROOT / name) != digest
                   for name, digest in protocol['source_sha256'].items())):
        raise ValueError('fresh protocol or frozen evidence differs')
    with (output_dir / 'execution.json').open('x', encoding='utf-8') as stream:
        json.dump({'status': 'consumed', 'protocol_sha256': _digest(path),
                   'started_utc': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.write('\n')
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        models = _models(protocol)
        state_before = {name: {key: value.clone() for key, value in model.state_dict().items()}
                        for name, model in models.items()}
        validation = _evaluate('validation', models)
        _save_split(output_dir, 'validation', validation)
        passed_validation = validation_gate(validation['trained'])
        result = {'protocol_sha256': _digest(path),
                  'validation_successes': {name: sum(row['success'] for row in rows)
                                           for name, rows in validation.items()},
                  'validation_gate_passed': passed_validation,
                  'final_gates': None, 'contact_intervention': None,
                  'final_evaluated': False, 'passed': False,
                  'actual_episodes': 48, 'actual_world_ticks': 48000,
                  'actual_neural_steps': 960000, 'optimizer_steps': 0}
        if passed_validation:
            final = _evaluate('final', models)
            _save_split(output_dir, 'final', final)
            result['final_evaluated'] = True
            result['final_gates'] = final_gates(
                final['trained'], final['untrained'], final['fixed'])
            result['actual_episodes'] += 192
            result['actual_world_ticks'] += 192000
            result['actual_neural_steps'] += 3840000
            result['contact_intervention'] = _contact_intervention(models['trained'])
            result['actual_neural_steps'] += 400
        result['parameters_unchanged'] = all(
            torch.equal(value, model.state_dict()[key])
            for name, model in models.items() for key, value in state_before[name].items())
        result['passed'] = bool(passed_validation and result['final_gates']['passed']
                                and result['contact_intervention']['changes_stop']
                                and result['parameters_unchanged'])
        result['completed_utc'] = datetime.now(timezone.utc).isoformat()
        (output_dir / 'results.json').write_text(
            json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return result
    finally:
        torch.set_num_threads(previous_threads)
