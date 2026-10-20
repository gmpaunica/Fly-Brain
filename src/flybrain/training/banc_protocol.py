"""Prepare and validate one immutable, train-only FB-10 protocol; never fit."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

from flybrain.neural.banc_readout import readout_contract
from .banc_readiness import (PROJECT_ROOT, capture_provenance, save_new_json,
                             summarize_features, verify_provenance)
from .evaluation import EvaluationConfig, _git_commit, locked_cases
from .readout_experiment import TrainingDataset
from .records import ensure_new_run_directory


SEED = 1010
MAX_UPDATES = 500
LEARNING_RATE = .01
DEFAULT_OUTPUT = Path('experiments/fb10-banc-readout')
DEFAULT_READINESS = PROJECT_ROOT / 'experiments/fb09-banc-readiness'


def sha256(path: Path) -> str:
    """Hash the actual bytes of a required file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def software_record() -> dict:
    """Record the environment to which exact replay claims apply."""
    return {'python': sys.version.split()[0], 'torch': str(torch.__version__),
            'platform': platform.platform(), 'device': 'cpu', 'dtype': 'float32',
            'execution_cpu_threads': 1}


def load_readiness(directory: Path) -> TrainingDataset:
    """Accept verified train-only samples; never expose odors as model features."""
    directory = Path(directory)
    result = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
    contract = readout_contract()
    if (sha256(directory / 'samples.csv') != result.get('samples_sha256')
            or sha256(directory / 'protocol.json') != result.get('protocol_sha256')):
        raise ValueError('readiness sample/protocol hash differs')
    if (any(result.get(key) is not True for key in (
            'readiness_passed', 'effective_weights_unchanged', 'provenance_preserved',
            'execution_budget_passed', 'protocol_unchanged'))
            or result.get('optimizer_steps') != 0 or result.get('network_size') != 83
            or result.get('dynamics_config') != contract['dynamics_config']
            or result.get('effective_weights_sha256') != contract['effective_weights_sha256']
            or result.get('feature_names') != contract['feature_names']
            or result.get('teacher_summary', {}).get('successes') != 8
            or not verify_provenance(result['provenance'])):
        raise ValueError('readiness evidence is not accepted for this frozen BANC circuit')
    names = contract['feature_names']
    expected_cases = [case.case_id for case in locked_cases()['train']]
    expected_ticks = {row['case_id']: row['ticks'] for row in result['teacher_episodes']}
    counts = {case: 0 for case in expected_cases}
    features, targets, case_order = [], [], []
    with (directory / 'samples.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['case_id', 'tick', 'left_odor', 'right_odor', *names,
                                'target_forward', 'target_turn']:
            raise ValueError('readiness feature columns/order differ')
        for row in reader:
            case = row['case_id']
            if case not in counts or int(row['tick']) != counts[case]:
                raise ValueError('samples must contain contiguous ticks from locked train cases')
            if not case_order or case_order[-1] != case:
                case_order.append(case)
            counts[case] += 1
            features.append([float(row[name]) for name in names])
            targets.append([float(row['target_forward']), float(row['target_turn'])])
    x = torch.tensor(features, dtype=torch.float64, device='cpu')
    y = torch.tensor(targets, dtype=torch.float64, device='cpu')
    if (case_order != expected_cases or counts != expected_ticks or len(x) != 1008
            or torch.any(x < 0) or torch.any(x > 1000)
            or y.shape != (1008, 2) or torch.any(y[:, 0] < 0) or torch.any(y[:, 0] > 1)
            or torch.any(y[:, 1].abs() > 1)):
        raise ValueError('readiness samples differ from the eight training trajectories')
    metrics = summarize_features(x, y)
    recorded = result['features']
    for key in ('samples', 'unique_feature_rows'):
        if metrics[key] != recorded[key]:
            raise ValueError('readiness sample statistics differ')
    for key in ('feature_std', 'mean_command', 'constant_mse', 'exact_alias_turn_mse_lower_bound'):
        if not torch.allclose(torch.tensor(metrics[key], dtype=torch.float64),
                              torch.tensor(recorded[key], dtype=torch.float64), rtol=1e-10, atol=1e-12):
            raise ValueError('readiness baseline/feature statistics differ')
    return TrainingDataset(x, y, counts)


def protocol_definition(dataset: TrainingDataset) -> dict:
    """Return the single predeclared candidate, normalization, cases and gates."""
    metrics = summarize_features(dataset.features, dataset.targets)
    return {
        'schema': 'fb10_banc_training_protocol_v1', 'experiment': 'fb10_banc_endpoint_readout',
        'hypothesis': 'An endpoint-rate-only nonlinear readout imitates sensory turns and improves training arrivals over matched untrained and fixed BANC controls.',
        'seed': SEED, 'contract': readout_contract(), 'network_size': 83,
        'trainable_parameters': 178, 'software': software_record(),
        'optimizer': 'Adam full batch', 'learning_rate': LEARNING_RATE,
        'optimizer_step_ceiling': MAX_UPDATES,
        'objective': 'forward MSE + turn MSE; no gradient through the spiking core',
        'normalization': {'rule': 'train population mean/std; std floor 1 Hz; float64 calculation, float32 storage',
            'mean': dataset.features.mean(0).float().tolist(),
            'scale': dataset.features.std(0, correction=0).clamp_min(1).float().tolist()},
        'sample_count': len(dataset.features), 'case_counts': dataset.case_counts,
        'train_only_baseline': metrics,
        'cases': [asdict(case) for case in locked_cases()['train']],
        'world_config': asdict(EvaluationConfig()),
        'max_world_ticks': 15000, 'max_neural_steps': 300000, 'max_episodes': 25,
        'gates': {'turn_mse_max': .25 * metrics['constant_mse'][1],
                  'constant_turn_mse_fraction': .25, 'trained_arrivals_min': 7,
                  'strictly_more_arrivals_than_both_controls': True,
                  'strictly_greater_all_case_progress_than_both_controls': True,
                  'finite_nonzero_gradients_and_parameter_changes': True,
                  'frozen_core_normalization_and_provenance': True,
                  'exact_output_and_embodied_checkpoint_replay': True},
        'stop_condition': 'One 500-update attempt; stop on nonfinite numerics. Record training controls after a finite fit even if turn gate fails. No retries, sweeps or held-out execution.',
        'scope': 'Historical training cases only; no learned-navigation or topology-benefit claim',
    }


def prepare_protocol(output_dir: Path, *, readiness_dir: Path = DEFAULT_READINESS) -> Path:
    """Freeze a reviewable protocol with zero scientific steps or updates."""
    output_dir = Path(output_dir).resolve()
    ensure_new_run_directory(output_dir)
    readiness_dir = Path(readiness_dir).resolve()
    dataset = load_readiness(readiness_dir)
    protocol = {
        **protocol_definition(dataset), 'prepared_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(), 'output_dir': str(output_dir),
        'readiness_dir': str(readiness_dir),
        'readiness_sha256': {name: sha256(readiness_dir / name)
            for name in ('samples.csv', 'results.json', 'protocol.json')},
        'provenance': capture_provenance(PROJECT_ROOT, output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'protocol.json'
    save_new_json(path, protocol)
    with path.with_suffix('.sha256').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(sha256(path) + '\n')
    return path


def load_protocol(path: Path) -> tuple[dict, str, TrainingDataset]:
    """Fail closed on stale bytes, changed settings, source, history or data."""
    path = Path(path)
    digest = sha256(path)
    if path.with_suffix('.sha256').read_text(encoding='utf-8').strip() != digest:
        raise ValueError('protocol hash differs; do not reseal a changed preparation')
    protocol = json.loads(path.read_text(encoding='utf-8'))
    if path.resolve() != Path(protocol['output_dir']).resolve() / 'protocol.json':
        raise ValueError('protocol destination differs from its single-use preparation')
    readiness = Path(protocol['readiness_dir'])
    if any(sha256(readiness / name) != value for name, value in protocol['readiness_sha256'].items()):
        raise ValueError('frozen readiness hash differs')
    dataset = load_readiness(readiness)
    if any(protocol.get(key) != value for key, value in protocol_definition(dataset).items()):
        raise ValueError('protocol settings/normalization/cases/environment differ')
    if not verify_provenance(protocol['provenance']):
        raise ValueError('protocol source/bundle/history provenance differs')
    return protocol, digest, dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--readiness-dir', type=Path, default=DEFAULT_READINESS)
    args = parser.parse_args(argv)
    path = prepare_protocol(args.output_dir, readiness_dir=args.readiness_dir)
    print(json.dumps({'protocol': str(path), 'sha256': sha256(path),
                      'experiment_optimizer_steps': 0, 'experiment_world_ticks': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
