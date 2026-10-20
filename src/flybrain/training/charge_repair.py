"""One train-only M6 repair using measured neural charge features."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import torch

from flybrain.neural.charge_readout import (
    CHARGE_EPSILON, ENCODER_GAIN, FEATURE_NAMES, ChargeFeatureFilter, ChargeMotorDecoder,
)
from flybrain.neural.controller import NeuralController
from flybrain.neural.network import SyntheticNetwork
from flybrain.neural.readout import TrainableMotorReadout
from flybrain.world.sensing import SensoryObservation

from .evaluation import (
    EpisodeCase, EvaluationConfig, _git_commit, evaluate_cases, locked_cases,
    run_episode, summarize,
)
from .fb07e_probe import CANDIDATE_SCALE
from .m6_recovery import MAX_UPDATES, TURN_GATE_MSE, fit_conditioned_readout
from .readout_experiment import (
    FEATURE_ALPHA, TRAIN_SAMPLE_TICKS, TrainingDataset, _RecordingController,
    teacher_command, topology_sha256,
)
from .records import ensure_new_run_directory


def collect_charge_samples(cases: Sequence[EpisodeCase], config: EvaluationConfig,
                           sample_ticks: int = TRAIN_SAMPLE_TICKS) -> TrainingDataset:
    """Replay unchanged fixed-controller train observations through the neural ledger."""
    allowed = set(locked_cases()['train'])
    if not cases or any(case not in allowed for case in cases):
        raise ValueError('sample cases must be locked train cases')
    if type(sample_ticks) is not int or not 0 < sample_ticks <= config.horizon_ticks:
        raise ValueError('sample_ticks must fit the evaluation horizon')
    sampled = replace(config, horizon_ticks=sample_ticks)
    features, targets, counts = [], [], {}
    for case in cases:
        recorder = _RecordingController(config.network_seed)
        run_episode(case, sampled, lambda: recorder)
        network = SyntheticNetwork(seed=config.network_seed, encoder_mode='rational',
                                   encoder_scale=CANDIDATE_SCALE, record_charge=True)
        feature_filter = ChargeFeatureFilter(network.topology, FEATURE_ALPHA)
        for observation, _ in recorder.records:
            features.append(feature_filter.update(network.advance(observation)))
            targets.append(teacher_command(observation))
        counts[case.case_id] = len(recorder.records)
    return TrainingDataset(torch.stack(features), torch.stack(targets), counts)


def _contract(network_seed: int) -> dict:
    return {
        'schema': 3, 'feature_names': FEATURE_NAMES, 'feature_count': 9,
        'network_seed': network_seed, 'topology_sha256': topology_sha256(network_seed),
        'encoder_mode': 'rational', 'encoder_scale': CANDIDATE_SCALE,
        'feature_alpha': FEATURE_ALPHA, 'charge_epsilon': CHARGE_EPSILON,
        'encoder_gain': ENCODER_GAIN,
        'charge_rule': 'rounded membrane changes plus leak and reset losses / interval input fraction',
    }


def save_charge_checkpoint(path: Path, model: TrainableMotorReadout, *,
                           network_seed: int) -> None:
    """Save charge semantics separately from historical voltage checkpoints."""
    if model.feature_count != 9:
        raise ValueError('charge contract requires nine input features')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as file:
        torch.save({**_contract(network_seed), 'state_dict': model.state_dict()}, file)


def load_charge_checkpoint(path: Path, *, network_seed: int) -> TrainableMotorReadout:
    """Validate the complete charge contract before loading weights on CPU."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if any(payload.get(key) != value for key, value in _contract(network_seed).items()):
        raise ValueError('checkpoint charge feature/topology contract differs')
    model = TrainableMotorReadout(9)
    model.load_state_dict(payload['state_dict'], strict=True)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('checkpoint contains nonfinite readout weights')
    model.eval()
    return model


def charge_controller_factory(model: TrainableMotorReadout,
                              network_seed: int) -> Callable[[], NeuralController]:
    """Create fresh neural, feature and motor state for each episode."""
    def create() -> NeuralController:
        network = SyntheticNetwork(seed=network_seed, encoder_mode='rational',
                                   encoder_scale=CANDIDATE_SCALE, record_charge=True)
        return NeuralController(network, ChargeMotorDecoder(network.topology, model,
                                                            FEATURE_ALPHA))
    return create


def _charge_example(network_seed: int) -> list[dict]:
    """Record a tiny sensory-input example, outside the controller interface."""
    network = SyntheticNetwork(seed=network_seed, encoder_mode='rational',
                               encoder_scale=CANDIDATE_SCALE, record_charge=True)
    rows = []
    for left, right in [(0.008, 0.010)] * 3 + [(0.0, 0.0)]:
        observation = SensoryObservation(left, right)
        activity = network.advance(observation)
        indices = [list(network.topology.indices(name))
                   for name in ('sensory_left', 'sensory_right')]
        rows.append({
            'odor_left_right': [left, right],
            'endpoint_voltage_left_right': [float(activity.snapshot.voltage[i].mean())
                                            for i in indices],
            'charge_left_right': [float(activity.integrated_charge[i].mean())
                                   for i in indices],
            'teacher_forward_turn': teacher_command(observation).tolist(),
        })
    return rows


def run_repair(output_dir: Path, *, config: EvaluationConfig | None = None,
               sample_ticks: int = TRAIN_SAMPLE_TICKS, steps: int = MAX_UPDATES) -> Path:
    """Fit one candidate and evaluate only matched original training cases."""
    ensure_new_run_directory(output_dir)
    if type(steps) is not int or not 1 <= steps <= MAX_UPDATES:
        raise ValueError('steps must be an integer from 1 through 500')
    output_dir = Path(output_dir)
    config = config or EvaluationConfig()
    cases = locked_cases()['train']
    topology_before = topology_sha256(config.network_seed)
    protocol = {
        'experiment': 'm6_neural_charge_repair',
        'hypothesis': 'reset-corrected neural charge features enable the same readout to fit sensory turns',
        'optimizer': 'Adam full batch; train-only standardization folded into raw weights',
        'learning_rate': 0.05, 'optimizer_step_ceiling': MAX_UPDATES,
        'planned_updates': steps, 'sample_ticks_per_case': sample_ticks,
        'config': asdict(config), 'cases': [asdict(case) for case in cases],
        'contract': _contract(config.network_seed),
        'stop_rule': 'stop after one candidate; require turn MSE <= .016752 and <= .1 constant, train arrivals >=7/8',
        'scope': 'train only; no validation/final or live app promotion',
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n',
                                             encoding='utf-8')
    dataset = collect_charge_samples(cases, config, sample_ticks)
    untrained = TrainableMotorReadout(9)
    save_charge_checkpoint(output_dir / 'untrained.pt', untrained,
                           network_seed=config.network_seed)
    outcome = fit_conditioned_readout(dataset.features, dataset.targets, steps=steps)
    checkpoint = output_dir / 'trained.pt'
    save_charge_checkpoint(checkpoint, outcome.model, network_seed=config.network_seed)
    reloaded = load_charge_checkpoint(checkpoint, network_seed=config.network_seed)
    with torch.no_grad():
        predictions = reloaded(dataset.features)
        errors = (predictions - dataset.targets).square().mean(dim=0)
        mean_command = dataset.targets.mean(dim=0)
        constant_errors = (mean_command - dataset.targets).square().mean(dim=0)
        reload_outputs_exact = torch.equal(predictions, outcome.model(dataset.features))
    rows = {
        'trained': evaluate_cases(cases, config,
                                   charge_controller_factory(reloaded, config.network_seed)),
        'untrained': evaluate_cases(cases, config,
                                     charge_controller_factory(untrained, config.network_seed)),
        'fixed': evaluate_cases(cases, config),
    }
    summaries = {name: summarize(values) for name, values in rows.items()}
    replay = run_episode(cases[0], config,
                         charge_controller_factory(outcome.model, config.network_seed))
    checks = {
        'topology_unchanged': topology_before == topology_sha256(config.network_seed),
        'checkpoint_outputs_exact': reload_outputs_exact,
        'checkpoint_replay_exact': replay == rows['trained'][0],
        'finite_parameters': all(bool(torch.isfinite(p).all()) for p in reloaded.parameters()),
        'gradient_and_update_observed': (outcome.gradient_nonzero_scalars > 0
                                        and outcome.parameter_changed_scalars > 0),
        'both_sensory_turn_weights_changed': bool((reloaded.linear.weight[1, 7:] != 0).all()),
    }
    gate = {
        'turn_fit': float(errors[1]) <= min(TURN_GATE_MSE, 0.1 * float(constant_errors[1])),
        'training_arrivals': summaries['trained']['successes'] >= 7,
        'numerical_checks': all(checks.values()),
    }
    gate['passed'] = all(gate.values())
    payload = {
        **protocol, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(), 'network_size': 100,
        'optimizer_steps': steps, 'sample_count': len(dataset.features),
        'sample_counts': dataset.case_counts,
        'features_sha256': hashlib.sha256(dataset.features.numpy().tobytes()).hexdigest(),
        'targets_sha256': hashlib.sha256(dataset.targets.numpy().tobytes()).hexdigest(),
        'feature_mean': outcome.mean.tolist(), 'feature_scale': outcome.scale.tolist(),
        'gradient_nonzero_scalars': outcome.gradient_nonzero_scalars,
        'gradient_min_abs': outcome.gradient_min_abs,
        'parameter_changed_scalars': outcome.parameter_changed_scalars,
        'parameter_change_min_abs': outcome.parameter_change_min_abs,
        'readout_mse_forward_turn': errors.tolist(),
        'constant_mse_forward_turn': constant_errors.tolist(),
        'mean_command': mean_command.tolist(),
        'turn_reduction_vs_constant': float(1 - errors[1] / constant_errors[1]),
        'checkpoints': {name: {'file': f'{name}.pt', 'sha256': hashlib.sha256(
            (output_dir / f'{name}.pt').read_bytes()).hexdigest()}
                        for name in ('trained', 'untrained')},
        'checks': checks, 'gate': gate, 'summaries': summaries,
        'results': {name: [asdict(row) for row in values] for name, values in rows.items()},
        'charge_example': _charge_example(config.network_seed),
        'validation_cases_evaluated': False, 'final_cases_evaluated': False,
        'software': {'python': platform.python_version(), 'torch': torch.__version__,
                     'pygame': version('pygame'), 'platform': platform.platform(), 'device': 'cpu'},
        'scientific_limit': 'engineered reconstruction of sensory drive; no learned spiking dynamics or demonstrated topology benefit',
    }
    with (output_dir / 'curve.csv').open('x', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=('step', 'forward_mse', 'turn_mse'))
        writer.writeheader()
        writer.writerows(outcome.curve)
    with (output_dir / 'train_episodes.csv').open('x', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=('controller', *asdict(rows['trained'][0])))
        writer.writeheader()
        for name, values in rows.items():
            writer.writerows({'controller': name, **asdict(row)} for row in values)
    path = output_dir / 'results.json'
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the predeclared single repair and report its training gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/m6-charge-repair'))
    args = parser.parse_args(argv)
    path = run_repair(args.output_dir)
    result = json.loads(path.read_text(encoding='utf-8'))
    print(f"Wrote {path}: turn MSE {result['readout_mse_forward_turn'][1]:.6f}, "
          f"train arrivals {result['summaries']['trained']['successes']}/8, "
          f"gate passed {result['gate']['passed']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
