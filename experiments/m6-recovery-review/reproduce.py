"""Training-only controls for the M6 project review; never a deployed policy."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

from flybrain.neural.controller import NeuralController, NeuralDecision
from flybrain.neural.motor import MotorDecoder
from flybrain.neural.network import SyntheticNetwork
from flybrain.training.evaluation import (
    EvaluationConfig, evaluate_cases, locked_cases, run_episode, summarize,
)
from flybrain.training.fb07h_training import collect_paired_samples, load_matched_checkpoint
from flybrain.training.readout_experiment import teacher_command, topology_sha256
from flybrain.world.fly import MotorCommand
from flybrain.world.sensing import SensoryObservation


class TeacherReference(NeuralController):
    """Diagnostic sensory-rule control; computed spikes do not drive movement."""

    def decide(self, observation: SensoryObservation) -> NeuralDecision:
        """Take only bilateral odor; bypass the decoder with its teacher target."""
        decision = super().decide(observation)
        target = teacher_command(observation)
        command = MotorCommand(float(target[0]), float(target[1]))
        if not self._motor_enabled:
            command = MotorCommand()
        return replace(decision, command=command)


def main() -> None:
    """Reproduce both controls once and preserve inputs and per-case evidence."""
    root = Path(__file__).resolve().parents[2]
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'results.json'
    if output.exists() or output.with_suffix('.csv').exists():
        raise FileExistsError('Choose a new result path to preserve the recorded audit')
    source = root / 'experiments/fb07h-matched-readout'
    recorded = json.loads((source / 'results.json').read_text(encoding='utf-8'))
    gain = json.loads((root / 'experiments/fb07j-turn-gain/results.json').read_text(encoding='utf-8'))
    config = EvaluationConfig()
    cases = locked_cases()['train']
    fixed_replay = asdict(run_episode(cases[0], config))
    assert fixed_replay == gain['results']['train']['fixed_1'][0]

    def factory() -> TeacherReference:
        network = SyntheticNetwork(seed=config.network_seed)
        return TeacherReference(network, MotorDecoder(network.topology))

    # Check the diagnostic's sensory-only decision and reset on a tiny case.
    probe = factory()
    observation = SensoryObservation(0.02, 0.01)
    expected = teacher_command(observation)
    first = probe.decide(observation)
    assert first.command == MotorCommand(float(expected[0]), float(expected[1]))
    probe.reset()
    assert probe.decide(observation).command == first.command
    probe.set_motor_enabled(False)
    assert probe.decide(observation).command == MotorCommand()

    rows = evaluate_cases(cases, config, factory)
    data = dict(zip(('seven', 'nine'), collect_paired_samples(cases, config)))
    fit = {}
    for name, dataset in data.items():
        evidence = recorded['models'][name]
        path = source / evidence['checkpoint']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == evidence['checkpoint_sha256']
        model = load_matched_checkpoint(path, network_seed=config.network_seed,
                                        feature_count=dataset.features.shape[1])
        with torch.no_grad():
            predicted = model(dataset.features)
            average = dataset.targets.mean(dim=0)
            model_mse = (predicted - dataset.targets).square().mean(dim=0)
            constant_mse = (average - dataset.targets).square().mean(dim=0)
        mean_mse = float(model_mse.mean())
        assert abs(mean_mse - evidence['curve'][-1]['loss']) < 1e-7
        fit[name] = {
            'samples': len(dataset.features),
            'checkpoint_sha256': digest,
            'constant_command': average.tolist(),
            'model_mse_forward_turn': model_mse.tolist(),
            'constant_mse_forward_turn': constant_mse.tolist(),
            'model_mean_mse': mean_mse,
            'constant_mean_mse': float(constant_mse.mean()),
            'turn_mse_reduction_vs_constant_fraction':
                float(1 - model_mse[1] / constant_mse[1]),
        }
    payload = {
        'experiment': 'm6_recovery_training_only_review',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'objective': 'Audit teacher reachability and saved readout fit versus constant',
        'training_run': False, 'optimizer_steps': 0, 'learning_rate': None,
        'validation_cases_evaluated': False, 'final_cases_evaluated': False,
        'git_commit': None, 'device': 'cpu',
        'python': platform.python_version(), 'torch': torch.__version__,
        'platform': platform.platform(), 'config': asdict(config),
        'network_seed': config.network_seed, 'network_size': 100,
        'topology_sha256': topology_sha256(config.network_seed),
        'fixed_train_0_replay_exact': True,
        'teacher': {
            'rule': 'forward=0.7; turn=clip(8*(right-left)/max(right+left,1e-9),-1,1)',
            'kind': 'hand-written sensory reference; bypasses neural decoder',
            'spike_metric_note': 'recorded activity is incidental, not causal for this reference',
            'cases': [asdict(case) for case in cases],
            'results': [asdict(row) for row in rows], 'summary': summarize(rows),
        },
        'saved_readout_fit': fit,
        'source_training_result': 'experiments/fb07h-matched-readout/results.json',
        'source_episode_result': 'experiments/fb07j-turn-gain/results.json',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=tuple(payload['teacher']['results'][0]))
        writer.writeheader()
        writer.writerows(payload['teacher']['results'])
    print(json.dumps({'teacher': payload['teacher']['summary'], 'fit': fit}, indent=2))


if __name__ == '__main__':
    main()
