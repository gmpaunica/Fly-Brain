"""FB-07J: a single matched, sensory-only motor turn-gain intervention."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
from .records import ensure_new_run_directory
import platform
import sys
from collections.abc import Callable, Sequence

import torch

from flybrain.neural.controller import MotorReadout, NeuralController
from flybrain.neural.network import NeuralActivity
from flybrain.neural.readout import TrainableMotorReadout
from flybrain.world.fly import MotorCommand

from .evaluation import (
    EpisodeCase, EpisodeResult, EvaluationConfig, _git_commit, evaluate_cases,
    fixed_controller, locked_cases, summarize,
)
from .fb07h_training import candidate_controller_factory, load_matched_checkpoint
from .readout_experiment import topology_sha256


TURN_GAIN = 2.0


class TurnGainDecoder:
    """Scale only a decoded sensory-derived turn, then clamp to motor bounds."""

    def __init__(self, decoder: MotorReadout, gain: float) -> None:
        if (isinstance(gain, bool) or not isinstance(gain, (int, float))
                or not math.isfinite(gain) or gain <= 0):
            raise ValueError('gain must be positive and finite')
        self.decoder = decoder
        self.gain = float(gain)

    def reset(self) -> None:
        """Reset the wrapped decoder's episode state."""
        self.decoder.reset()

    def decode(self, activity: NeuralActivity) -> MotorCommand:
        """Preserve forward command and increase clockwise turn magnitude."""
        command = self.decoder.decode(activity)
        turn = max(-1.0, min(1.0, command.turn * self.gain))
        return MotorCommand(command.forward, turn)


def gain_controller_factory(
    kind: str, model: TrainableMotorReadout | None,
    network_seed: int, *, gain: float,
) -> Callable[[], NeuralController]:
    """Build a fresh fixed or learned controller with one motor gain."""
    if kind == 'fixed' and model is None:
        base = lambda: fixed_controller(network_seed)
    elif kind == 'nine' and model is not None and model.feature_count == 9:
        base = candidate_controller_factory(model, network_seed)
    else:
        raise ValueError('kind/model must be fixed/None or nine/nine-feature')

    def create() -> NeuralController:
        controller = base()
        controller.decoder = TurnGainDecoder(controller.decoder, gain)
        return controller
    return create


def validation_gate(nine_2: dict, nine_1: dict,
                    fixed_2: dict) -> dict[str, bool]:
    """Open final starts only after learned gain-2 beats both controls."""
    arrivals = nine_2['successes'] > max(nine_1['successes'], fixed_2['successes'])
    progress = nine_2['mean_progress_fraction'] > max(
        nine_1['mean_progress_fraction'], fixed_2['mean_progress_fraction'])
    return {'arrival_above_both': arrivals, 'progress_above_both': progress,
            'passed': arrivals and progress}


def _write_cases(path: Path, cases: Sequence[EpisodeCase],
                 rows: dict[str, Sequence[EpisodeResult]]) -> None:
    """Write matched per-case outcomes and progress for every controller."""
    names = tuple(rows)
    with path.open('w', newline='', encoding='utf-8') as file:
        fields = ('case_id', *[f'{name}_{field}' for name in names
                               for field in ('outcome', 'progress_fraction',
                                             'final_distance')])
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, case in enumerate(cases):
            if any(rows[name][index].case_id != case.case_id for name in names):
                raise RuntimeError('comparison case IDs differ')
            writer.writerow({
                'case_id': case.case_id,
                **{f'{name}_{field}': getattr(rows[name][index], field)
                   for name in names
                   for field in ('outcome', 'progress_fraction', 'final_distance')},
            })


def run_intervention(
    output_dir: Path, training_result_path: Path, baseline_path: Path, *,
    config: EvaluationConfig | None = None,
) -> Path:
    """Evaluate one gain-2 intervention and gate final cases on validation."""
    ensure_new_run_directory(output_dir)
    config = config or EvaluationConfig()
    cases = locked_cases()
    trained_path = Path(training_result_path)
    trained = json.loads(trained_path.read_text(encoding='utf-8'))
    baseline = json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    expected_cases = {name: [asdict(case) for case in group]
                      for name, group in cases.items()}
    expected_ids = [case.case_id for case in cases['validation'] + cases['final']]
    if (trained.get('experiment') != 'fb07h_matched_voltage_readout'
            or trained.get('config') != asdict(config)
            or trained.get('final_cases_evaluated') is not False
            or baseline.get('experiment') != 'fb06_fixed_baseline'
            or baseline.get('config') != asdict(config)
            or baseline.get('cases') != expected_cases
            or [row['case_id'] for row in baseline.get('results', [])] != expected_ids):
        raise ValueError('FB-07H or FB-06 protocol differs')
    checkpoint = trained_path.parent / trained['models']['nine']['checkpoint']
    if (hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != trained['models']['nine']['checkpoint_sha256']):
        raise ValueError('nine-feature checkpoint hash differs')
    model = load_matched_checkpoint(checkpoint, network_seed=config.network_seed,
                                    feature_count=9)
    topology_before = topology_sha256(config.network_seed)
    factories = {
        'fixed_1': gain_controller_factory('fixed', None, config.network_seed,
                                           gain=1.0),
        'fixed_2': gain_controller_factory('fixed', None, config.network_seed,
                                           gain=TURN_GAIN),
        'nine_1': gain_controller_factory('nine', model, config.network_seed,
                                          gain=1.0),
        'nine_2': gain_controller_factory('nine', model, config.network_seed,
                                          gain=TURN_GAIN),
    }
    train_rows = {name: evaluate_cases(cases['train'], config, factory)
                  for name, factory in factories.items()}
    validation_rows = {
        'fixed_1': tuple(EpisodeResult(**row) for row in baseline['results'][:4]),
        'fixed_2': evaluate_cases(cases['validation'], config,
                                  factories['fixed_2']),
        'nine_1': tuple(EpisodeResult(**row)
                        for row in trained['results']['validation']['nine']),
        'nine_2': evaluate_cases(cases['validation'], config,
                                 factories['nine_2']),
    }
    gain_one_exact = (
        evaluate_cases(cases['validation'], config, factories['fixed_1'])
        == validation_rows['fixed_1']
        and evaluate_cases(cases['validation'], config, factories['nine_1'])
        == validation_rows['nine_1']
    )
    if not gain_one_exact:
        raise RuntimeError('gain-1 validation replay differs from recorded rows')
    train_summary = {name: summarize(rows) for name, rows in train_rows.items()}
    validation_summary = {name: summarize(rows)
                          for name, rows in validation_rows.items()}
    gate = validation_gate(validation_summary['nine_2'],
                           validation_summary['nine_1'],
                           validation_summary['fixed_2'])
    results = {
        'train': {name: [asdict(row) for row in rows]
                  for name, rows in train_rows.items()},
        'validation': {name: [asdict(row) for row in rows]
                       for name, rows in validation_rows.items()},
    }
    summaries = {'train': train_summary, 'validation': validation_summary}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_cases(output_dir / 'train.csv', cases['train'], train_rows)
    _write_cases(output_dir / 'validation.csv', cases['validation'], validation_rows)
    if gate['passed']:
        final_rows = {
            'fixed_1': tuple(EpisodeResult(**row) for row in baseline['results'][4:]),
            'fixed_2': evaluate_cases(cases['final'], config, factories['fixed_2']),
            'nine_2': evaluate_cases(cases['final'], config, factories['nine_2']),
        }
        _write_cases(output_dir / 'final.csv', cases['final'], final_rows)
        results['final'] = {name: [asdict(row) for row in rows]
                            for name, rows in final_rows.items()}
        summaries['final'] = {name: summarize(rows)
                              for name, rows in final_rows.items()}
    topology_unchanged = topology_before == topology_sha256(config.network_seed)
    if not topology_unchanged:
        raise RuntimeError('fixed topology changed')
    payload = {
        'experiment': 'fb07j_turn_gain',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'objective': 'test one motor turn gain after FB-07I early retreat',
        'turn_gain': TURN_GAIN,
        'intervention': 'clamp(2*decoded_turn,-1,1); forward unchanged',
        'validation_gate_rule': ('nine gain-2 arrivals and mean progress above '
                                 'nine gain-1 and fixed gain-2'),
        'validation_gate': gate,
        'gain_one_replay_exact': gain_one_exact,
        'training_run': False,
        'final_cases_evaluated': gate['passed'],
        'config': asdict(config),
        'network_seed': config.network_seed,
        'network_size': 100,
        'topology_sha256': topology_before,
        'topology_unchanged': topology_unchanged,
        'checkpoint_sha256': trained['models']['nine']['checkpoint_sha256'],
        'source_training_result': str(trained_path),
        'baseline': str(baseline_path),
        'cases': {split: [asdict(case) for case in cases[split]]
                  for split in ('train', 'validation') +
                  (('final',) if gate['passed'] else ())},
        'summaries': summaries,
        'results': results,
        'software': {'python': sys.version.split()[0], 'torch': torch.__version__,
                     'pygame': version('pygame'), 'platform': platform.platform(),
                     'device': 'cpu'},
    }
    path = output_dir / 'results.json'
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the declared single-gain intervention."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/fb07j-turn-gain'))
    parser.add_argument('--training-result', type=Path,
                        default=Path('experiments/fb07h-matched-readout/results.json'))
    parser.add_argument('--baseline', type=Path,
                        default=Path('experiments/fb06-fixed-baseline/results.json'))
    args = parser.parse_args(argv)
    path = run_intervention(args.output_dir, args.training_result, args.baseline)
    result = json.loads(path.read_text(encoding='utf-8'))
    print(f'Wrote turn-gain intervention: {path}')
    for split, rows in result['summaries'].items():
        print(f"{split}: fixed1={rows['fixed_1']['successes']} "
              f"fixed2={rows['fixed_2']['successes']} "
              f"nine2={rows['nine_2']['successes']} arrivals")
    print(f"Validation gate passed: {result['validation_gate']['passed']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
