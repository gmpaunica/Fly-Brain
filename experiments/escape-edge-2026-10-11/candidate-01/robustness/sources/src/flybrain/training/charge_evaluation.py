"""Execute the approved, frozen M6 charge evaluation without any training."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import torch

from .charge_repair import charge_controller_factory, load_charge_checkpoint
from .evaluation import EpisodeCase, EpisodeResult, EvaluationConfig, _git_commit, run_episode, summarize
from .records import ensure_new_run_directory


APPROVED_PROTOCOL_SHA256 = '3f3f7a9c32eecb7c591265420dfae1c3b0398bfbd2bf9733131524e1af576a71'
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTROLLERS = ('trained', 'untrained', 'fixed')


def paired_arrival_test(trained: Sequence[bool], control: Sequence[bool]) -> dict:
    """One-sided exact binomial probability among discordant paired arrivals."""
    if (len(trained) != len(control) or not trained
            or any(type(value) is not bool for value in (*trained, *control))):
        raise ValueError('paired outcomes must be equally sized nonempty boolean sequences')
    wins = sum(a and not b for a, b in zip(trained, control))
    losses = sum(b and not a for a, b in zip(trained, control))
    discordant = wins + losses
    probability = (sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
                   / 2 ** discordant if discordant else 1.0)
    return {'trained_only': wins, 'control_only': losses,
            'both_success': sum(a and b for a, b in zip(trained, control)),
            'neither_success': sum(not a and not b for a, b in zip(trained, control)),
            'arrival_difference': wins - losses, 'one_sided_p': probability}


def assess_split(split: str, outcomes: Mapping[str, Sequence[bool]]) -> dict:
    """Apply the reviewed 16-case validation or 32-case final gate exactly."""
    if split not in ('validation', 'final') or set(outcomes) != set(CONTROLLERS):
        raise ValueError('expected validation/final and all three matched controllers')
    count = 16 if split == 'validation' else 32
    if any(len(values) != count for values in outcomes.values()):
        raise ValueError(f'{split} requires exactly {count} outcomes per controller')
    comparisons = {name: paired_arrival_test(outcomes['trained'], outcomes[name])
                   for name in ('untrained', 'fixed')}
    successes = {name: sum(values) for name, values in outcomes.items()}
    minimum = successes['trained'] >= (13 if split == 'validation' else 26)
    if split == 'validation':
        comparison_passed = all(row['arrival_difference'] > 0 for row in comparisons.values())
    else:
        comparison_passed = all(row['arrival_difference'] >= 8
                                and row['one_sided_p'] <= 0.025
                                for row in comparisons.values())
    return {'successes': successes, 'comparisons': comparisons,
            'minimum_arrivals_passed': minimum,
            'comparisons_passed': comparison_passed,
            'passed': minimum and comparison_passed}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _software() -> dict[str, str]:
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'pygame': version('pygame'), 'platform': platform.platform(), 'device': 'cpu'}


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('protocol path must stay inside the project root')
    return path


def verify_protocol(path: Path, *, expected_sha256: str = APPROVED_PROTOCOL_SHA256,
                    project_root: Path = PROJECT_ROOT) -> dict:
    """Reject changed cases, checkpoints, source or recorded software before a run."""
    path, project_root = Path(path), Path(project_root)
    if _sha256(path) != expected_sha256:
        raise ValueError('protocol SHA-256 differs from the approved record')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload['protocol'] != 'm6_charge_fresh_matched_evaluation':
        raise ValueError('unexpected evaluation protocol')
    for relative, digest in payload['frozen_source_sha256'].items():
        if _sha256(_inside(project_root, relative)) != digest:
            raise ValueError(f'frozen source differs: {relative}')
    checkpoint_dir = _inside(project_root, payload['checkpoint_directory'])
    for name, record in payload['candidate_checkpoints'].items():
        if _sha256(_inside(checkpoint_dir, record['file'])) != record['sha256']:
            raise ValueError(f'checkpoint hash differs: {name}')
    if 'software' in payload and payload['software'] != _software():
        raise ValueError('software/device environment differs from the frozen protocol')
    config = EvaluationConfig(**payload['config'])
    if config.network_seed != payload['network_seed']:
        raise ValueError('network seed differs within protocol')
    if set(payload['candidate_checkpoints']) != {'trained', 'untrained'}:
        raise ValueError('both frozen candidate checkpoints are required')
    if set(payload['cases']) != {'validation', 'final'}:
        raise ValueError('protocol requires validation and final cases')
    all_cases = []
    for split, count in (('validation', 16), ('final', 32)):
        group = [EpisodeCase(**row) for row in payload['cases'][split]]
        if len(group) != count or any(case.split != split for case in group):
            raise ValueError('case split/count differs from the reviewed protocol')
        all_cases.extend(group)
    if (len({case.seed for case in all_cases}) != 48
            or len({case.case_id for case in all_cases}) != 48
            or any(case.seed in {*range(8), *range(100, 104), *range(200, 208)}
                   for case in all_cases)):
        raise ValueError('fresh case identities must be unique and disjoint from historical splits')
    for case in all_cases:
        if (not all(math.isfinite(value) for value in (case.x, case.y, case.heading))
                or not 6 <= case.x <= config.arena_width - 6
                or not 6 <= case.y <= config.arena_height - 6
                or not 0 <= case.heading < math.tau):
            raise ValueError('case pose must be finite and within arena bounds')
    budget = payload['budget']
    if (budget['training_updates'] != 0 or budget['validation_episodes'] != 48
            or budget['final_episodes_if_validation_passes'] != 96
            or budget['horizon_ticks_per_episode'] != config.horizon_ticks
            or 144 * config.horizon_ticks > budget['maximum_world_ticks']
            or budget['maximum_world_ticks'] > 86400):
        raise ValueError('evaluation budget differs from the reviewed limits')
    return payload


def run_evaluation(protocol_path: Path, output_dir: Path, *,
                   expected_sha256: str = APPROVED_PROTOCOL_SHA256,
                   project_root: Path = PROJECT_ROOT) -> Path:
    """Consume one reviewed protocol; persist every episode and gate final access."""
    protocol_path, output_dir = Path(protocol_path), Path(output_dir)
    ensure_new_run_directory(output_dir)
    protocol = verify_protocol(protocol_path, expected_sha256=expected_sha256,
                               project_root=project_root)
    config = EvaluationConfig(**protocol['config'])
    checkpoint_dir = _inside(Path(project_root), protocol['checkpoint_directory'])
    # The approved content fixes this directory: renaming/copying the JSON
    # cannot create another opportunity to evaluate the same fresh cases.
    receipt_path = checkpoint_dir / f'heldout-{expected_sha256}.execution.json'
    if receipt_path.exists():
        raise FileExistsError(f'This protocol already started evaluation: {receipt_path}')
    models = {name: load_charge_checkpoint(checkpoint_dir / record['file'],
                                            network_seed=config.network_seed)
              for name, record in protocol['candidate_checkpoints'].items()}
    for model in models.values():
        model.requires_grad_(False)
    initial = {name: {key: value.clone() for key, value in model.state_dict().items()}
               for name, model in models.items()}
    factories = {name: charge_controller_factory(model, config.network_seed)
                 for name, model in models.items()}
    factories['fixed'] = None
    started = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('x', encoding='utf-8') as file:
        json.dump({'started_utc': started, 'protocol_sha256': expected_sha256,
                   'output_directory': str(output_dir.resolve()),
                   'meaning': 'protocol consumed; inspect recorded run before any further evaluation'},
                  file, indent=2)
        file.write('\n')
    results, summaries, gates = {}, {}, {}
    completed_episodes = 0
    actual_ticks = 0
    try:
        with (output_dir / 'episodes.jsonl').open('x', encoding='utf-8') as journal:
            for split in ('validation', 'final'):
                # This branch precedes construction or evaluation of final cases.
                if split == 'final' and not gates['validation']['passed']:
                    break
                verify_protocol(protocol_path, expected_sha256=expected_sha256,
                                 project_root=project_root)
                cases = tuple(EpisodeCase(**row) for row in protocol['cases'][split])
                rows: dict[str, list[EpisodeResult]] = {name: [] for name in CONTROLLERS}
                for name in CONTROLLERS:
                    for index, case in enumerate(cases, start=1):
                        row = run_episode(case, config, factories[name])
                        if row.outcome not in ('success', 'timeout'):
                            raise RuntimeError('unexpected episode outcome')
                        entry = {'controller': name, 'result': asdict(row)}
                        journal.write(json.dumps(entry, allow_nan=False) + '\n')
                        journal.flush()
                        rows[name].append(row)
                        completed_episodes += 1
                        actual_ticks += row.ticks
                        if actual_ticks > protocol['budget']['maximum_world_ticks']:
                            raise RuntimeError('evaluation tick budget exceeded')
                        if index % 4 == 0:
                            print(f'{split}/{name}: {index}/{len(cases)} cases', flush=True)
                outcomes = {name: [row.outcome == 'success' for row in values]
                            for name, values in rows.items()}
                gates[split] = assess_split(split, outcomes)
                summaries[split] = {name: summarize(values) for name, values in rows.items()}
                results[split] = {name: [asdict(row) for row in values] for name, values in rows.items()}
                for name, model in models.items():
                    if any(not torch.equal(value, initial[name][key])
                           for key, value in model.state_dict().items()):
                        raise RuntimeError('frozen readout parameters changed during evaluation')
                with (output_dir / f'{split}.csv').open('x', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=('controller', *EpisodeResult.__dataclass_fields__))
                    writer.writeheader()
                    for name, values in results[split].items():
                        writer.writerows({'controller': name, **row} for row in values)
                (output_dir / f'{split}.json').write_text(json.dumps({
                    'gate': gates[split], 'summaries': summaries[split], 'results': results[split],
                }, indent=2, allow_nan=False) + '\n', encoding='utf-8')
                print(f"{split}: arrivals={gates[split]['successes']}; "
                      f"gate passed={gates[split]['passed']}", flush=True)
        verify_protocol(protocol_path, expected_sha256=expected_sha256, project_root=project_root)
        payload = {
            'experiment': 'm6_charge_fresh_matched_evaluation',
            'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'protocol': protocol, 'protocol_sha256': expected_sha256,
            'runner_sha256': _sha256(Path(__file__)), 'git_commit': _git_commit(),
            'optimizer_steps': 0, 'learning_rate': None, 'network_size': 100,
            'frozen_parameters_unchanged': True, 'integrity_checks_passed': True,
            'actual_episodes': completed_episodes, 'actual_world_ticks': actual_ticks,
            'software': _software(), 'results': results, 'summaries': summaries, 'gates': gates,
            'final_cases_evaluated': 'final' in results,
            'm6_gate_passed': gates.get('final', {}).get('passed', False),
        }
        path = output_dir / 'results.json'
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return path
    except Exception as error:
        (output_dir / 'aborted.json').write_text(json.dumps({
            'error_type': type(error).__name__, 'error': str(error),
            'completed_episodes': completed_episodes, 'actual_world_ticks': actual_ticks,
            'protocol_sha256': expected_sha256,
            'instruction': 'stop; preserve the journal and request a decision before another run',
        }, indent=2) + '\n', encoding='utf-8')
        raise


def main(argv: list[str] | None = None) -> int:
    """Run the approved hash-bound protocol exactly once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path,
                        default=Path('experiments/m6-charge-repair/heldout_protocol.json'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/m6-charge-evaluation'))
    args = parser.parse_args(argv)
    path = run_evaluation(args.protocol, args.output_dir)
    print(f'Wrote evaluation: {path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
