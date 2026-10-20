"""Bounded, matched FB-08 interventions on the frozen M6 charge readout."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random

import torch

from flybrain.neural.charge_readout import ChargeMotorDecoder
from flybrain.neural.controller import NeuralController
from flybrain.neural.lif import LIFPopulation
from flybrain.neural.network import SyntheticNetwork
from flybrain.neural.readout import TrainableMotorReadout
from flybrain.neural.sensory import OdorObservation
from .charge_evaluation import PROJECT_ROOT, _software
from .charge_repair import load_charge_checkpoint
from .evaluation import EpisodeCase, EpisodeResult, EvaluationConfig, _git_commit, run_episode, summarize
from .fb07e_probe import CANDIDATE_SCALE
from .readout_experiment import FEATURE_ALPHA
from .records import ensure_new_run_directory


CONDITIONS = ('intact', 'left_off', 'right_off', 'both_off', 'synapses_off',
              'pathway_half', 'pathway_double', 'rewired_701', 'rewired_702', 'rewired_703')
LABELS = ('Intact', 'Left sensory off', 'Right sensory off', 'Both sensory off',
          'All synapses off', 'Sensory pathway x0.5', 'Sensory pathway x2',
          'Rewired 701', 'Rewired 702', 'Rewired 703')


class InterventionNetwork(SyntheticNetwork):
    """Fresh experimental copy; alter currents/weights without changing LIF rules.

    Indices follow the fixed 100-neuron synthetic group contract. Rewiring
    retains each destination row's weight multiset within its source layer.
    It destroys side-specific blocks and need not preserve source out-degree.
    """

    def __init__(self, condition: str) -> None:
        if condition not in CONDITIONS:
            raise ValueError('unknown FB-08 condition')
        super().__init__(seed=7, encoder_mode='rational',
                         encoder_scale=CANDIDATE_SCALE, record_charge=True)
        self.condition = condition
        weights = self.topology.weights
        if condition == 'synapses_off':
            weights.zero_()
        elif condition in ('pathway_half', 'pathway_double'):
            weights[20:70, :20] *= .5 if condition == 'pathway_half' else 2.
        elif condition.startswith('rewired_'):
            generator = torch.Generator(device='cpu').manual_seed(int(condition[-3:]))
            for post in range(20, 70):
                weights[post, :20] = weights[post, :20][torch.randperm(20, generator=generator)]
            for post in (*range(70, 80), *range(90, 100)):
                weights[post, 20:70] = weights[post, 20:70][torch.randperm(50, generator=generator)]
        # Labels/indices and encoder are unchanged. Both visible topology and
        # actual dynamics own the same intervention; no private-weight mutation.
        self.topology = replace(self.topology, _weights=weights)
        self.population = LIFPopulation(weights, self.population.config, record_charge=True)

    def encode(self, observation: OdorObservation) -> torch.Tensor:
        """Remove specified external sensory current before neural integration."""
        current = super().encode(observation)
        if self.condition in ('left_off', 'both_off'):
            current[:10] = 0
        if self.condition in ('right_off', 'both_off'):
            current[10:20] = 0
        return current


def controller_factory(model: TrainableMotorReadout, condition: str) -> Callable[[], NeuralController]:
    """Create independent state for one experimental episode, sharing frozen weights."""
    def create() -> NeuralController:
        network = InterventionNetwork(condition)
        return NeuralController(network, ChargeMotorDecoder(network.topology, model, FEATURE_ALPHA))
    return create


def diagnostic_cases() -> tuple[EpisodeCase, ...]:
    """Sixteen deterministic new starts; these are diagnostic, not a final split."""
    cases = []
    for seed in range(51000, 51016):
        rng = random.Random(seed)
        angle = rng.uniform(0, math.tau)
        radius = math.sqrt(rng.uniform(60**2, 180**2))
        cases.append(EpisodeCase(f'fb08-{seed}', 'diagnostic', seed,
                                600 + radius * math.cos(angle),
                                300 + radius * math.sin(angle), rng.uniform(0, math.tau)))
    return tuple(cases)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_results(result_path: Path) -> None:
    """Generate standalone PNG/SVG solely from recorded result rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    payload = json.loads(Path(result_path).read_text(encoding='utf-8'))
    names = list(payload['results'])
    labels = [LABELS[CONDITIONS.index(name)] for name in names]
    counts = [payload['summaries'][name]['successes'] for name in names]
    n = len(payload['protocol']['cases'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout='constrained')
    colors = ['#2563eb' if name == 'intact' else '#607d8b' for name in names]
    axes[0].barh(labels, counts, color=colors)
    axes[0].set_xlim(0, n + 2)
    axes[0].set_xlabel(f'Food arrivals / {n} matched starts')
    axes[0].set_title('Navigation under intervention')
    for i, count in enumerate(counts):
        axes[0].text(count + .15, i, f'{count}/{n}', va='center')
    axes[0].invert_yaxis()
    for i, name in enumerate(names):
        values = [r['progress_fraction'] for r in payload['results'][name]]
        offsets = [(j - (len(values)-1)/2) * .025 for j in range(len(values))]
        axes[1].scatter(values, [i+offset for offset in offsets], s=15, alpha=.65,
                        color=colors[i])
        axes[1].scatter([sum(values)/len(values)], [i], marker='D', s=40, color='black')
    axes[1].set_yticks(range(len(names)), labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color='grey', lw=.8)
    axes[1].set_xlabel('Progress toward arrival radius (all cases)')
    axes[1].set_title('Dots: cases; black diamonds: means')
    fig.suptitle('FB-08: frozen readout, zero training updates\n'
                 'Same 16 starts per condition; three rewired graphs are separate controls')
    for extension in ('png', 'svg'):
        target = Path(result_path).with_name(f'interventions.{extension}')
        with target.open('xb') as file:
            fig.savefig(file, format=extension, dpi=160)
    plt.close(fig)


def run_experiment(output_dir: Path, *,
                   checkpoint: Path = PROJECT_ROOT / 'experiments/m6-charge-repair/trained.pt',
                   config: EvaluationConfig | None = None,
                   cases: tuple[EpisodeCase, ...] | None = None,
                   make_plots: bool = True) -> Path:
    """Record one bounded assay; stop if intact reliability or integrity fails."""
    output_dir, checkpoint = Path(output_dir), Path(checkpoint)
    ensure_new_run_directory(output_dir)
    config = config or EvaluationConfig()
    cases = diagnostic_cases() if cases is None else cases
    if (config.network_seed != 7 or not 1 <= config.horizon_ticks <= 600
            or len(cases) != 16 or len({c.case_id for c in cases}) != 16
            or any(c.split != 'diagnostic' for c in cases)):
        raise ValueError('FB-08 requires graph seed 7, 16 diagnostic cases and <=600 ticks')
    if make_plots:
        version('matplotlib')  # Fail before episodes if optional plot extra is absent.
    model = load_charge_checkpoint(checkpoint, network_seed=7)
    model.requires_grad_(False)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    checkpoint_sha = _sha(checkpoint)
    sources = {p.relative_to(PROJECT_ROOT).as_posix(): _sha(p)
               for p in sorted((PROJECT_ROOT/'src').rglob('*.py'))}
    prior = {p.relative_to(PROJECT_ROOT).as_posix(): _sha(p)
             for p in sorted((PROJECT_ROOT/'experiments').rglob('*'))
             if p.is_file() and p.suffix in ('.json', '.csv', '.pt')}
    conditions = {}
    for name in CONDITIONS:
        weights = InterventionNetwork(name).population.weights
        conditions[name] = {'description': LABELS[CONDITIONS.index(name)],
                            'weights_sha256': hashlib.sha256(weights.numpy().tobytes()).hexdigest(),
                            'nonzero_synapses': int(torch.count_nonzero(weights))}
    protocol = {
        'experiment': 'FB-08 frozen charge readout interventions',
        'hypothesis': 'both sensory off loses >=8/16 arrivals; synapses off loses <=2/16 versus intact',
        'interpretation': 'descriptive thresholds; no population test or topology-benefit claim',
        'config': asdict(config), 'cases': [asdict(c) for c in cases],
        'conditions': conditions, 'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': checkpoint_sha, 'frozen_source_sha256': sources,
        'prior_artifact_sha256': prior, 'budget': {'optimizer_steps': 0,
            'maximum_episodes': 160, 'maximum_world_ticks': 96000},
        'stop_rule': 'intact arrivals >=13/16 before interventions; abort on integrity/numerical mismatch',
        'rewiring': 'seeds 701/702/703; independent row permutation in sensory->interneuron and interneuron->side-motor layers; row multiset preserved, out-degree not preserved',
        'scientific_limit': 'fixed trained readout, annular starts, graph seed 7; engineered sensory reconstruction; no retraining, biological claim or new final claim',
        'software': {**_software(), 'matplotlib': version('matplotlib') if make_plots else None},
        'git_commit': _git_commit(), 'started_utc': datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    results, summaries, comparisons = {}, {}, {}
    total_ticks = 0
    episodes = 0
    try:
        with (output_dir/'episodes.jsonl').open('x', encoding='utf-8') as journal:
            for name in CONDITIONS:
                if name != 'intact' and summaries['intact']['successes'] < 13:
                    break
                rows = []
                for index, case in enumerate(cases, 1):
                    result = run_episode(case, config, controller_factory(model, name))
                    row = asdict(result)
                    journal.write(json.dumps({'condition': name, **row}, allow_nan=False)+'\n')
                    journal.flush()
                    rows.append(row)
                    total_ticks += result.ticks
                    episodes += 1
                    if total_ticks > 96000 or episodes > 160:
                        raise RuntimeError('FB-08 budget exceeded')
                    if index % 4 == 0:
                        print(f'{name}: {index}/16', flush=True)
                results[name] = rows
                summaries[name] = summarize([EpisodeResult(**r) for r in rows])
                base = results['intact']
                comparisons[name] = {
                    'successes_lost': sum(a['outcome']=='success' and b['outcome']!='success'
                                          for a,b in zip(base, rows)),
                    'successes_gained': sum(a['outcome']!='success' and b['outcome']=='success'
                                            for a,b in zip(base, rows)),
                    'mean_paired_progress_delta': sum(b['progress_fraction']-a['progress_fraction']
                                                      for a,b in zip(base, rows))/16,
                }
                print(f"{name}: arrivals {summaries[name]['successes']}/16", flush=True)
        checks = {
            'checkpoint_unchanged': _sha(checkpoint) == checkpoint_sha,
            'parameters_unchanged': all(torch.equal(v, initial[k]) for k,v in model.state_dict().items()),
            'source_unchanged': all(_sha(PROJECT_ROOT/p)==digest for p,digest in sources.items()),
            'prior_artifacts_unchanged': all(_sha(PROJECT_ROOT/p)==digest for p,digest in prior.items()),
        }
        if not all(checks.values()):
            raise RuntimeError('frozen artifact integrity failed')
        assay = summaries['intact']['successes'] >= 13
        supported = (summaries['intact']['successes']-summaries['both_off']['successes'] >= 8
                     and summaries['intact']['successes']-summaries['synapses_off']['successes'] <= 2
                     if assay else None)
        payload = {'protocol': protocol, 'optimizer_steps': 0, 'learning_rate': None,
                   'network_size': 100, 'actual_episodes': episodes, 'actual_world_ticks': total_ticks,
                   'assay_passed': assay, 'hypothesis_supported': supported,
                   'checks': checks, 'results': results, 'summaries': summaries,
                   'paired_comparisons': comparisons,
                   'completed_utc': datetime.now(timezone.utc).isoformat()}
        with (output_dir/'episodes.csv').open('x', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=('condition', *results['intact'][0]))
            writer.writeheader()
            for name, rows in results.items():
                writer.writerows({'condition': name, **r} for r in rows)
        path = output_dir/'results.json'
        path.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        if make_plots:
            plot_results(path)
        return path
    except Exception as error:
        (output_dir/'aborted.json').write_text(json.dumps({
            'error': str(error), 'completed_episodes': episodes, 'actual_world_ticks': total_ticks,
            'next': 'preserve records; stop and request a decision before any rerun'}, indent=2),
            encoding='utf-8')
        raise


def main(argv: list[str] | None = None) -> int:
    """Run the single predeclared FB-08 assay without any training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/fb08-interventions'))
    args = parser.parse_args(argv)
    path = run_experiment(args.output_dir)
    print(f'Wrote {path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
