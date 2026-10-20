"""Approved 4,500-step observation-only replay of the BANC PN plateau."""

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from flybrain.data.banc import load_banc_subset
from flybrain.neural.banc_network import BancDynamicsConfig, BancNetwork
from flybrain.neural.lif import LIFPopulation
from flybrain.world.sensing import SensoryObservation
from .banc_recovery import PROJECT_ROOT, make_protocol, sha256
from .records import ensure_new_run_directory


def trace_step(population: LIFPopulation, current: torch.Tensor,
               indices: tuple[int, ...]) -> list[dict]:
    """Observe one real LIF transition; charge recording never changes dynamics."""
    if not population.record_charge:
        raise ValueError('diagnostic requires charge recording')
    before = population.voltage
    jumps = population.weights @ population.spikes.to(population.dtype)
    spikes = population.step(current)
    after = population.voltage
    losses = population.charge_losses
    return [{
        'step': population.step_count, 'index': i,
        'voltage_before': float(before[i]), 'synaptic_jump': float(jumps[i]),
        'current_contribution': float(current[i]) * -math.expm1(-population.config.dt / population.config.tau),
        'leak_loss': float(losses.leak[i]), 'reset_loss': float(losses.reset[i]),
        'pre_reset_voltage': float(after[i]) + float(losses.reset[i]),
        'overshoot': max(0., float(after[i]) + float(losses.reset[i]) - population.config.threshold) if spikes[i] else 0.,
        'spike': int(spikes[i]), 'voltage_after': float(after[i]),
    } for i in indices]


def run_diagnosis(output_dir: Path) -> dict:
    """Replay exactly three recorded high-band cases; reject reused output."""
    ensure_new_run_directory(output_dir)
    original_path = PROJECT_ROOT / 'experiments/fb09-dynamics-recovery/results.json'
    original = json.loads(original_path.read_text(encoding='utf-8'))
    candidate = next(row for row in original['candidates'] if row['config']['synaptic_budget'] == 3)
    cases = [row for row in candidate['cases'] if row['case']['center'] == .25
             and row['case']['pattern'] in ('equal', 'right_stronger', 'left_stronger')]
    if len(cases) != 3:
        raise ValueError('expected exactly three recorded high-band cases')
    protocol = make_protocol()
    protocol.update({
        'name': 'fb09-pn-plateau-diagnosis',
        'hypothesis': 'Compressed distinct presynaptic drive is mapped onto a PN count plateau by integration/reset dynamics.',
        'cases': [row['case'] for row in cases],
        'config': asdict(BancDynamicsConfig(3)),
        'budget': {'neural_substeps': 4500, 'optimizer_steps': 0},
        'candidate_budgets': [3.], 'conditional_causal_cases': None,
        'stop_condition': 'Stop after three unchanged-parameter replays; no new candidate or training.',
        'original_results_sha256': sha256(original_path),
        'diagnostic_code_sha256': sha256(Path(__file__)),
    })
    prior = {str(p.resolve()): sha256(p) for p in (PROJECT_ROOT / 'experiments').rglob('*') if p.is_file()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (('protocol.json', protocol), ('preservation_before.json', prior)):
        with (output_dir / name).open('x', encoding='utf-8') as file:
            json.dump(payload, file, indent=2)
    summaries = []
    traces = []
    for original_case in cases:
        brain = BancNetwork(load_banc_subset(connectome_gain=1.), dynamics_config=BancDynamicsConfig(3))
        brain.population = LIFPopulation(brain.topology.weights, brain.population.config, record_charge=True)
        indices = tuple(brain.topology.indices('projection_left') + brain.topology.indices('projection_right'))
        names = {i: brain.topology.groups[i] for i in indices}
        case = original_case['case']
        pulse = brain.encode(SensoryObservation(case['left'], case['right']))
        zero = brain.encode(SensoryObservation(0., 0.))
        counts = torch.zeros(83, dtype=torch.int64, device='cpu')
        recorded = []
        for step in range(1500):
            rows = trace_step(brain.population, pulse if 250 <= step < 1250 else zero, indices)
            if 500 <= step < 1250:
                counts += brain.population.spikes.to(torch.int64)
            for row in rows:
                row.update(case=case['pattern'], role=names[row['index']], steady_window=500 <= step < 1250)
            recorded.extend(rows)
        replay = counts.tolist() == original_case['steady_per_neuron_counts']
        case_summary = {'case': case, 'exact_count_replay': replay, 'projections': {}}
        for index in indices:
            all_rows = [row for row in recorded if row['index'] == index]
            selected = [row for row in all_rows if row['steady_window']]
            spikes = [row['step'] for row in selected if row['spike']]
            intervals = [b-a for a, b in zip(spikes, spikes[1:])]
            case_summary['projections'][names[index]] = {
                'spikes': len(spikes), 'spike_steps': spikes,
                'interval_histogram_ms': {str(dt): intervals.count(dt) for dt in sorted(set(intervals))},
                **{key + '_sum': sum(row[key] for row in selected) for key in
                   ('synaptic_jump', 'leak_loss', 'reset_loss', 'overshoot')},
                'initial_voltage': selected[0]['voltage_before'],
                'final_voltage': selected[-1]['voltage_after'],
            }
        summaries.append(case_summary)
        traces.extend(recorded)
        print(f"{case['pattern']}: exact count replay={replay}", flush=True)
    with (output_dir / 'projection_trace.csv').open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    result = {
        'protocol_sha256': sha256(output_dir / 'protocol.json'),
        'trace_sha256': sha256(output_dir / 'projection_trace.csv'),
        'neural_substeps': 4500, 'optimizer_steps': 0, 'learning_rate': None, 'checkpoint': None,
        'cases': summaries, 'exact_count_replay': all(row['exact_count_replay'] for row in summaries),
        'previous_files_preserved': all(sha256(Path(path)) == digest for path, digest in prior.items()),
        'source_code_unchanged': all(sha256(PROJECT_ROOT / p) == digest for p, digest in protocol['source_code_sha256'].items()),
        'diagnostic_code_unchanged': sha256(Path(__file__)) == protocol['diagnostic_code_sha256'],
    }
    with (output_dir / 'results.json').open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments/fb09-pn-diagnosis')
    result = run_diagnosis(parser.parse_args().output_dir)
    print('Closed diagnostic: 4500 steps; exact replay=', result['exact_count_replay'])


if __name__ == '__main__':
    main()
