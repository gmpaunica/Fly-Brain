"""Approved 3,000-step observation-only PN-to-DNa02 timing diagnostic.

No candidate selection, parameter changes, extra windows or training. Records
the original two mid-band cases and compares against their frozen counts.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import torch

from flybrain.data.banc import load_banc_subset
from flybrain.neural.banc_network import BancDynamicsConfig, BancNetwork
from flybrain.world.sensing import SensoryObservation
from .banc_recovery import PROJECT_ROOT, sha256
from .records import ensure_new_run_directory


def observe_step(brain: BancNetwork, observation: SensoryObservation) -> list[dict]:
    """Advance exactly once and observe actual losses at nonsensory cells.

    Step numbers refer to the end of a 1 ms transition. The original steady
    interval is therefore steps 501..1250 inclusive. Actual recorded losses
    already account for the different PN time constant and reset rule.
    """
    population = brain.population
    if not population.record_charge:
        raise ValueError('diagnostic requires charge recording')
    before = population.voltage
    jumps = population.weights @ population.spikes.to(population.dtype)
    activity = brain.advance(observation, substeps=1)
    after = population.voltage
    losses = population.charge_losses
    rows = []
    for i, role in enumerate(brain.topology.groups):
        if role.startswith('sensory_'):
            continue
        v0, v1 = float(before[i]), float(after[i])
        jump, leak, reset = float(jumps[i]), float(losses.leak[i]), float(losses.reset[i])
        rows.append({
            'step': activity.end_step, 'root_id': brain.topology.root_ids[i], 'role': role,
            'voltage_before': v0, 'synaptic_jump': jump, 'leak_loss': leak,
            'reset_loss': reset, 'pre_reset_voltage': v1 + reset,
            'spike': int(activity.spike_counts[i]), 'voltage_after': v1,
            # Nonsensory external current is zero; small errors are float32 rounding.
            'balance_error': (v1 - v0) - (jump - leak - reset),
        })
    return rows


def relay_accounting(parent_steps: list[int], child_steps: list[int], delay: int) -> dict:
    """Check event correspondence and fixed-window boundary crossings, not a new gate."""
    expected = {step + delay for step in parent_steps if step + delay <= 1500}
    actual = set(child_steps)
    in_window = lambda step: 501 <= step <= 1250
    return {
        'delay_ms': delay,
        'exact_delayed_replay': expected == actual,
        'unmatched_expected_child_steps': sorted(expected - actual),
        'unmatched_actual_child_steps': sorted(actual - expected),
        'parent_steady_count': sum(in_window(step) for step in parent_steps),
        'child_steady_count': sum(in_window(step) for step in child_steps),
        'entering_window_parent_steps': [step for step in parent_steps
            if not in_window(step) and in_window(step + delay)],
        'leaving_window_parent_steps': [step for step in parent_steps
            if in_window(step) and not in_window(step + delay)],
    }


def projection_path(graph, side: str) -> list[str]:
    """Follow the actual unique central corridor from PN to steering endpoint."""
    start, = graph.indices(f'projection_{side}')
    end, = graph.indices(f'steering_{side}')
    path = [graph.root_ids[start]]
    while path[-1] != graph.root_ids[end]:
        successors = [edge.post for edge in graph.edges if edge.pre == path[-1]]
        if len(successors) != 1 or successors[0] in path:
            raise ValueError('diagnostic expects one acyclic PN-to-steering corridor')
        path.append(successors[0])
    return path


def run_diagnosis(output_dir: Path) -> dict:
    """Replay the two authorized cases once and preserve all earlier artifacts."""
    output_dir = Path(output_dir).resolve()
    ensure_new_run_directory(output_dir)
    parent_dir = PROJECT_ROOT / 'experiments/fb09-pn-charge'
    original = json.loads((parent_dir / 'results.json').read_text(encoding='utf-8'))
    parent_protocol = json.loads((parent_dir / 'protocol.json').read_text(encoding='utf-8'))
    if sha256(parent_dir / 'protocol.json') != original['protocol_sha256']:
        raise ValueError('original protocol integrity mismatch')
    if not all(sha256(PROJECT_ROOT / path) == digest
               for path, digest in parent_protocol['source_code_sha256'].items()):
        raise ValueError('core source differs from the frozen failed experiment')
    candidate, = original['candidates']
    cases = [row for row in candidate['cases'] if row['case']['name']
             in ('0.1-equal', '0.1-right_stronger')]
    if len(cases) != 2:
        raise ValueError('expected exactly the two approved mid-band cases')
    config = BancDynamicsConfig(**candidate['config'])
    graph = load_banc_subset(connectome_gain=1.)
    probe = BancNetwork(graph, dynamics_config=config)
    paths = {side: projection_path(probe.topology, side) for side in ('left', 'right')}
    protocol = {
        'name': 'fb09-pn-relay-timing-diagnosis',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'The six-to-four steady contrast change can be localized by exact event matching and charge accounting along the sourced PN-to-DNa02 corridors.',
        'cases': [row['case'] for row in cases], 'config': candidate['config'],
        'timing_substeps': parent_protocol['timing_substeps'],
        'budget': {'neural_substeps': 3000, 'optimizer_steps': 0},
        'stop_condition': 'Exactly two unchanged-parameter replays; no candidate, window, threshold or training changes.',
        'acceptance': 'Diagnostic fidelity only: exact full-pulse and steady counts for all83 cells; frozen source and previous files; no training-readiness promotion.',
        'source_code_sha256': parent_protocol['source_code_sha256'],
        'diagnostic_code_sha256': sha256(Path(__file__)),
        'original_results_sha256': sha256(parent_dir / 'results.json'),
        'original_protocol_sha256': original['protocol_sha256'],
        'subset_sha256': original['subset_sha256'],
        'software': {'python': platform.python_version(), 'torch': torch.__version__,
                     'platform': platform.platform(), 'device': 'cpu', 'commit': None},
        'seed': None, 'seed_reason': 'No random draws; fixed graph, input and reset.',
        'paths': paths,
    }
    prior = {str(path.resolve()): sha256(path) for path in (PROJECT_ROOT / 'experiments').rglob('*')
             if path.is_file() and not path.resolve().is_relative_to(output_dir)}
    output_dir.mkdir(parents=True, exist_ok=True)
    def save(name: str, payload: dict) -> None:
        with (output_dir / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')
    save('protocol.json', protocol)
    save('preservation_before.json', prior)
    summaries = []
    with (output_dir / 'relay_trace.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = None
        for recorded in cases:
            brain = BancNetwork(graph, dynamics_config=config)
            brain.population.record_charge = True
            case = recorded['case']
            pulse = SensoryObservation(case['left'], case['right'])
            zero = SensoryObservation(0., 0.)
            events = {root_id: [] for root_id in brain.topology.root_ids}
            pulse_counts = torch.zeros(83, dtype=torch.int64, device='cpu')
            steady_counts = torch.zeros_like(pulse_counts)
            totals = {}
            for index in range(1500):
                rows = observe_step(brain, pulse if 250 <= index < 1250 else zero)
                spikes = brain.population.spikes.to(torch.int64)
                if 250 <= index < 1250:
                    pulse_counts += spikes
                if 500 <= index < 1250:
                    steady_counts += spikes
                for cell in spikes.nonzero().flatten().tolist():
                    events[brain.topology.root_ids[cell]].append(index + 1)
                for row in rows:
                    row.update(case=case['name'], steady_window=500 <= index < 1250)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    total = totals.setdefault(row['root_id'], {'role': row['role'],
                        'steady_spikes': 0, 'jump_sum': 0., 'leak_sum': 0., 'reset_sum': 0.,
                        'max_balance_error': 0.})
                    total['max_balance_error'] = max(total['max_balance_error'], abs(row['balance_error']))
                    if row['steady_window']:
                        total['steady_spikes'] += row['spike']
                        total['jump_sum'] += row['synaptic_jump']
                        total['leak_sum'] += row['leak_loss']
                        total['reset_sum'] += row['reset_loss']
            replay = (pulse_counts.tolist() == recorded['pulse_per_neuron_counts']
                      and steady_counts.tolist() == recorded['steady_per_neuron_counts'])
            relays = {}
            for side, path in paths.items():
                relays[side] = {
                    'whole_path': relay_accounting(events[path[0]], events[path[-1]], len(path) - 1),
                    'edges': [{'pre': pre, 'post': post,
                               **relay_accounting(events[pre], events[post], 1)}
                              for pre, post in zip(path, path[1:])],
                }
            summaries.append({'case': case, 'exact_count_replay': replay, 'relays': relays,
                              'neurons': totals, 'spike_steps': events})
            stream.flush()
            print(f"{case['name']}: exact all-cell replay={replay}", flush=True)
    result = {
        'protocol_sha256': sha256(output_dir / 'protocol.json'),
        'trace_sha256': sha256(output_dir / 'relay_trace.csv'),
        'neural_substeps': 3000, 'optimizer_steps': 0, 'learning_rate': None, 'checkpoint': None,
        'cases': summaries, 'exact_count_replay': all(row['exact_count_replay'] for row in summaries),
        'previous_file_count': len(prior),
        'previous_files_preserved': all(Path(path).is_file() and sha256(Path(path)) == digest
                                        for path, digest in prior.items()),
        'source_code_unchanged': all(sha256(PROJECT_ROOT / path) == digest
                                   for path, digest in protocol['source_code_sha256'].items()),
        'diagnostic_code_unchanged': sha256(Path(__file__)) == protocol['diagnostic_code_sha256'],
        'claim': 'Observation-only event timing, not a new candidate or passed FB10 entry gate.',
    }
    result['diagnostic_valid'] = all(result[key] for key in ('exact_count_replay',
        'previous_files_preserved', 'source_code_unchanged', 'diagnostic_code_unchanged'))
    save('results.json', result)
    return result


def main() -> None:
    """Run this approved diagnostic once in an unused destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments/fb09-relay-diagnosis')
    result = run_diagnosis(parser.parse_args().output_dir)
    print(f"Closed diagnostic: 3000 steps; valid={result['diagnostic_valid']}")
    raise SystemExit(0 if result['diagnostic_valid'] else 2)


if __name__ == '__main__':
    main()
