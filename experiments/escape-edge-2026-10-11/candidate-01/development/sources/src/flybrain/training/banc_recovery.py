"""One capped, non-learning BANC dynamics experiment and causal controls.

The protocol is stored before the first neural step. Existing results are
never overwritten. A failed gate closes this experiment, without a new sweep.
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import torch

from flybrain.data.banc import BANC_SUBSET_SHA256, BANC_TRANSFORM_VERSION, load_banc_subset
from flybrain.neural.banc_network import BancDynamicsConfig, BancNetwork
from flybrain.neural.lif import LIFPopulation
from flybrain.world.sensing import SensoryObservation
from .records import ensure_new_run_directory


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_BUDGETS = (1.25, 2.0, 3.0)
CENTERS = (.025, .1, .25)
BASELINE_STEPS = 250
PULSE_STEPS = 1000
WASHOUT_STEPS = 250
STEADY_START = 250
WASHOUT_GRACE = 20


def sha256(path: Path) -> str:
    """Hash bytes for source and experiment preservation receipts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pulse_cases() -> list[dict]:
    """Fixed zero case and five bilateral patterns at each of three levels."""
    cases = [{'name': 'zero', 'center': 0., 'pattern': 'zero', 'left': 0., 'right': 0.}]
    for center in CENTERS:
        for pattern, left, right in (
            ('left', center, 0.), ('right', 0., center),
            ('equal', center, center), ('right_stronger', .8 * center, 1.2 * center),
            ('left_stronger', 1.2 * center, .8 * center),
        ):
            cases.append({'name': f'{center:g}-{pattern}', 'center': center,
                          'pattern': pattern, 'left': left, 'right': right})
    return cases


def contrast_gate(equal: tuple[float, float], right_stronger: tuple[float, float],
                  left_stronger: tuple[float, float]) -> dict:
    """Require signed >=1Hz and >=2% response shifts around equal-input bias."""
    baseline = equal[1] - equal[0]
    right_shift = right_stronger[1] - right_stronger[0] - baseline
    left_shift = baseline - (left_stronger[1] - left_stronger[0])
    right_minimum = max(1., .02 * max(sum(right_stronger), 1.))
    left_minimum = max(1., .02 * max(sum(left_stronger), 1.))
    return {'equal_input_bias_hz': baseline,
            'right_shift_hz': right_shift, 'left_shift_hz': left_shift,
            'right_minimum_hz': right_minimum, 'left_minimum_hz': left_minimum,
            'passed': right_shift >= right_minimum and left_shift >= left_minimum}


def make_protocol(*, wide_encoder: bool = False, pn_filter: bool = False, phase_rate: bool = False,
                  pn_charge: bool = False, latency_aligned: bool = False) -> dict:
    """Construct the exact predeclared hypothesis, gates and maximum budget."""
    if latency_aligned:
        if any((wide_encoder, pn_filter, phase_rate, pn_charge)):
            raise ValueError('latency-aligned protocol is a separate experiment mode')
        pn_charge = True
    code_paths = ('src/flybrain/data/banc.py', 'src/flybrain/neural/banc_network.py',
                  'src/flybrain/neural/banc_motor.py', 'src/flybrain/neural/lif.py',
                  'src/flybrain/training/banc_recovery.py', 'src/flybrain/neural/banc_synapses.py')
    protocol = {
        'schema_version': 1, 'name': 'fb09-normalized-nonself-recovery',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'At least one fixed row budget allows bounded downstream propagation, bilateral contrast and finite-latency washout in the declared approach-odor band.',
        'subset_sha256': BANC_SUBSET_SHA256, 'transform_version': BANC_TRANSFORM_VERSION,
        'weight_rule': 'W[post,pre]=budget*count/sum(selected nonself incoming counts); source self-edges masked only in simulator; all cross-side edges retained',
        'encoder': {'rule': '2.5*c/(c+s_i)', 'scales': 'logspace(.005,.05,N_side) in sorted original root-ID order',
                    'tonic_drive': 0, 'assumption': 'engineered multiscale sensitivity, not measured ORN physiology'},
        'lif': {'dt_seconds': .001, 'tau_seconds': .02, 'threshold': 1., 'reset': 0., 'rest': 0.},
        'candidate_budgets': list(CANDIDATE_BUDGETS), 'cases': pulse_cases(),
        'timing_substeps': {'baseline': BASELINE_STEPS, 'pulse': PULSE_STEPS,
                            'washout': WASHOUT_STEPS, 'pulse_steady_start': STEADY_START,
                            'washout_grace': WASHOUT_GRACE},
        'gates': {
            'propagation': 'At least one steady-window spike in each intended-side PN, DNa02 and motor group; contralateral activity allowed.',
            'contrast': 'Right-minus-left DNa02 steady Hz changes in the appropriate direction relative to equal-input bias at each center, by >=1Hz and >=2% of max(total steering Hz,1Hz).',
            'washout': 'No spikes after 20ms grace; max absolute voltage <1e-4 after 250ms zero input.',
            'stability': 'Finite voltage throughout, zero-only and pre-pulse baseline silent, maximum per-cell steady-window rate <=250Hz.',
            'causal': 'All-synapse ablation preserves ORN counts and eliminates all downstream spikes; each DNa02 incoming-edge cut eliminates that DNa02 and its ipsilateral motor spikes, without renormalization.'},
        'selection': 'Run all three candidates; choose the smallest budget passing every pulse gate. No navigation outcomes used.',
        'budget': {'selection_neural_substeps': 72000, 'conditional_causal_neural_substeps': 4500,
                   'maximum_total_neural_substeps': 76500, 'optimizer_steps': 0,
                   'learning_rate': None, 'checkpoint': None, 'behavioral_evaluation_cases': []},
        'conditional_causal_cases': {'conditions': ['synapses_off', 'left_dna02_input_off', 'right_dna02_input_off'],
                                    'left': .1, 'right': .1, 'substeps_each': 1500},
        'stop_condition': 'Close after these candidates and conditional controls. If none pass, do not choose a config, append candidates, tune encoder, or start training.',
        'software': {'python': platform.python_version(), 'torch': torch.__version__,
                     'device': 'cpu', 'platform': platform.platform(), 'commit': None},
        'seed': None, 'seed_reason': 'Fixed graph, inputs and reset; no random draws.',
        'source_code_sha256': {path: sha256(PROJECT_ROOT / path) for path in code_paths},
    }
    protocol['candidate_configs'] = [asdict(BancDynamicsConfig(b)) for b in CANDIDATE_BUDGETS]
    if wide_encoder or pn_filter:
        protocol.update({
            'name': 'fb09-wide-range-encoder-repair',
            'hypothesis': 'Wider ORN sensitivities reduce high-band compression and shared volleys, restoring transient directional endpoint activity with unchanged graph and LIF rules.',
            'candidate_budgets': [3.],
            'candidate_configs': [asdict(BancDynamicsConfig(3., encoder_max_scale=.375))],
            'selection': 'One predeclared candidate; every pulse and contrast gate must pass.',
            'stop_condition': 'Close after one candidate and conditional controls; no added gain/scale candidates or training.',
        })
        protocol['encoder']['scales'] = 'logspace(.005,.375,N_side) in sorted original root-ID order'
        protocol['budget'].update(selection_neural_substeps=24000, maximum_total_neural_substeps=28500)
    if pn_filter:
        protocol.update({
            'name': 'fb09-pn-filter-mechanism',
            'hypothesis': 'Area-preserving 5 ms PN-only synaptic filtering reduces peak/reset loss and restores contrast without changing effective weights or downstream transmission.',
            'candidate_configs': [asdict(BancDynamicsConfig(3., encoder_max_scale=.375, projection_filter_tau=.005))],
            'projection_filter': 'z_next=exp(-dt/.005)*z+(1-exp(-dt/.005))*(W@previous_spikes); replace jumps at two PN rows only; clear state on reset',
            'stop_condition': 'Close after one fixed 5 ms PN-filter candidate and conditional controls; no tau/gain/range sweep or training.',
        })
    if phase_rate or pn_charge:
        if wide_encoder or pn_filter:
            raise ValueError('phase-rate candidate cannot be combined with other experiment modes')
        protocol.update({
            'name': 'fb09-phase-rate-representation',
            'hypothesis': 'Independent square-root sensory rate coding with distributed fractional phases preserves contrast and avoids synchronized quantized ORN volleys.',
            'candidate_budgets': [3.],
            'candidate_configs': [asdict(BancDynamicsConfig(3., sensory_mode='phase_rate'))],
            'encoder': {'rule': 'rate_Hz=min(96,176*sqrt(c)); phase+=rate*dt; phase>=1 triggers one ORN LIF current pulse; phase-=1',
                        'initial_phases': '(j+.5)/N in sorted root-ID order, separately per side',
                        'pulse_current': '1.01*(threshold-rest)/(1-exp(-dt/tau)); zero otherwise',
                        'tonic_drive': 0, 'assumption': 'engineered sensory spike schedule; no downstream, geometry or teacher inputs'},
            'selection': 'One phase-rate candidate; every original pulse and contrast gate must pass.',
            'stop_condition': 'Close after this representation candidate and conditional controls; no coefficient/cap/exponent/phase tuning or training.',
        })
        protocol['conditional_causal_cases']['conditions'].append('downstream_synapses_off')
        protocol['budget'].update(selection_neural_substeps=24000, conditional_causal_neural_substeps=6000,
                                  maximum_total_neural_substeps=30000)
        protocol['gates']['causal'] = ('All-synapse cut preserves exact external encoding waveform and silences downstream; '
            'the retained ORN-to-ORN recipient may change. A separate downstream-only cut retains sensory wiring, '
            'preserves all ORN counts and silences downstream. Each DNa02 incoming cut silences its endpoint/motors '
            'without renormalization. All controls must preserve external drive and stability/washout.')
    if pn_charge:
        if phase_rate:
            raise ValueError('choose only one experiment mode')
        protocol.update({
            'name': 'fb09-pn-charge-preserving',
            'hypothesis': 'PN-only excess-charge retention with explicitly assumed membrane parameters preserves directional information lost at hard reset.',
            'candidate_configs': [asdict(BancDynamicsConfig(3., sensory_mode='phase_rate', projection_dynamics='charge_preserving'))],
            'projection_dynamics': {'tau_seconds': .025, 'threshold': 1.1,
                'reset': 'candidate - spike*1.1; at most one spike per millisecond; suprathreshold residual retained',
                'scope': 'two projection neurons only; other cells use unchanged base LIF; no synaptic filter'},
            'selection': 'One user-approved PN-only candidate; every original gate must pass.',
            'stop_condition': 'Close after this candidate and conditional controls; no tau/threshold/encoder sweep or training.',
        })
    if latency_aligned:
        protocol.update({
            'name': 'fb09-latency-aligned-sustained-rate',
            'hypothesis': 'The unchanged PN-charge candidate preserves sustained directional rate after accounting for sourced relay delays.',
            'measurement': {
                'definition': 'Count each DNa02 over the common PN interval delayed by its sourced corridor length; original simultaneous counts also recorded.',
                'steering_delays_substeps': steering_delays(load_banc_subset(connectome_gain=1.)),
                'pn_end_step_window_inclusive': [501, 1250],
                'duration_seconds': .75,
                'limitation': 'Historically informed change of measurement definition; not independent confirmation or simultaneous-window robustness.',
            },
            'selection': 'One unchanged candidate; all original pulse gates plus latency-aligned directional thresholds must pass.',
            'stop_condition': 'Close after one capped assay and conditional controls; no window/parameter selection or training.',
        })
        protocol['gates']['contrast'] += ' Endpoint windows are aligned to the fixed PN interval by source-derived delays; thresholds unchanged.'
    return protocol


def steering_delays(topology) -> dict[str, int]:
    """Derive unique PN-to-DNa02 corridor lengths; never use observed spike times."""
    delays = {}
    for side in ('left', 'right'):
        start, = topology.indices(f'projection_{side}')
        target, = topology.indices(f'steering_{side}')
        path = [topology.root_ids[start]]
        while path[-1] != topology.root_ids[target]:
            next_ids = [edge.post for edge in topology.edges
                        if edge.pre == path[-1] and edge.pre != edge.post]
            if len(next_ids) != 1 or next_ids[0] in path:
                raise ValueError('alignment requires a unique acyclic sourced corridor')
            path.append(next_ids[0])
        delays[side] = len(path) - 1
    return delays


def aligned_steering_rates(events: torch.Tensor, delays: dict[str, int]) -> list[float]:
    """Count offline endpoint events in shifted equal-duration windows only."""
    start, end = BASELINE_STEPS + STEADY_START, BASELINE_STEPS + PULSE_STEPS
    seconds = (PULSE_STEPS - STEADY_START) * .001
    return [int(events[start + delays[side]:end + delays[side], column].sum()) / seconds
            for column, side in enumerate(('left', 'right'))]


def measure_pulse(brain: BancNetwork, case: dict, *, endpoint_delays: dict[str, int] | None = None) -> dict:
    """Measure one reset pulse at 1ms resolution, observing every transition."""
    brain.reset()
    size = brain.population.size
    zero = brain.encode(SensoryObservation(0., 0.))
    pulse = brain.encode(SensoryObservation(case['left'], case['right']))
    baseline = torch.zeros(size, dtype=torch.int64, device='cpu')
    pulse_counts = torch.zeros_like(baseline)
    steady = torch.zeros_like(baseline)
    silence = torch.zeros_like(baseline)
    late_silence = torch.zeros_like(baseline)
    peak_voltage = 0.
    finite = True
    drive_hash = hashlib.sha256()
    steering_indices = [brain.topology.indices(f'steering_{side}')[0] for side in ('left', 'right')]
    steering_events = torch.zeros((BASELINE_STEPS + PULSE_STEPS + WASHOUT_STEPS, 2),
                                  dtype=torch.int64, device='cpu') if endpoint_delays is not None else None
    for step in range(BASELINE_STEPS + PULSE_STEPS + WASHOUT_STEPS):
        on = BASELINE_STEPS <= step < BASELINE_STEPS + PULSE_STEPS
        if brain.dynamics_config is not None and brain.dynamics_config.sensory_mode == 'phase_rate':
            observation = SensoryObservation(case['left'], case['right']) if on else SensoryObservation(0., 0.)
            drive_hash.update(brain.encode(observation).numpy().tobytes())
            spikes = brain.advance(observation, substeps=1).spike_counts
        else:
            drive_hash.update((pulse if on else zero).numpy().tobytes())
            spikes = brain.population.step(pulse if on else zero).to(torch.int64)
        voltage = brain.population.voltage
        if steering_events is not None:
            steering_events[step] = spikes[steering_indices]
        finite = finite and bool(torch.isfinite(voltage).all().item())
        peak_voltage = max(peak_voltage, voltage.abs().max().item())
        if step < BASELINE_STEPS:
            baseline += spikes
        elif on:
            pulse_counts += spikes
            if step >= BASELINE_STEPS + STEADY_START:
                steady += spikes
        else:
            silence += spikes
            if step >= BASELINE_STEPS + PULSE_STEPS + WASHOUT_GRACE:
                late_silence += spikes
    groups = sorted(set(brain.topology.groups))

    def grouped(counts: torch.Tensor) -> dict:
        return {group: int(counts[list(brain.topology.indices(group))].sum().item()) for group in groups}

    steady_seconds = (PULSE_STEPS - STEADY_START) * brain.population.config.dt
    steady_groups = grouped(steady)
    intended = [side for side in ('left', 'right') if case[side] > 0]
    propagation = all(steady_groups[f'{group}_{side}'] > 0
                      for side in intended for group in ('projection', 'steering', 'motor'))
    tail_voltage = brain.population.voltage.abs().max().item()
    peak_rate = steady.max().item() / steady_seconds
    zero_only = case['left'] == case['right'] == 0
    gates = {'finite': finite, 'baseline_silent': baseline.sum().item() == 0,
             'zero_case_silent': not zero_only or pulse_counts.sum().item() + silence.sum().item() == 0,
             'propagation': propagation,
             'washout_spikes': late_silence.sum().item() == 0,
             'washout_voltage': tail_voltage < 1e-4,
             'bounded_rate': peak_rate <= 250.}
    simultaneous_rates = [steady_groups[f'steering_{side}'] / steady_seconds for side in ('left', 'right')]
    measured_rates = (aligned_steering_rates(steering_events, endpoint_delays)
                      if endpoint_delays is not None else simultaneous_rates)
    return {'case': dict(case), 'external_drive_sha256': drive_hash.hexdigest(), 'pulse_counts': grouped(pulse_counts),
            'pulse_per_neuron_counts': pulse_counts.tolist(),
            'steady_counts': steady_groups, 'steady_per_neuron_counts': steady.tolist(),
            'steady_seconds': steady_seconds,
            'steering_rates_hz': measured_rates,
            'simultaneous_steering_rates_hz': simultaneous_rates,
            'steering_window_delays_substeps': endpoint_delays,
            'silence_counts': grouped(silence), 'late_silence_counts': grouped(late_silence),
            'max_voltage': peak_voltage, 'final_max_abs_voltage': tail_voltage,
            'max_steady_rate_hz': peak_rate, 'gates': gates, 'passed': all(gates.values())}


def _new_json(path: Path, payload: dict) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sensory_counts_match(topology, intact: dict, measured: dict, *,
                         allow_sensory_edge_receivers: bool = False) -> bool:
    """Compare every ORN, exempting only sourced sensory-edge receivers on all-cut."""
    sensory = set(topology.indices('sensory_left') + topology.indices('sensory_right'))
    exempt = {topology.index(edge.post) for edge in topology.edges
              if topology.index(edge.pre) in sensory and topology.index(edge.post) in sensory
              and edge.pre != edge.post} if allow_sensory_edge_receivers else set()
    return all(intact['pulse_per_neuron_counts'][i] == measured['pulse_per_neuron_counts'][i]
               for i in sensory - exempt)


def run_recovery(output_dir: Path, *, wide_encoder: bool = False, pn_filter: bool = False, phase_rate: bool = False,
                 pn_charge: bool = False, latency_aligned: bool = False) -> dict:
    """Execute only this fixed protocol and preserve every earlier artifact."""
    output_dir = Path(output_dir).resolve()
    ensure_new_run_directory(output_dir)
    previous = {str(path.resolve()): sha256(path) for path in (PROJECT_ROOT / 'experiments').rglob('*')
                if path.is_file() and not path.resolve().is_relative_to(output_dir)}
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = make_protocol(wide_encoder=wide_encoder, pn_filter=pn_filter, phase_rate=phase_rate,
                             pn_charge=pn_charge, latency_aligned=latency_aligned)
    endpoint_delays = protocol.get('measurement', {}).get('steering_delays_substeps')
    _new_json(output_dir / 'protocol.json', protocol)
    _new_json(output_dir / 'preservation_before.json', previous)
    graph = load_banc_subset(connectome_gain=1.)
    candidates = []
    causal = []
    steps = 0
    journal = output_dir / 'journal.jsonl'
    with journal.open('x', encoding='utf-8', newline='\n') as log:
        for config_record in protocol['candidate_configs']:
            config = BancDynamicsConfig(**config_record)
            budget = config.synaptic_budget
            brain = BancNetwork(graph, dynamics_config=config)
            rows = []
            for case in protocol['cases']:
                measured = measure_pulse(brain, case, endpoint_delays=endpoint_delays)
                rows.append(measured)
                steps += 1500
                log.write(json.dumps({'budget': budget, **measured}, allow_nan=False) + '\n')
                log.flush()
            contrasts = {}
            simultaneous_contrasts = {}
            for center in CENTERS:
                rates = {row['case']['pattern']: row['steering_rates_hz'] for row in rows if row['case']['center'] == center}
                contrasts[str(center)] = contrast_gate(rates['equal'], rates['right_stronger'], rates['left_stronger'])
                raw_rates = {row['case']['pattern']: row['simultaneous_steering_rates_hz'] for row in rows if row['case']['center'] == center}
                simultaneous_contrasts[str(center)] = contrast_gate(raw_rates['equal'], raw_rates['right_stronger'], raw_rates['left_stronger'])
            passed = all(row['passed'] for row in rows) and all(row['passed'] for row in contrasts.values())
            candidates.append({'config': asdict(config), 'cases': rows,
                               'contrasts': contrasts, 'simultaneous_contrasts': simultaneous_contrasts, 'passed': passed})
            print(f'budget={budget:g}: pulse gates={all(row["passed"] for row in rows)}, contrast gates={all(row["passed"] for row in contrasts.values())}', flush=True)
        selected = next((row for row in candidates if row['passed']), None)
        if selected is not None:
            config = BancDynamicsConfig(**selected['config'])
            case = next(case for case in protocol['cases'] if case['pattern'] == 'equal' and case['center'] == .1)
            intact = next(row for row in selected['cases'] if row['case'] == case)
            for condition in protocol['conditional_causal_cases']['conditions']:
                brain = BancNetwork(graph, dynamics_config=config)
                weights = brain.topology.weights
                if condition == 'synapses_off':
                    weights.zero_()
                elif condition == 'downstream_synapses_off':
                    # Retain the sourced ORN->ORN edge while cutting every
                    # route from sensory cells into central/endpoint groups.
                    downstream = [i for i, group in enumerate(brain.topology.groups) if not group.startswith('sensory_')]
                    weights[downstream, :] = 0
                else:
                    side = condition.split('_')[0]
                    weights[list(brain.topology.indices(f'steering_{side}')), :] = 0
                # Interventions follow normalization and never redistribute removed drive.
                kept = tuple(edge for edge in brain.topology.edges
                             if weights[brain.topology.index(edge.post), brain.topology.index(edge.pre)] != 0)
                brain.topology = replace(brain.topology, edges=kept, _weights=weights)
                brain.population = brain.population_for_weights(weights)
                measured = measure_pulse(brain, case, endpoint_delays=endpoint_delays)
                steps += 1500
                counts = measured['pulse_counts']
                sensory_unchanged = sensory_counts_match(graph, intact, measured)
                target_groups = ([group for group in counts if not group.startswith('sensory_')]
                                 if condition in ('synapses_off', 'downstream_synapses_off') else [f'steering_{side}', f'motor_{side}'])
                external_unchanged = measured['external_drive_sha256'] == intact['external_drive_sha256']
                sensory_comparison_passed = sensory_counts_match(
                    graph, intact, measured,
                    allow_sensory_edge_receivers=(phase_rate or pn_charge or latency_aligned) and condition == 'synapses_off')
                stable = all(value for gate, value in measured['gates'].items() if gate != 'propagation')
                passed = (external_unchanged and stable and sensory_comparison_passed
                          and all(counts[group] == 0 for group in target_groups))
                record = {'condition': condition, 'sensory_unchanged': sensory_unchanged,
                          'per_neuron_sensory_comparison_passed': sensory_comparison_passed,
                          'external_drive_unchanged': external_unchanged,
                          'target_groups': target_groups, 'passed': passed, 'measurement': measured}
                causal.append(record)
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
    preserved = all(Path(path).is_file() and sha256(Path(path)) == digest for path, digest in previous.items())
    code_unchanged = all(sha256(PROJECT_ROOT / path) == digest for path, digest in protocol['source_code_sha256'].items())
    causal_passed = len(causal) == len(protocol['conditional_causal_cases']['conditions']) and all(row['passed'] for row in causal)
    result = {'protocol_sha256': sha256(output_dir / 'protocol.json'),
              'subset_sha256': BANC_SUBSET_SHA256, 'transform_version': BANC_TRANSFORM_VERSION,
              'candidates': candidates, 'selected_config': selected['config'] if selected is not None else None,
              'causal_controls': causal, 'dynamics_gate_passed': selected is not None,
              'causal_gate_passed': causal_passed, 'neural_substeps': steps,
              'optimizer_steps': 0, 'learning_rate': None, 'checkpoint': None,
              'previous_files_preserved': preserved, 'previous_file_count': len(previous),
              'source_code_unchanged': code_unchanged,
              'passed': selected is not None and causal_passed and preserved and code_unchanged,
              'claim': 'Dynamics and causal endpoint interface only; no training or navigation-improvement claim.'}
    _new_json(output_dir / 'results.json', result)
    with (output_dir / 'summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('budget', 'case', 'left_dna02_hz', 'right_dna02_hz', 'max_cell_hz', 'tail_voltage', 'passed'))
        for candidate in candidates:
            for row in candidate['cases']:
                writer.writerow((candidate['config']['synaptic_budget'], row['case']['name'],
                                 *row['steering_rates_hz'], row['max_steady_rate_hz'],
                                 row['final_max_abs_voltage'], row['passed']))
    print(f'closed: passed={result["passed"]}, neural_substeps={steps}, optimizer_steps=0', flush=True)
    return result


def main() -> None:
    """Run the predeclared recovery in an unused output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'experiments/fb09-dynamics-recovery')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--wide-encoder', action='store_true', help='run only the separately predeclared wide-range candidate')
    modes.add_argument('--pn-filter', action='store_true', help='run only the predeclared 5 ms PN-filter mechanism test')
    modes.add_argument('--phase-rate', action='store_true', help='run only the predeclared phase-rate representation candidate')
    modes.add_argument('--pn-charge', action='store_true', help='run only the approved charge-preserving PN candidate')
    modes.add_argument('--latency-aligned', action='store_true', help='run the approved delay-aligned sustained-rate protocol')
    args = parser.parse_args()
    result = run_recovery(args.output_dir, wide_encoder=args.wide_encoder, pn_filter=args.pn_filter,
                          phase_rate=args.phase_rate, pn_charge=args.pn_charge, latency_aligned=args.latency_aligned)
    raise SystemExit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
