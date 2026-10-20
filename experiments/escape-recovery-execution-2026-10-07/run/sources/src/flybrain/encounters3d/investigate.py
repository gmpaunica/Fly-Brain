"""One locked 120-presentation diagnostic, without fitting or motor decoding."""
import argparse
import csv
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import numpy as np
import torch
from flybrain.flight3d.anatomy import CACHE, digest, load_manifest
from flybrain.flight3d.contracts import FlightObservation
from .diagnostics import DiagnosticEscapeBrain, retained_paths, stage_indices
from .scene import SceneSpec, encounter_eyes


def cases() -> list[dict]:
    """24 predeclared development cases, offset from the historical azimuth grid."""
    return [dict(label=f'{kind}-{sector}', kind=kind, sector=sector,
                 azimuth_degrees=22.5+45*sector, seed=990000+sector)
            for kind in ('far', 'near', 'sequence') for sector in range(8)]


def interventions() -> tuple[str, ...]:
    return ('intact', 'visual-silent', 'DNp03-silent', 'DNp26-silent', 'both-DN-silent')


def presentation_images(case: dict) -> list[np.ndarray]:
    """Evaluation geometry produces RGB only; occlusion is an opaque eye shutter.

    The shutter covers both images, using the same sky RGB. No visibility label,
    coordinate, direction, distance or precomputed steering crosses into inference.
    """
    angle = math.radians(case['azimuth_degrees'])
    images = []
    for interval in range(50):
        radius = .17 if case['kind'] == 'far' else .09
        if case['kind'] == 'sequence':
            radius = .17 if interval < 10 else .17-.08*(interval-9)/10 if interval < 20 else .09
        scene = SceneSpec(case['seed'], 'predator',
                         (.3+radius*math.sin(angle), .3+radius*math.cos(angle), .1))
        eyes = encounter_eyes(np.array((.3, .3, .1)), np.zeros(3), scene)
        if case['kind'] == 'sequence' and 20 <= interval < 30:
            eyes = np.empty_like(eyes); eyes[:] = (.035, .055, .09)
        images.append(eyes)
    return images


def _silenced(brain, intervention: str) -> set[str]:
    kinds = {'DNp03', 'DNp26'} if intervention == 'both-DN-silent' else {intervention.split('-')[0]}
    return {n['id'] for n in brain.nodes if n['metadata'].get('cell_type') in kinds or
            (intervention == 'visual-silent' and 'vision' in n['roles'])}


def _direction_summary(rows: list[dict], groups: dict) -> list[dict]:
    """Descriptive signal collisions, never a fitted decoder or success gate."""
    summary = []
    for kind in ('far', 'near', 'sequence'):
        for intervention in interventions():
            cohort = [r for r in rows if r['kind'] == kind and r['intervention'] == intervention]
            for name, indices in groups.items():
                spikes = [tuple(r['counts'][i] for i in indices) for r in cohort]
                voltage = [tuple(round(r['peak_voltage'][i], 12) for i in indices) for r in cohort]
                timing = [tuple(tuple(interval[i] for i in indices) for interval in r['interval_counts']) for r in cohort]
                summary.append(dict(kind=kind, intervention=intervention, stage=name,
                    directions=len(cohort), distinct_spike_count_vectors=len(set(spikes)),
                    distinct_temporal_spike_vectors=len(set(timing)),
                    distinct_peak_voltage_vectors=len(set(voltage)),
                    nonzero_spike_directions=sum(any(v) for v in spikes),
                    minimum_peak_voltage_linf_difference=min(
                        max((abs(x-y) for x, y in zip(a, b)), default=0.)
                        for k, a in enumerate(voltage) for b in voltage[k+1:])))
    return summary


def investigate(directory: Path) -> dict:
    """Lock before onset, stream all intervals, and refuse any existing directory."""
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    brain = DiagnosticEscapeBrain(load_manifest())
    groups = stage_indices(brain.manifest)
    measured = sorted({i for indices in groups.values() for i in indices})
    inputs = set(brain.roles['vision'])
    topology = {}
    for name, indices in groups.items():
        starts = {brain.ids[i] for i in (brain.roles['vision'] if name == 'Visual neurons' else indices)}
        reachable = retained_paths(brain.manifest, starts)
        topology[name] = dict(ids=[brain.ids[i] for i in indices],
            reachable_wings=sorted(reachable & {brain.ids[i] for i in groups['Wing-control neurons']}),
            direct_visual_edges=[e for e in brain.manifest['edges']
                if e['pre'] in {brain.ids[i] for i in inputs} and e['post'] in starts and e['pre'] != e['post']],
            incoming_nonself_edges=sum(e['post'] in starts and e['pre'] != e['post'] for e in brain.manifest['edges']))
    source_files = [Path(__file__), *[Path(__file__).with_name(n) for n in
                   ('diagnostics.py', 'neural.py', 'scene.py')],
                   Path('src/flybrain/flight3d/neural.py'), Path('src/flybrain/flight3d/sensors.py')]
    config = dict(schema='brain-to-wing-diagnostic-v1', date='2026-10-05',
        hypothesis='Current sensory assignment and assumed dynamics fail to deliver a usable directional signal to the wings.',
        budget_presentations=120, seconds_per_presentation=1., neural_steps_per_presentation=1000,
        stop_condition='Stop at 120 presentations regardless of outcome. No sweep, training, retry or decoder.',
        case_status='New diagnostic development cases; historical inspected placements remain historical.',
        cases=cases(), interventions=interventions(), seed_base=990000,
        distances_m=[.17, .09], angle_offset_degrees=22.5,
        sequence='0-.20 far; .20-.40 linear growth; .40-.60 opaque binocular shutter; .60-1 near reappearance',
        pose={'position_m': [.3, .3, .1], 'orientation_radians': [0., 0., 0.]},
        objective='Find retained-path absence, subthreshold electrical drive and directional signal collisions.',
        control_dt_seconds=.02, neural_dt_seconds=.001, leak_seconds=.1,
        threshold=1., incoming_weight_sum=.9, reset='subtractive', delay_seconds=.001,
        input_rule='unchanged cyan encoder and ID-ranked fields; deterministic 0..240 Hz phase input',
        current_units='normalized voltage increment per 1 ms; sums over interval, not amperes',
        device='cpu', dtype='float64', platform=platform.platform(), processor=platform.processor(),
        software={n: importlib.metadata.version(n) for n in ('numpy', 'torch', 'panda3d', 'pygame')},
        python=platform.python_version(), torch_threads=1, commit=None,
        optimizer_updates=0, learning_rate=None, checkpoint=None,
        network_cells=len(brain.ids), retained_edges=len(brain.edge_order),
        circuit_sha256=digest(CACHE/'circuit.json'),
        source_sha256={p.as_posix(): digest(p) for p in source_files},
        neuron_ids=brain.ids, stage_indices=groups, topology=topology,
        trace_schema='JSONL all neurons in manifest order; CSV per-stage-cell per-interval; sparse source-edge totals in manifest order')
    (directory/'config.json').write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    rows = []
    fields = ['presentation', 'label', 'intervention', 'interval', 'neuron_id',
              'end_seconds', 'current_sum', 'current_peak', 'discarded_current',
              'peak_voltage', 'peak_phase', 'spike_count', 'first_spike_seconds']
    with gzip.open(directory/'traces.jsonl.gz', 'wt', encoding='utf-8') as trace, \
         gzip.open(directory/'traces.csv.gz', 'wt', encoding='utf-8', newline='') as table:
        writer = csv.DictWriter(table, fieldnames=fields); writer.writeheader()
        for case in cases():
            images = presentation_images(case)
            image_hash = hashlib.sha256(np.stack(images).tobytes()).hexdigest()
            np.savez_compressed(directory/f"images-{case['label']}.npz", eyes=np.stack(images))
            for intervention in interventions():
                brain.reset(); brain.silenced = _silenced(brain, intervention)
                counts = np.zeros(len(brain.ids), dtype=np.int64)
                voltage = np.zeros(len(brain.ids)); currents = np.zeros(len(brain.ids))
                current_peak = np.zeros(len(brain.ids)); driven = set(); interval_counts = []
                for interval, eyes in enumerate(images):
                    observation = FlightObservation(0., 0., eyes, (0., 0., 0.), (0., 0., 9.81), False, False)
                    rates = brain.sensory_rates(observation)
                    driven.update(int(i) for i in np.flatnonzero(rates))
                    brain.advance_rates(rates)
                    record = brain.diagnostic
                    counts += record.spike_counts
                    voltage = np.maximum(voltage, record.peak_voltage)
                    currents += record.current_sum; current_peak = np.maximum(current_peak, record.current_peak)
                    interval_counts.append(record.spike_counts)
                    trace.write(json.dumps(dict(presentation=len(rows), label=case['label'],
                        intervention=intervention, interval=interval, **asdict(record),
                        input_rates_hz=rates.tolist(),
                        edge_transmissions=[(i, x) for i, x in enumerate(brain.edge_transmissions) if x]),
                        separators=(',', ':'))+'\n')
                    for i in measured:
                        writer.writerow(dict(presentation=len(rows), label=case['label'], intervention=intervention,
                            interval=interval, neuron_id=brain.ids[i], end_seconds=record.end_seconds,
                            current_sum=record.current_sum[i], current_peak=record.current_peak[i],
                            discarded_current=record.discarded_current[i], peak_voltage=record.peak_voltage[i],
                            peak_phase=record.peak_phase[i], spike_count=record.spike_counts[i],
                            first_spike_seconds=record.first_spike_seconds[i]))
                reached = retained_paths(brain.manifest, {brain.ids[i] for i in driven}, brain.silenced)
                stages = {}
                for name, indices in groups.items():
                    stage_counts = [int(counts[i]) for i in indices]
                    first = [record.episode_first_spike_seconds[i] for i in indices]
                    unreachable = [brain.ids[i] for i in indices if brain.ids[i] not in reached]
                    below = [brain.ids[i] for i in indices if brain.ids[i] in reached and counts[i] == 0]
                    stages[name] = dict(spikes=sum(stage_counts), cell_counts=stage_counts,
                        peak_voltage=max((float(voltage[i]) for i in indices), default=0.),
                        current_sum=sum(float(currents[i]) for i in indices),
                        current_peak=max((float(current_peak[i]) for i in indices), default=0.),
                        first_spike_seconds=min((x for x in first if x is not None), default=None),
                        no_retained_path_under_intervention=unreachable,
                        reachable_but_no_spikes=below,
                        silenced=[brain.ids[i] for i in indices if brain.ids[i] in brain.silenced])
                rows.append(dict(**case, intervention=intervention, eyes_sha256=image_hash,
                    driven_visual_ids=sorted(brain.ids[i] for i in driven), stages=stages,
                    counts=counts.tolist(), peak_voltage=voltage.tolist(), incoming_current_sum=currents.tolist(),
                    first_spike_seconds=list(record.episode_first_spike_seconds), interval_counts=interval_counts))
                print(f"{len(rows)}/120 {case['label']} {intervention} | wing spikes {stages['Wing-control neurons']['spikes']}", flush=True)
    direction = _direction_summary(rows, groups)
    # Per-trial failure categories retain intervention/structural qualifiers.
    for row in rows:
        wing = row['stages']['Wing-control neurons']
        cohort = next(d for d in direction if d['kind'] == row['kind'] and
                      d['intervention'] == row['intervention'] and d['stage'] == 'Wing-control neurons')
        row['failure_categories'] = dict(
            retained_path_absence=bool(wing['no_retained_path_under_intervention']),
            reachable_insufficient_drive=bool(wing['reachable_but_no_spikes']),
            non_distinguishing_wing_spikes=cohort['distinct_temporal_spike_vectors'] < 8,
            non_distinguishing_peak_voltage=cohort['distinct_peak_voltage_vectors'] < 8,
            interpretation='Intervention path absence is causal blocking, not a missing source connection; reachability is only structural.')
    result = dict(config='config.json', presentations=len(rows), completed_intervals=6000,
        neural_steps=120000, rows=rows, direction=direction, optimizer_updates=0,
        repair_promoted=False, no_behavior_claim=True)
    (directory/'results.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    keys = ['label', 'intervention', 'stage', 'spikes', 'peak_voltage', 'current_sum', 'current_peak', 'first_spike_seconds']
    with (directory/'responses.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in rows:
            for name, values in row['stages'].items():
                writer.writerow(dict(label=row['label'], intervention=row['intervention'], stage=name,
                                     **{k: values[k] for k in keys[3:]}))
    (directory/'receipt.json').write_text(json.dumps(dict(status='closed', presentations=120,
        neural_steps=120000, retry_authorized=False,
        sha256={p.name: digest(p) for p in directory.iterdir() if p.is_file()}), indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    result = investigate(parser.parse_args().directory)
    print(json.dumps(dict(presentations=result['presentations'], neural_steps=result['neural_steps'])))
