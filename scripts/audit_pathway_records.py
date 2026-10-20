"""Audit existing diagnostic records without another neural presentation."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(directory: Path) -> dict:
    """Reconcile locked budget, image/source binding and interval measurements."""
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    result = json.loads((directory/'results.json').read_text(encoding='utf-8'))
    receipt = json.loads((directory/'receipt.json').read_text(encoding='utf-8'))
    for name, expected in receipt['sha256'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected, name
    for name, expected in config['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    assert len(result['rows']) == receipt['presentations'] == config['budget_presentations'] == 120
    assert result['neural_steps'] == receipt['neural_steps'] == 120000
    assert result['optimizer_updates'] == 0 and not result['repair_promoted']
    n = config['network_cells']
    aggregates = [dict(counts=np.zeros(n, dtype=np.int64), current=np.zeros(n),
                      voltage=np.zeros(n), first=[None]*n, intervals=0) for _ in range(120)]
    json_count = 0
    with gzip.open(directory/'traces.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line); a = aggregates[record['presentation']]
            assert record['interval'] == a['intervals']
            assert record['start_seconds'] == record['interval'] * 20 * .001
            assert record['end_seconds'] == (record['interval']+1) * 20 * .001
            assert len(record['spike_counts']) == len(record['current_sum']) == n
            assert np.isclose(sum(record['current_sum']), sum(x[1] for x in record['edge_transmissions']), atol=1e-12)
            for i, time in enumerate(record['first_spike_seconds']):
                if time is not None:
                    assert record['start_seconds'] < time <= record['end_seconds']
                    assert abs(time*1000-round(time*1000)) < 1e-10
                    if a['first'][i] is None: a['first'][i] = time
            a['counts'] += record['spike_counts']; a['current'] += record['current_sum']
            a['voltage'] = np.maximum(a['voltage'], record['peak_voltage'])
            a['intervals'] += 1; json_count += 1
    for row, a in zip(result['rows'], aggregates):
        assert a['intervals'] == 50
        np.testing.assert_array_equal(a['counts'], row['counts'])
        np.testing.assert_array_equal(a['current'], row['incoming_current_sum'])
        np.testing.assert_array_equal(a['voltage'], row['peak_voltage'])
        assert a['first'] == row['first_spike_seconds']
    assert json_count == result['completed_intervals'] == 6000
    groups = config['stage_indices']
    measured = sorted({i for indices in groups.values() for i in indices})
    csv_count = 0
    with gzip.open(directory/'traces.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        for record in csv.DictReader(stream):
            csv_count += 1
    assert csv_count == 6000*len(measured)
    for case in config['cases']:
        cohort = [r for r in result['rows'] if r['label'] == case['label']]
        assert len(cohort) == 5 and len({r['eyes_sha256'] for r in cohort}) == 1
        with np.load(directory/f"images-{case['label']}.npz") as images:
            assert hashlib.sha256(images['eyes'].tobytes()).hexdigest() == cohort[0]['eyes_sha256']
    outcome = dict(presentations=120, neural_steps=120000, json_intervals=json_count,
                   csv_cell_intervals=csv_count, source_and_image_hashes_verified=True,
                   interval_aggregates_verified=True, no_scientific_rerun=True)
    print(json.dumps(outcome, indent=2))
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory)
