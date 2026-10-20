"""Reconcile saved scientific traces without any brain/world stepping."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from flybrain.flight3d.anatomy import digest
from flybrain.retreat3d.neural import FrozenDecoder
from flybrain.retreat3d.experiment import score_windows, validation_gate


def audit(directory: Path) -> dict:
    """Validate counts, windows, source bindings and failed gate from disk only."""
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    result = json.loads((directory/'results.json').read_text(encoding='utf-8'))
    receipt = json.loads((directory/'receipt.json').read_text(encoding='utf-8'))
    calibration = json.loads((directory/'calibration.json').read_text(encoding='utf-8'))
    for name, expected in receipt['sha256'].items():
        assert digest(directory/name) == expected, name
    for name, expected in config['source_sha256'].items():
        assert digest(Path(name)) == expected, name
    from flybrain.flight3d.anatomy import load_manifest
    manifest = load_manifest()
    indices = [i for i, n in enumerate(manifest['nodes']) if n['id'] in config['wing_ids']]
    decoder = FrozenDecoder.load(directory/'templates.json', tuple(config['wing_ids']))
    records = calibration+result['validation']
    offset = 0; presented = Counter(); totals = Counter(); windows = []
    current = np.zeros(48, dtype=np.int64); pooled = [[] for _ in range(8)]
    with gzip.open(directory/'traces.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); case = records[offset//75]; interval = offset % 75
            assert row['interval'] == interval
            assert all(row[k] == case[k] for k in ('phase', 'azimuth', 'elevation', 'profile', 'seed'))
            assert abs(row['pathway']['start_seconds']-interval*.02) < 1e-12
            assert abs(row['pathway']['end_seconds']-(interval+1)*.02) < 1e-12
            assert row['actual_counts'] == row['pathway']['spike_counts']
            if interval < 25:
                assert row['actual_counts'] == row['reference_counts']
                totals['motion_baseline_intervals'] += 1
            if interval == 25:
                current[:] = 0
            actual = np.array(row['actual_counts'])[indices]
            reference = np.array(row['reference_counts'])[indices]
            totals[f"{row['phase']}_actual_wing_spikes"] += int(actual.sum())
            totals[f"{row['phase']}_reference_wing_spikes"] += int(reference.sum())
            current += actual-reference
            if row['wing_vector'] is not None:
                np.testing.assert_array_equal(current, row['wing_vector'])
                if interval >= 25:
                    windows.append(current.copy())
                current[:] = 0
            if interval == 74:
                np.testing.assert_array_equal(windows, case['windows'])
                assert len(windows) == 5
                if row['phase'] == 'calibration':
                    pooled[int(row['azimuth']/45)].extend(windows)
                else:
                    score = score_windows(windows, decoder, case['azimuth'])
                    assert score['success'] == case['success']
                    assert score['usable_windows'] == case['usable_windows']
                    totals['usable_validation_windows'] += score['usable_windows']
                    totals['nonzero_validation_windows'] += sum(np.linalg.norm(w) >= 1 for w in windows)
                windows = []; presented[row['phase']] += 1
            offset += 1
    pooled_values = np.array([np.mean(v, axis=0) for v in pooled])
    norms = np.linalg.norm(pooled_values, axis=1)
    expected_templates = np.divide(pooled_values, norms[:, None], out=np.zeros_like(pooled_values), where=norms[:, None]>0)
    np.testing.assert_allclose(expected_templates, decoder.templates, atol=1e-15, rtol=0)
    assert presented == {'calibration':48, 'validation':96}
    assert offset == 10800 and result['neural_steps'] == offset*40 == 432000
    assert result['simulated_seconds'] == offset*.02 == 216
    assert not validation_gate(result['validation']) and not result['validation_passed']
    assert result['validation_successes'] == 9
    assert result['closed_loop_cases'] == 0 and result['optimizer_updates'] == 0
    assert result['status'] == receipt['status'] == 'closed-validation-failed'
    with (directory/'traces.csv').open(newline='', encoding='utf-8') as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == offset
    report = dict(verified=True, cases=dict(presented), intervals=offset,
        simulated_seconds=216, actual_reference_neural_steps=432000,
        validation_successes=9, validation_groups=result['groups'], totals={k:int(v) for k,v in totals.items()},
        template_raw_norms=norms.tolist(), postprocessing_neural_steps=0,
        source_hashes_match=True, receipt_hashes_match=True, closed_loop_cases=0,
        retreat_delivered=False)
    (directory/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Archive source bytes named by the locked config, with their original paths.
    for name in config['source_sha256']:
        target = directory/'source'/name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
        assert digest(target) == config['source_sha256'][name]
    return report


if __name__ == '__main__':
    print(json.dumps(audit(Path(sys.argv[1])), indent=2))
