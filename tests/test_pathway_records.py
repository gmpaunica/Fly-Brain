"""Read the single closed diagnostic; never rerun it or historical experiments."""
import hashlib
import json
from pathlib import Path
import numpy as np

RUN = Path('experiments/brain-to-wing-diagnostic-2026-10-05')


def test_single_locked_budget_and_source_binding():
    config = json.loads((RUN/'config.json').read_text())
    result = json.loads((RUN/'results.json').read_text())
    receipt = json.loads((RUN/'receipt.json').read_text())
    assert receipt['status'] == 'closed' and receipt['presentations'] == 120
    assert result['completed_intervals'] == 6000 and result['neural_steps'] == 120000
    assert len(config['cases']) == 24 and len(config['interventions']) == 5
    assert config['optimizer_updates'] == 0 and config['checkpoint'] is None
    assert not result['repair_promoted'] and result['no_behavior_claim']
    for name, expected in config['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    for row in result['rows']:
        assert len(row['counts']) == 589 and len(row['interval_counts']) == 50
        np.testing.assert_array_equal(np.sum(row['interval_counts'], axis=0), row['counts'])
        if row['intervention'] == 'visual-silent': assert not any(row['counts'])
        for name in ('DNp03', 'DNp26'):
            if row['intervention'] in (name+'-silent', 'both-DN-silent'):
                assert row['stages'][name]['spikes'] == 0
    for case in config['cases']:
        cohort = [r for r in result['rows'] if r['label'] == case['label']]
        assert len(cohort) == 5 and len({r['eyes_sha256'] for r in cohort}) == 1


def test_historical_integrations_are_archived_byte_for_byte():
    base = Path('artifacts/pathway-diagnostics-2026-10-05')
    before = json.loads((base/'before.json').read_text())
    for name in ('session.py', 'scene.py', 'viewer.py'):
        path = 'src/flybrain/encounters3d/'+name
        assert hashlib.sha256((base/'original'/name).read_bytes()).hexdigest() == before[path]
