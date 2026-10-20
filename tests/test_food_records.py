"""Offline audit of completed validation and the interrupted final attempt."""

import hashlib
import json
from pathlib import Path

from flybrain.training.food_preflight import PROJECT_ROOT


def test_validation_is_complete_paired_and_final_remains_unclaimed():
    directory = PROJECT_ROOT / 'experiments/food-stop-fresh-2026-09-28'
    protocol_bytes = (directory / 'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    validation = json.loads((directory / 'validation.json').read_text())
    interruption = json.loads((directory / 'interruption.json').read_text())
    expected = protocol['cases']['validation']
    assert interruption['protocol_sha256'] == hashlib.sha256(protocol_bytes).hexdigest()
    for name, rows in validation.items():
        assert len(rows) == 16
        assert [(r['case_id'], r['seed'], r['x'], r['y'], r['heading']) for r in rows] == [
            (c['case_id'], c['seed'], c['x'], c['y'], c['heading']) for c in expected]
        assert all(r['world_ticks'] == 1000 and r['neural_steps'] == 20000 for r in rows)
        assert sum(r['success'] for r in rows) == interruption['validation_successes'][name]
    assert interruption['validation_successes'] == {
        'trained': 16, 'untrained': 0, 'fixed': 0}
    assert interruption['scientific_acceptance'] is False
    assert not (directory / 'final.json').exists()
    assert not (directory / 'results.json').exists()


def test_predeclared_fit_and_fresh_sources_remain_unchanged():
    for directory in ('food-stop-fit-2026-09-28', 'food-stop-fresh-2026-09-28'):
        protocol = json.loads((PROJECT_ROOT / 'experiments' / directory / 'protocol.json').read_text())
        for name, digest in protocol['source_sha256'].items():
            assert hashlib.sha256((PROJECT_ROOT / Path(name)).read_bytes()).hexdigest() == digest
