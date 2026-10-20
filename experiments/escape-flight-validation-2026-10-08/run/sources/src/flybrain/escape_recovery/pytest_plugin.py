"""Test-only output isolation; never change historical proof/source bytes."""
from pathlib import Path
import tempfile
import os
import hashlib
import pytest


@pytest.fixture(autouse=True)
def isolated_diagnostic_ledger(request,monkeypatch):
    """Original synthetic tests use a fresh per-test ledger instead of archive."""
    if request.node.path.name!='test_escape_diagnostic.py':
        yield
        return
    import flybrain.escape_diagnostic.recorder as recorder
    with tempfile.TemporaryDirectory(prefix='flybrain-test-ledger-') as folder:
        ledger=Path(folder)/'engineering.json'
        monkeypatch.setattr(recorder,'ENGINEERING',ledger)
        yield
        if ledger.exists() and os.environ.get('FLYBRAIN_TEST_LEDGERS'):
            target=Path(os.environ['FLYBRAIN_TEST_LEDGERS']);target.mkdir(parents=True,exist_ok=True)
            (target/(hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]+'.json')).write_bytes(ledger.read_bytes())
