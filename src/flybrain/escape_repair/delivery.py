"""Repaired app activation binds the audited run, replay and engineering proof."""
import json
from pathlib import Path
from flybrain.escape_diagnostic_recovery.runner import verify_hashes
from flybrain.flight3d.anatomy import digest
from flybrain.escape_recovery.config import CandidateConfig
from .session import source_hashes

ART = Path('artifacts/escape-repair-2026-10-08')


def verified_candidate(art: Path = ART) -> CandidateConfig | None:
    """Fail closed if no complete, source-bound delivery exists."""
    path = art / 'delivery.json'
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    if r['status'] != 'passed':
        return None
    verify_hashes(r['proofs'])
    if r['repair_sources'] != source_hashes():
        raise ValueError('repaired application source mismatch')
    verify_hashes(r['repair_sources'])
    for name in (r['audit'], r['replay'], r['receipt']):
        if r['proofs'].get(name) != digest(Path(name)):
            raise ValueError('unbound delivery evidence')
    receipt_path = Path(r['receipt'])
    receipt = json.loads(receipt_path.read_text())
    result_path = receipt_path.parent / 'results.json'
    result = json.loads(result_path.read_text())
    if (digest(receipt_path) != r['receipt_sha256'] or receipt['status'] != 'passed'
            or receipt['outputs'].get('results.json') != digest(result_path)
            or result['status'] != 'passed' or result['cases'] != 96
            or result['intact_successes'] != 48 or result['silent_triggers'] != 0):
        raise ValueError('physical delivery receipt mismatch')
    audit = json.loads(Path(r['audit']).read_text())
    replay = json.loads(Path(r['replay']).read_text())
    if not (audit['status'] == 'audit-passed' and audit['experiment_status'] == 'passed'
            and audit['intact_successes'] == 48 and audit['silent_triggers'] == 0
            and replay['status'] == 'exact-replay-passed' and replay['cases'] == 4
            and audit['receipt_sha256'] == replay['receipt_sha256'] == r['receipt_sha256']):
        raise ValueError('repair delivery gates not satisfied')
    return CandidateConfig.from_record(r['candidate'])
