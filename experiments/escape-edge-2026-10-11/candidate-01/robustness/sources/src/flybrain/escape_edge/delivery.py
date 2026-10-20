"""Fail-closed activation bound to both measured batches and exact replay."""
import json
from pathlib import Path
from flybrain.escape_diagnostic_recovery.runner import verify_hashes
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.flight3d.anatomy import digest
from .session import source_hashes
from .validation import ART, gate


def verified_candidate(art: Path = ART) -> CandidateConfig | None:
    """Accept only complete, source-bound engineering and physical evidence."""
    path = art/'delivery.json'
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    if r['status']!='passed':
        return None
    verify_hashes(r['proofs'])
    verify_hashes(r['bound'])
    if r['edge_sources']!=source_hashes():
        raise ValueError('edge delivery source mismatch')
    receipts = {}
    for phase, directory in r['runs'].items():
        directory = Path(directory)
        receipt_path = directory/'receipt.json'
        receipts[phase] = digest(receipt_path)
        if r['proofs'].get(receipt_path.as_posix())!=receipts[phase]:
            raise ValueError('unbound physical receipt')
        receipt = json.loads(receipt_path.read_text())
        result = json.loads((directory/'results.json').read_text())
        config = json.loads((directory/'config.json').read_text())
        audit_path = art/f'{phase}-audit.json'
        audit = json.loads(audit_path.read_text())
        if (receipt['status']!='passed' or result['status']!='passed'
                or receipt['outputs'].get('results.json')!=digest(directory/'results.json')
                or receipt['outputs'].get('config.json')!=digest(directory/'config.json')
                or config['bound']!=r['bound'] or result['reserved_kernel_steps']!=1152000
                or result['optimizer_updates']!=0 or not gate(config['cases'],result['closed_loop'])
                or audit['status']!='audit-passed' or audit['experiment_status']!='passed'
                or audit['receipt_sha256']!=receipts[phase]
                or r['proofs'].get(audit_path.as_posix())!=digest(audit_path)):
            raise ValueError('edge physical delivery mismatch')
    if set(receipts)!={'development','robustness'}:
        raise ValueError('both physical batches required')
    replay_path = art/'replay.json'
    replay = json.loads(replay_path.read_text())
    if (replay['status']!='exact-replay-passed' or replay['cases']!=4 or replay['neural_steps']!=96000
            or replay['new_scientific_cases']!=0 or replay['receipts']!=receipts
            or r['proofs'].get(replay_path.as_posix())!=digest(replay_path)):
        raise ValueError('edge replay delivery mismatch')
    return CandidateConfig.from_record(r['candidate'])
