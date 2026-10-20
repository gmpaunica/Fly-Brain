"""Promotion requires passed behavioral record, audit and explicit user choice."""
import json
from pathlib import Path
from flybrain.flight3d.anatomy import digest
from .config import CandidateConfig

RUN=Path('experiments/escape-recovery-2026-10-07')
MARK=Path('artifacts/escape-recovery-2026-10-07/activation.json')


def verified_candidate(require_activation:bool=True) -> CandidateConfig|None:
    """Never substitute an unverified development decoder for the failed one."""
    if not (RUN/'receipt.json').exists():return None
    receipt=json.loads((RUN/'receipt.json').read_text())
    if receipt['status']!='passed':return None
    for name,h in receipt['outputs'].items():
        if digest(RUN/name)!=h:raise ValueError('recovery receipt mismatch')
    c=CandidateConfig.from_record(json.loads((RUN/'config.json').read_text())['candidate'])
    from flybrain.turnflee.protocol import validation_gate,closed_gate
    results=json.loads((RUN/'results.json').read_text())
    if not validation_gate(results['validation']) or not closed_gate(results['closed_loop']):raise ValueError('activation behavioral gate failed')
    if require_activation:
        if not MARK.exists():return None
        mark=json.loads(MARK.read_text())
        if mark['receipt_sha256']!=digest(RUN/'receipt.json') or mark['candidate_identity']!=c.identity:raise ValueError('activation receipt mismatch')
        audit_path=MARK.parent/'audit.json'
        if digest(audit_path)!=mark['audit_sha256']:raise ValueError('activation audit binding mismatch')
        verified=json.loads(audit_path.read_text())
        if verified['status']!='audit-passed' or verified['experiment_status']!='passed':raise ValueError('activation audit failed')
    return c
