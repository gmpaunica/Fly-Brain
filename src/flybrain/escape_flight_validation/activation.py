"""Separate prototype verifier; historical directional gate stays failed."""
import json
from pathlib import Path
from flybrain.flight3d.anatomy import digest
from flybrain.escape_diagnostic_recovery.runner import write
from .acceptance import ART,LABEL,verify_exception
from .runner import RUN
from .audit import audit

MARK=ART/'activation.json'


def verify_delivery_evidence(directory:Path,art:Path) -> None:
    """Bind exact replay and separate saved visual review to this physical run."""
    receipt_hash=digest(directory/'receipt.json')
    replay=json.loads((art/'replay.json').read_text());visual=json.loads((art/'visual-review.json').read_text())
    if (replay['status']!='exact-replay-passed' or replay['cases']!=48
        or replay['replay_neural_steps']!=576000 or replay['new_scientific_cases']!=0
        or replay['receipt_sha256']!=receipt_hash):raise ValueError('prototype replay evidence mismatch')
    if visual['saved_review']!='reviewed' or visual['receipt_sha256']!=receipt_hash:
        raise ValueError('prototype visual review evidence mismatch')
    for name,h in visual['figures'].items():
        if digest(Path(name))!=h:raise ValueError('saved visual hash mismatch')


def verified_candidate(directory:Path=RUN,mark:Path=MARK):
    """Refuse absent, failing, forged or source-mismatched physical activation."""
    if not mark.exists():return None
    activation=json.loads(mark.read_text());config=json.loads((directory/'config.json').read_text())
    candidate=verify_exception(config['exception'])
    if activation['label']!=LABEL or activation['candidate_identity']!=candidate.identity:
        raise ValueError('prototype activation identity mismatch')
    paths={'physical_receipt_sha256':directory/'receipt.json','audit_sha256':mark.parent/'audit.json',
           'replay_sha256':mark.parent/'replay.json','visual_review_sha256':mark.parent/'visual-review.json',
           'prototype_acceptance_sha256':mark.parent/'prototype-acceptance.json'}
    if any(activation[k]!=digest(p) for k,p in paths.items()):raise ValueError('prototype activation evidence mismatch')
    physical=json.loads((directory/'results.json').read_text());saved_audit=json.loads((mark.parent/'audit.json').read_text())
    if physical['status']!='passed' or saved_audit['status']!='audit-passed' or saved_audit['experiment_status']!='passed':
        raise ValueError('physical activation gate failed')
    verify_delivery_evidence(directory,mark.parent)
    final=json.loads((mark.parent/'prototype-acceptance.json').read_text())
    if (final['exception_sha256']!=config['exception_sha256'] or final['candidate_identity']!=candidate.identity
        or final['physical_receipt_sha256']!=digest(directory/'receipt.json')
        or final['directional_receipt_sha256']!=config['exception']['directional_receipt_sha256']):
        raise ValueError('prototype acceptance binding mismatch')
    # Reconcile seal and every saved output on each new app launch.
    current=audit(directory)
    if current!=saved_audit:raise ValueError('current physical audit mismatch')
    return candidate


def activate(directory:Path=RUN,art:Path=ART) -> None:
    """Explicit user authorization is contingent on all recorded physical gates."""
    checked=audit(directory)
    if checked['status']!='audit-passed' or checked['experiment_status']!='passed':raise ValueError('physical gate failed; activation disabled')
    config=json.loads((directory/'config.json').read_text());candidate=verify_exception(config['exception'])
    verify_delivery_evidence(directory,art)
    write(art/'prototype-acceptance.json',dict(schema='escape-prototype-acceptance-v1',label=LABEL,
        candidate_identity=candidate.identity,exception_sha256=config['exception_sha256'],
        directional_receipt_sha256=config['exception']['directional_receipt_sha256'],
        physical_receipt_sha256=digest(directory/'receipt.json'),known_failures=[1100018,1100021],
        original_directional_status='closed-direction-failed',directional_successes=94,learning=False))
    write(art/'activation.json',dict(label=LABEL,candidate_identity=candidate.identity,
        physical_receipt_sha256=digest(directory/'receipt.json'),audit_sha256=digest(art/'audit.json'),
        replay_sha256=digest(art/'replay.json'),visual_review_sha256=digest(art/'visual-review.json'),
        prototype_acceptance_sha256=digest(art/'prototype-acceptance.json')))
    verified_candidate(directory,art/'activation.json')
