"""Narrow prospective prototype exception; historical gates remain untouched."""
import json
from pathlib import Path
from flybrain.flight3d.anatomy import CACHE, digest, load_manifest
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.escape_diagnostic_recovery.runner import verify_hashes

DIRECTIONAL = Path('experiments/escape-recovery-execution-2026-10-07/run')
ART = Path('artifacts/escape-flight-validation-2026-10-08')
IDENTITY = 'ea70e22d8879874f14df4299ba797170d7624e0b26c6e0baeddf720d3494dcab'
LABEL = 'prototype: 94/96 directional result accepted; physical flight validated'


def required_files() -> dict[str,str]:
    """Bind the failed original and its already verified motion-only evidence."""
    paths=[DIRECTIONAL/n for n in ('config.json','results.json','receipt.json','circuit.json')]
    paths += [DIRECTIONAL.parent/'escape-recovery-seal.json',
              Path('artifacts/escape-recovery-execution-2026-10-07/audit.json'),
              Path('artifacts/escape-recovery-execution-2026-10-07/analysis.json'),
              Path('experiments/turnflee-2026-10-06/receipt.json'),
              Path('experiments/turnflee-2026-10-06/results.json'),
              Path('experiments/turnflee-2026-10-06/calibration.json')]
    return {p.as_posix():digest(p) for p in paths}


def prospective_record() -> dict:
    """Create the user-authorized exception before any physical exposure."""
    return dict(schema='escape-prototype-exception-v1',authorized_date='2026-10-08',
        candidate_identity=IDENTITY,directional_directory=DIRECTIONAL.as_posix(),
        directional_receipt_sha256=digest(DIRECTIONAL/'receipt.json'),
        original_status='closed-direction-failed',directional_successes=94,directional_cases=96,
        failed_azimuth=86.25,azimuth_successes=4,azimuth_cases=6,original_required=5,
        known_failures=[1100018,1100021],scope='this frozen prototype only',
        directional_rerun=False,retuning=False,learning=False,
        motion_baseline='previously verified fixed-body directional evidence; no new closed-case baseline',
        physical_receipt_sha256=None,bound_files=required_files())


def verify_exception(record:dict) -> CandidateConfig:
    """Verify sources, complete archive, circuit, calibration and failed outcome."""
    expected=prospective_record()
    if any(record.get(k)!=v for k,v in expected.items() if k!='bound_files'):
        raise ValueError('prototype exception binding mismatch')
    bound=record.get('bound_files',{})
    if any(bound.get(k)!=v for k,v in expected['bound_files'].items()):
        raise ValueError('prototype evidence binding mismatch')
    verify_hashes(bound)
    config=json.loads((DIRECTIONAL/'config.json').read_text())
    candidate=CandidateConfig.from_record(config['candidate'])
    if candidate.identity!=IDENTITY or config['candidate_identity']!=IDENTITY:
        raise ValueError('frozen candidate identity mismatch')
    receipt=json.loads((DIRECTIONAL/'receipt.json').read_text())
    seal=json.loads((DIRECTIONAL.parent/'escape-recovery-seal.json').read_text())
    if seal['receipt_sha256']!=digest(DIRECTIONAL/'receipt.json') or receipt['candidate_identity']!=IDENTITY:
        raise ValueError('directional receipt mismatch')
    verify_hashes({str(DIRECTIONAL/n):h for n,h in receipt['outputs'].items()})
    for name,h in candidate.source_sha256:
        if digest(DIRECTIONAL/'sources'/name)!=h:raise ValueError('frozen source archive mismatch')
    inventory={p.relative_to(DIRECTIONAL/'sources').as_posix() for p in (DIRECTIONAL/'sources').rglob('*.py')}
    if inventory!={n for n,h in candidate.source_sha256 if n.endswith('.py')}:
        raise ValueError('frozen source inventory mismatch')
    if digest(DIRECTIONAL/'circuit.json')!=candidate.response.circuit_sha256 or digest(CACHE/'circuit.json')!=candidate.response.circuit_sha256:
        raise ValueError('circuit binding mismatch')
    manifest=load_manifest()
    if len(manifest['nodes'])!=589 or len(manifest['edges'])!=4139:
        raise ValueError('frozen anatomical size mismatch')
    result=json.loads((DIRECTIONAL/'results.json').read_text())
    rows=result['validation']
    if (result['status']!='closed-direction-failed' or receipt['status']!=result['status']
        or len(rows)!=96 or sum(r['success'] for r in rows)!=94
        or [r['seed'] for r in rows if not r['success']]!=[1100018,1100021]
        or sum(r['success'] for r in rows if r['azimuth']==86.25)!=4
        or result['closed_loop'] or result['reserved_kernel_steps']!=288000):
        raise ValueError('original directional result mismatch')
    evidence=json.loads(Path('artifacts/escape-recovery-execution-2026-10-07/audit.json').read_text())
    if evidence['status']!='audit-passed' or evidence['experiment_status']!=result['status']:
        raise ValueError('original directional audit mismatch')
    return candidate
