"""Seal an externally terminated run without altering any saved source/chunk."""
import json
from pathlib import Path
import numpy as np
from flybrain.escape_diagnostic_recovery.audit import sha

ROOT=Path('experiments/escape-diagnostic-recovery-2026-10-07')


def main() -> None:
    """Fresh terminal receipt only; this is never a scientific resume."""
    if (ROOT/'receipt.json').exists(): raise FileExistsError('run already sealed')
    reserved=json.loads((ROOT/'progress.json').read_text())['reserved_steps']
    chunks=sorted(ROOT.glob('[0-9][0-9]-*/*.npz'))
    for p in chunks:
        with np.load(p,allow_pickle=False) as a:
            assert a['actual_spikes'].shape==a['reference_spikes'].shape==(20,589)
            assert all(np.isfinite(a[k]).all() for k in a.files)
    assert len(chunks)*40==reserved==22120
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file()}
    receipt=dict(status='interrupted',error='Execution process disappeared externally; tool session unknown and no Python process remained. No terminal exception or scientific mismatch logged.',
        completed_cases=7,recorded_intervals=552,complete_neural_chunks=553,reserved_kernel_steps=reserved,
        observed_kernel_steps=reserved,simulated_seconds=11.06,fully_logged_seconds=11.04,
        retry_authorized=False,terminal_receipt_origin='external saved-record finalizer after verified process termination',outputs=hashes)
    with (ROOT/'receipt.json').open('x',encoding='utf-8') as f: json.dump(receipt,f,indent=2)
    seal=dict(directory=str(ROOT.resolve()),receipt_sha256=sha(ROOT/'receipt.json'),status='interrupted',external_finalizer=True)
    with (ROOT.parent/'escape-diagnostic-recovery-seal.json').open('x',encoding='utf-8') as f: json.dump(seal,f,indent=2)
    print(json.dumps({k:v for k,v in receipt.items() if k!='outputs'},indent=2))


if __name__=='__main__': main()
