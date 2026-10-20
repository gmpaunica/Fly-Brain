"""Detached single-child worker with append-only monitoring and fallback sealing."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib
import numpy as np
from flybrain.escape_diagnostic_journal import HeartbeatJournal
from .contract import sha

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'experiments/escape-diagnostic-final-five-2026-10-07'
ART=ROOT/'artifacts/escape-diagnostic-final-five-2026-10-07'


def reservation_upper_bound(directory: Path) -> int:
    """Read completed reservation lines and conservatively include partial tail."""
    manifest=directory/'progress.json'
    if not manifest.exists(): return 0
    record=json.loads(manifest.read_text()); limit=int(record['limit'])
    journal=directory/record['journal']
    if not journal.exists(): return 0
    lines=journal.read_bytes().splitlines(); previous=0
    for k,line in enumerate(lines):
        try:
            r=json.loads(line)
            if r['limit']!=limit or r['reserved_steps']!=previous+40: return limit
            previous=int(r['reserved_steps'])
        except (ValueError,KeyError,TypeError):
            return min(limit,previous+40) if k==len(lines)-1 else limit
    return min(limit,previous)


def seal_missing_receipt(directory: Path, reason: str) -> None:
    """Observe terminal child once; retain all partial files and never resume."""
    if not directory.exists() or (directory/'receipt.json').exists(): return
    steps=reservation_upper_bound(directory); completed=0; intervals=0; chunks=0
    for p in directory.glob('[0-9][0-9]-*/*.npz'):
        try:
            with np.load(p,allow_pickle=False) as a:
                valid=a['actual_spikes'].shape==a['reference_spikes'].shape==(20,589) and all(np.isfinite(a[k]).all() for k in a.files)
                chunks+=int(valid)
        except (OSError,ValueError,KeyError): pass
    for d in directory.glob('[0-9][0-9]-*'):
        p=d/'intervals.jsonl.gz'
        if not p.exists(): continue
        decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
        lines=decoder.decompress(p.read_bytes()).splitlines(); valid=0
        for line in lines:
            try: json.loads(line); valid+=1
            except json.JSONDecodeError: break
        intervals+=valid; completed+=int(valid==75 and decoder.eof)
    outputs={p.relative_to(directory).as_posix():sha(p) for p in directory.rglob('*') if p.is_file()}
    receipt=dict(status='interrupted',error=reason,completed_cases=completed,recorded_intervals=intervals,
        complete_neural_chunks=chunks,reserved_kernel_steps=steps,recorded_kernel_steps_lower_bound=chunks*40,
        simulated_seconds_upper_bound=steps/2000,retry_authorized=False,terminal_receipt_origin='append-only detached supervisor',outputs=outputs)
    with (directory/'receipt.json').open('x',encoding='utf-8') as f: json.dump(receipt,f,indent=2)
    seal=dict(directory=str(directory.resolve()),receipt_sha256=sha(directory/'receipt.json'),status='interrupted')
    with (directory.parent/'escape-diagnostic-final-seal.json').open('x',encoding='utf-8') as f: json.dump(seal,f,indent=2)


def supervise(command: list[str], directory: Path, log: Path, heartbeat: Path) -> int:
    """Run one child; append heartbeat and preserve existing terminal receipt."""
    journal=HeartbeatJournal(heartbeat.with_suffix(''))
    with log.open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                journal.observe(dict(status='running',supervisor_pid=os.getpid(),child_pid=child.pid,
                                     updated_unix_seconds=time.time(),retry_authorized=False))
                time.sleep(1.)
            code=child.returncode
        except BaseException as error:
            if child.poll() is None:
                child.terminate();child.wait()
            seal_missing_receipt(directory,f'supervision failed: {error}')
            journal.stream.close()
            raise
    seal_missing_receipt(directory,f'child exit {code} without normal receipt')
    journal.finish(dict(status='exited',exit_code=code,supervisor_pid=os.getpid(),child_pid=child.pid,
                        updated_unix_seconds=time.time(),retry_authorized=False))
    return code


if __name__=='__main__':
    ART.mkdir(parents=True,exist_ok=True)
    raise SystemExit(supervise([sys.executable,'-m','flybrain.escape_diagnostic_final.runner'],RUN,ART/'scientific-run.log',ART/'worker.json'))
