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
from flybrain.flight3d.anatomy import digest as sha

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'experiments/escape-recovery-2026-10-07'
ART=ROOT/'artifacts/escape-recovery-2026-10-07'


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
    steps=reservation_upper_bound(directory);completed=0;intervals=0;chunks=0
    records=directory/'traces.jsonl.gz'
    if records.exists():
        decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
        groups={}
        for line in decoder.decompress(records.read_bytes()).splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:break
            key=(r['phase'],r['seed'],r.get('state'),r.get('intervention'))
            groups[key]=groups.get(key,0)+1;intervals+=1
        completed=sum(n==(75 if key[0]=='validation' else 300) for key,n in groups.items())

    if not (directory/'results.json').exists():
        with (directory/'results.json').open('x') as f:json.dump(dict(status='interrupted',error=reason,validation=[],closed_loop=[],cases=completed,reserved_kernel_steps=steps,retry_authorized=False),f)
    outputs={p.relative_to(directory).as_posix():sha(p) for p in directory.rglob('*') if p.is_file()}
    receipt=dict(status='interrupted',error=reason,completed_cases=completed,recorded_intervals=intervals,
        complete_neural_chunks=chunks,reserved_kernel_steps=steps,recorded_kernel_steps_lower_bound=intervals*40,
        simulated_seconds_upper_bound=steps/2000,retry_authorized=False,terminal_receipt_origin='append-only detached supervisor',outputs=outputs)
    with (directory/'receipt.json').open('x',encoding='utf-8') as f: json.dump(receipt,f,indent=2)
    seal=dict(directory=str(directory.resolve()),receipt_sha256=sha(directory/'receipt.json'),status='interrupted')
    with (directory.parent/'escape-recovery-seal.json').open('x',encoding='utf-8') as f: json.dump(seal,f,indent=2)


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
    raise SystemExit(supervise([sys.executable,'-m','flybrain.escape_recovery.experiment'],RUN,ART/'experiment.log',ART/'worker.json'))
