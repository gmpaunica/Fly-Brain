"""Detached process supervisor: durable heartbeat and terminal fallback seal."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib
from .contract import sha

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'experiments/escape-diagnostic-remaining-2026-10-07'
ART=ROOT/'artifacts/escape-diagnostic-remaining-2026-10-07'


def atomic(path: Path, value: dict) -> None:
    """Replace only mutable worker heartbeat; scientific chunks stay immutable."""
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2),encoding='utf-8'); os.replace(temp,path)


def seal_missing_receipt(directory: Path, reason: str) -> None:
    """Observe process termination once; never resume or repeat any simulation."""
    if not directory.exists() or (directory/'receipt.json').exists(): return
    progress=directory/'progress.json'
    steps=json.loads(progress.read_text())['reserved_steps'] if progress.exists() else 0
    completed=0; intervals=0; chunks=len(list(directory.glob('[0-9][0-9]-*/*.npz')))
    for d in directory.glob('[0-9][0-9]-*'):
        p=d/'intervals.jsonl.gz'
        if not p.exists(): continue
        decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
        lines=decoder.decompress(p.read_bytes()).splitlines()
        valid=0
        for line in lines:
            try: json.loads(line); valid+=1
            except json.JSONDecodeError: break
        intervals+=valid
        if valid==75 and decoder.eof: completed+=1
    outputs={p.relative_to(directory).as_posix():sha(p) for p in directory.rglob('*') if p.is_file()}
    receipt=dict(status='interrupted',error=reason,completed_cases=completed,recorded_intervals=intervals,
        complete_neural_chunks=chunks,reserved_kernel_steps=steps,recorded_kernel_steps_lower_bound=chunks*40,
        simulated_seconds_upper_bound=steps/2000,retry_authorized=False,terminal_receipt_origin='detached supervisor',outputs=outputs)
    with (directory/'receipt.json').open('x',encoding='utf-8') as f: json.dump(receipt,f,indent=2)
    seal=dict(directory=str(directory.resolve()),receipt_sha256=sha(directory/'receipt.json'),status='interrupted')
    with (directory.parent/'escape-diagnostic-remaining-seal.json').open('x',encoding='utf-8') as f: json.dump(seal,f,indent=2)


def supervise(command: list[str], directory: Path, log: Path, heartbeat: Path) -> int:
    """Own one child, record terminal state, and never automatically retry it."""
    with log.open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        while child.poll() is None:
            atomic(heartbeat,dict(status='running',supervisor_pid=os.getpid(),child_pid=child.pid,
                                  updated_unix_seconds=time.time(),retry_authorized=False))
            time.sleep(1.)
        code=child.returncode
    seal_missing_receipt(directory,f'child terminated with exit {code} without normal receipt')
    atomic(heartbeat,dict(status='exited',exit_code=code,supervisor_pid=os.getpid(),child_pid=child.pid,
                          updated_unix_seconds=time.time(),retry_authorized=False))
    return code


if __name__=='__main__':
    ART.mkdir(parents=True,exist_ok=True)
    raise SystemExit(supervise([sys.executable,'-m','flybrain.escape_diagnostic_remaining.runner'],
                              RUN,ART/'scientific-run.log',ART/'worker.json'))
