"""Detached sequential worker with terminal sealing and no automatic retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from flybrain.escape_diagnostic_journal import HeartbeatJournal
from flybrain.escape_diagnostic_recovery.runner import write
from flybrain.flight3d.anatomy import ROOT, digest
from .validation import ART, BASE


def seal_missing(directory: Path, reason: str) -> None:
    """Record an interrupted child without rewriting any existing terminal file."""
    if not directory.exists() or (directory/'receipt.json').exists():
        return
    reserved = 0
    journal = directory/'progress.reservations.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                reserved = max(reserved,json.loads(line)['reserved_steps'])
            except (ValueError,KeyError):
                break
    if not (directory/'results.json').exists():
        write(directory/'results.json',dict(status='interrupted',error=reason,cases=0,
            closed_loop=[],reserved_kernel_steps=reserved,optimizer_updates=0))
    outputs = {p.relative_to(directory).as_posix():digest(p) for p in directory.rglob('*') if p.is_file()}
    write(directory/'receipt.json',dict(status='interrupted',error=reason,outputs=outputs))
    seal = directory.parent/f'{directory.name}-seal.json'
    if not seal.exists():
        write(seal,dict(status='interrupted',receipt_sha256=digest(directory/'receipt.json')))


def supervise() -> int:
    """Evaluate development, then frozen seed robustness, audit and exact replay."""
    ART.mkdir(parents=True,exist_ok=True)
    heartbeat = HeartbeatJournal(ART/'worker')
    child = None
    phase = None
    status = 'failed'
    try:
        stages = [
            ('development',['-m','flybrain.escape_edge.validation','development']),
            ('development-audit',['-m','flybrain.escape_edge.audit',str(BASE/'development'),'--output',str(ART/'development-audit.json')]),
            ('robustness',['-m','flybrain.escape_edge.validation','robustness']),
            ('robustness-audit',['-m','flybrain.escape_edge.audit',str(BASE/'robustness'),'--output',str(ART/'robustness-audit.json')]),
            ('replay',['-m','flybrain.escape_edge.replay'])]
        for phase,args in stages:
            with (ART/f'{phase}.log').open('x',encoding='utf-8') as stream:
                child = subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    heartbeat.observe(dict(stage=phase,supervisor_pid=os.getpid(),child_pid=child.pid,
                                           updated_unix_seconds=time.time()))
                    time.sleep(1)
                code = child.returncode
            if phase in ('development','robustness'):
                seal_missing(BASE/phase,f'child exit {code} without receipt')
                result = json.loads((BASE/phase/'results.json').read_text())
                if result['status']!='passed':
                    status = result['status']
                    return 1
            if code:
                return code
        status = 'verification-passed'
        return 0
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        if phase in ('development','robustness'):
            seal_missing(BASE/phase,f'{type(exc).__name__}: {exc}')
        raise
    finally:
        heartbeat.finish(dict(status=status,last_stage=phase,supervisor_pid=os.getpid(),automatic_retry=False))


if __name__ == '__main__':
    raise SystemExit(supervise())
