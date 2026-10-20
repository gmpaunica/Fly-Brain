"""Detached single-child supervisor with append-only heartbeat and fallback seal."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib
from flybrain.escape_diagnostic_journal import HeartbeatJournal
from flybrain.escape_diagnostic_recovery.runner import write
from flybrain.escape_recovery.worker import reservation_upper_bound
from flybrain.flight3d.anatomy import ROOT,digest
from .runner import RUN,ART


def seal_missing_receipt(directory:Path,reason:str) -> None:
    """Conservatively seal disappeared or failed child; never overwrite receipts."""
    if not directory.exists() or (directory/'receipt.json').exists():return
    intervals=0;groups={};records=directory/'traces.jsonl.gz'
    if records.exists():
        decoder=zlib.decompressobj(16+zlib.MAX_WBITS)
        try:lines=decoder.decompress(records.read_bytes()).splitlines()
        except zlib.error:lines=[]
        for line in lines:
            try:r=json.loads(line)
            except (ValueError,TypeError):break
            key=(r['seed'],r['state'],r['intervention']);groups[key]=groups.get(key,0)+1;intervals+=1
    steps=reservation_upper_bound(directory)
    if not (directory/'results.json').exists():
        write(directory/'results.json',dict(status='interrupted',error=reason,closed_loop=[],
            cases=sum(v==300 for v in groups.values()),recorded_intervals=intervals,
            reserved_kernel_steps=steps,recorded_kernel_steps_lower_bound=intervals*40,
            simulated_seconds_upper_bound=steps/2000,retry_authorized=False))
    outputs={p.relative_to(directory).as_posix():digest(p) for p in directory.rglob('*') if p.is_file()}
    config=json.loads((directory/'config.json').read_text()) if (directory/'config.json').exists() else {}
    write(directory/'receipt.json',dict(status='interrupted',error=reason,candidate_identity=config.get('candidate_identity'),
        outputs=outputs,reserved_kernel_steps=steps,retry_authorized=False,terminal_receipt_origin='detached supervisor'))
    write(directory.parent/'flight-validation-seal.json',dict(status='interrupted',receipt_sha256=digest(directory/'receipt.json')))


def supervise(command:list[str],directory:Path,log:Path,heartbeat:Path) -> int:
    """Start one child; no automatic restart or scientific retry."""
    journal=HeartbeatJournal(heartbeat.with_suffix(''));child=None
    try:
        with log.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            while child.poll() is None:
                journal.observe(dict(status='running',supervisor_pid=os.getpid(),child_pid=child.pid,updated_unix_seconds=time.time()))
                time.sleep(1)
            code=child.returncode
        seal_missing_receipt(directory,f'child exit {code} without normal receipt')
        journal.finish(dict(status='exited',exit_code=code,supervisor_pid=os.getpid(),child_pid=child.pid,retry_authorized=False))
        return code
    except BaseException as exc:
        if child is not None and child.poll() is None:child.terminate();child.wait()
        seal_missing_receipt(directory,f'supervision failed: {type(exc).__name__}: {exc}')
        journal.stream.close()
        raise


if __name__=='__main__':
    ART.mkdir(parents=True,exist_ok=True)
    raise SystemExit(supervise([sys.executable,'-m','flybrain.escape_flight_validation.runner'],RUN,ART/'experiment.log',ART/'worker.json'))
