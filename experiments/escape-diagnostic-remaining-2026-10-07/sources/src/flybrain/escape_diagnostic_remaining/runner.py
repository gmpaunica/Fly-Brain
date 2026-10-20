"""Accepted eleven remaining cases; original kernels and comparisons unchanged."""
import argparse
from collections import deque
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import numpy as np
import torch
from flybrain.flight3d.anatomy import CACHE, load_manifest
from flybrain.flight3d.contracts import FlightObservation
from flybrain.encounters3d.scene import SceneSpec, encounter_eyes
from flybrain.retreat3d.controller import KinematicEstimate
from flybrain.retreat3d.threat import ThreatMonitor
from flybrain.turnflee.control import EscapePolicy, WingScores
from flybrain.turnflee.neural import ResponseConfig
from flybrain.turnflee.protocol import cases
from flybrain.escape_diagnostic.recorder import RecordedPair, chunk_arrays, ENGINEERING
from flybrain.escape_diagnostic_recovery.runner import preflight as original_preflight
from flybrain.escape_diagnostic_recovery.comparator import assert_historical_equal
from .contract import remaining_cases, additional_budget, verify_prior_receipt

ORIGINAL=Path('experiments/turnflee-2026-10-06')
OUT=Path('experiments/escape-diagnostic-remaining-2026-10-07')
ART=Path('artifacts/escape-diagnostic-remaining-2026-10-07')
PRIOR=Path('experiments/escape-diagnostic-recovery-2026-10-07')
PAIRS=((1100000,1100004),(1100001,1100003),(1100002,1100005),
       (1100010,1100008),(1100012,1100014),(1100013,1100015),
       (1100017,1100016),(1100032,1100033),(1100036,1100037))


def sha(path: Path) -> str:
    """Stream an input/output hash, including large source archives."""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write(path: Path, value, fresh: bool=True) -> None:
    """Finite JSON; immutable outputs or atomic progress receipt replacement."""
    if fresh:
        with path.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)
    else:
        p=path.with_suffix('.tmp'); p.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(p,path)


def verify_hashes(bound: dict[str,str]) -> None:
    """Reject changed or missing protected bytes before another interval."""
    for name,h in bound.items():
        if not Path(name).is_file() or sha(Path(name))!=h: raise ValueError(f'protected hash mismatch: {name}')


def consume(directory: Path) -> None:
    """Parent-scoped exclusive consumption forbids interruption/renaming retries."""
    if directory.exists(): raise FileExistsError(directory)
    directory.parent.mkdir(parents=True,exist_ok=True)
    write(directory.parent/'escape-diagnostic-remaining-once.json',dict(directory=str(directory.resolve()),
          consumed=True,retry_authorized=False))
    directory.mkdir()


class StepBudget:
    """Reserve both brains' complete next interval before any kernel call."""
    def __init__(self, path: Path, limit: int=33000):
        self.path=path; self.limit=limit; self.steps=0
        write(path,dict(reserved_steps=0,limit=limit))

    def reserve_interval(self) -> None:
        if self.steps+40>self.limit: raise ValueError('scientific budget exceeded')
        self.steps+=40
        write(self.path,dict(reserved_steps=self.steps,limit=self.limit),False)


def historical_records() -> dict[int,list[dict]]:
    """Read only the 18 exact validation identities, checking complete endpoints."""
    selected={s:[] for pair in PAIRS for s in pair}
    with gzip.open(ORIGINAL/'traces.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['phase']=='validation' and r['candidate']==1 and r['seed'] in selected:
                selected[r['seed']].append(r)
    protocol={c['seed']:c for c in cases('validation')}
    for seed,rows in selected.items():
        if [r['interval'] for r in rows]!=list(range(75)): raise ValueError('incomplete historical case')
        if any(any(r[k]!=v for k,v in protocol[seed].items()) for r in rows):
            raise ValueError('historical identity mismatch')
    return selected


def preflight() -> dict:
    """Bind both consumed attempts and verify exact remaining-case approval."""
    config=original_preflight()
    seal=json.loads((PRIOR.parent/'escape-diagnostic-recovery-seal.json').read_text())
    prior=verify_prior_receipt(PRIOR/'receipt.json',seal['receipt_sha256'])
    for p,h in prior['outputs'].items():
        if sha(PRIOR/p)!=h: raise ValueError('interrupted prior output hash mismatch')
    old_config=json.loads((PRIOR/'config.json').read_text())
    for p,h in {**dict(old_config['response']['source_sha256']),**old_config['added_sources'],**old_config['test_sources']}.items():
        if sha(Path(p))!=h or sha(PRIOR/'sources'/p)!=h: raise ValueError('prior source archive mismatch')
    all_cases=config['cases']
    config['aggregate_cases']=all_cases
    config['cases']=remaining_cases(all_cases)
    if additional_budget(config['cases'])!=(16.5,33000): raise ValueError('remaining budget mismatch')
    frozen=config['protected'].copy()
    for folder in (PRIOR,Path('artifacts/escape-diagnostic-recovery-2026-10-07')):
        frozen.update({p.as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    frozen.update({p.as_posix():sha(p) for p in [PRIOR.parent/'escape-diagnostic-recovery-once.json',PRIOR.parent/'escape-diagnostic-recovery-seal.json']})
    frozen.update(config['added_sources']); frozen.update(config['test_sources'])
    config['protected']=frozen
    config['added_sources'].update({p.as_posix():sha(p) for p in sorted(Path('src/flybrain/escape_diagnostic_remaining').glob('*.py'))})
    config['test_sources']['tests/test_escape_diagnostic_remaining.py']=sha(Path('tests/test_escape_diagnostic_remaining.py'))
    evidence=json.loads((ART/'acceptance.json').read_text())
    if evidence['status']!='passed' or evidence['added_kernel_steps']!=0 or evidence['engineering_steps']!=1747:
        raise ValueError('saved-only remaining acceptance missing')
    verify_hashes(evidence['source_hashes']); verify_hashes(evidence['verification_hashes'])
    config.update(acceptance_sha256=sha(ART/'acceptance.json'),prior_completion=dict(directory=PRIOR.as_posix(),
        receipt_sha256=sha(PRIOR/'receipt.json'),scientific_steps=22120,simulated_seconds=11.06,completed_cases=7),
        max_cumulative_diagnostic_steps=55160,max_cumulative_diagnostic_seconds=27.58,
        approved_proposal_sha256=sha(Path('docs/ESCAPE-DIAGNOSTIC-REMAINING-PROPOSAL.md')))
    return config


def interval_record(case,interval,pair,response,threat,policy,rgb,window,cumulative):
    """Original interval schema plus exact entry/expiry and all-cell cumulative."""
    expired=window[0] if len(window)==10 else None
    increment=pair.actual.last_counts-pair.reference.last_counts
    window.append((interval,increment.copy())); cumulative+=increment
    base=dict(**case,interval=interval,end_seconds=(interval+1)*.02,
        eyes_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        actual_counts=pair.actual.last_counts.tolist(),reference_counts=pair.reference.last_counts.tolist(),
        wing_vector=pair.last_vector.tolist(),wing_scores=asdict(response),threat=asdict(threat),
        selected_side=policy.side,escape_state=policy.state,triggers=policy.trigger_count,
        actual_wings=int(pair.actual.last_counts[list(pair.indices)].sum()),
        reference_wings=int(pair.reference.last_counts[list(pair.indices)].sum()),
        finite=pair.finite,max_downstream_hz=pair.max_downstream_hz,pathway=asdict(pair.actual.diagnostic))
    return dict(original=base,start_step=interval*20,end_step=(interval+1)*20,
        reference_pathway=asdict(pair.reference.diagnostic),
        entered_interval=interval,expired_interval=expired[0] if expired else None,
        entered_increment=increment.tolist(),expired_increment=expired[1].tolist() if expired else [0]*589,
        rolling_all=np.sum([v for _,v in window],axis=0).tolist(),
        cumulative_increment=cumulative.tolist(),pending_until=policy.pending_until,
        first_spike_step_actual=[round(t*1000) if t is not None else None for t in pair.actual._first_spike],
        first_spike_step_reference=[round(t*1000) if t is not None else None for t in pair.reference._first_spike])


def run(directory: Path=OUT) -> dict:
    """Consume once after preflight, stop at first mismatch, always seal outputs."""
    torch.set_num_threads(1)
    config=preflight(); expected=historical_records(); consume(directory)
    status='interrupted'; error=None; completed=0; intervals=0; budget=None
    output_hashes={}
    try:
        config.update(schema='escape-diagnostic-remaining-v3',hypothesis='Vision shifts tonic left b3 timing through retained right-DN delivery; rolling expiration creates the five transient deficits.',
            max_cases=11,max_seconds=16.5,max_kernel_steps=33000,baseline_steps=500,
            interval_steps=20,neural_dt_seconds=.001,rolling_intervals=10,
            downstream_guard_hz=240,commit=None,optimizer_updates=0,learning_rate=None,checkpoint=None,
            python=platform.python_version(),platform=platform.platform(),device='cpu',
            software={n:importlib.metadata.version(n) for n in ('torch','numpy','panda3d','pygame')},
            stop='first mismatch/hash/measurement/nonfinite/rate/budget violation; otherwise exactly 11; never retry')
        write(directory/'config.json',config)
        for name in ('acceptance.json','remaining-tests.xml','offline-call-guard.json'):
            target=directory/'preflight'/name; target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ART/name).read_bytes())
        for name,h in {**dict(ResponseConfig.from_record(config['response']).source_sha256),**config['added_sources'],**config['test_sources']}.items():
            p=directory/'sources'/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(Path(name).read_bytes())
            if sha(p)!=h: raise ValueError('source archive hash mismatch')
        manifest=load_manifest(); (directory/'circuit.json').write_bytes((CACHE/'circuit.json').read_bytes())
        if sha(directory/'circuit.json')!=config['response']['circuit_sha256']: raise ValueError('circuit archive mismatch')
        response=ResponseConfig.from_record(config['response']); pair=RecordedPair(manifest,response)
        graph=dict(pre=pair.actual.pre.numpy(),post=pair.actual.post.numpy(),weights=pair.actual.weights.numpy(),
                   inputs=pair.actual.inputs.numpy(),live=np.ones(589,dtype=bool),
                   effective=((~pair.actual.inputs[pair.actual.post])).numpy(),dynamic_edges=pair.actual.dynamic_edges,
                   wing_indices=np.array(pair.indices),ids=np.array(pair.actual.ids))
        np.savez_compressed(directory/'graph.npz',**graph)
        budget=StepBudget(directory/'progress.json')
        # Fast source checks each interval; large immutable archives checked at
        # preflight, each case boundary and completion rather than reread per ms.
        fast={**dict(response.source_sha256),**config['added_sources'],**config['test_sources']}
        metadata={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns) for p in config['protected']}
        for case_number,case in enumerate(config['cases']):
            response.verify(manifest)
            pair.reset(); monitor=ThreatMonitor(); scores=WingScores(manifest)
            policy=EscapePolicy(case['seed']); estimate=KinematicEstimate(); estimate.altitude=.1
            window=deque(maxlen=10); cumulative=np.zeros(589,dtype=np.int64)
            case_dir=directory/f'{case_number:02d}-{case["seed"]}'; case_dir.mkdir()
            with gzip.open(case_dir/'intervals.jsonl.gz','xt',encoding='utf-8') as trace:
                for interval in range(75):
                    verify_hashes(fast)
                    for name,stamp in metadata.items():
                        stat=Path(name).stat()
                        if (stat.st_size,stat.st_mtime_ns)!=stamp:
                            verify_hashes({name:config['protected'][name]})
                            metadata[name]=(stat.st_size,stat.st_mtime_ns)
                    rgb=np.zeros((2,24,48,3),dtype=np.float32)
                    if interval>=25:
                        start,end=((.17,.08),(.12,.06))[case['profile']]
                        distance=start+(end-start)*(interval-25)/49
                        az=math.radians(case['azimuth']); el=math.radians(case['elevation'])
                        position=(.3+distance*math.sin(az)*math.cos(el),.3+distance*math.cos(az)*math.cos(el),.1+distance*math.sin(el))
                        rgb=encounter_eyes(np.array((.3,.3,.1)),np.zeros(3),SceneSpec(case['seed'],'predator',position))
                    budget.reserve_interval()
                    obs=FlightObservation(0.,0.,rgb,(0.,0.,0.),(0.,0.,9.81),False,False)
                    vector=pair.advance(obs); response_value=scores(vector); threat=monitor.update(rgb)
                    policy.command(estimate,obs,response_value,threat)
                    record=interval_record(case,interval,pair,response_value,threat,policy,rgb,window,cumulative)
                    chunk=case_dir/f'{interval:03d}.npz'
                    arrays=chunk_arrays(pair,rgb)
                    if any(not np.isfinite(v).all() for v in arrays.values()): raise ValueError('nonfinite measurement')
                    np.savez_compressed(chunk,**arrays)
                    record['chunk_sha256']=sha(chunk); output_hashes[chunk.relative_to(directory).as_posix()]=record['chunk_sha256']
                    trace.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n'); trace.flush()
                    intervals+=1
                    if not pair.finite or not pair.stable: raise ValueError('nonfinite state or downstream rate >240 Hz')
                    assert_historical_equal(record['original'],expected[case['seed']][interval])
            completed+=1
            print(f'case {completed}/11 seed {case["seed"]}: 75/75 intervals identical | steps {budget.steps}',flush=True)
        verify_hashes(config['protected']); verify_hashes(fast)
        if (completed,intervals,budget.steps)!=(11,825,33000): raise ValueError('exact budget mismatch')
        status='completed'
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'; status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
        raise
    finally:
        # Include partial files on failure/interruption, and never resume them.
        for p in directory.rglob('*'):
            if p.is_file(): output_hashes[p.relative_to(directory).as_posix()]=sha(p)
        receipt=dict(status=status,error=error,completed_cases=completed,recorded_intervals=intervals,
                     reserved_kernel_steps=budget.steps if budget else 0,simulated_seconds=intervals*.02,
                     retry_authorized=False,outputs=output_hashes)
        write(directory/'receipt.json',receipt)
        write(directory.parent/'escape-diagnostic-remaining-seal.json',dict(directory=str(directory.resolve()),receipt_sha256=sha(directory/'receipt.json'),status=status))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--preflight',action='store_true'); args=parser.parse_args()
    if args.preflight:
        c=preflight(); print(json.dumps(dict(status='preflight-passed',cases=c['cases'],protected=len(c['protected'])),indent=2))
    else: print(json.dumps(run(),indent=2))
