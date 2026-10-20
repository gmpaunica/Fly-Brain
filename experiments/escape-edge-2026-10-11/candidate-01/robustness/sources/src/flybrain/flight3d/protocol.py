"""Freeze and authorize the single October 1 landing experiment; never retry."""
from dataclasses import asdict
import csv
import json
import os
from pathlib import Path
import platform
import re
import shutil
from datetime import datetime, timezone
import numpy as np
import torch
from .anatomy import ROOT, CACHE, digest, load_manifest
from .world import make_trial

AUTHORIZATION='flight3d-visual-repair-2026-10-01'
RECEIPT=ROOT/'experiments'/f'.{AUTHORIZATION}-consumed.json'
DEVELOPMENT=ROOT/'experiments'/'flight3d-development-2026-10-01'


def durable_json(path: Path,value) -> None:
    """Flush a complete JSON replacement before atomically publishing it."""
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def check_overlap(root: Path,seeds: set[int],exclude: Path | None=None) -> dict:
    """Check structured historical/development records, including reserved seeds.

    Inspect integer tokens conservatively in JSON/JSONL/CSV (also catches seed
    lists and encoded dictionaries). Hash strings and prose are not case IDs.
    A collision blocks preparation; cases are never silently regenerated.
    """
    collisions=[]; scanned={}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix not in ('.json','.jsonl','.csv'):continue
        if exclude and (path==exclude or exclude in path.parents):continue
        content=path.read_text(encoding='utf-8-sig')
        matches={int(v) for v in re.findall(r'(?<![\w.])\d{6}(?![\w.])',content)} & seeds
        scanned[str(path.relative_to(root))]=digest(path)
        for seed in sorted(matches):collisions.append({'seed':seed,'file':str(path.relative_to(root))})
    if collisions:raise ValueError(f'case identity overlap: {collisions}')
    return {'collisions':collisions,'scanned_sha256':scanned}


def consume_authorization(directory: Path,protocol_hash: str,receipt: Path=RECEIPT) -> None:
    """Exclusive receipt lives outside run directory, so renaming cannot retry."""
    with receipt.open('x',encoding='utf-8') as stream:
        json.dump({'authorization':AUTHORIZATION,'consumed':True,'directory':str(directory.resolve()),
            'protocol_sha256':protocol_hash,'utc':datetime.now(timezone.utc).isoformat(),
            'interruption':'incomplete; authorization remains consumed'},stream,indent=2)
        stream.flush();os.fsync(stream.fileno())


def freeze(directory: Path) -> dict:
    """Freeze sources, conditions, fresh coordinates and unchanged scientific gates."""
    if RECEIPT.exists():raise FileExistsError('one-use authorization already consumed')
    if directory.exists():raise FileExistsError('protocol directory already exists')
    development=DEVELOPMENT/'sampling-search-v1'/'results.json'
    evidence=json.loads(development.read_text())
    expected=set(range(970000,970048))|{960000}
    if not evidence['passed'] or {r['seed'] for r in evidence['episodes']}!=expected or len(evidence['episodes'])!=49 or not all(r['success'] for r in evidence['episodes']):
        raise ValueError('development landing prerequisite requires 49/49')
    cases={split:[asdict(make_trial(start+r*24+h*8+s,sector=s,height=h))
        for r in range(repeats) for h in range(3) for s in range(8)]
        for split,start,repeats in [('train',980000,2),('validation',981000,1),('final',982000,4)]}
    seeds={c['seed'] for rows in cases.values() for c in rows}
    overlap=check_overlap(ROOT/'experiments',seeds)
    manifest=load_manifest();source=Path(__file__).parent
    protocol={'schema':'flight3d-experiment-v2','authorization':AUTHORIZATION,
        'hypothesis':'Motor-endpoint readouts can learn safe flower landing from local sensory activity, with fixed topology and dynamics.',
        'seed':940404,'device':'cpu','python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'platform':platform.platform(),'commit':None,
        'circuit_sha256':digest(CACHE/'circuit.json'),'cells':len(manifest['nodes']),'edges':len(manifest['edges']),
        'source_sha256':{p.name:digest(p) for p in sorted(source.glob('*.py'))},
        'development_sha256':digest(development),'cases':cases,
        'sensors':{'rgb_shape':[2,24,48,3],'horizontal_samples':2,'vertical_samples':4,
            'occlusion':'nearest surface separately per sample, then RGB average','contrast':'positive R-(G+B)/2','contrast_threshold':.01},
        'search':{'period_seconds':8,'amplitude_m':.015,'entry_altitude':'integrated local specific-force estimate','yaw':.15},
        'dynamics':'fixed 1 ms unsigned count-normalized LIF; 100 ms leak; threshold 1; subtractive reset; one step delay; sensory clamps; 100/500 ms endpoint histories',
        'episode_seconds':30,'success':'safe landing and >=2 s stationary food contact maintained through end',
        'optimizer':'Adam','updates':2000,'learning_rate':.001,'batch_size':512,
        'collection_fit_seconds_limit':7200,'normalization':'training mean/std only; scale floor 1 Hz',
        'objective':'mean squared error across bounded five-command vector','sampling':'uniform training ticks',
        'teacher_gate':48,'training_gate':44,'validation_gate':20,'final_gate':82,'final_per_height':26,
        'continuous_channels':['power','pitch','roll','yaw'],'mean_predictor_gate':'strictly smaller MSE in every continuous channel',
        'paired_p_limit':.025,'paired_tests':'one-sided exact discordant-pair binomial against matched untrained and fixed',
        'prerequisites':'signal distinction and propagation, each DN intervention, motor silence, mechanics',
        'stopping_rule':'stop on first failed prerequisite/gate; interruption or budget incomplete, never restart',
        'controls':['identically initialized untrained head with train normalization','fixed neural decoder']}
    directory.mkdir(parents=True)
    shutil.copytree(source,directory/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(CACHE/'circuit.json',directory/'circuit.json')
    durable_json(directory/'overlap.json',overlap)
    durable_json(directory/'protocol.json',protocol)
    durable_json(directory/'protocol.sha256.json',{'sha256':digest(directory/'protocol.json')})
    durable_json(directory/'results.json',{'status':'not_started','actual_optimizer_updates':0,'learned_landing_accepted':False})
    return protocol


def load_frozen(directory: Path) -> dict:
    """Reject edited source/configuration or any identity collision since freeze."""
    protocol=json.loads((directory/'protocol.json').read_text())
    if protocol.get('authorization')!=AUTHORIZATION:raise ValueError('wrong authorization')
    if digest(directory/'protocol.json')!=json.loads((directory/'protocol.sha256.json').read_text())['sha256']:
        raise ValueError('protocol digest mismatch')
    for name,sha in protocol['source_sha256'].items():
        if digest(Path(__file__).parent/name)!=sha or digest(directory/'source'/name)!=sha:
            raise ValueError(f'frozen source mismatch: {name}')
    if digest(CACHE/'circuit.json')!=protocol['circuit_sha256'] or digest(directory/'circuit.json')!=protocol['circuit_sha256']:
        raise ValueError('frozen circuit mismatch')
    seeds={c['seed'] for rows in protocol['cases'].values() for c in rows}
    check_overlap(ROOT/'experiments',seeds,exclude=directory.resolve())
    return protocol
