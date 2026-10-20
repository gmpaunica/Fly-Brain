"""Bounded, read-only pathway characterization; never selects or fits weights."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import numpy as np
import torch
from flybrain.flight3d.anatomy import CACHE, digest, load_manifest
from flybrain.flight3d.contracts import FlightObservation
from .neural import EscapeBrain
from .scene import SceneSpec, EncounterWorld


def stimulus_cases() -> list[tuple[str, SceneSpec]]:
    """Eight azimuths at two sizes; front/rear are symmetric binocular stimuli."""
    return [(f'{sector}-{size}', SceneSpec(980000+sector, 'predator',
             (.3+radius*math.sin(sector*math.tau/8),
              .3+radius*math.cos(sector*math.tau/8), .1)))
            for sector in range(8) for size,radius in (('far',.15),('expanded',.06))]


def static_response(brain: EscapeBrain, eyes: np.ndarray, intervention: str) -> dict:
    """One second of matched static RGB, with intervention fixed before onset."""
    brain.reset()
    dn=[i for i,n in enumerate(brain.nodes) if n['metadata'].get('cell_type')=='DNp03']
    motors=sorted(set(brain.roles['wing_steering'])|set(brain.roles['wing_power']))
    brain.silenced=({brain.ids[i] for i in brain.roles['vision']} if intervention=='visual-silent'
                    else {brain.ids[i] for i in dn} if intervention=='DNp03-silent' else set())
    observation=FlightObservation(0,0,eyes,(0,0,0),(0,0,9.81),False,False)
    counts=np.zeros(len(brain.ids),dtype=np.int64)
    first_dn=None; first_motor=None; max_voltage=0.
    for interval in range(50):
        current=brain.advance(observation)
        counts+=current
        if first_dn is None and current[dn].any(): first_dn=(interval+1)*.02
        if first_motor is None and current[motors].any(): first_motor=(interval+1)*.02
        max_voltage=max(max_voltage,float(brain.voltage[motors].max()))
    return {'intervention':intervention,'counts':counts.tolist(),
            'visual_spikes':int(counts[brain.roles['vision']].sum()),
            'DNp03_spikes':counts[dn].tolist(),'motor_spikes':int(counts[motors].sum()),
            'motor_counts':counts[motors].tolist(), 'max_endpoint_voltage':max_voltage,
            'first_DNp03_seconds':first_dn,'first_motor_seconds':first_motor}


def characterize(directory: Path) -> dict:
    """Write a fresh, source-bound report, refusing existing run directories."""
    # Permit only an empty directory left by a setup failure before any steps.
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('existing scientific records cannot be overwritten')
    directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    brain=EscapeBrain(load_manifest())
    config={'schema':'encounters3d-characterization-v1','date':'2026-10-05',
        'hypothesis':'Existing visually driven wing spike histories distinguish escape directions.',
        'budget':'16 physical RGB stimuli plus symmetric saturation; three matched interventions; 1 s each.',
        'stop_condition':'Zero or directionally ambiguous wing responses close the gate; no decoder fitting/training.',
        'objective':'Characterize visual -> DNp03 -> wing contributions before motor decoding.',
        'seed_base':980000,'software':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__},
        'device':'cpu','commit':None,'neural_steps_per_trial':1000,'control_dt_seconds':.02,
        'neural_dt_seconds':.001,'learning_rate':None,'optimizer_updates':0,'checkpoint':None,
        'network_cells':len(brain.ids),'network_edges':len(brain.edge_order),
        'circuit_sha256':digest(CACHE/'circuit.json'),
        'encoder':'Cyan min(G,B)-R > .05; 240*clip(.75 peak+.25 occupancy+.5 positive growth,0,1); existing ID-ranked 8x6 fields.',
        'cases':[{'label':label,**asdict(scene)} for label,scene in stimulus_cases()],
        'source_sha256':{str(p.resolve().relative_to(Path.cwd())):digest(p) for p in
                         [Path(__file__),Path(__file__).with_name('neural.py'),Path(__file__).with_name('scene.py'),
                          Path('src/flybrain/flight3d/neural.py'),Path('src/flybrain/flight3d/sensors.py')]}}
    (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows=[]
    for label,scene in stimulus_cases():
        eyes=EncounterWorld(scene).observe().eyes
        for intervention in ('intact','visual-silent','DNp03-silent'):
            row=static_response(brain,eyes,intervention)
            rows.append({'label':label, 'scene':asdict(scene),
                         'eyes_sha256':hashlib.sha256(eyes.tobytes()).hexdigest(), **row})
    # Diagnostic upper-drive stimulus, synthetic and distinct from one-object scenes.
    eyes=np.empty((2,24,48,3),dtype=np.float32); eyes[:]=(.05,.95,.95)
    for intervention in ('intact','visual-silent','DNp03-silent'):
        rows.append({'label':'synthetic-bilateral-full-field',**static_response(brain,eyes,intervention)})
    intact=[r for r in rows if r['intervention']=='intact' and 'scene' in r]
    result={'config':'config.json','neuron_ids':brain.ids,'rows':rows,
            'directional_escape_supported':False,
            'reason':'Physical one-object stimuli do not establish a reliable directional wing spike code; no motor decoder selected.',
            'physical_motor_spikes':[r['motor_spikes'] for r in intact],
            'optimizer_updates':0,'checkpoint':None}
    (directory/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    keys=('label','intervention','visual_spikes','DNp03_spikes','motor_spikes','max_endpoint_voltage',
          'first_DNp03_seconds','first_motor_seconds')
    with (directory/'responses.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader()
        writer.writerows({key:row[key] for key in keys} for row in rows)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    result=characterize(parser.parse_args().directory)
    print(json.dumps({'physical_motor_spikes':result['physical_motor_spikes'],
                      'supported':result['directional_escape_supported']}))
