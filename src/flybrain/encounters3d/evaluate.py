"""Evaluator-only geometry for matched gate-failed encounter trials."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import numpy as np
import torch
from flybrain.flight3d.anatomy import CACHE, digest
from .characterize import stimulus_cases
from .session import EncounterSession


def measure(scene, intervention: str, steps: int=250) -> dict:
    """Measure a frozen controller; geometry never enters its observation."""
    session=EncounterSession(scene)
    b=session.brain
    vision=[int(i) for i in b.roles['vision']]
    dn=[i for i,n in enumerate(b.nodes) if n['metadata'].get('cell_type')=='DNp03']
    motor=sorted(set(b.roles['wing_power'])|set(b.roles['wing_steering']))
    b.silenced=({b.ids[i] for i in vision} if intervention=='visual-silent' else
                {b.ids[i] for i in dn} if intervention=='DNp03-silent' else set())
    initial=float(np.linalg.norm(session.world.position-np.array(scene.position)))
    separation=[initial]; counts=np.zeros(len(b.ids),dtype=np.int64)
    visual_delay=None; dn_delay=None; command_delay=None
    for _ in range(steps):
        snapshot=session.advance()
        counts+=np.array(snapshot.spike_counts)
        separation.append(float(np.linalg.norm(session.world.position-np.array(scene.position))))
        if visual_delay is None and b.last_counts[vision].any(): visual_delay=snapshot.time
        if dn_delay is None and b.last_counts[dn].any(): dn_delay=snapshot.time
        if command_delay is None and max(abs(snapshot.command.pitch),abs(snapshot.command.roll))>.01:
            command_delay=snapshot.time
    return {'seed':scene.seed,'scene':asdict(scene),'intervention':intervention,
            'duration_seconds':session.world.time,'initial_separation_m':initial,
            'final_separation_m':separation[-1],'minimum_separation_m':min(separation),
            'separation_change_m':separation[-1]-initial,'collision':session.world.crashed,
            'first_visual_spikes_seconds':visual_delay,'first_DNp03_spikes_seconds':dn_delay,
            'first_horizontal_command_seconds':command_delay,
            'escape_response_delay_seconds':None,'avoidance_supported':False,
            'final_speed_m_per_s':float(np.linalg.norm(session.world.velocity)),
            'visual_spikes':int(counts[vision].sum()),'DNp03_spikes':counts[dn].tolist(),
            'motor_spikes':int(counts[motor].sum())}


def evaluate(directory: Path) -> dict:
    """Append distinct evaluator files to this characterization; refuse overwrite."""
    directory.mkdir(parents=True,exist_ok=True)
    scenes=[(label,scene) for label,scene in stimulus_cases() if label.endswith('-far')]
    config={'schema':'encounters3d-closed-loop-v1','date':'2026-10-05',
            'objective':'Measure the failed-gate preview transparently; no avoidance success claim.',
            'budget':'8 placements x 3 interventions x 5 seconds; stop after these 24 trials.',
            'cases':[{'label':label,**asdict(scene)} for label,scene in scenes],
            'steps':250,'neural_steps_per_case':5000,'control_dt_seconds':.02,
            'software':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
            'device':'cpu','commit':None,'learning_rate':None,'optimizer_updates':0,'checkpoint':None,
            'circuit_sha256':digest(CACHE/'circuit.json'),
            'source_sha256':{str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('controller.py'),
                                                      Path(__file__).with_name('session.py')]}}
    with (directory/'closed_loop_config.json').open('x') as stream: json.dump(config,stream,indent=2)
    torch.set_num_threads(1)
    rows=[{'label':label,**measure(scene,intervention)} for label,scene in scenes
          for intervention in ('intact','visual-silent','DNp03-silent')]
    result={'rows':rows,'avoidance_supported':False,
            'reason':'No escape decoder selected after failed static wing spike gate; stable hover is scaffolding.',
            'optimizer_updates':0}
    with (directory/'closed_loop_results.json').open('x') as stream: json.dump(result,stream,indent=2)
    keys=('label','seed','intervention','duration_seconds','initial_separation_m','final_separation_m',
          'separation_change_m','collision','first_visual_spikes_seconds','first_DNp03_spikes_seconds',
          'escape_response_delay_seconds','final_speed_m_per_s','motor_spikes')
    with (directory/'closed_loop.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader()
        writer.writerows({key:row[key] for key in keys} for row in rows)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('directory',type=Path)
    result=evaluate(parser.parse_args().directory)
    print(json.dumps({'trials':len(result['rows']),'collisions':sum(r['collision'] for r in result['rows']),
                      'escape_responses':sum(r['escape_response_delay_seconds'] is not None for r in result['rows'])}))
