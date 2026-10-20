"""Development-only coverage evidence; never reads reserved scientific cases."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import numpy as np
from flybrain.flight3d.world import FlightWorld, make_trial
from flybrain.flight3d.teacher import FlightTeacher, visual_measurement

ROOT = Path(__file__).parent
cases = [make_trial(970000+r*24+h*8+s,sector=s,height=h)
         for r in range(2) for h in range(3) for s in range(8)]
cases += [make_trial(960000,sector=0,height=0)]

def run(name: str):
    folder=ROOT/name
    folder.mkdir(exist_ok=False)
    (folder/'manifest.json').write_text(json.dumps({'kind':'development; 960000 historical regression',
        'cases':[asdict(c) for c in cases]},indent=2))
    rows=[]
    for case in cases:
        world=FlightWorld(case); teacher=FlightTeacher(); trace=[]; detections=0; first=None
        for tick in range(1500):
            obs=world.observe(); sight=visual_measurement(obs.eyes)
            if sight is not None:
                detections+=1
                if first is None: first=world.time
            command=teacher.command(obs); snap=world.step(command)
            trace.append((*snap.position,*snap.velocity,*command.vector(),
                sight is not None,snap.foot_contact,snap.food_contact,snap.crashed,snap.dwell,
                *(sight[0] if sight is not None else [0,0,0])))
        values=np.array(trace)
        row={'seed':case.seed,'success':bool(world.food_contact and not world.crashed and world.dwell>=2
              and np.linalg.norm(world.velocity)<1e-4),'dwell':world.dwell,'crashed':world.crashed,
             'first_detection':first,'detections':detections,'first_contact':world.first_contact,
             'final_position':world.position.tolist(),'flower':case.flower,
             'minimum_horizontal_error':float(np.linalg.norm(values[:,:2]-np.array(case.flower[:2]),axis=1).min())}
        np.savez_compressed(folder/f'{case.seed}.npz',trace=values)
        with (folder/'episodes.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        rows.append(row);print(json.dumps(row),flush=True)
    (folder/'results.json').write_text(json.dumps({'passed':all(r['success'] for r in rows),'episodes':rows},indent=2))

if __name__=='__main__': run(sys.argv[1])
