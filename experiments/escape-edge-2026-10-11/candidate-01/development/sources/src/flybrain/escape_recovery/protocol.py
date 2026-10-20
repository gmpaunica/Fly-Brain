"""Fixed numerical gates; trajectories, not state labels, establish movement."""
import math
import numpy as np
from flybrain.flight3d.contracts import FOOT_LENGTH
from flybrain.turnflee.experiment import presentation_success
from flybrain.turnflee.protocol import cases,closed_cases,validation_gate,closed_gate


def trajectory_metrics(case: dict,row: dict,trajectory: list[dict]) -> dict:
    """Evaluator-only poses verify turn/takeoff/separation/braking/recovery."""
    trigger=row.get('trigger_seconds');side=row.get('side');collision=any(r['collided'] for r in trajectory)
    turn=False;takeoff=case['state']!='resting';separation=0.;braked=False;stable=False
    weak=False;recovered=False;takeoff_ticks=0;recovery_ticks=0
    anchor=None;initial_distance=0.;sign=1 if side=='right' else -1;turn_window=True
    for r in trajectory:
        if trigger is None or r['end_seconds']+1e-12<trigger:continue
        pos=np.array(r['position']);pred=np.array(r['predator_position']);velocity=np.array(r['velocity'])
        if anchor is None:
            anchor=float(r['pre_orientation'][2]);initial_distance=float(np.linalg.norm(np.array(r['pre_position'])-pred))
        yaw=sign*(float(r['orientation'][2])-anchor)
        if turn_window and r['escape_state'] in ('turning','fleeing') and abs(yaw-math.pi/2)<=math.radians(3):turn=True
        if r['escape_state'] in ('fleeing','braking','idle'):turn_window=False
        if not weak:separation=max(separation,float(np.linalg.norm(pos-pred))-initial_distance)
        if case['state']=='resting':
            takeoff_ticks=takeoff_ticks+1 if not r['foot_contact'] and pos[2]>=FOOT_LENGTH+.005 else 0
            takeoff|=takeoff_ticks>=3
        weak|=bool(r['threat']['weak_continuous'])
        if weak and r['escape_state'] in ('braking','idle'):
            braked|=np.linalg.norm(velocity[:2])<.02 and abs(velocity[2])<.02
        recovered|=bool(r['recovered'])
        safe=(recovered and r['escape_state']=='idle' and r['behavior_state']=='cruising' and not r['collided']
              and .075<=pos[2]<=.125 and abs(velocity[2])<.02 and r['triggers']==1)
        recovery_ticks=recovery_ticks+1 if safe else 0;stable|=recovery_ticks>=15
    physical=bool(turn and takeoff and separation>=.02-1e-12 and not collision and weak and braked and recovered and stable)
    return dict(measured_turn=bool(turn),measured_takeoff=bool(takeoff),separation_gain_m=separation,
        measured_braking=bool(braked),collision=collision,weak_reached=bool(weak),recovered=bool(recovered),
        stable_recovery=bool(stable),physical_success=physical)


def closed_success(row: dict) -> bool:
    """Both timely signed selection and all measured physical criteria required."""
    return presentation_success(row) and row['physical_success']
