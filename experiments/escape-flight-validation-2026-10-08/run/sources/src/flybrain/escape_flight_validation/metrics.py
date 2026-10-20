"""Evaluator-only physical scoring, using measured SI-unit trajectories."""
import math
import numpy as np

MEASUREMENTS=('pre_position','pre_orientation','position','orientation','velocity',
              'predator_position','foot_contact','collided','behavior_state',
              'escape_state','recovered','triggers','threat','end_seconds')


def validate_measurements(record:dict) -> None:
    """Missing or nonfinite measurements stop execution, rather than score false."""
    if any(k not in record for k in MEASUREMENTS):raise ValueError('missing physical measurement')
    for k in ('pre_position','pre_orientation','position','orientation','velocity','predator_position'):
        value=np.asarray(record[k],dtype=float)
        if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('nonfinite physical measurement')
    if not math.isfinite(record['end_seconds']):raise ValueError('nonfinite measurement time')
    if 'weak_continuous' not in record['threat']:raise ValueError('missing threat measurement')


def physical_metrics(case:dict,trigger:float|None,side:str|None,records:list[dict]) -> dict:
    """Require actual completed turn before flee and measured stable recovery."""
    for r in records:validate_measurements(r)
    collision=any(r['collided'] for r in records)
    turn=False;takeoff=case['state']!='resting';separation=0.;braked=False;stable=False
    weak=False;recovered=False;takeoff_streak=0;recovery_streak=0;max_recovery=0
    first_flee=None;turn_time=None;weak_time=None;takeoff_time=None;brake_time=None
    anchor=None;initial_distance=None;sign=1 if side=='right' else -1
    for r in records:
        if trigger is None or r['end_seconds']+1e-12<trigger:continue
        pos=np.array(r['position']);pred=np.array(r['predator_position']);vel=np.array(r['velocity'])
        if anchor is None:
            anchor=float(r['pre_orientation'][2]);initial_distance=float(np.linalg.norm(np.array(r['pre_position'])-pred))
        # Flee begins when its command is issued: only pre-command yaw can
        # establish the turn on that interval. Later turns cannot repair it.
        if first_flee is None:
            yaw=float(r['pre_orientation'][2] if r['escape_state']=='fleeing' else r['orientation'][2])
            if r['escape_state'] in ('turning','fleeing') and abs(sign*(yaw-anchor)-math.pi/2)<=math.radians(3)+1e-12:
                turn=True;turn_time=turn_time or r['end_seconds']
            if r['escape_state']=='fleeing':first_flee=r['end_seconds']
        now_weak=bool(r['threat']['weak_continuous'])
        if not weak and not now_weak:separation=max(separation,float(np.linalg.norm(pos-pred))-initial_distance)
        if case['state']=='resting':
            takeoff_streak=takeoff_streak+1 if not r['foot_contact'] and pos[2]>=.008-1e-12 else 0
            if takeoff_streak>=3:takeoff=True;takeoff_time=takeoff_time or r['end_seconds']
        if now_weak and not weak:weak_time=r['end_seconds']
        weak|=now_weak
        if weak and r['escape_state'] in ('braking','idle') and np.linalg.norm(vel[:2])<.02 and abs(vel[2])<.02:
            braked=True;brake_time=brake_time or r['end_seconds']
        recovered|=bool(r['recovered'])
        safe=(recovered and r['escape_state']=='idle' and r['behavior_state']=='cruising'
              and not r['collided'] and .075<=pos[2]<=.125 and abs(vel[2])<.02 and r['triggers']==1)
        recovery_streak=recovery_streak+1 if safe else 0
        max_recovery=max(max_recovery,recovery_streak);stable|=recovery_streak>=15
    passed=bool(turn and first_flee is not None and takeoff and separation>=.02-1e-12
                and not collision and weak and braked and recovered and stable)
    return dict(measured_turn=turn,measured_takeoff=takeoff,separation_gain_m=separation,
        measured_braking=braked,collision=collision,weak_reached=weak,recovered=recovered,
        stable_recovery=stable,physical_success=passed,turn_seconds=turn_time,
        first_flee_seconds=first_flee,weak_seconds=weak_time,takeoff_seconds=takeoff_time,
        braking_seconds=brake_time,max_stable_recovery_intervals=max_recovery,
        pre_command_yaw=anchor,initial_predator_separation_m=initial_distance)
