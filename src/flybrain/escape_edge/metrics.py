"""Evaluator-only scoring from actual motion; no measurements enter control."""
import math
import numpy as np
from flybrain.escape_flight_validation.metrics import validate_measurements


def edge_metrics(case: dict, trigger: float | None, side: str | None,
                 records: list[dict]) -> dict:
    """Require a real turn, continued travel and a two-second safe edge hover."""
    for r in records:
        validate_measurements(r)
    collision = any(r['collided'] for r in records)
    turn = False
    takeoff = case['state'] != 'resting'
    takeoff_streak = 0
    first_flee = None
    arrival = None
    stable_ticks = 0
    max_stable = 0
    premature_stop = False
    reversal = False
    initial_distance = None
    separation = 0.
    anchor = None
    direction = None
    hover_unsafe = False
    for r in records:
        if trigger is None or r['end_seconds']+1e-12 < trigger:
            continue
        pos, pre, vel = (np.asarray(r[k]) for k in ('position', 'pre_position', 'velocity'))
        pred = np.asarray(r['predator_position'])
        if anchor is None:
            anchor = float(r['pre_orientation'][2])
            heading = anchor+(math.pi/2 if side == 'right' else -math.pi/2)
            direction = np.array((math.sin(heading), math.cos(heading)))
            initial_distance = float(np.linalg.norm(pre-pred))
        state = r['escape_state']
        if first_flee is None and state != 'turning':
            yaw = (r['pre_orientation'][2]-anchor)*(1 if side == 'right' else -1)
            turn = abs(yaw-math.pi/2) <= math.radians(3)+1e-12
            first_flee = r['end_seconds']
        if not r['foot_contact'] and pos[2] >= .008:
            takeoff_streak += 1
        else:
            takeoff_streak = 0
        takeoff |= takeoff_streak >= 3
        separation = max(separation, float(np.linalg.norm(pos-pred))-initial_distance)
        wall_distance = min(*pos[:2], .6-pos[0], .6-pos[1])
        speed = float(np.linalg.norm(vel[:2]))
        if state in ('fleeing', 'edge_braking'):
            reversal |= float((pos-pre)[:2]@direction) < -.002
            if first_flee is not None and r['end_seconds'] > first_flee+.4 and wall_distance > .06:
                premature_stop |= speed < .04
        safe = (state == 'edge_hover' and .005 <= wall_distance <= .025
                and speed <= .01 and abs(pos[2]-.1) <= .015 and not r['collided'])
        if state == 'edge_hover':
            hover_unsafe |= not safe
        stable_ticks = stable_ticks+1 if safe else 0
        max_stable = max(max_stable, stable_ticks)
        if safe and arrival is None:
            arrival = r['end_seconds']
    success = bool(turn and takeoff and separation >= .02-1e-12 and not collision
        and not premature_stop and not reversal and not hover_unsafe
        and arrival is not None and arrival <= 10.+1e-12 and max_stable >= 100)
    return dict(measured_turn=turn, measured_takeoff=takeoff, separation_gain_m=separation,
        collision=collision, premature_stop=premature_stop, inward_reversal=reversal,
        arrival_seconds=arrival, stable_hover_seconds=max_stable*.02,
        hover_unsafe=hover_unsafe, physical_success=success)
