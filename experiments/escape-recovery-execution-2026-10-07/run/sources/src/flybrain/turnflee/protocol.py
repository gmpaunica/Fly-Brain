"""Predeclared candidate order, signed responses and exact case gates."""
import json
import math
from pathlib import Path
from flybrain.flight3d.anatomy import digest
from .neural import GAIN_PAIRS


def cases(phase: str) -> list[dict]:
    """Historical development vs offset validation; angle is evaluator-only."""
    if phase not in ('calibration','validation'):
        raise ValueError('calibration or validation required')
    azimuths=list(range(0,360,45)) if phase=='calibration' else [18.75+22.5*k for k in range(16)]
    elevations=(0,) if phase=='calibration' else (-20,0,20)
    return [dict(phase=phase,azimuth=az,elevation=el,profile=p,seed=1100000+i*6+j*2+p)
            for i,az in enumerate(azimuths) for j,el in enumerate(elevations) for p in range(2)]


def closed_cases() -> list[dict]:
    """Exactly eight directions, three initial states and a visual intervention."""
    return [dict(azimuth=45*i,state=state,intervention=condition,seed=1110000+i*3+j)
            for i in range(8) for j,state in enumerate(('resting','hovering','cruising'))
            for condition in ('intact','visual-silent')]


def correct_side(azimuth: float, side: str | None) -> bool:
    """Positive azimuth is predator on right: evasive yaw must be left."""
    if side not in ('left','right'): return False
    angle=azimuth%360
    if min(angle,360-angle,abs(angle-180))<=15: return True
    return side==('left' if math.sin(math.radians(angle))>0 else 'right')


def select_candidate(rows: list[dict]) -> tuple[int,int] | None:
    """No adaptive additions; choose minimum declared eligible gains."""
    if [tuple(r['gains']) for r in rows]!=list(GAIN_PAIRS):
        raise ValueError('exactly three candidates in predeclared order required')
    return next((tuple(r['gains']) for r in rows if r['eligible']),None)


def validation_gate(rows: list[dict]) -> bool:
    expected=cases('validation')
    fields=('azimuth','elevation','profile','seed')
    key=lambda r:tuple(r[f] for f in fields)
    return (len(rows)==96 and {key(r) for r in rows}=={key(r) for r in expected}
            and sum(r['success'] for r in rows)>=90
            and all(sum(r['success'] for r in rows if r['azimuth']==az)>=5
                    for az in {r['azimuth'] for r in expected}))


def closed_gate(rows: list[dict]) -> bool:
    key=lambda r:(r['azimuth'],r['state'],r['intervention'],r['seed'])
    intact=[r for r in rows if r['intervention']=='intact']
    return (len(rows)==48 and {key(r) for r in rows}=={key(r) for r in closed_cases()}
            and sum(r['success'] for r in intact)>=22
            and all(sum(r['success'] for r in intact if r['state']==s)>=7
                    for s in ('resting','hovering','cruising'))
            and all(r['triggers']==0 for r in rows if r['intervention']=='visual-silent'))


def history_audit() -> dict:
    """Read/hash prior protocol/config records; reuse is never called unseen.

    Nested recorded angle fields are audited even outside the retreat task.
    This is an angle audit, not proof of absence of earlier hand interactions.
    """
    files={}; angles=set()
    def visit(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('azimuth','azimuth_degrees') and isinstance(v,(int,float)):
                    angles.add(float(v)%360)
                visit(v)
        elif isinstance(value,list):
            for v in value: visit(v)
    for name in ('config.json','protocol.json'):
        for path in sorted(Path('experiments').rglob(name)):
            if 'turnflee' in path.as_posix(): continue
            files[path.as_posix()]=digest(path)
            visit(json.loads(path.read_text(encoding='utf-8-sig')))
    calibration={c['azimuth'] for c in cases('calibration')}
    validation={c['azimuth'] for c in cases('validation')}
    return dict(files=files,recorded_azimuths=sorted(angles),
                calibration_historical=calibration<=angles,
                validation_overlaps=sorted(validation&angles),
                validation_label='offset validation; no unseen claim',
                limit='angle fields in saved protocols/configs; manual placements not proven absent')
