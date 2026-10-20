"""Saved-only scientific delivery tables; no brain/world stepping."""
import csv,gzip,json
from pathlib import Path
from flybrain.escape_flight_validation.runner import RUN
from flybrain.escape_flight_validation.acceptance import ART
from audit_serialized import audit
from flybrain.escape_flight_validation.visuals import render
from flybrain.escape_diagnostic_recovery.runner import write,verify_hashes
from flybrain.flight3d.anatomy import digest
checked=audit(RUN);write(ART/'audit.json',checked)
result=json.loads((RUN/'results.json').read_text());config=json.loads((RUN/'config.json').read_text())
write(ART/('prototype-acceptance-pending.json' if result['status']=='passed' else 'prototype-acceptance.json'),dict(schema='escape-prototype-acceptance-v1',candidate_identity=config['candidate_identity'],
    exception_sha256=config['exception_sha256'],directional_receipt_sha256=config['exception']['directional_receipt_sha256'],
    physical_receipt_sha256=digest(RUN/'receipt.json'),original_directional_status='closed-direction-failed',
    directional_successes=94,known_failures=[1100018,1100021],physical_status=result['status'],
    prototype_accepted=False,activation='disabled: physical gate failed; experiment closed',learning=False))
fields=['azimuth','state','intervention','seed','success','timely_correct_side','side','first_strong_seconds','trigger_seconds',
    'measured_turn','measured_takeoff','separation_gain_m','collision','measured_braking','weak_reached','recovered','stable_recovery',
    'triggers','turn_seconds','first_flee_seconds','takeoff_seconds','weak_seconds','braking_seconds','max_stable_recovery_intervals','max_downstream_hz']
with (ART/'physical-cases.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for row in result['closed_loop']:writer.writerow({k:row[k] for k in fields})
first_collisions={}
with gzip.open(RUN/'traces.jsonl.gz','rt') as f:
    for line in f:
        row=json.loads(line);key=f'{row["seed"]}-{row["state"]}-{row["intervention"]}'
        if row['collided'] and key not in first_collisions:
            first_collisions[key]={k:row[k] for k in ('end_seconds','pre_position','position','velocity','orientation','escape_state','behavior_state','command')}
criteria=[('timely_correct_side',lambda r:r['timely_correct_side']),('actual_turn',lambda r:r['measured_turn']),
    ('takeoff',lambda r:r['measured_takeoff']),('separation',lambda r:r['separation_gain_m']>=.02-1e-12),
    ('collision_freedom',lambda r:not r['collision']),('weak_threat',lambda r:r['weak_reached']),
    ('braking',lambda r:r['measured_braking']),('recovery',lambda r:r['recovered']),('stable_recovery',lambda r:r['stable_recovery'])]
failed=[dict(case={k:r[k] for k in ('azimuth','state','intervention','seed')},unmet=[name for name,check in criteria if not check(r)])
    for r in result['closed_loop'] if r['intervention']=='intact' and not r['success']]
write(ART/'analysis.json',dict(status=result['status'],intact_successes=result['intact_successes'],per_state=result['per_state'],
    silent_triggers=result['silent_triggers'],failed_intact=failed,first_collisions=first_collisions,
    max_downstream_hz=max(r['max_downstream_hz'] for r in result['closed_loop']),analysis_neural_steps=0))
paths=render(RUN,ART/'flight-visuals')
verify_hashes(config['protected'])
print(json.dumps(dict(audit=checked,failed_intact=failed,figures=[str(p) for p in paths],protected_files=len(config['protected'])),indent=2))
