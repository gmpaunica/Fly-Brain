"""Numerical escape pathway and engineered visual input tests, no optimizer."""
import math
import numpy as np
import torch
from flybrain.flight3d.contracts import FlightObservation
from flybrain.flight3d.anatomy import load_manifest
from flybrain.encounters3d.neural import EscapeBrain
from flybrain.encounters3d.scene import SceneSpec, EncounterWorld


def observation(eyes):
    return FlightObservation(0,0,eyes,(0,0,0),(0,0,9.81),False,False)


def test_escape_encoder_only_drives_existing_visual_cells_growth_and_reset():
    b=EscapeBrain(load_manifest())
    far=EncounterWorld(SceneSpec(3,'predator',(.3,.45,.1))).observe()
    near=EncounterWorld(SceneSpec(3,'predator',(.3,.37,.1))).observe()
    rates=b.sensory_rates(far)
    assert rates.max()>0
    assert set(np.flatnonzero(rates))<=set(b.roles['vision'])
    growing=b.sensory_rates(near)
    stable=b.sensory_rates(near)
    assert growing.sum()>stable.sum()>rates.sum()
    b.advance(near)
    b.reset()
    assert b.previous_eyes is None and b.steps==0
    assert not b.voltage.any() and not b.previous.any() and not b.phase.any()
    assert not b.history.any() and not b.last_counts.any()
    flower=EncounterWorld(SceneSpec(3,'flower',(.3,.4,.1))).observe()
    assert not b.sensory_rates(flower).any()


def test_actual_dnp03_leak_threshold_subtractive_reset_and_one_ms_delay():
    b=EscapeBrain(load_manifest())
    d=next(i for i,n in enumerate(b.nodes) if n['metadata'].get('cell_type')=='DNp03')
    zero=np.zeros(len(b.ids))
    b.voltage[d]=.5
    b.advance_rates(zero,steps=1)
    assert math.isclose(float(b.voltage[d]),.5*math.exp(-.001/.1),rel_tol=1e-13)
    b.reset(); b.voltage[d]=1.1
    counts=b.advance_rates(zero,steps=1)
    assert counts[d]==1
    assert math.isclose(float(b.voltage[d]),1.1*math.exp(-.01)-1,abs_tol=1e-13)
    # Actual LPLC4 -> DNp03 source edge and delayed current, with no other drive.
    edge=next(k for k,(pre,post) in enumerate(zip(b.pre,b.post))
              if int(post)==d and 'vision' in b.nodes[int(pre)]['roles'])
    pre=int(b.pre[edge]); weight=float(b.weights[edge])
    b.reset(); b.phase[pre]=.99
    rates=zero.copy(); rates[pre]=240
    b.advance_rates(rates,steps=1)
    assert b.previous[pre]==1 and b.voltage[d]==0
    b.advance_rates(zero,steps=1)
    assert math.isclose(float(b.voltage[d]),weight,rel_tol=1e-13)
    # Post never injects into this upstream input cell.
    b.reset(); b.previous[d]=1
    b.advance_rates(zero,steps=1)
    assert not b.voltage[pre] and not b.last_counts[pre]
    endpoint=next(k for k,(source,post) in enumerate(zip(b.pre,b.post))
                  if int(source)==d and set(b.nodes[int(post)]['roles']) & {'wing_power','wing_steering'})
    post=int(b.post[endpoint])
    assert math.isclose(float(b.voltage[post]),float(b.weights[endpoint]),rel_tol=1e-13)


def test_saturated_visual_pathway_and_matched_interventions():
    b=EscapeBrain(load_manifest())
    rates=np.zeros(len(b.ids)); rates[b.roles['vision']]=240
    dn={n['id'] for n in b.nodes if n['metadata'].get('cell_type')=='DNp03'}
    motor=np.array(sorted(set(b.roles['wing_power'])|set(b.roles['wing_steering'])))
    intact=b.advance_rates(rates,steps=1000)
    b.reset(); b.silenced=dn
    cut=b.advance_rates(rates,steps=1000)
    assert sum(intact[b.index[i]] for i in dn)>0
    assert not any(cut[b.index[i]] for i in dn)
    assert intact[motor].sum()>cut[motor].sum()>0  # DNp03 contributes; not sole route.
    b.reset(); b.silenced={b.ids[i] for i in b.roles['vision']}
    silent=b.advance_rates(rates,steps=1000)
    assert not silent.any()
