"""Catch unauthorized weight changes, lost parity and window/reset errors."""
import importlib
from dataclasses import FrozenInstanceError
import numpy as np
import pytest
import torch
from flybrain.flight3d.anatomy import load_manifest
from flybrain.flight3d.contracts import FlightObservation
from flybrain.retreat3d.neural import BrainPair


def api():
    assert importlib.util.find_spec('flybrain.turnflee'), 'additive response package required'
    assert importlib.util.find_spec('flybrain.turnflee.neural'), 'new gain-bound brain required'
    return importlib.import_module('flybrain.turnflee.neural')


def observation():
    return FlightObservation(0., 0., np.zeros((2,24,48,3)), (.2,-.3,.4), (1.,2.,9.81), False, False)


def test_gain_one_bit_parity_motion_baseline_and_reset():
    m = api(); manifest = load_manifest()
    candidate = m.ResponsePair(manifest, m.ResponseConfig.create(manifest, (1,1)))
    old = BrainPair(manifest)
    for _ in range(12):
        vector = candidate.advance(observation()); old.advance(observation())
        for name in ('voltage','previous','phase','history'):
            assert torch.equal(getattr(candidate.actual,name), getattr(old.actual,name))
        np.testing.assert_array_equal(candidate.actual.last_counts, candidate.reference.last_counts)
        assert candidate.actual.edge_transmissions == old.actual.edge_transmissions
        assert vector.shape == (48,) and not vector.any()
    assert len(candidate.window) == 10
    candidate.reset()
    assert not candidate.window and not candidate.last_vector.any()
    assert candidate.actual.steps == candidate.reference.steps == 0
    assert candidate.actual.diagnostic is None and candidate.max_downstream_hz == 0
    for brain in (candidate.actual,candidate.reference):
        for field in ('voltage','previous','phase','history'):
            assert not getattr(brain,field).any()
        assert brain.previous_eyes is None and not any(brain.edge_transmissions)


def test_exact_edge_gain_mask_and_original_anatomy():
    m = api(); manifest = load_manifest(); original = BrainPair(manifest)
    config = m.ResponseConfig.create(manifest, (4,2))
    pair = m.ResponsePair(manifest, config)
    nodes = {n['id']:n['metadata']['cell_type'] for n in manifest['nodes']}
    for j, edge_index in enumerate(original.actual.dynamic_edges):
        pre, post = original.actual.edge_order[edge_index]
        gain = 4 if nodes[pre]=='LPLC4' and nodes[post] in ('DNp03','DNp26') else 2 if nodes[pre] in ('DNp03','DNp26') else 1
        assert pair.actual.weights[j] == original.actual.weights[j]*gain
    assert pair.actual.ids == original.actual.ids and pair.actual.edge_order == original.actual.edge_order
    assert pair.actual.manifest == manifest
    assert torch.equal(pair.actual.weights,pair.reference.weights)
    assert config.incoming_edges and config.outgoing_edges
    with pytest.raises(FrozenInstanceError): config.gains = (8,4)
    with pytest.raises(ValueError): m.ResponseConfig.create(manifest,(16,8))
    for _ in range(10):
        np.testing.assert_array_equal(pair.advance(observation()),np.zeros(48))
        assert torch.equal(pair.actual.voltage,pair.reference.voltage)


def test_rolling_window_expires_and_downstream_guard():
    m=api(); pair=m.ResponsePair(load_manifest(),m.ResponseConfig.create(load_manifest(),(1,1)))
    a=np.zeros(589,dtype=np.int64); b=a.copy()
    a[pair.indices[0]]=1
    pair.record_counts(a,b)
    assert pair.last_vector[0]==1
    for _ in range(9): pair.record_counts(b,b)
    assert pair.last_vector[0]==1
    pair.record_counts(b,b)
    assert pair.last_vector[0]==0
    index=int(np.flatnonzero(~pair.actual.inputs.numpy())[0]); a[:]=0; a[index]=5
    for _ in range(10): pair.record_counts(a,b)
    assert pair.max_downstream_hz==250 and not pair.stable


def test_directed_one_ms_delay_and_unchanged_threshold():
    m=api(); config=m.ResponseConfig.create(load_manifest(),(8,4)); p=m.ResponsePair(load_manifest(),config)
    # Existing DN→motor edge with largest effective weight. A spike is delivered
    # on the next millisecond, never when it first crosses threshold.
    edges=[j for j,k in enumerate(p.actual.dynamic_edges) if p.actual.edge_order[k] in config.outgoing_edges]
    j=max(edges,key=lambda j:float(p.actual.weights[j])); pre=int(p.actual.pre[j]); post=int(p.actual.post[j])
    p.actual.previous[pre]=1
    rates=np.zeros(589); weight=float(p.actual.weights[j])
    p.actual.advance_rates(rates,1)
    assert float(p.actual.voltage[post])+int(p.actual.last_counts[post])==pytest.approx(weight)
    assert p.actual.steps==1


def test_calibration_receipt_is_checked_before_brain_construction(tmp_path):
    m=api(); manifest=load_manifest(); path=tmp_path/'calibration.json'
    import json
    from flybrain.flight3d.anatomy import digest
    path.write_text(json.dumps([dict(gains=[2,1],eligible=True),dict(gains=[4,2],eligible=True),dict(gains=[8,4],eligible=False)]))
    config=m.ResponseConfig.create(manifest,(2,1),f'{path.as_posix()}#{digest(path)}')
    m.ResponsePair(manifest,config)
    wrong=m.ResponseConfig.create(manifest,(4,2),f'{path.as_posix()}#{digest(path)}')
    with pytest.raises(ValueError,match='lowest eligible'): m.ResponsePair(manifest,wrong)
    path.write_text('[]')
    with pytest.raises(ValueError,match='receipt mismatch'): m.ResponsePair(manifest,config)


def test_visual_silent_preserves_motion_spikes_and_cannot_trigger():
    m=api(); manifest=load_manifest(); p=m.ResponsePair(manifest,m.ResponseConfig.create(manifest,(8,4)))
    from flybrain.turnflee.control import WingScores,EscapePolicy
    from flybrain.retreat3d.threat import ThreatMonitor
    from flybrain.retreat3d.controller import KinematicEstimate
    p.actual.silenced={p.actual.ids[i] for i in p.actual.roles['vision']}
    monitor=ThreatMonitor(); scores=WingScores(manifest); policy=EscapePolicy(1150015)
    estimate=KinematicEstimate()
    obs=FlightObservation(0.,0.,np.broadcast_to([.05,.95,.95],(2,24,48,3)),(0.,0.,0.),(0.,0.,9.81),True,False)
    for _ in range(15):
        vector=p.advance(obs)
        np.testing.assert_array_equal(p.actual.last_counts,p.reference.last_counts)
        assert policy.command(estimate,obs,scores(vector),monitor.update(obs.eyes)) is None
    assert policy.trigger_count==0 and p.actual.last_counts.sum()>0
