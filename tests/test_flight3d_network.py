"""Delivered currents, fixed source accounting and simulation-time display."""
import numpy as np
import torch
from flybrain.flight3d.neural import FlightBrain
from test_flight3d_neural import fixture


def test_transmission_is_delayed_current_not_presynaptic_spike_count():
    manifest=fixture()
    manifest['edges'] += [{'pre':'1','post':'1','count':5},{'pre':'2','post':'1','count':5}]
    brain=FlightBrain(manifest)
    brain.previous[0]=1
    brain.advance_rates(np.zeros(4),1)
    totals=getattr(brain,'edge_transmissions',None)
    assert totals is not None, 'immutable fixed-edge-order transmission interval required'
    assert isinstance(totals,tuple)
    np.testing.assert_allclose(totals,(.9,0,0,0,0))
    assert brain.voltage[1] == .9
    brain.reset();brain.previous[0]=1;brain.silenced={'2'}
    brain.advance_rates(np.zeros(4),1)
    assert not any(brain.edge_transmissions)
    brain.reset(); assert not any(brain.edge_transmissions)


def test_group_membership_boundaries_and_all_source_edges_are_accounted():
    from flybrain.flight3d import network
    from flybrain.flight3d.anatomy import load_manifest
    manifest=load_manifest(); grouped=network.group_network(manifest)
    assert len(grouped.groups)==96
    members=[i for g in grouped.groups for i in g.members]
    assert sorted(members)==sorted(n['id'] for n in manifest['nodes'])
    lookup={n['id']:n for n in manifest['nodes']}
    for group in grouped.groups:
        assert len({(lookup[i]['metadata'].get('side'),lookup[i]['metadata'].get('region')) for i in group.members})==1
    assert sorted(i for b in grouped.bundles for i in b.edge_indices)==list(range(len(manifest['edges'])))
    assert grouped == network.group_network(manifest)


def test_transmission_filter_exact_decay_duplicate_pause_and_replay():
    from flybrain.flight3d import network
    trace=network.TransmissionTrace(2)
    trace.update(.02,(.02,0))
    value=trace.values.copy()
    np.testing.assert_allclose(value,[(1-np.exp(-.02/.3)),0])
    for _ in range(10): trace.update(.02,(.02,0))
    np.testing.assert_array_equal(trace.values,value)
    trace.update(.32,(0,0))
    np.testing.assert_allclose(trace.values,value*np.exp(-1))
    trace.update(0,(0,0)); assert not trace.values.any()
