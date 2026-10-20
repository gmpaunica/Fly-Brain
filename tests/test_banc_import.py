"""Behavior tests for the pinned BANC subset and its LIF connection."""

import copy
from importlib.resources import files
import json
import math

import pytest
import torch

from flybrain.data.banc import build_banc_topology, load_banc_subset
from flybrain.neural.banc_network import BancNetwork
from flybrain.world.sensing import SensoryObservation


def tiny_rows():
    nodes = [
        {'root_888': '11', 'role': 'sensory_left', 'side': 'left',
         'cell_type': 'ORN_VA2', 'proofread': 'TRUE'},
        {'root_888': '12', 'role': 'projection_left', 'side': 'left',
         'cell_type': 'VA2_adPN', 'proofread': 'TRUE'},
        {'root_888': '21', 'role': 'sensory_right', 'side': 'right',
         'cell_type': 'ORN_VA2', 'proofread': 'TRUE'},
    ]
    edges = [{'pre': '11', 'post': '12', 'count': 10, 'norm': 1.0,
              'post_count': 10, 'pre_count': 10}]
    return nodes, edges


def test_directed_count_maps_to_post_row_pre_column():
    nodes, edges = tiny_rows()
    graph = build_banc_topology(nodes, edges, connectome_gain=1.5)
    assert graph.root_ids == ('11', '12', '21')
    assert graph.weights[graph.index('12'), graph.index('11')].item() == 1.5
    assert graph.weights[graph.index('11'), graph.index('12')].item() == 0
    assert graph.edges[0].count == 10
    assert graph.edges[0].norm == 1.0
    assert graph.nodes[0].metadata['proofread'] == 'TRUE'
    graph.weights.fill_(99)
    assert graph.weights[graph.index('12'), graph.index('11')].item() == 1.5


def test_banc_network_drives_only_matching_orns_then_propagates_with_delay():
    nodes, edges = tiny_rows()
    brain = BancNetwork(build_banc_topology(nodes, edges, connectome_gain=1.0))
    current = brain.encode(SensoryObservation(0.025, 0))
    assert current.tolist() == [2.0, 0.0, 0.0]
    first = brain.advance(SensoryObservation(0.025, 0), substeps=20)
    assert first.spike_counts[brain.topology.index('11')] > 0
    assert first.spike_counts[brain.topology.index('12')] > 0
    assert first.spike_counts[brain.topology.index('21')] == 0
    assert first.end_step == 20
    brain.reset()
    replay = brain.advance(SensoryObservation(0.025, 0), substeps=20)
    assert torch.equal(first.spike_counts, replay.spike_counts)
    assert torch.equal(first.snapshot.voltage, replay.snapshot.voltage)


@pytest.mark.parametrize('change', [
    lambda n, e: n.append(copy.deepcopy(n[0])),
    lambda n, e: e.append(copy.deepcopy(e[0])),
    lambda n, e: e[0].update(post='99'),
    lambda n, e: e[0].update(norm=0.2),
    lambda n, e: e[0].update(count=4),
    lambda n, e: e[0].update(pre=11),
])
def test_malformed_or_unqualified_subset_is_rejected(change):
    nodes, edges = tiny_rows()
    change(nodes, edges)
    with pytest.raises(ValueError):
        build_banc_topology(nodes, edges, connectome_gain=1.0)


@pytest.mark.parametrize('gain', [0, -1, math.nan, math.inf])
def test_gain_must_be_finite_positive(gain):
    nodes, edges = tiny_rows()
    with pytest.raises(ValueError):
        build_banc_topology(nodes, edges, connectome_gain=gain)


def test_bundled_banc_subset_reproduces_source_audit_and_anchors():
    graph = load_banc_subset(connectome_gain=1.0)
    assert len(graph.nodes) == 83
    assert len(graph.edges) == 176
    assert sum(edge.count for edge in graph.edges) == 6062
    assert sum(edge.pre == edge.post for edge in graph.edges) == 31
    assert len(graph.indices('sensory_left')) == 31
    assert len(graph.indices('sensory_right')) == 37
    assert graph.weights[graph.index('720575941539370133'),
                         graph.index('720575941653217813')] == pytest.approx(0.012055)
    assert graph.weights[graph.index('720575941653217813'),
                         graph.index('720575941539370133')] == 0
    assert graph.source['materialization'] == '888'
    assert graph.source['license'] == 'CC BY 4.0'


def test_bundled_graph_runs_finitely_and_reset_replays():
    graph = load_banc_subset(connectome_gain=1.0)
    brain = BancNetwork(graph)
    observation = SensoryObservation(0.025, 0.0)
    first = brain.advance(observation, substeps=100)
    assert first.spike_counts[list(graph.indices('sensory_left'))].sum() > 0
    assert first.spike_counts[list(graph.indices('sensory_right'))].sum() == 0
    assert torch.isfinite(first.snapshot.voltage).all()
    assert first.spike_counts.max().item() <= 100
    brain.reset()
    replay = brain.advance(observation, substeps=100)
    assert torch.equal(first.spike_counts, replay.spike_counts)
    assert torch.equal(first.snapshot.voltage, replay.snapshot.voltage)


@pytest.mark.parametrize('field', ['source', 'nodes', 'edges'])
def test_loader_rejects_missing_document_sections_with_value_error(tmp_path, field):
    payload = json.loads(files('flybrain.data').joinpath(
        'banc_v888_va2_dna02.json').read_text(encoding='utf-8'))
    payload[field] = None
    path = tmp_path / 'corrupt.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        load_banc_subset(connectome_gain=1.0, path=path)


def test_loader_detects_changed_metadata_even_when_audit_totals_match(tmp_path):
    payload = json.loads(files('flybrain.data').joinpath(
        'banc_v888_va2_dna02.json').read_text(encoding='utf-8'))
    payload['nodes'][0]['neurotransmitter_verified'] = 'invented'
    path = tmp_path / 'edited.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError, match='SHA-256'):
        load_banc_subset(connectome_gain=1.0, path=path)


def test_edge_source_totals_must_be_consistent_for_shared_endpoint():
    nodes, edges = tiny_rows()
    edges.append({'pre': '21', 'post': '12', 'count': 5, 'norm': .25,
                  'post_count': 20, 'pre_count': 5})
    with pytest.raises(ValueError, match='inconsistent'):
        build_banc_topology(nodes, edges, connectome_gain=1)


@pytest.mark.parametrize('which', ['nodes', 'edges'])
def test_nonobject_rows_raise_value_error(which):
    nodes, edges = tiny_rows()
    (nodes if which == 'nodes' else edges).append(None)
    with pytest.raises(ValueError):
        build_banc_topology(nodes, edges, connectome_gain=1)
