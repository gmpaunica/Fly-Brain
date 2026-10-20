"""Numerical contracts for the engineered BANC recovery, without a gain sweep."""

import importlib
import json
import math
import hashlib
from pathlib import Path
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from flybrain.data import banc
from flybrain.neural import banc_network
from flybrain.neural.network import NeuralActivity, NeuralSnapshot
from flybrain.world.sensing import SensoryObservation


def config(budget=2.0):
    assert hasattr(banc_network, 'BancDynamicsConfig'), 'recovery config is missing'
    return banc_network.BancDynamicsConfig(synaptic_budget=budget)


def test_normalization_preserves_direction_cross_edges_and_source():
    source = banc.load_banc_subset(connectome_gain=1)
    raw = source.weights
    brain = banc_network.BancNetwork(source, dynamics_config=config())
    weights = brain.topology.weights
    assert len(brain.source_topology.edges) == 176
    assert len(brain.topology.edges) == 145
    assert torch.count_nonzero(weights.diag()) == 0
    assert torch.equal(source.weights, raw)
    assert torch.equal(weights != 0, (raw - torch.diag(raw.diag())) != 0)
    row_sums = weights.sum(1)
    assert torch.allclose(row_sums[row_sums != 0], torch.full_like(row_sums[row_sums != 0], 2.0))
    # The original 741 left-ORN synapses onto the right PN remain present.
    pn = source.index('720575941497888384')
    assert weights[pn, list(source.indices('sensory_left'))].sum().item() == pytest.approx(2 * 741 / 2166)
    # Single-input DNa02 rows are explicit engineered amplification, not source norm.
    assert weights[source.index('720575941510475536'), source.index('720575941500284745')].item() == 2


def test_multiscale_encoder_maps_sorted_ids_independently_without_saturation_overflow():
    source = banc.load_banc_subset(connectome_gain=1)
    brain = banc_network.BancNetwork(source, dynamics_config=config())
    encoded = brain.encode(SensoryObservation(.005, 0))
    left = source.indices('sensory_left')
    assert encoded[left[0]].item() == pytest.approx(1.25)
    assert encoded[left[-1]].item() == pytest.approx(2.5 / 11)
    assert torch.all(encoded[list(source.indices('sensory_right'))] == 0)
    nonsensory = [i for i, role in enumerate(source.groups) if not role.startswith('sensory_')]
    assert torch.all(encoded[nonsensory] == 0)
    huge = brain.encode(SensoryObservation(1.7e308, 1.7e308))
    assert torch.isfinite(huge).all()
    assert huge.max().item() == 2.5
    for observation in (SimpleNamespace(odor_left=-1, odor_right=0),
                        SimpleNamespace(odor_left=math.nan, odor_right=0)):
        with pytest.raises(ValueError):
            brain.encode(observation)


def test_wide_range_encoder_spans_high_band_threshold_without_changing_graph():
    source = banc.load_banc_subset(connectome_gain=1)
    wide = banc_network.BancNetwork(source, dynamics_config=banc_network.BancDynamicsConfig(
        3., encoder_max_scale=.375))
    old = banc_network.BancNetwork(source, dynamics_config=config(3.))
    assert torch.equal(wide.topology.weights, old.topology.weights)
    current = wide.encode(SensoryObservation(.25, .3))
    assert current[source.indices('sensory_left')[-1]].item() == pytest.approx(1.)
    assert current[source.indices('sensory_right')[-1]].item() == pytest.approx(10/9)
    assert current[source.indices('sensory_left')[0]].item() == pytest.approx(2.5*.25/.255)
    assert not wide.encode(SensoryObservation(0., 0.)).any()


@pytest.mark.parametrize('budget', [0, -1, math.nan, math.inf, True])
def test_invalid_budget_is_rejected(budget):
    assert hasattr(banc_network, 'BancDynamicsConfig'), 'recovery config is missing'
    with pytest.raises(ValueError):
        banc_network.BancDynamicsConfig(synaptic_budget=budget)


def activity(topology, counts, elapsed=.02, start=0):
    count_tensor = torch.tensor(counts, dtype=torch.int64)
    snapshot = NeuralSnapshot(start + 20, torch.zeros(len(counts)), torch.zeros(len(counts), dtype=torch.bool))
    return NeuralActivity(start, start + 20, elapsed, count_tensor, snapshot)


def test_endpoint_decoder_observes_only_eight_spike_rates_and_resets():
    assert importlib.util.find_spec('flybrain.neural.banc_motor'), 'BANC motor module is missing'
    decoder_cls = importlib.import_module('flybrain.neural.banc_motor').BancMotorDecoder
    topology = banc.load_banc_subset(connectome_gain=1)
    decoder = decoder_cls(topology)
    counts = [0] * 83
    for index in topology.indices('sensory_left'):
        counts[index] = 20
    silent = decoder.decode(activity(topology, counts))
    assert silent.forward == silent.turn == 0
    counts = [0] * 83
    counts[topology.indices('steering_left')[0]] = 2
    for index in topology.indices('motor_left'):
        counts[index] = 2
    command = decoder.decode(activity(topology, counts, start=20))
    # 100Hz for one 20ms interval with tau100ms -> 18.1269247Hz.
    assert decoder.feature_rates[0].item() == pytest.approx(18.1269247)
    assert command.turn == pytest.approx(-18.1269247 / 28.1269247)
    assert command.forward == pytest.approx(9.06346235 / 29.06346235)
    assert len(decoder.feature_names) == 8
    before = decoder.feature_rates
    decoder.feature_rates.fill_(999)
    assert torch.equal(decoder.feature_rates, before)
    decoder.reset()
    assert not decoder.feature_rates.any()


def test_decoder_rejects_bad_intervals_atomically():
    assert importlib.util.find_spec('flybrain.neural.banc_motor'), 'BANC motor module is missing'
    decoder = importlib.import_module('flybrain.neural.banc_motor').BancMotorDecoder(banc.load_banc_subset(connectome_gain=1))
    topology = banc.load_banc_subset(connectome_gain=1)
    for elapsed in (0, math.nan, -.02):
        with pytest.raises(ValueError):
            decoder.decode(activity(topology, [1] * 83, elapsed))
    with pytest.raises(ValueError):
        decoder.decode(activity(topology, [21] * 83))
    assert not decoder.feature_rates.any()


def test_controller_replays_all_neural_and_decoder_state():
    assert importlib.util.find_spec('flybrain.neural.banc_motor'), 'BANC motor module is missing'
    controller = importlib.import_module('flybrain.neural.banc_motor').make_banc_controller(config())
    obs = SensoryObservation(.1, .025)
    first = [controller.decide(obs) for _ in range(5)]
    controller.reset()
    second = [controller.decide(obs) for _ in range(5)]
    assert [x.command for x in first] == [x.command for x in second]
    assert all(torch.equal(a.activity.spike_counts, b.activity.spike_counts) for a, b in zip(first, second))


def test_completed_run_refuses_reexecution_before_scientific_work(tmp_path):
    assert importlib.util.find_spec('flybrain.training.banc_recovery'), 'recovery runner is missing'
    path = tmp_path / 'prior'
    path.mkdir()
    sentinel = path / 'results.json'
    sentinel.write_text('{"prior": true}', encoding='utf-8')
    with pytest.raises(FileExistsError):
        importlib.import_module('flybrain.training.banc_recovery').run_recovery(path)
    assert sentinel.read_text(encoding='utf-8') == '{"prior": true}'


def test_unverified_config_cannot_be_loaded_as_an_app_default(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'config': {'synaptic_budget': 2}}), encoding='utf-8')
    with pytest.raises(ValueError, match='not verified'):
        banc_network.load_verified_banc_config(path)


def test_contrast_gate_rejects_flat_and_reversed_endpoints():
    assert importlib.util.find_spec('flybrain.training.banc_recovery'), 'recovery runner is missing'
    assess = importlib.import_module('flybrain.training.banc_recovery').contrast_gate
    # Baseline source-side bias can be nonzero. Responses must move in both
    # appropriate directions relative to equal odor, by >=1Hz and >=2%.
    assert assess((10, 12), (8, 14), (13, 9))['passed']
    assert not assess((10, 12), (10, 12), (10, 12))['passed']
    assert not assess((10, 12), (13, 9), (8, 14))['passed']


def test_banc_controller_keeps_cpu_contract_under_ambient_meta_device():
    factory = importlib.import_module('flybrain.neural.banc_motor').make_banc_controller
    with torch.device('meta'):
        controller = factory(config())
        decision = controller.decide(SensoryObservation(.1, .025))
    assert decision.activity.spike_counts.device.type == 'cpu'
    assert controller.decoder.feature_rates.device.type == 'cpu'


def test_receipt_cannot_promote_the_actual_failed_experiment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    evidence = (root / 'experiments/fb09-dynamics-recovery/results.json').read_bytes()
    (tmp_path / 'banc_dynamics_results.json').write_bytes(evidence)
    (tmp_path / 'banc_dynamics_protocol.json').write_bytes(
        (root / 'experiments/fb09-dynamics-recovery/protocol.json').read_bytes())
    receipt = {
        'schema_version': 1, 'subset_sha256': banc.BANC_SUBSET_SHA256,
        'transform_version': banc.BANC_TRANSFORM_VERSION,
        'dynamics_gate_passed': True, 'causal_gate_passed': True,
        'probe_results_sha256': hashlib.sha256(evidence).hexdigest(),
        'config': {'synaptic_budget': 2},
    }
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError, match='evidence'):
        banc_network.load_verified_banc_config(path)


@pytest.mark.parametrize('mutation', [None, 'hash', 'config', 'source', 'missing'])
def test_receipt_requires_consistent_evidence_and_current_code(tmp_path, mutation):
    # Synthetic file-format fixture only; this does not simulate or assert a
    # scientific pass for any real candidate.
    from flybrain.training.banc_recovery import make_protocol
    protocol = make_protocol()
    if mutation == 'source':
        protocol['source_code_sha256']['src/flybrain/neural/lif.py'] = '0' * 64
    protocol_bytes = json.dumps(protocol).encode()
    result = {
        'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
        'subset_sha256': banc.BANC_SUBSET_SHA256,
        'transform_version': banc.BANC_TRANSFORM_VERSION,
        'passed': True, 'dynamics_gate_passed': True, 'causal_gate_passed': True,
        'previous_files_preserved': True, 'source_code_unchanged': True,
        'selected_config': asdict(config()), 'optimizer_steps': 0,
    }
    result_bytes = json.dumps(result).encode()
    receipt = {
        'schema_version': 1, 'subset_sha256': banc.BANC_SUBSET_SHA256,
        'transform_version': banc.BANC_TRANSFORM_VERSION,
        'dynamics_gate_passed': True, 'causal_gate_passed': True,
        'probe_results_sha256': hashlib.sha256(result_bytes).hexdigest(),
        'config': asdict(config(3 if mutation == 'config' else 2)),
    }
    if mutation == 'hash':
        receipt['probe_results_sha256'] = '0' * 64
    if mutation != 'missing':
        (tmp_path / 'banc_dynamics_results.json').write_bytes(result_bytes)
    (tmp_path / 'banc_dynamics_protocol.json').write_bytes(protocol_bytes)
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(receipt), encoding='utf-8')
    if mutation is None:
        assert banc_network.load_verified_banc_config(path) == config()
    else:
        with pytest.raises(ValueError, match='evidence'):
            banc_network.load_verified_banc_config(path)
