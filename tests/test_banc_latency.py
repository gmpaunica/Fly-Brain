"""Delay alignment changes offline counting only, never controller stepping."""

import pytest
import torch

from flybrain.data.banc import load_banc_subset
from flybrain.neural.banc_network import BancDynamicsConfig, BancNetwork
from flybrain.training import banc_recovery as recovery


def test_alignment_uses_source_corridor_lengths_and_fixed_budget():
    assert hasattr(recovery, 'steering_delays'), 'source-derived latency measurement missing'
    graph = load_banc_subset(connectome_gain=1.)
    assert recovery.steering_delays(graph) == {'left': 4, 'right': 3}
    protocol = recovery.make_protocol(latency_aligned=True)
    assert protocol['measurement']['steering_delays_substeps'] == {'left': 4, 'right': 3}
    assert protocol['budget']['maximum_total_neural_substeps'] == 30000
    assert len(protocol['candidate_configs']) == 1
    assert protocol['candidate_configs'][0]['projection_dynamics'] == 'charge_preserving'


def test_aligned_windows_include_both_boundaries_and_exclude_adjacent_events():
    assert hasattr(recovery, 'aligned_steering_rates'), 'aligned event counter missing'
    events = torch.zeros((1500, 2), dtype=torch.int64)
    # One-based end steps: left505..1254, right504..1253, inclusive.
    events[[503, 504, 1253, 1254], 0] = 1
    events[[502, 503, 1252, 1253], 1] = 1
    assert recovery.aligned_steering_rates(events, {'left': 4, 'right': 3}) == pytest.approx([2/.75, 2/.75])


def test_real_pulse_keeps_original_counts_and_clock_alongside_aligned_rates():
    assert hasattr(recovery, 'steering_delays'), 'latency measurement missing'
    brain = BancNetwork(load_banc_subset(connectome_gain=1.), dynamics_config=BancDynamicsConfig(
        3., sensory_mode='phase_rate', projection_dynamics='charge_preserving'))
    case = next(c for c in recovery.pulse_cases() if c['name'] == '0.1-equal')
    result = recovery.measure_pulse(brain, case, endpoint_delays={'left': 4, 'right': 3})
    assert brain.population.step_count == 1500
    assert result['steady_counts']['steering_left'] == 100
    assert result['steady_counts']['steering_right'] == 101
    assert result['simultaneous_steering_rates_hz'] == pytest.approx([100/.75, 101/.75])
    assert result['steering_rates_hz'] == pytest.approx([101/.75, 100/.75])
    assert result['passed']
