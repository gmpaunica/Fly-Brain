"""Observation-only tracing must preserve the actual phase-rate transition."""

import importlib.util

import pytest
import torch

from flybrain.data.banc import load_banc_subset, normalize_banc_topology
from flybrain.neural.banc_network import BancDynamicsConfig, BancNetwork
from flybrain.world.sensing import SensoryObservation


def diagnostic():
    assert importlib.util.find_spec('flybrain.training.banc_relay_diagnosis'), 'relay observer missing'
    from flybrain.training import banc_relay_diagnosis
    return banc_relay_diagnosis


def test_observation_keeps_phase_rate_state_and_actual_charge_accounting():
    observe_step = diagnostic().observe_step
    config = BancDynamicsConfig(3., sensory_mode='phase_rate', projection_dynamics='charge_preserving')
    graph = load_banc_subset(connectome_gain=1.)
    brain = BancNetwork(graph, dynamics_config=config)
    reference = BancNetwork(graph, dynamics_config=config)
    brain.population.record_charge = True
    for _ in range(60):
        observation = SensoryObservation(.1, .12)
        rows = observe_step(brain, observation)
        reference.advance(observation, 1)
        assert torch.equal(brain.population.voltage, reference.population.voltage)
        assert torch.equal(brain.population.spikes, reference.population.spikes)
        assert torch.equal(brain.encode(observation), reference.encode(observation))
        assert len(rows) == 15
        assert max(abs(row['balance_error']) for row in rows) < 1e-6
        for row in rows:
            if row['spike'] and row['role'].startswith('projection_'):
                assert row['reset_loss'] == pytest.approx(1.1, abs=2e-7)
    assert brain.population.step_count == 60


def test_observer_rejects_missing_charge_recording_without_advancing():
    observe_step = diagnostic().observe_step
    brain = BancNetwork(load_banc_subset(connectome_gain=1.))
    with pytest.raises(ValueError, match='recording'):
        observe_step(brain, SensoryObservation(.1, .1))
    assert brain.population.step_count == 0


def test_relay_accounting_distinguishes_boundary_delay_from_lost_spikes():
    account = diagnostic().relay_accounting
    result = account([498, 600, 1248], [502, 604, 1252], 4)
    assert result['exact_delayed_replay']
    assert result['parent_steady_count'] == result['child_steady_count'] == 2
    assert result['entering_window_parent_steps'] == [498]
    assert result['leaving_window_parent_steps'] == [1248]
    assert result['unmatched_expected_child_steps'] == []
    missing = account([498, 600, 1248], [502, 1252], 4)
    assert not missing['exact_delayed_replay']
    assert missing['unmatched_expected_child_steps'] == [604]
    assert not account([500], [501, 700], 1)['exact_delayed_replay']


def test_paths_use_actual_source_corridors_and_existing_runs_are_protected(tmp_path):
    module = diagnostic()
    graph = normalize_banc_topology(load_banc_subset(connectome_gain=1.), 3.)
    assert len(module.projection_path(graph, 'left')) - 1 == 4
    assert len(module.projection_path(graph, 'right')) - 1 == 3
    protected = tmp_path / 'existing'
    protected.mkdir()
    marker = protected / 'results.json'
    marker.write_text('protected', encoding='utf-8')
    with pytest.raises(FileExistsError):
        module.run_diagnosis(protected)
    assert marker.read_text(encoding='utf-8') == 'protected'
