"""Numerical controls and bounded recording for FB-08 experiments."""

from dataclasses import replace
import json

import pytest
import torch

from flybrain.neural.readout import TrainableMotorReadout
from flybrain.training.charge_repair import charge_controller_factory, save_charge_checkpoint
from flybrain.training.evaluation import EpisodeCase, EvaluationConfig, run_episode
from flybrain.training.fb08_experiments import (
    CONDITIONS, InterventionNetwork, controller_factory, diagnostic_cases,
    run_experiment,
)
from flybrain.world.sensing import SensoryObservation


def test_intact_equivalence_and_real_sensory_ablation():
    model = TrainableMotorReadout(9)
    with torch.no_grad():
        model.linear.weight[0, :7] = .1
        model.linear.weight[1, 7:] = torch.tensor([-4., 4.])
    case = EpisodeCase('toy', 'diagnostic', 900000, 450, 300, 0)
    config = EvaluationConfig(horizon_ticks=5)
    assert run_episode(case, config, controller_factory(model, 'intact')) == (
        run_episode(case, config, charge_controller_factory(model, 7)))
    odor = SensoryObservation(.008, .010)
    for name, silent in [('left_off', list(range(10))),
                         ('right_off', list(range(10, 20))),
                         ('both_off', list(range(20)))]:
        network = InterventionNetwork(name)
        assert torch.count_nonzero(network.encode(odor)[silent]) == 0
        activity = network.advance(odor)
        assert torch.count_nonzero(activity.spike_counts[silent]) == 0
        assert torch.count_nonzero(activity.integrated_charge[silent]) == 0
        # Tonic current needs more than 20 ms to fire from rest; charge is
        # already present on the first tick even while spike count is zero.
        assert torch.allclose(activity.integrated_charge[80:90],
                              torch.full((10,), 1.3, dtype=torch.float64))
        if name != 'both_off':
            active = slice(10, 20) if name == 'left_off' else slice(0, 10)
            assert (activity.integrated_charge[active] > 0).all()


def test_weight_interventions_preserve_untargeted_values_rng_and_reset():
    torch.manual_seed(123)
    rng = torch.random.get_rng_state().clone()
    original = InterventionNetwork('intact').topology.weights
    for name, gain in [('pathway_half', .5), ('pathway_double', 2.)]:
        expected = original.clone()
        expected[20:70, :20] *= gain
        net = InterventionNetwork(name)
        assert torch.equal(net.population.weights, expected)
        assert torch.equal(net.topology.weights, expected)
    assert not InterventionNetwork('synapses_off').population.weights.any()
    fingerprints = []
    for name in ('rewired_701', 'rewired_702', 'rewired_703'):
        net = InterventionNetwork(name)
        weights = net.topology.weights
        assert torch.equal(weights.sort(dim=1).values, original.sort(dim=1).values)
        assert not weights[:20].any()
        assert not weights[20:70, 20:].any()
        assert not weights[70:, :20].any()
        assert not weights[:, 70:].any()
        assert not torch.equal(weights, original)
        assert torch.equal(weights, InterventionNetwork(name).population.weights)
        first = net.advance(SensoryObservation(.008, .01))
        net.reset()
        again = net.advance(SensoryObservation(.008, .01))
        assert torch.equal(first.spike_counts, again.spike_counts)
        assert torch.equal(first.snapshot.voltage, again.snapshot.voltage)
        fingerprints.append(weights.numpy().tobytes())
    assert len(set(fingerprints)) == 3
    assert torch.equal(rng, torch.random.get_rng_state())
    with pytest.raises(ValueError):
        InterventionNetwork('unknown')


def test_cases_are_new_deterministic_and_bounded():
    cases = diagnostic_cases()
    assert cases == diagnostic_cases() and len(cases) == 16
    assert {c.seed for c in cases} == set(range(51000, 51016))
    assert all(c.split == 'diagnostic' and 60 <= ((c.x-600)**2+(c.y-300)**2)**.5 <= 180
               for c in cases)


@pytest.mark.parametrize('near_food', [False, True])
def test_runner_records_or_stops_without_mutating_checkpoint(tmp_path, near_food):
    model = TrainableMotorReadout(9)
    with torch.no_grad():
        model.linear.bias[0] = 4
    checkpoint = tmp_path / 'trained.pt'
    save_charge_checkpoint(checkpoint, model, network_seed=7)
    original = checkpoint.read_bytes()
    cases = tuple(replace(c, x=578.5 if near_food else 400., y=300., heading=0.)
                  for c in diagnostic_cases())
    output = tmp_path / 'run'
    path = run_experiment(output, checkpoint=checkpoint,
                          config=EvaluationConfig(horizon_ticks=1), cases=cases,
                          make_plots=False)
    result = json.loads(path.read_text())
    assert result['optimizer_steps'] == 0
    assert result['checks']['checkpoint_unchanged']
    assert checkpoint.read_bytes() == original
    assert result['assay_passed'] is near_food
    assert set(result['results']) == (set(CONDITIONS) if near_food else {'intact'})
    assert result['actual_episodes'] == (160 if near_food else 16)
    assert all(len(rows) == 16 for rows in result['results'].values())
    assert (output / 'protocol.json').exists()
    with pytest.raises(FileExistsError):
        run_experiment(output, checkpoint=checkpoint, make_plots=False)
