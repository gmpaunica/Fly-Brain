"""Neural charge preserves sensory ordering across membrane resets."""

from dataclasses import replace
import math
import random

import pytest
import torch

from flybrain.neural.lif import LIFConfig, LIFPopulation
from flybrain.neural.network import SyntheticNetwork
from flybrain.neural.readout import SpikeFeatureFilter, TrainableMotorReadout
from flybrain.world.sensing import SensoryObservation


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_charge_recovers_drive_through_spikes_decay_and_episode_reset(dtype):
    measured = SyntheticNetwork(dtype=dtype, encoder_mode='rational',
                                encoder_scale=0.005, record_charge=True)
    original = SyntheticNetwork(dtype=dtype, encoder_mode='rational',
                                encoder_scale=0.005)
    sequence = [(0.008, 0.010), (0.008, 0.010), (0.15, 0.18), (0, 0)]
    first = None
    for values in sequence:
        observation = SensoryObservation(*values)
        activity = measured.advance(observation)
        reference = original.advance(observation)
        assert torch.equal(activity.snapshot.voltage, reference.snapshot.voltage)
        assert torch.equal(activity.snapshot.spikes, reference.snapshot.spikes)
        assert torch.equal(activity.spike_counts, reference.spike_counts)
        assert reference.integrated_charge is None
        charge = activity.integrated_charge
        assert charge.dtype == torch.float64 and charge.shape == (100,)
        assert charge.device.type == 'cpu' and not charge.requires_grad
        expected = measured.encode(observation)[:20].double()
        torch.testing.assert_close(charge[:20], expected, rtol=2e-7, atol=2e-7)
        if values == (0, 0):
            assert torch.equal(charge[:20], torch.zeros(20, dtype=torch.float64))
            assert activity.snapshot.voltage[:20].any()
        if first is None:
            first = charge.clone()
    measured.reset()
    replay = measured.advance(SensoryObservation(*sequence[0]))
    assert torch.equal(replay.integrated_charge, first)


def test_charge_ledger_retains_threshold_overshoot():
    population = LIFPopulation(torch.zeros((1, 1)),
                               LIFConfig(dt=math.log(2), tau=1), record_charge=True)
    population.step(torch.tensor([3.0]))
    losses = population.charge_losses
    assert population.spikes.item() and population.voltage.item() == 0
    assert losses.leak.item() == 0
    assert losses.reset.item() == pytest.approx(1.5)
    losses.reset.zero_()
    assert population.charge_losses.reset.item() == pytest.approx(1.5)
    population.reset()
    assert population.charge_losses is None


def test_charge_features_remove_reset_phase_reversal_and_keep_spike_ema():
    from flybrain.neural.charge_readout import ChargeFeatureFilter

    network = SyntheticNetwork(encoder_mode='rational', encoder_scale=0.005,
                               record_charge=True)
    feature_filter = ChargeFeatureFilter(network.topology)
    spike_filter = SpikeFeatureFilter(network.topology)
    for _ in range(3):
        activity = network.advance(SensoryObservation(0.008, 0.010))
        assert activity.snapshot.voltage[10] < activity.snapshot.voltage[0]
        features = feature_filter.update(activity)
        assert torch.equal(features[:7], spike_filter.update(activity))
        assert features[8] > features[7]
        assert features[-2:].tolist() == pytest.approx(
            [math.log(0.008 / 0.005), math.log(0.010 / 0.005)], abs=2e-6)
    zero = feature_filter.update(network.advance(SensoryObservation(0, 0)))
    assert torch.isfinite(zero).all()
    assert zero[7] == zero[8]


def test_charge_decoder_uses_neural_activity_and_reset_replays():
    from flybrain.neural.charge_readout import ChargeMotorDecoder

    network = SyntheticNetwork(encoder_mode='rational', encoder_scale=0.005,
                               record_charge=True)
    model = TrainableMotorReadout(9)
    with torch.no_grad():
        model.linear.weight[1, 7:] = torch.tensor([-0.25, 0.25])
    decoder = ChargeMotorDecoder(network.topology, model)
    activity = network.advance(SensoryObservation(0.008, 0.010))
    command = decoder.decode(activity)
    assert command.forward == pytest.approx(0.5)
    assert command.turn == pytest.approx(math.tanh(0.25 * math.log(1.25)), abs=1e-6)
    decoder.reset()
    assert decoder.decode(activity) == command
    with pytest.raises(ValueError, match='nine'):
        ChargeMotorDecoder(network.topology, TrainableMotorReadout(7))
    with pytest.raises(ValueError, match='charge'):
        decoder.decode(replace(activity, integrated_charge=None))


def test_charge_filter_rejects_missing_or_invalid_charge_without_advancing_ema():
    from flybrain.neural.charge_readout import ChargeFeatureFilter

    network = SyntheticNetwork(record_charge=True)
    activity = network.advance(SensoryObservation(0.025, 0.01))
    feature_filter = ChargeFeatureFilter(network.topology)
    reference = ChargeFeatureFilter(network.topology)
    for invalid in (None, torch.zeros(3), torch.full((100,), math.nan)):
        with pytest.raises(ValueError, match='charge'):
            feature_filter.update(replace(activity, integrated_charge=invalid))
    assert torch.equal(feature_filter.update(activity), reference.update(activity))
    feature_filter.reset()
    reference.reset()
    assert torch.equal(feature_filter.update(activity), reference.update(activity))


def test_charge_feature_clamps_extreme_drive_to_finite_values():
    from flybrain.neural.charge_readout import ChargeFeatureFilter

    network = SyntheticNetwork(record_charge=True, encoder_mode='rational')
    feature_filter = ChargeFeatureFilter(network.topology)
    features = feature_filter.update(network.advance(SensoryObservation(0, 1e308)))
    assert torch.isfinite(features).all()
    assert features[7] < 0 < features[8]


def test_charge_measurements_are_copies_and_do_not_change_network_state():
    left = SyntheticNetwork(record_charge=True)
    right = SyntheticNetwork(record_charge=True)
    observation = SensoryObservation(0.02, 0.01)
    left.advance(observation).integrated_charge.fill_(99)
    right.advance(observation)
    assert torch.equal(left.advance(observation).integrated_charge,
                       right.advance(observation).integrated_charge)


def test_optional_charge_measurement_consumes_no_global_randomness():
    torch_before = torch.random.get_rng_state().clone()
    python_before = random.getstate()
    network = SyntheticNetwork(record_charge=True)
    network.advance(SensoryObservation(0.02, 0.01))
    assert torch.equal(torch_before, torch.random.get_rng_state())
    assert python_before == random.getstate()
