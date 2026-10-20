import math

import pytest
import torch

from flybrain.neural.motor import MotorDecoder
from flybrain.neural.network import NeuralActivity, NeuralSnapshot
from flybrain.neural.topology import build_synthetic_topology
from flybrain.world.fly import MotorCommand


def activity_with_counts(**group_totals: int) -> NeuralActivity:
    """Build literal motor counts without using decoder implementation logic."""
    topology = build_synthetic_topology()
    counts = torch.zeros(100, dtype=torch.int64, device='cpu')
    for group, total in group_totals.items():
        indices = topology.indices(group)
        whole, remainder = divmod(total, len(indices))
        counts[list(indices)] = whole
        if remainder:
            counts[list(indices[:remainder])] += 1
    snapshot = NeuralSnapshot(
        step_count=20,
        voltage=torch.zeros(100, dtype=torch.float32, device='cpu'),
        spikes=torch.zeros(100, dtype=torch.bool, device='cpu'),
    )
    return NeuralActivity(0, 20, 0.020, counts, snapshot)


def replace_counts(activity: NeuralActivity, counts: torch.Tensor) -> NeuralActivity:
    return NeuralActivity(
        activity.start_step,
        activity.end_step,
        activity.elapsed_seconds,
        counts,
        activity.snapshot,
    )


def test_left_and_right_population_counts_have_documented_turn_sign():
    left = MotorDecoder(build_synthetic_topology(), smoothing=1.0)
    right = MotorDecoder(build_synthetic_topology(), smoothing=1.0)

    assert left.decode(activity_with_counts(motor_left=10)).turn == -1.0
    assert right.decode(activity_with_counts(motor_right=10)).turn == 1.0


def test_decoder_uses_complete_counts_not_last_substep_spikes():
    decoder = MotorDecoder(build_synthetic_topology(), smoothing=1.0)
    activity = activity_with_counts(motor_forward=10)
    assert not activity.snapshot.spikes.any()

    assert decoder.decode(activity) == MotorCommand(1.0, 0.0)


def test_decoder_smooths_repeated_commands_and_reset_clears_history():
    decoder = MotorDecoder(build_synthetic_topology(), smoothing=0.5)
    activity = activity_with_counts(motor_left=10, motor_forward=10)

    assert decoder.decode(activity) == MotorCommand(0.5, -0.5)
    assert decoder.decode(activity) == MotorCommand(0.75, -0.75)
    decoder.reset()
    assert decoder.decode(activity) == MotorCommand(0.5, -0.5)


def test_equal_turn_counts_cancel_and_large_counts_stay_bounded():
    decoder = MotorDecoder(build_synthetic_topology(), smoothing=1.0)
    equal = activity_with_counts(motor_left=10, motor_right=10)
    assert decoder.decode(equal).turn == 0.0

    saturated = activity_with_counts(motor_forward=100, motor_right=100)
    assert decoder.decode(saturated) == MotorCommand(1.0, 1.0)


@pytest.mark.parametrize(
    'counts',
    [
        torch.zeros(99, dtype=torch.int64),
        torch.zeros(100, dtype=torch.float32),
        torch.zeros(100, dtype=torch.bool),
        torch.tensor([-1] + [0] * 99, dtype=torch.int64),
        torch.empty(100, dtype=torch.int64, device='meta'),
    ],
)
def test_invalid_counts_are_rejected_without_changing_smoothing(counts):
    decoder = MotorDecoder(build_synthetic_topology(), smoothing=0.5)
    invalid = replace_counts(activity_with_counts(), counts)

    with pytest.raises(ValueError):
        decoder.decode(invalid)

    valid = activity_with_counts(motor_forward=10)
    assert decoder.decode(valid) == MotorCommand(0.5, 0.0)


@pytest.mark.parametrize('smoothing', [True, 0.0, -0.1, 1.1, math.nan, math.inf])
def test_invalid_smoothing_is_rejected(smoothing):
    with pytest.raises(ValueError):
        MotorDecoder(build_synthetic_topology(), smoothing=smoothing)
