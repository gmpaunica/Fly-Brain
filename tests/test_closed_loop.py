from dataclasses import asdict
import inspect
import math

import pytest
import torch

from flybrain.closed_loop import ClosedLoopSimulation
from flybrain.neural.controller import NeuralController
from flybrain.neural.motor import MotorDecoder
from flybrain.neural.network import SyntheticNetwork
from flybrain.simulation import FixedStepSimulation
from flybrain.world.arena import Arena
from flybrain.world.fly import Fly, MotorCommand
from flybrain.world.odor import OdorSource
from flybrain.world.sensing import SensoryObservation, sample_odor


def make_loop(
    *,
    body_seed: int = 7,
    network_seed: int = 7,
    source: OdorSource | None = None,
    dt: float = 0.02,
    smoothing: float = 0.5,
) -> ClosedLoopSimulation:
    arena = Arena()
    world = FixedStepSimulation(
        arena,
        arena.spawn(body_seed),
        dt=dt,
        odor_source=source or OdorSource(600, 300),
    )
    network = SyntheticNetwork(seed=network_seed)
    decoder = MotorDecoder(network.topology, smoothing=smoothing)
    return ClosedLoopSimulation(world, NeuralController(network, decoder))


def tick_signature(tick):
    return (
        tick.tick_id,
        tick.pre_movement_observation,
        tuple(tick.activity.spike_counts.tolist()),
        tick.activity.start_step,
        tick.activity.end_step,
        tick.command,
        tick.collision,
        tick.body,
    )


def test_controller_per_tick_boundary_accepts_only_sensory_observation():
    parameters = tuple(inspect.signature(NeuralController.decide).parameters)
    assert parameters == ('self', 'observation')

    network = SyntheticNetwork()
    controller = NeuralController(network, MotorDecoder(network.topology))
    decision = controller.decide(SensoryObservation(0.025, 0.0))
    assert decision.activity.end_step == 20
    assert decision.command.turn < 0


def test_tick_samples_before_movement_and_publishes_resulting_body():
    source = OdorSource(600, 300)
    loop = make_loop(source=source, smoothing=1.0)
    before = sample_odor(loop.world.fly, source)

    tick = loop.step()

    assert tick.pre_movement_observation == before
    assert (tick.activity.start_step, tick.activity.end_step) == (0, 20)
    assert tick.activity.elapsed_seconds == pytest.approx(0.020)
    assert tick.tick_id == loop.world.tick_count == 1
    assert asdict(tick.body) == asdict(loop.world.fly)
    assert loop.last_snapshot is tick
    assert loop.world.observation == sample_odor(loop.world.fly, source)


def test_tonic_forward_activity_changes_pose_and_future_sensory_input():
    loop = make_loop(body_seed=7, source=OdorSource(600, 300))
    initial_pose = asdict(loop.world.fly)
    initial_observation = loop.world.observation

    ticks = [loop.step() for _ in range(6)]

    assert any(tick.command.forward > 0 for tick in ticks)
    assert asdict(loop.world.fly) != initial_pose
    assert loop.world.observation != initial_observation


def test_real_network_left_and_right_activity_have_opposite_turn_signs():
    left_network = SyntheticNetwork()
    right_network = SyntheticNetwork()
    left = NeuralController(
        left_network, MotorDecoder(left_network.topology, smoothing=1.0)
    )
    right = NeuralController(
        right_network, MotorDecoder(right_network.topology, smoothing=1.0)
    )

    assert left.decide(SensoryObservation(0.025, 0.0)).command.turn < 0
    assert right.decide(SensoryObservation(0.0, 0.025)).command.turn > 0


def test_motor_gate_stops_motion_while_brain_advances_then_restores_output():
    loop = make_loop(source=OdorSource(600, 300))
    initial_pose = asdict(loop.world.fly)
    loop.controller.set_motor_enabled(False)

    silent_ticks = [loop.step() for _ in range(3)]

    assert all(tick.command == MotorCommand() for tick in silent_ticks)
    assert loop.controller.network.snapshot().step_count == 60
    assert any(tick.activity.spike_counts.sum() > 0 for tick in silent_ticks)
    assert asdict(loop.world.fly) == initial_pose

    loop.controller.set_motor_enabled(True)
    restored = loop.step()
    assert restored.activity.start_step == 60
    assert restored.command.forward > 0
    assert asdict(loop.world.fly) != initial_pose


@pytest.mark.parametrize('enabled', [0, 1, None, 'yes'])
def test_motor_gate_requires_a_boolean(enabled):
    loop = make_loop()
    with pytest.raises(ValueError):
        loop.controller.set_motor_enabled(enabled)


def test_world_tick_must_equal_exactly_twenty_neural_substeps():
    arena = Arena()
    world = FixedStepSimulation(arena, arena.spawn(7), dt=0.01)
    network = SyntheticNetwork()
    controller = NeuralController(network, MotorDecoder(network.topology))

    with pytest.raises(ValueError):
        ClosedLoopSimulation(world, controller)

    assert world.tick_count == 0
    assert network.snapshot().step_count == 0


def test_elapsed_frame_partitions_produce_identical_closed_loop_state():
    whole = make_loop(body_seed=9, network_seed=9)
    split = make_loop(body_seed=9, network_seed=9)

    assert whole.advance(0.1) == 5
    assert sum(split.advance(value) for value in (0.01, 0.03, 0.06)) == 5

    assert asdict(whole.world.fly) == asdict(split.world.fly)
    assert whole.world.observation == split.world.observation
    assert tick_signature(whole.last_snapshot) == tick_signature(split.last_snapshot)
    assert torch.equal(
        whole.controller.network.snapshot().voltage,
        split.controller.network.snapshot().voltage,
    )


def test_advance_snapshots_returns_every_tick_for_render_aggregation():
    loop = make_loop(body_seed=9, network_seed=9)

    snapshots = loop.advance_snapshots(0.06)

    assert tuple(snapshot.tick_id for snapshot in snapshots) == (1, 2, 3)
    assert tuple(
        (snapshot.activity.start_step, snapshot.activity.end_step)
        for snapshot in snapshots
    ) == ((0, 20), (20, 40), (40, 60))
    assert loop.world.tick_count == 3
    assert loop.last_snapshot is snapshots[-1]


@pytest.mark.parametrize('elapsed', [-1.0, math.nan, math.inf])
def test_invalid_elapsed_is_atomic_and_preserves_partial_time(elapsed):
    loop = make_loop()
    assert loop.advance(0.01) == 0

    with pytest.raises(ValueError):
        loop.advance(elapsed)

    assert loop.world.tick_count == 0
    assert loop.controller.network.snapshot().step_count == 0
    assert loop.last_snapshot is None
    assert loop.advance(0.01) == 1


def test_reset_clears_every_episode_state_and_replays_exactly():
    loop = make_loop(body_seed=11, network_seed=13)

    def replay():
        loop.reset(11)
        assert loop.last_snapshot is None
        assert loop.world.tick_count == 0
        assert loop.controller.network.snapshot().step_count == 0
        assert loop.advance(0.01) == 0
        signatures = []
        for elapsed in (0.01, 0.04, 0.03, 0.02):
            before = loop.world.tick_count
            loop.advance(elapsed)
            if loop.world.tick_count != before:
                signatures.append(tick_signature(loop.last_snapshot))
        return signatures, asdict(loop.world.fly), loop.world.observation

    first = replay()
    loop.advance(0.08)
    second = replay()

    assert first == second
