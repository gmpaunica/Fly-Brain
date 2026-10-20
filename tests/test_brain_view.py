from dataclasses import asdict
import math
import random

import pygame
import pytest
import torch

from flybrain.closed_loop import BodyState, ClosedLoopSimulation, ClosedLoopSnapshot
from flybrain.neural.controller import NeuralController
from flybrain.neural.motor import MotorDecoder
from flybrain.neural.network import NeuralActivity, NeuralSnapshot, SyntheticNetwork
from flybrain.neural.topology import build_synthetic_topology
from flybrain.simulation import FixedStepSimulation
from flybrain.visualization.brain_view import BrainActivityModel, BrainView, select_edges
from flybrain.world.arena import Arena
from flybrain.world.fly import Fly, MotorCommand
from flybrain.world.odor import OdorSource
from flybrain.world.sensing import SensoryObservation


def make_snapshot(
    tick_id: int,
    start_step: int,
    end_step: int,
    spike_indices: tuple[int, ...] = (),
    command: MotorCommand = MotorCommand(),
) -> ClosedLoopSnapshot:
    counts = torch.zeros(100, dtype=torch.int64)
    counts[list(spike_indices)] = 1
    neural_snapshot = NeuralSnapshot(
        end_step,
        torch.zeros(100),
        counts.to(torch.bool),
    )
    activity = NeuralActivity(
        start_step,
        end_step,
        (end_step - start_step) * 0.001,
        counts,
        neural_snapshot,
    )
    return ClosedLoopSnapshot(
        tick_id,
        SensoryObservation(0.01, 0.02),
        activity,
        command,
        False,
        BodyState(400.0, 300.0, 0.0, 8.0, 80.0, 3.0),
    )


def make_loop(seed: int = 7) -> ClosedLoopSimulation:
    arena = Arena()
    world = FixedStepSimulation(
        arena,
        arena.spawn(seed),
        odor_source=OdorSource(600, 300),
    )
    network = SyntheticNetwork(seed=seed)
    return ClosedLoopSimulation(
        world,
        NeuralController(network, MotorDecoder(network.topology)),
    )


def loop_signature(loop: ClosedLoopSimulation):
    activity = loop.last_snapshot.activity
    neural = loop.controller.network.snapshot()
    return (
        asdict(loop.world.fly),
        loop.world.observation,
        loop.world.tick_count,
        activity.start_step,
        activity.end_step,
        tuple(activity.spike_counts.tolist()),
        tuple(neural.voltage.tolist()),
        tuple(neural.spikes.tolist()),
    )


def test_real_spikes_flash_and_fade_only_when_simulation_time_advances():
    topology = build_synthetic_topology(seed=4)
    model = BrainActivityModel(topology, fade_seconds=0.2)
    model.observe(make_snapshot(1, 0, 20, (0, 45, 99)))

    assert model.activity_levels[0] == 1.0
    assert model.activity_levels[45] == 1.0
    assert model.activity_levels[99] == 1.0
    before_draw = model.activity_levels

    pygame.font.init()
    try:
        view = BrainView(pygame.Surface((400, 600)), topology, model=model)
        for _ in range(5):
            view.draw()
        assert model.activity_levels == before_draw
    finally:
        pygame.quit()

    model.observe(make_snapshot(2, 20, 120))
    assert model.activity_levels[0] == pytest.approx(0.5)
    assert model.simulation_seconds == pytest.approx(0.12)


def test_activity_model_reset_clears_visual_episode_state():
    topology = build_synthetic_topology(seed=4)
    model = BrainActivityModel(topology)
    model.observe(make_snapshot(3, 40, 60, (2, 3), MotorCommand(0.7, -0.2)))

    model.reset()

    assert model.activity_levels == (0.0,) * 100
    assert model.tick_id == 0
    assert model.end_step == 0
    assert model.simulation_seconds == 0.0
    assert model.command == MotorCommand()


def test_activity_model_rejects_noncontiguous_or_repeated_time_atomically():
    topology = build_synthetic_topology(seed=4)
    model = BrainActivityModel(topology)
    model.observe(make_snapshot(1, 0, 20, (0,)))
    before = (
        model.activity_levels,
        model.tick_id,
        model.end_step,
        model.simulation_seconds,
    )

    with pytest.raises(ValueError, match='contiguous'):
        model.observe(make_snapshot(2, 40, 60, (1,)))

    assert (
        model.activity_levels,
        model.tick_id,
        model.end_step,
        model.simulation_seconds,
    ) == before


def test_edge_selection_is_deterministic_limited_and_follows_weight_direction():
    topology = build_synthetic_topology(seed=17)

    first = select_edges(topology, limit=37)
    second = select_edges(topology, limit=37)

    assert first == second
    assert len(first) == 37
    assert all(topology.weights[post, pre].item() == pytest.approx(weight)
               for pre, post, weight in first)
    assert all(weight > 0 for _, _, weight in first)


def test_panel_draws_groups_edges_activity_and_motor_output_read_only():
    pygame.font.init()
    try:
        topology = build_synthetic_topology(seed=8)
        surface = pygame.Surface((400, 600))
        model = BrainActivityModel(topology)
        model.observe(make_snapshot(1, 0, 20, (0, 20, 70, 80, 90),
                                     MotorCommand(0.6, 0.25)))
        view = BrainView(surface, topology, model=model, edge_limit=30)
        before = model.activity_levels
        rng_before = random.getstate()

        view.draw()
        first_image = pygame.image.tobytes(surface, 'RGB')
        view.draw()

        assert pygame.image.tobytes(surface, 'RGB') == first_image
        assert model.activity_levels == before
        assert random.getstate() == rng_before
        assert len(view.edges) == 30
        assert len(view.positions) == 100
        assert len(set(view.group_colors.values())) >= 3
        assert any(channel > 30 for channel in surface.get_at(view.positions[0])[:3])
        active_group_pixels = {
            surface.get_at(view.positions[index])[:3]
            for index in (0, 20, 70)
        }
        assert len(active_group_pixels) == 3
    finally:
        pygame.quit()


def test_extra_brain_draws_do_not_change_world_or_neural_trajectory():
    pygame.font.init()
    try:
        drawn = make_loop(seed=21)
        reference = make_loop(seed=21)
        surface = pygame.Surface((400, 600))
        view = BrainView(
            surface,
            drawn.controller.network.topology,
        )

        for _ in range(8):
            snapshot = drawn.step()
            view.observe(snapshot)
            for _ in range(4):
                view.draw()
            reference.step()

        assert loop_signature(drawn) == loop_signature(reference)
    finally:
        pygame.quit()
