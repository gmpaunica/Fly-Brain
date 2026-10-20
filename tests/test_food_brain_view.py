"""Engineering contact cells appear without changing neural simulation state."""

import pygame
import torch

from flybrain.foraging import ForagingSimulation
from flybrain.neural.food_readout import FoodReadout, make_food_controller
from flybrain.simulation import FixedStepSimulation
from flybrain.visualization.food_brain_view import FoodBrainView
from flybrain.world.arena import Arena
from flybrain.world.fly import Fly
from flybrain.world.odor import OdorSource


def test_brain_panel_tracks_both_engineered_cells_without_stepping():
    model = FoodReadout(torch.zeros(22), torch.ones(22))
    controller = make_food_controller(model)
    world = FixedStepSimulation(Arena(), Fly(600, 300),
                                odor_source=OdorSource(600, 300))
    loop = ForagingSimulation(world, controller)
    pygame.font.init()
    try:
        view = FoodBrainView(pygame.Surface((400, 600)),
                             controller.network.source_topology)
        frame = loop.step()
        view.observe(frame)
        assert view.model.tick_id == 1
        assert view.engineered_spike_counts[0] > 0
        assert view.engineered_spike_counts[1] > 0
        before = controller.network.snapshot()
        for _ in range(3):
            view.draw()
        after = controller.network.snapshot()
        assert before.step_count == after.step_count
        assert torch.equal(before.voltage, after.voltage)
        view.reset()
        assert view.model.tick_id == 0
        assert view.engineered_spike_counts == (0, 0)
    finally:
        pygame.font.quit()
