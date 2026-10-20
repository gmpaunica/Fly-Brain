"""Read-only panel for 83 sourced cells plus two engineered contact cells."""

from dataclasses import replace

import pygame

from flybrain.closed_loop import ClosedLoopSnapshot

from .brain_view import BrainView


class FoodBrainView:
    """Reuse the sourced panel and label real contact/relay spike activity."""

    def __init__(self, surface: pygame.Surface, source_topology) -> None:
        if len(source_topology.root_ids) != 83:
            raise ValueError('83 sourced cells required')
        self.surface = surface
        self.sourced = BrainView(surface, source_topology, kind='banc')
        self.model = self.sourced.model
        self.engineered_spike_counts = (0, 0)
        self._levels = [0., 0.]

    def observe(self, snapshot: ClosedLoopSnapshot) -> None:
        """Read all 85 counts; pass only the original 83 to the source view."""
        activity = snapshot.activity
        if activity.spike_counts.shape != (85,):
            raise ValueError('85-cell contact activity required')
        source_activity = replace(activity, spike_counts=activity.spike_counts[:83].clone())
        self.sourced.observe(replace(snapshot, activity=source_activity))
        self.engineered_spike_counts = tuple(
            int(value) for value in activity.spike_counts[83:].tolist())
        fade = activity.elapsed_seconds / .2
        self._levels = [min(1., max(0., old - fade) + (1. if count else 0.))
                        for old, count in zip(self._levels, self.engineered_spike_counts)]

    def reset(self) -> None:
        """Clear visual traces independently of the neural controller."""
        self.sourced.reset()
        self.engineered_spike_counts = (0, 0)
        self._levels = [0., 0.]

    def draw(self) -> None:
        """Draw sourced and engineered activity without stepping simulation."""
        self.sourced.draw()
        width, height = self.surface.get_size()
        pygame.draw.rect(self.surface, (13, 18, 29), (4, 4, width - 8, 34))
        self.surface.blit(self.sourced.title_font.render(
            '85 CELLS | 83 SOURCED + 2 ENGINEERED', True, (230, 235, 245)), (14, 12))
        y = height - 186
        pygame.draw.rect(self.surface, (20, 29, 39), (12, y, width - 24, 50))
        self.surface.blit(self.sourced.small_font.render(
            'ENGINEERED FOOD CONTACT PATH', True, (185, 205, 205)), (20, y + 4))
        points = ((50, y + 33), (width - 120, y + 33))
        pygame.draw.line(self.surface, (110, 135, 120), points[0], points[1], 2)
        for index, (point, name, base) in enumerate((
            (points[0], 'sensor', (75, 215, 125)),
            (points[1], 'relay', (225, 170, 95)),
        )):
            brightness = round(40 * self._levels[index])
            color = tuple(min(255, value + brightness) for value in base)
            pygame.draw.circle(self.surface, color, point, 7 if self._levels[index] else 5)
            self.surface.blit(self.sourced.small_font.render(
                f'{name} {self.engineered_spike_counts[index]} spikes', True, (220, 230, 225)),
                (point[0] + 11, point[1] - 6))
        pygame.draw.rect(self.surface, (13, 18, 29), (3, height - 52, width - 6, 49))
        self.surface.blit(self.sourced.small_font.render(
            'Frozen learned readout | 419 parameters | no online updates',
            True, (175, 190, 210)), (14, height - 43))
