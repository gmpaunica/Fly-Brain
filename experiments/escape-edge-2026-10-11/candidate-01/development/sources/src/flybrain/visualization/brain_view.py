"""Read-only functional view of actual synthetic or BANC neural activity."""

from collections import defaultdict
import math

import pygame

from flybrain.closed_loop import ClosedLoopSnapshot
from flybrain.neural.topology import NeuralTopology
from flybrain.world.fly import MotorCommand


GROUP_COLORS = {
    'sensory_left': (65, 165, 235),
    'sensory_right': (75, 205, 225),
    'interneuron_left': (155, 105, 225),
    'interneuron_right': (190, 115, 215),
    'motor_left': (245, 145, 75),
    'motor_forward': (245, 195, 75),
    'motor_right': (235, 105, 85),
    'projection_left': (115, 165, 235),
    'projection_right': (125, 195, 235),
    'corridor_left': (155, 105, 225),
    'corridor_right': (190, 115, 215),
    'steering_left': (235, 140, 155),
    'steering_right': (240, 110, 170),
}


def select_edges(
    topology: NeuralTopology,
    limit: int = 80,
) -> tuple[tuple[int, int, float], ...]:
    """Return a deterministic, group-balanced subset as (pre, post, weight)."""
    if type(limit) is not int or limit < 0:
        raise ValueError('edge limit must be a nonnegative integer')
    weights = topology.weights
    buckets: dict[tuple[str, str], list[tuple[int, int, float]]] = defaultdict(list)
    for post, pre in weights.nonzero(as_tuple=False).tolist():
        buckets[(topology.groups[pre], topology.groups[post])].append(
            (pre, post, float(weights[post, pre].item()))
        )
    for edges in buckets.values():
        edges.sort(key=lambda edge: (-edge[2], edge[1], edge[0]))

    selected: list[tuple[int, int, float]] = []
    ordered_buckets = [buckets[key] for key in sorted(buckets)]
    depth = 0
    while len(selected) < limit:
        added = False
        for edges in ordered_buckets:
            if depth < len(edges):
                selected.append(edges[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return tuple(selected)


class BrainActivityModel:
    """Simulation-time activity traces independent of display frame count."""

    def __init__(self, topology: NeuralTopology,
                 fade_seconds: float = 0.2) -> None:
        if not math.isfinite(fade_seconds) or fade_seconds <= 0:
            raise ValueError('fade_seconds must be positive and finite')
        self.topology = topology
        self.fade_seconds = fade_seconds
        self._levels = [0.0] * len(topology.labels)
        self._spike_counts = [0] * len(topology.labels)
        self._observed = False
        self.tick_id = 0
        self.end_step = 0
        self.simulation_seconds = 0.0
        self.command = MotorCommand()

    @property
    def activity_levels(self) -> tuple[float, ...]:
        """Immutable per-neuron brightness values in [0, 1]."""
        return tuple(self._levels)

    @property
    def spike_counts(self) -> tuple[int, ...]:
        """Counts from the most recently observed neural interval."""
        return tuple(self._spike_counts)

    def observe(self, snapshot: ClosedLoopSnapshot) -> None:
        """Accumulate one contiguous tick and decay using neural elapsed time."""
        activity = snapshot.activity
        counts = activity.spike_counts
        size = len(self.topology.labels)
        if counts.ndim != 1 or counts.numel() != size:
            raise ValueError('spike_counts must match topology size')
        if activity.end_step <= activity.start_step:
            raise ValueError('activity step interval must be positive')
        if not math.isfinite(activity.elapsed_seconds) or activity.elapsed_seconds <= 0:
            raise ValueError('activity elapsed_seconds must be positive and finite')
        if self._observed and activity.start_step != self.end_step:
            raise ValueError('activity intervals must be contiguous')
        integer_counts = tuple(int(value) for value in counts.tolist())
        if any(value < 0 for value in integer_counts):
            raise ValueError('spike counts must be nonnegative')

        decay = activity.elapsed_seconds / self.fade_seconds
        levels = [max(0.0, level - decay) for level in self._levels]
        for index, count in enumerate(integer_counts):
            if count:
                levels[index] = min(1.0, levels[index] + count)
        dt = activity.elapsed_seconds / (activity.end_step - activity.start_step)

        self._levels = levels
        self._spike_counts = list(integer_counts)
        self._observed = True
        self.tick_id = snapshot.tick_id
        self.end_step = activity.end_step
        self.simulation_seconds = activity.end_step * dt
        self.command = snapshot.command

    def reset(self) -> None:
        """Clear episode-specific visual traces and displayed output."""
        self._levels = [0.0] * len(self.topology.labels)
        self._spike_counts = [0] * len(self.topology.labels)
        self._observed = False
        self.tick_id = 0
        self.end_step = 0
        self.simulation_seconds = 0.0
        self.command = MotorCommand()


class BrainView:
    """Draw labeled neural groups, selected edges, traces, and motor output."""

    def __init__(
        self,
        surface: pygame.Surface,
        topology: NeuralTopology,
        *,
        model: BrainActivityModel | None = None,
        edge_limit: int = 80,
        kind: str = 'synthetic',
    ) -> None:
        if surface.get_width() < 320 or surface.get_height() < 480:
            raise ValueError('brain panel must be at least 320x480 pixels')
        if model is not None and model.topology.labels != topology.labels:
            raise ValueError('activity model topology must match the view topology')
        if kind not in ('synthetic', 'banc'):
            raise ValueError('brain view kind must be synthetic or banc')
        self.kind = kind
        self.surface = surface
        self.topology = topology
        self.model = model or BrainActivityModel(topology)
        self.edges = select_edges(topology, edge_limit)
        self.positions = self._build_positions()
        self.group_colors = dict(GROUP_COLORS)
        self.title_font = pygame.font.Font(None, 27)
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)

    def _build_positions(self) -> tuple[tuple[int, int], ...]:
        width, height = self.surface.get_size()
        top = 100
        bottom = height - 150
        quarter = (bottom - top) / 4
        centers = {
            'sensory_left': (0.15 * width, top + quarter),
            'sensory_right': (0.15 * width, top + 3 * quarter),
            'interneuron_left': (0.49 * width, top + quarter),
            'interneuron_right': (0.49 * width, top + 3 * quarter),
            'motor_left': (0.83 * width, top + 0.35 * quarter),
            'motor_forward': (0.83 * width, top + 2 * quarter),
            'motor_right': (0.83 * width, top + 3.65 * quarter),
        }
        if self.kind == 'banc':
            centers = {
                f'{role}_{side}': (fraction * width, top + level * quarter)
                for role, fraction in (('sensory', .13), ('projection', .32),
                                       ('corridor', .5), ('steering', .69),
                                       ('motor', .88))
                for side, level in (('left', 1), ('right', 3))
            }
        positions: list[tuple[int, int]] = []
        for group in self.topology.groups:
            index_in_group = sum(
                previous == group for previous in self.topology.groups[:len(positions)]
            )
            group_size = len(self.topology.indices(group))
            columns = 5 if group_size >= 20 else 2
            rows = math.ceil(group_size / columns)
            column = index_in_group % columns
            row = index_in_group // columns
            center_x, center_y = centers[group]
            x = center_x + (column - (columns - 1) / 2) * 10
            y = center_y + (row - (rows - 1) / 2) * 10
            positions.append((round(x), round(y)))
        return tuple(positions)

    def observe(self, snapshot: ClosedLoopSnapshot) -> None:
        """Add one simulation tick to the visual trace model."""
        self.model.observe(snapshot)

    def reset(self) -> None:
        """Clear activity traces for an episode reset."""
        self.model.reset()

    def draw(self) -> None:
        """Render current visual state without advancing or decaying it."""
        surface = self.surface
        width, height = surface.get_size()
        surface.fill((13, 18, 29))
        pygame.draw.rect(surface, (90, 110, 130), surface.get_rect(), 2)
        surface.blit(
            self.title_font.render('BANC v888 | 83-CELL SUBSET' if self.kind == 'banc'
                                   else 'LIVE SYNTHETIC BRAIN', True, (230, 235, 245)),
            (14, 12),
        )
        timing = (f'tick {self.model.tick_id}  neural step {self.model.end_step}  '
                  f't={self.model.simulation_seconds:.3f}s')
        surface.blit(self.font.render(timing, True, (175, 190, 210)), (14, 42))
        surface.blit(
            self.small_font.render(
                'Functional layout | assumed LIF dynamics' if self.kind == 'banc'
                else 'Diagram layout | activity is real simulated spikes',
                True,
                (135, 150, 170),
            ),
            (14, 65),
        )

        for pre, post, _weight in self.edges:
            pygame.draw.line(
                surface,
                (42, 55, 78),
                self.positions[pre],
                self.positions[post],
                1,
            )

        levels = self.model.activity_levels
        for index, (position, group) in enumerate(
            zip(self.positions, self.topology.groups)
        ):
            base = GROUP_COLORS[group]
            level = levels[index]
            highlight = round(70 * level)
            color = tuple(min(255, channel + highlight) for channel in base)
            radius = 5 if level > 0 else 3
            pygame.draw.circle(surface, color, position, radius)

        label_positions = {
            'SENSORY': (18, 82),
            'INTERNEURONS': (round(width * 0.37), 82),
            'MOTOR': (round(width * 0.75), 82),
        }
        if self.kind == 'banc':
            label_positions = {
                text: (round(width * fraction), 82)
                for text, fraction in (('ORN', .08), ('PN', .29), ('PATH', .45),
                                        ('DNa02', .64), ('MOTOR', .82))
            }
        for label, position in label_positions.items():
            surface.blit(self.small_font.render(label, True, (180, 190, 205)), position)

        active = sum(level > 0 for level in levels)
        surface.blit(
            self.font.render(
                f'Recently active neurons: {active}/{len(self.topology.labels)}',
                True,
                (210, 220, 235),
            ),
            (14, height - 128),
        )
        self._draw_motor_activity(height - 98)

    def _draw_motor_activity(self, y: int) -> None:
        surface = self.surface
        levels = self.model.activity_levels
        groups = (
            ('L', 'motor_left'),
            ('F', 'motor_forward'),
            ('R', 'motor_right'),
        )
        if self.kind == 'banc':
            groups = (('DL', 'steering_left'), ('DR', 'steering_right'),
                      ('M', 'motor'))
        for offset, (short, group) in enumerate(groups):
            indices = (self.topology.indices('motor_left')
                       + self.topology.indices('motor_right') if group == 'motor'
                       else self.topology.indices(group))
            activity = sum(levels[index] for index in indices) / len(indices)
            x = 16 + offset * 82
            pygame.draw.rect(surface, (35, 43, 58), (x + 18, y, 54, 12))
            pygame.draw.rect(
                surface,
                GROUP_COLORS['motor_forward' if group == 'motor' else group],
                (x + 18, y, round(54 * activity), 12),
            )
            surface.blit(self.small_font.render(short, True, (220, 225, 235)),
                         (x, y - 1))
        command = self.model.command
        text = f'output  forward={command.forward:.2f}  turn={command.turn:+.2f}'
        surface.blit(self.font.render(text, True, (225, 205, 145)), (14, y + 26))
        surface.blit(
            self.small_font.render('Fixed hand-written decoder | no learning', True,
                                   (135, 150, 170)),
            (14, y + 51),
        )
