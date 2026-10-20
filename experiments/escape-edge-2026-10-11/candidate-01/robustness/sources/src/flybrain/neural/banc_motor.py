"""Fixed motor scaffolding that can observe only BANC downstream spikes."""

import math

import torch
from torch import Tensor

from flybrain.data.banc import BancTopology, load_banc_subset
from flybrain.world.fly import MotorCommand

from .banc_network import BancDynamicsConfig, BancNetwork
from .controller import NeuralController
from .network import NeuralActivity


class BancMotorDecoder:
    """Eight endpoint spike-rate features, in Hz, with a 100 ms tick EMA.

    Features are left/right DNa02 followed by three left and three right
    motor endpoints, each sorted by source root ID. The interval EMA treats
    counts/elapsed as a constant rate during that interval; it approximates
    fine spike timing and is used at the declared 20 ms world tick.
    No voltage, sensory charge, encoder current or geometry is observed.
    """

    filter_tau_seconds = .1

    def __init__(self, topology: BancTopology) -> None:
        groups = ('steering_left', 'steering_right', 'motor_left', 'motor_right')
        indices = []
        for group, size in zip(groups, (1, 1, 3, 3)):
            selected = sorted(topology.indices(group), key=lambda i: topology.root_ids[i])
            if len(selected) != size:
                raise ValueError('BANC decoder requires two steering and six motor endpoints')
            indices.extend(selected)
        self._indices = torch.tensor(indices, dtype=torch.int64, device='cpu')
        self._size = len(topology.root_ids)
        self.feature_names = tuple(f'{topology.groups[i]}:{topology.root_ids[i]}' for i in indices)
        self.reset()

    @property
    def feature_rates(self) -> Tensor:
        """Detached eight-feature copy, ordered like feature_names, in Hz."""
        return self._rates.clone()

    def reset(self) -> None:
        """Clear all eight endpoint filter states at the episode boundary."""
        self._rates = torch.zeros(8, dtype=torch.float64, device='cpu')

    def decode(self, activity: NeuralActivity) -> MotorCommand:
        """Update endpoint rates once, then emit a bounded engineering proxy."""
        counts = activity.spike_counts
        steps = activity.end_step - activity.start_step
        elapsed = activity.elapsed_seconds
        if (not isinstance(counts, Tensor) or counts.device.type != 'cpu'
                or counts.layout != torch.strided or counts.dtype != torch.int64
                or counts.shape != (self._size,) or steps <= 0
                or not math.isfinite(elapsed) or elapsed <= 0
                or torch.any(counts < 0).item() or torch.any(counts > steps).item()):
            raise ValueError('BANC decoder requires valid complete-interval spike counts')
        instantaneous = counts[self._indices].double() / elapsed
        alpha = -math.expm1(-elapsed / self.filter_tau_seconds)
        rates = (1 - alpha) * self._rates + alpha * instantaneous
        if not torch.isfinite(rates).all().item():
            raise ValueError('BANC endpoint rate overflowed')
        left, right = rates[:2].tolist()
        motor = rates[2:].mean().item()
        command = MotorCommand(motor / (motor + 20), (right - left) / (right + left + 10))
        self._rates = rates
        return command


def make_banc_controller(config: BancDynamicsConfig) -> NeuralController:
    """Create independent source, neural and motor state for one episode."""
    brain = BancNetwork(load_banc_subset(connectome_gain=1.0), dynamics_config=config)
    return NeuralController(brain, BancMotorDecoder(brain.topology))
