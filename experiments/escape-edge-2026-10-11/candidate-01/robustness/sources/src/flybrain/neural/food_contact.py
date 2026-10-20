"""Add an explicitly engineered contact pathway to the sourced BANC subset.

The first 83 cells and their effective weights are unchanged. The two appended
cells and their single fixed edge are engineering scaffolding, not BANC data.
"""

from dataclasses import replace
import math
from types import MappingProxyType

import torch

from flybrain.data.banc import BancNode, load_banc_subset
from flybrain.world.food import FoodObservation

from .banc_network import load_verified_banc_config
from .banc_projection import ProjectionNetwork


class FoodContactNetwork(ProjectionNetwork):
    """85 cells with causal odor rates and a 100 Hz contact-to-relay path."""

    CONTACT = 83
    RELAY = 84
    CONTACT_RATE_HZ = 100.0
    RELAY_JUMP = 2.0

    def __init__(self) -> None:
        super().__init__(load_banc_subset(connectome_gain=1.),
                         dynamics_config=load_verified_banc_config())
        self.reference_weights = self.population.weights
        weights = torch.zeros((85, 85), dtype=self.population.dtype, device='cpu')
        weights[:83, :83] = self.reference_weights
        weights[self.RELAY, self.CONTACT] = self.RELAY_JUMP
        nodes = self.topology.nodes + (
            BancNode('engineered:food-contact', 'contact_sensory',
                     MappingProxyType({'provenance': 'engineered'})),
            BancNode('engineered:contact-relay', 'contact_relay',
                     MappingProxyType({'provenance': 'engineered'})),
        )
        self.topology = replace(self.topology, nodes=nodes,
                                root_ids=self.topology.root_ids + tuple(node.root_id for node in nodes[-2:]),
                                groups=self.topology.groups + ('contact_sensory', 'contact_relay'),
                                _weights=weights)
        self.population = self.population_for_weights(weights)
        self._initial_phases = torch.cat((self._initial_phases,
                                          torch.zeros(2, dtype=torch.float64, device='cpu')))
        self._phases = self._initial_phases.clone()
        self.contact_blocked = False

    @staticmethod
    def odor_rate(concentration: float) -> float:
        """Engineering rate map, in Hz, shared by both antenna channels."""
        if not math.isfinite(concentration) or concentration < 0:
            raise ValueError('odor concentration must be finite and nonnegative')
        return 96.0 * concentration / (concentration + .1)

    def _sensory_rates(self, observation: FoodObservation) -> torch.Tensor:
        if not isinstance(observation, FoodObservation):
            raise ValueError('odor and local food contact required')
        rates = torch.zeros(85, dtype=torch.float64, device='cpu')
        for side, value in (('left', observation.odor_left),
                            ('right', observation.odor_right)):
            rates[list(self._scales[side][0])] = self.odor_rate(value)
        if observation.food_contact and not self.contact_blocked:
            rates[self.CONTACT] = self.CONTACT_RATE_HZ
        return rates

    def reset(self) -> None:
        """Clear voltages, pending spikes, phases and neural clock."""
        super().reset()
