"""Locally seeded, hand-wired synthetic graph; no biological provenance."""

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class NeuralTopology(Protocol):
    """Read-only graph interface shared by synthetic and sourced networks."""

    @property
    def labels(self) -> tuple[str, ...]: ...

    @property
    def groups(self) -> tuple[str, ...]: ...

    @property
    def weights(self) -> Tensor: ...

    def indices(self, group: str) -> tuple[int, ...]: ...


GROUP_SIZES = (
    ('sensory_left', 10), ('sensory_right', 10),
    ('interneuron_left', 25), ('interneuron_right', 25),
    ('motor_left', 10), ('motor_forward', 10), ('motor_right', 10),
)


@dataclass(frozen=True)
class SyntheticTopology:
    """Immutable labels and copied weights for the fixed synthetic graph."""

    labels: tuple[str, ...]
    groups: tuple[str, ...]
    seed: int
    _weights: Tensor

    @property
    def weights(self) -> Tensor:
        """Copy of end-of-step synaptic jumps in W[post,pre] orientation."""
        return self._weights.clone()

    def indices(self, group: str) -> tuple[int, ...]:
        """Ordered neuron indices; unknown groups are errors."""
        found = tuple(i for i, name in enumerate(self.groups) if name == group)
        if not found:
            raise ValueError(f'unknown neuron group: {group}')
        return found


def build_synthetic_topology(seed: int = 7,
                             dtype: torch.dtype = torch.float32) -> SyntheticTopology:
    """Build 100 labeled neurons using only a local CPU generator.

    Same-side sensory→interneuron→motor blocks have positive weights in
    [0.096,0.144); all other edges remain zero. This predisposed circuit is
    engineering scaffolding, not a sourced connectome or learned structure.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('only float32 and float64 are supported')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    labels = tuple(f'{group}_{i:02d}' for group, size in GROUP_SIZES for i in range(size))
    groups = tuple(group for group, size in GROUP_SIZES for _ in range(size))
    weights = torch.zeros(len(labels), len(labels), dtype=dtype, device='cpu')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    graph = SyntheticTopology(labels, groups, seed, weights)
    for side in ('left', 'right'):
        for post_group, pre_group in (
            (f'interneuron_{side}', f'sensory_{side}'),
            (f'motor_{side}', f'interneuron_{side}'),
        ):
            post, pre = graph.indices(post_group), graph.indices(pre_group)
            block = 0.096 + 0.048 * torch.rand(len(post), len(pre),
                                             dtype=dtype, device='cpu', generator=generator)
            weights[torch.tensor(post, device='cpu')[:, None],
                    torch.tensor(pre, device='cpu')[None, :]] = block
    return graph
