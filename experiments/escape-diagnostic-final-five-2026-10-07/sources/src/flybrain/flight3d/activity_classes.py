"""Display categories from model roles, independent of biological classification.

Membership and one-hop neighbors are cached once from the fixed manifest. This
module observes assignments; it neither changes roles nor calculates activity.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class ActivityClass:
    """One display switch and its RGB activation endpoint (unit interval)."""
    key: str
    label: str
    roles: frozenset[str]
    rgb: tuple[float, float, float]


def _category(key: str, label: str, roles: str, color: str) -> ActivityClass:
    return ActivityClass(key, label, frozenset(roles.split()),
                         tuple(int(color[i:i+2],16)/255 for i in (0,2,4)))


CLASSES = (
    _category('odor', 'Odor inputs', 'odor', 'E6B450'),
    _category('vision', 'Vision', 'vision', '86D97A'),
    _category('optic_flow', 'Optic flow', 'optic_flow', '3A9D6D'),
    _category('motion', 'Motion sensing', 'motion', 'E889B5'),
    _category('taste', 'Taste', 'taste', 'E75C5C'),
    _category('touch', 'Touch', 'touch', 'C7DC66'),
    _category('odor_projection', 'Odor relays', 'odor_projection', 'C59BE8'),
    _category('motor', 'Motor', 'wing_power wing_steering wing_tension landing_motor', 'F18A48'),
    _category('descending', 'Descending cells', 'flight_power_dn flight_steering_dn landing_dn', 'B378D9'),
    _category('structural', 'Structural relays', 'structural_corridor', 'D7D7DB'),
)
SENSORY = frozenset(c.key for c in CLASSES[:6])


class DisplayActivityIndex:
    """Immutable membership/index data, aligned with manifest neuron order."""
    classes = CLASSES

    def __init__(self, manifest: dict[str, Any]):
        self.ids = tuple(n['id'] for n in manifest['nodes'])
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('display neuron IDs must be unique')
        positions = {id: i for i, id in enumerate(self.ids)}
        members = {c.key: set() for c in CLASSES}
        palette = []
        for i, node in enumerate(manifest['nodes']):
            matches = [c for c in CLASSES if c.roles.intersection(node['roles'])]
            if len(matches) != 1 or set(node['roles']) - matches[0].roles:
                raise ValueError(f'{node["id"]} must map to exactly one display class')
            members[matches[0].key].add(i)
            palette.append(matches[0].rgb)
        neighbors = [{i} for i in range(len(self.ids))]
        for edge in manifest['edges']:
            a, b = positions[edge['pre']], positions[edge['post']]
            if a != b:
                neighbors[a].add(b); neighbors[b].add(a)
        self.members = MappingProxyType({k: frozenset(v) for k, v in members.items()})
        self.adjacency = MappingProxyType({id: frozenset(neighbors[i]) for i, id in enumerate(self.ids)})
        self.activation_colors = tuple(palette)

    def sensory_state(self, enabled: frozenset[str]) -> str:
        """Return off/on/mixed for the six sensory switches."""
        subset = enabled & SENSORY
        return 'on' if subset == SENSORY else 'mixed' if subset else 'off'

    def toggle_sensory(self, enabled: frozenset[str]) -> frozenset[str]:
        """An off/mixed parent enables all children; on disables all children."""
        return enabled - SENSORY if self.sensory_state(enabled) == 'on' else enabled | SENSORY

    def scope_members(self, scope: str, selected: str | None = None) -> frozenset[int]:
        """Whole model or selected neuron plus direct non-self partners."""
        if scope == 'whole': return frozenset(range(len(self.ids)))
        if scope != 'selected' or selected not in self.adjacency:
            raise ValueError('selected circuit requires a modeled neuron')
        return self.adjacency[selected]

    def mask(self, enabled: frozenset[str], scope: str = 'whole',
             selected: str | None = None) -> tuple[bool, ...]:
        """Intersect class membership and circuit scope into an immutable mask."""
        if enabled - self.members.keys(): raise ValueError('unknown display class')
        owners = frozenset().union(*(self.members[k] for k in enabled))
        owners &= self.scope_members(scope, selected)
        return tuple(i in owners for i in range(len(self.ids)))
