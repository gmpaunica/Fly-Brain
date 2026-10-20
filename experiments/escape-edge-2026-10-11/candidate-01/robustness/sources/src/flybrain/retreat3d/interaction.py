"""Immutable fixed-control-step inputs and spawn events for exact replay."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from flybrain.flight3d.anatomy import CACHE, digest
from .world import PredatorInput, EncounterObject


@dataclass(frozen=True)
class SpawnEvent:
    step: int
    kind: str
    position: tuple[float, float, float]


@dataclass
class InteractionLog:
    """One applied predator input per 20 ms; spawns occur before their step."""
    seed: int
    steps: list[PredatorInput] = field(default_factory=list)
    spawns: list[SpawnEvent] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Refuse overwrite; bind actions to the original source registry."""
        with path.open('x', encoding='utf-8') as stream:
            json.dump(dict(schema='retreat3d-actions-v1', seed=self.seed, dt_seconds=.02,
                circuit_sha256=digest(CACHE/'circuit.json'),
                steps=[c.axes for c in self.steps],
                spawns=[dict(step=e.step, kind=e.kind, position=e.position) for e in self.spawns]),
                stream, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'InteractionLog':
        record = json.loads(path.read_text(encoding='utf-8'))
        if (record['schema'] != 'retreat3d-actions-v1' or record['dt_seconds'] != .02 or
                record['circuit_sha256'] != digest(CACHE/'circuit.json') or
                type(record['seed']) is not int or record['seed'] < 0):
            raise ValueError('action log source/timestep/seed mismatch')
        steps = [PredatorInput(tuple(c)) for c in record['steps']]
        spawns = []
        for event in record['spawns']:
            if type(event['step']) is not int or not 0 <= event['step'] <= len(steps):
                raise ValueError('spawn outside recorded endpoint')
            if spawns and event['step'] < spawns[-1].step:
                raise ValueError('spawns must be recorded in application order')
            obj = EncounterObject(record['seed'], event['kind'], tuple(event['position']))
            spawns.append(SpawnEvent(event['step'], obj.kind, obj.position))
        return cls(record['seed'], steps, spawns)
