"""Unchanged visual encoding plus original motion rates; spike-only decoding."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
from flybrain.flight3d.contracts import FlightObservation
from flybrain.flight3d.neural import FlightBrain
from flybrain.encounters3d.neural import EscapeBrain
from flybrain.encounters3d.diagnostics import DiagnosticMixin


def wing_indices(manifest: dict) -> tuple[int, ...]:
    """All wing-control endpoints in original manifest order."""
    return tuple(i for i, n in enumerate(manifest['nodes'])
                 if set(n['roles']) & {'wing_power', 'wing_steering'})


def motion_rates(brain: FlightBrain, observation: FlightObservation) -> np.ndarray:
    """Exact historical motion-role formula, driving no other sensory role."""
    rates = np.zeros(len(brain.ids))
    motion = (*observation.angular_velocity, *(v/9.81 for v in observation.specific_force))
    for side in ('left', 'right'):
        group = [int(i) for i in brain.roles.get('motion', ())
                 if brain.nodes[i]['metadata'].get('side') == side]
        for rank, i in enumerate(group):
            signal = motion[(rank//2) % 6] * (1 if rank % 2 else -1)
            rates[i] = 120*(1+math.tanh(signal))
    return rates


class MotionVisionBrain(DiagnosticMixin, EscapeBrain):
    """Only add the original 16 motion inputs to the untouched predator encoder."""
    def sensory_rates(self, observation: FlightObservation) -> np.ndarray:
        return super().sensory_rates(observation) + motion_rates(self, observation)


class MotionBrain(FlightBrain):
    """Independent motion-only reference with identical dynamics and observations."""
    def sensory_rates(self, observation: FlightObservation) -> np.ndarray:
        return motion_rates(self, observation)


class BrainPair:
    """Two real brains; nonoverlapping 200 ms signed endpoint count windows."""
    def __init__(self, manifest: dict):
        self.actual = MotionVisionBrain(manifest)
        self.reference = MotionBrain(manifest)
        self.indices = wing_indices(manifest)
        self.reset()

    def reset(self) -> None:
        self.actual.reset(); self.reference.reset()
        self.intervals = 0
        self.accumulated = np.zeros(len(self.indices), dtype=np.int64)
        self.last_vector = np.zeros(len(self.indices), dtype=np.int64)

    def advance(self, observation: FlightObservation) -> np.ndarray | None:
        actual = self.actual.advance(observation)
        reference = self.reference.advance(observation)
        self.accumulated += actual[list(self.indices)] - reference[list(self.indices)]
        self.intervals += 1
        if self.intervals % 10:
            return None
        self.last_vector = self.accumulated.copy(); self.accumulated[:] = 0
        return self.last_vector.copy()


@dataclass(frozen=True)
class DecodeResult:
    """Confidence is a control calculation, not additional neural activity."""
    confident: bool = False
    heading_radians: float | None = None
    norm: float = 0.
    similarity: float = 0.
    margin: float = 0.


class FrozenDecoder:
    """Eight normalized signed-spike templates; no RGB, state or optimizer input."""
    def __init__(self, templates: np.ndarray, wing_ids: tuple[str, ...]):
        values = np.asarray(templates, dtype=np.float64)
        if values.shape != (8, 48) or len(wing_ids) != 48 or not np.isfinite(values).all():
            raise ValueError('eight finite manifest-ordered 48-wing templates required')
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values = np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)
        self.templates = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(8, 48)
        self.wing_ids = tuple(wing_ids)

    def decode(self, wing_vector: np.ndarray) -> DecodeResult:
        values = np.asarray(wing_vector, dtype=float)
        if values.shape != (48,) or not np.isfinite(values).all():
            raise ValueError('finite signed 48-wing vector required')
        norm = float(np.linalg.norm(values))
        if norm < 1:
            return DecodeResult(norm=norm)
        scores = self.templates @ (values/norm)
        ranking = np.argsort(scores, kind='stable')
        best = int(ranking[-1]); similarity = float(scores[best])
        margin = float(scores[best]-scores[ranking[-2]])
        confident = similarity >= .8 and margin >= .1
        return DecodeResult(confident, best*math.pi/4 if confident else None,
                            norm, similarity, margin)

    def save(self, path: Path) -> None:
        """Write a frozen manifest-bound decoder, refusing overwrite."""
        with path.open('x', encoding='utf-8') as stream:
            json.dump(dict(schema='retreat3d-spike-templates-v1', wing_ids=self.wing_ids,
                           templates=self.templates.tolist(), norm_min=1.,
                           similarity_min=.8, margin_min=.1), stream, indent=2)

    @classmethod
    def load(cls, path: Path, wing_ids: tuple[str, ...]) -> 'FrozenDecoder':
        record = json.loads(path.read_text(encoding='utf-8'))
        if (record['schema'] != 'retreat3d-spike-templates-v1' or
                tuple(record['wing_ids']) != tuple(wing_ids) or
                (record['norm_min'], record['similarity_min'], record['margin_min']) != (1., .8, .1)):
            raise ValueError('decoder identity/threshold binding mismatch')
        return cls(np.array(record['templates']), wing_ids)
