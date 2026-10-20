"""Paired frozen neurons, unchanged temporal readout, additive edge policy."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from flybrain.escape_repair.session import RepairSession, RepairLog
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.escape_diagnostic_recovery.runner import verify_hashes
from flybrain.flight3d.anatomy import digest
from flybrain.retreat3d.interaction import SpawnEvent
from flybrain.retreat3d.world import EncounterObject, PredatorInput
from .control import EdgeEscapeBehavior


def source_hashes() -> dict[str, str]:
    """Bind the complete additive implementation and inherited live dependencies."""
    paths = list(Path('src/flybrain/escape_edge').glob('*.py'))
    paths += list(Path('src/flybrain/escape_repair').glob('*.py'))
    return {p.as_posix(): digest(p) for p in sorted(paths)}


@dataclass
class EdgeLog(RepairLog):
    """Versioned replay records calibration separately from sensory inputs."""
    schema: str = 'escape-edge-actions-v1'

    def save(self, path: Path) -> None:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(dict(schema=self.schema, seed=self.seed, dt_seconds=.02,
                candidate=asdict(self.candidate), candidate_identity=self.candidate.identity,
                edge_sources=source_hashes(), initial_state=self.initial_state,
                intervention=self.intervention, steps=[v.axes for v in self.steps],
                spawns=[asdict(v) for v in self.spawns]), stream, indent=2, allow_nan=False)


class EdgeSession(RepairSession):
    """Only motor behavior changes; PyTorch forward dynamics stay frozen."""

    def _new_log(self) -> EdgeLog:
        return EdgeLog(self.seed, response_config=self.config, candidate=self.candidate,
                       initial_state=self.initial_state, intervention=self.intervention)

    def _install(self) -> None:
        super()._install()
        self.controller = EdgeEscapeBehavior(self.seed)
        if self.initial_state is not None:
            self.controller.calibrated_start(self.initial_state)
            if self.initial_state == 'resting':
                self.controller.idle.transition('resting')
                self.controller.idle.deadline = 10.
        self.mode = 'spike-triggered sustained edge escape | engineered flight'
        self.snapshot = self._snapshot()

    @classmethod
    def from_log(cls, path: Path, *, replay: bool = True) -> 'EdgeSession':
        """Validate bindings/actions and restore all episode state for replay."""
        r = json.loads(path.read_text(encoding='utf-8'))
        if (r['schema'] != 'escape-edge-actions-v1' or r['dt_seconds'] != .02
                or type(r['seed']) is not int or r['seed'] < 0
                or r['edge_sources'] != source_hashes()
                or r['intervention'] not in ('intact', 'visual-silent')):
            raise ValueError('edge action binding mismatch')
        verify_hashes(r['edge_sources'])
        candidate = CandidateConfig.from_record(r['candidate'])
        if candidate.identity != r['candidate_identity']:
            raise ValueError('candidate identity mismatch')
        steps = [PredatorInput(tuple(v)) for v in r['steps']]
        spawns = []
        previous = -1
        for event in r['spawns']:
            if type(event['step']) is not int or not previous <= event['step'] <= len(steps):
                raise ValueError('unordered spawn action')
            previous = event['step']
            obj = EncounterObject(r['seed'], event['kind'], tuple(event['position']))
            spawns.append(SpawnEvent(event['step'], obj.kind, obj.position))
        session = cls(r['seed'], candidate, r['initial_state'])
        session.intervention = r['intervention']
        session.log = session._new_log()
        session.log.steps, session.log.spawns = steps, spawns
        if replay:
            session.replay()
        return session
