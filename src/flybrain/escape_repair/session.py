"""Additive repaired policy integration with source-bound action replay."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from flybrain.escape_recovery.config import CandidateConfig
from flybrain.escape_recovery.session import RecoveryLog, RecoverySession
from flybrain.escape_diagnostic_recovery.runner import verify_hashes
from flybrain.flight3d.anatomy import digest
from flybrain.retreat3d.interaction import SpawnEvent
from flybrain.retreat3d.world import EncounterObject, PredatorInput
from .control import SafeEscapeBehavior
from .threat import ClearanceMonitor


def source_hashes() -> dict[str, str]:
    """Bind every adjacent implementation file, without changing old identities."""
    return {p.as_posix(): digest(p) for p in sorted(Path('src/flybrain/escape_repair').glob('*.py'))}


@dataclass
class RepairLog(RecoveryLog):
    """Replay identifies the repaired policy and optional named initial state."""
    initial_state: str | None = None

    def save(self, path: Path) -> None:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(dict(schema='escape-repair-actions-v1', seed=self.seed, dt_seconds=.02,
                candidate=asdict(self.candidate), candidate_identity=self.candidate.identity,
                repair_sources=source_hashes(), initial_state=self.initial_state,
                intervention=self.intervention, steps=[v.axes for v in self.steps],
                spawns=[asdict(v) for v in self.spawns]), stream, indent=2, allow_nan=False)


class RepairSession(RecoverySession):
    """Same actual/reference neurons and temporal readout; repaired flight only."""
    def __init__(self, seed: int, candidate: CandidateConfig, initial_state: str | None = None):
        if initial_state not in (None, 'resting', 'hovering', 'cruising'):
            raise ValueError('named calibrated start required')
        self.initial_state = initial_state
        super().__init__(seed, candidate)
        self.log = self._new_log()

    def _new_log(self) -> RepairLog:
        return RepairLog(self.seed, response_config=self.config, candidate=self.candidate,
                         initial_state=self.initial_state, intervention=self.intervention)

    def _install(self) -> None:
        super()._install()
        self.controller = SafeEscapeBehavior(self.seed)
        self.threat = ClearanceMonitor()
        if self.initial_state is not None:
            self.controller.calibrated_start(self.initial_state)
            if self.initial_state != 'resting':
                self.world.position[2] = .1
                self.world.foot_contact = False
                if self.initial_state == 'cruising':
                    self.world.velocity[1] = .06
            else:
                self.controller.idle.transition('resting')
                self.controller.idle.deadline = 10.
        self.snapshot = self._snapshot()

    def new_trial(self) -> None:
        super().new_trial()
        self.log = self._new_log()

    @classmethod
    def from_log(cls, path: Path, *, replay: bool = True) -> 'RepairSession':
        """Validate policy, input types/order and restore all state before replay."""
        r = json.loads(path.read_text(encoding='utf-8'))
        if (r['schema'] != 'escape-repair-actions-v1' or r['dt_seconds'] != .02
                or type(r['seed']) is not int or r['seed'] < 0
                or r['repair_sources'] != source_hashes()
                or r['intervention'] not in ('intact', 'visual-silent')):
            raise ValueError('repair action binding mismatch')
        verify_hashes(r['repair_sources'])
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
