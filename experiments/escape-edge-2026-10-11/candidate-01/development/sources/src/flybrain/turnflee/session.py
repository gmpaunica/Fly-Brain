"""Sensory-only candidate session, immutable diagnostics and versioned replay."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from flybrain.flight3d.anatomy import CACHE, digest, load_manifest
from flybrain.flight3d.network import TransmissionTrace
from flybrain.encounters3d.diagnostics import DiagnosticFlightBrain
from flybrain.retreat3d.interaction import InteractionLog, SpawnEvent
from flybrain.retreat3d.session import RetreatSession
from flybrain.retreat3d.neural import DecodeResult, MotionVisionBrain
from flybrain.retreat3d.threat import ThreatMonitor
from flybrain.retreat3d.world import RetreatSnapshot, RetreatWorld, PredatorInput, EncounterObject
from .neural import ResponseConfig, ResponsePair, apply_gains
from .control import EscapeBehavior, WingScores

RUN = Path('experiments/turnflee-2026-10-06')


def promoted_config() -> ResponseConfig | None:
    """Default activation requires recorded passage of every scientific gate."""
    path=RUN/'receipt.json'
    if not path.exists(): return None
    receipt=json.loads(path.read_text(encoding='utf-8'))
    if receipt['status']!='passed': return None
    for name,value in receipt['sha256'].items():
        if digest(RUN/name)!=value: raise ValueError('promotion receipt mismatch')
    result=json.loads((RUN/'results.json').read_text(encoding='utf-8'))
    from .protocol import validation_gate, closed_gate
    if not validation_gate(result['validation']) or not closed_gate(result['closed_loop']):
        raise ValueError('promotion gates failed')
    config=ResponseConfig.from_record(json.loads((RUN/'response.json').read_text(encoding='utf-8')))
    config.verify(load_manifest())
    return config


@dataclass(frozen=True)
class TurnFleeSnapshot(RetreatSnapshot):
    wing_scores: tuple[float,float] = (0.,0.)
    selected_side: str | None = None
    response_strength: float = 0.
    symmetry_breaking: bool = False
    escape_state: str = 'idle'
    home_guard: bool = False
    response_config: ResponseConfig | None = None


@dataclass
class ResponseLog(InteractionLog):
    """Original applied inputs, plus immutable full effective-response binding."""
    response_config: ResponseConfig | None = None

    def save(self, path: Path) -> None:
        if self.response_config is None: raise ValueError('response configuration required')
        with path.open('x',encoding='utf-8') as stream:
            json.dump(dict(schema='turnflee-actions-v2',seed=self.seed,dt_seconds=.02,
                circuit_sha256=digest(CACHE/'circuit.json'),
                response_config=asdict(self.response_config),response_identity=self.response_config.identity,
                steps=[c.axes for c in self.steps],
                spawns=[asdict(e) for e in self.spawns]),stream,indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ResponseLog':
        """Validate source, response identity and ordered endpoint events."""
        record=json.loads(path.read_text(encoding='utf-8'))
        if (record['schema']!='turnflee-actions-v2' or record['dt_seconds']!=.02
            or record['circuit_sha256']!=digest(CACHE/'circuit.json')
            or type(record['seed']) is not int or record['seed']<0):
            raise ValueError('response action log binding mismatch')
        config=ResponseConfig.from_record(record['response_config']); config.verify(load_manifest())
        if config.identity!=record['response_identity']: raise ValueError('response identity mismatch')
        steps=[PredatorInput(tuple(c)) for c in record['steps']]; spawns=[]
        for event in record['spawns']:
            if (type(event['step']) is not int or not 0<=event['step']<=len(steps)
                or spawns and event['step']<spawns[-1].step):
                raise ValueError('spawn outside ordered recorded endpoint')
            obj=EncounterObject(record['seed'],event['kind'],tuple(event['position']))
            spawns.append(SpawnEvent(event['step'],obj.kind,obj.position))
        return cls(record['seed'],steps,spawns,config)


class TurnFleeSession(RetreatSession):
    """Explicit candidate construction is for the authorized experiment only.

    Launchers use promoted_config, never a development candidate automatically.
    Anatomy records actual spikes; difference vectors are control diagnostics.
    """
    def __init__(self, seed: int, config: ResponseConfig):
        self.manifest=load_manifest(); config.verify(self.manifest)
        self.seed=seed; self.config=config; self.decoder=None
        self.verified_default=promoted_config()==config
        self.log=ResponseLog(seed,response_config=config); self._recording=True
        self._install()

    def _install(self) -> None:
        self.world=RetreatWorld(self.seed); self.controller=EscapeBehavior(self.seed)
        self.pair=ResponsePair(self.manifest,self.config); self.brain=self.pair.actual
        self.scores=WingScores(self.manifest)
        self.transmission_trace=TransmissionTrace(len(self.brain.edge_order))
        self.threat=ThreatMonitor(); self.decoded=DecodeResult()
        self.neural_origin=0.; self.step_index=0
        self.mode='turn-and-flee | engineered gain/flight policy'
        self.snapshot=self._snapshot()

    def _snapshot(self) -> TurnFleeSnapshot:
        base=super()._snapshot(); escape=self.controller.escape
        response=self.scores(self.pair.last_vector)
        values={f.name:getattr(base,f.name) for f in fields(base)}
        values['escape_supported']=self.verified_default
        return TurnFleeSnapshot(**values,wing_scores=(response.left,response.right),
            selected_side=escape.side,response_strength=response.strength,
            symmetry_breaking=escape.symmetry_breaking,escape_state=escape.state,
            home_guard=escape.home_guard,response_config=self.config)

    def spawn(self, kind: str, position: tuple | None = None) -> None:
        # Original spawn semantics preserve body/IMU and event ordering.
        super().spawn(kind,position)
        if kind!='flower': apply_gains(self.pair.actual,self.config)
        self.mode='retained flower teacher' if kind=='flower' else 'turn-and-flee | engineered gain/flight policy'
        self.snapshot=self._snapshot()

    def advance(self, predator: PredatorInput | None = None) -> TurnFleeSnapshot:
        """Advance paired neurons/control once, then apply four physics steps."""
        applied=predator or PredatorInput()
        if not isinstance(applied,PredatorInput): raise ValueError('environment-only PredatorInput required')
        observation=self.world.observe(); vector=self.pair.advance(observation)
        threat=self.threat.update(observation.eyes)
        command=self.controller.respond(observation,self.scores(vector),threat)
        self.world.step(command,applied)
        self.transmission_trace.update(self.world.time,self.brain.edge_transmissions)
        if self._recording: self.log.steps.append(applied)
        self.step_index+=1; self.snapshot=self._snapshot()
        return self.snapshot

    def new_trial(self) -> None:
        """Fresh idle episode with the same immutable response configuration."""
        self.seed+=1; self.log=ResponseLog(self.seed,response_config=self.config); self._recording=True
        self._install()

    @classmethod
    def from_log(cls, path: Path) -> RetreatSession:
        """Use historical v1 dynamics for old logs; bound v2 for candidate logs."""
        record=json.loads(path.read_text(encoding='utf-8'))
        if record['schema']=='retreat3d-actions-v1': return RetreatSession.from_log(path)
        log=ResponseLog.load(path); session=cls(log.seed,log.response_config)
        session.log=log; session.replay(); return session
