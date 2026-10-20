"""Real paired neurons -> temporal side impulse -> retained physical policy."""
from dataclasses import dataclass,asdict,replace
import json
from pathlib import Path
from flybrain.turnflee.session import TurnFleeSession,ResponseLog
from flybrain.turnflee.control import EscapeBehavior,WingScores,WingResponse
from flybrain.turnflee.neural import ResponsePair
from flybrain.retreat3d.world import RetreatWorld,PredatorInput
from flybrain.retreat3d.threat import ThreatMonitor
from flybrain.retreat3d.neural import DecodeResult
from flybrain.retreat3d.interaction import SpawnEvent
from flybrain.flight3d.network import TransmissionTrace
from .config import CandidateConfig
from .readout import TemporalReadout


@dataclass
class RecoveryLog(ResponseLog):
    """Versioned candidate-bound action log, preserving old log dispatch."""
    candidate: CandidateConfig|None=None
    intervention: str='intact'

    def save(self,path:Path) -> None:
        if self.candidate is None:raise ValueError('candidate binding required')
        with path.open('x',encoding='utf-8') as f:
            json.dump(dict(schema='escape-recovery-actions-v1',seed=self.seed,dt_seconds=.02,
                candidate=asdict(self.candidate),candidate_identity=self.candidate.identity,
                intervention=self.intervention,
                steps=[v.axes for v in self.steps],spawns=[asdict(v) for v in self.spawns]),f,indent=2)


class RecoverySession(TurnFleeSession):
    """Additive integration; only committed spike decision reaches side policy."""
    def __init__(self,seed:int,candidate:CandidateConfig):
        candidate.verify();self.candidate=candidate;self.intervention='intact'
        super().__init__(seed,candidate.response)
        self.verified_default=False
        self.log=RecoveryLog(seed,response_config=self.config,candidate=candidate)

    def _install(self) -> None:
        self.world=RetreatWorld(self.seed);self.controller=EscapeBehavior(self.seed)
        self.pair=ResponsePair(self.manifest,self.config);self.brain=self.pair.actual
        self._apply_intervention()
        self.scores=WingScores(self.manifest);self.readout=TemporalReadout(self.manifest)
        self.transmission_trace=TransmissionTrace(len(self.brain.edge_order));self.threat=ThreatMonitor();self.decoded=DecodeResult()
        self.neural_origin=0.;self.step_index=0;self.mode='temporal spike recovery | experimental'
        self.snapshot=self._snapshot()

    def _snapshot(self):
        base=super()._snapshot();d=self.readout.last
        return replace(base,wing_scores=(float(d.left),float(d.right)),response_strength=d.strength,symmetry_breaking=False)

    def spawn(self,kind:str,position:tuple|None=None) -> None:
        super().spawn(kind,position);self._apply_intervention();self.readout.reset();self.snapshot=self._snapshot()

    def _apply_intervention(self) -> None:
        for brain in (self.pair.actual,self.pair.reference):
            brain.silenced=({brain.ids[i] for i in brain.roles['vision']} if self.intervention=='visual-silent' else set())

    def set_intervention(self,value:str) -> None:
        """Evaluator-only episode intervention, bound in every action log."""
        if value not in ('intact','visual-silent') or self.brain.steps!=0:raise ValueError('intervention must precede neural episode')
        self.intervention=value;self.log.intervention=value;self._apply_intervention()

    def advance(self,predator:PredatorInput|None=None):
        applied=predator or PredatorInput()
        if not isinstance(applied,PredatorInput):raise ValueError('environment input type required')
        obs=self.world.observe();self.pair.advance(obs);threat=self.threat.update(obs.eyes)
        d=self.readout.update(self.brain.last_counts,self.pair.reference.last_counts,threat.strong,threat.weak_continuous,
                              self.controller.escape.state!='idle')
        # The retained policy cannot see a tie or an unconfirmed preference.
        impulse=WingResponse(left=1. if d.side=='left' else 0.,right=1. if d.side=='right' else 0.,strength=1.) if d.committed else WingResponse()
        command=self.controller.respond(obs,impulse,threat)
        if self.controller.escape.recovered:self.readout.recovered()
        self.world.step(command,applied);self.transmission_trace.update(self.world.time,self.brain.edge_transmissions)
        if self._recording:self.log.steps.append(applied)
        self.step_index+=1;self.snapshot=self._snapshot();return self.snapshot

    def new_trial(self) -> None:
        self.seed+=1;self.intervention='intact';self.log=RecoveryLog(self.seed,response_config=self.config,candidate=self.candidate);self._recording=True;self._install()

    @classmethod
    def from_log(cls,path:Path):
        r=json.loads(path.read_text())
        if r['schema']!='escape-recovery-actions-v1':return TurnFleeSession.from_log(path)
        c=CandidateConfig.from_record(r['candidate'])
        if c.identity!=r['candidate_identity'] or r['dt_seconds']!=.02 or type(r['seed']) is not int or r['seed']<0:raise ValueError('recovery log identity mismatch')
        if r['intervention'] not in ('intact','visual-silent'):raise ValueError('replay intervention mismatch')
        previous=-1
        for event in r['spawns']:
            if type(event['step']) is not int or not previous<=event['step']<=len(r['steps']):raise ValueError('unordered replay event')
            previous=event['step']
        s=cls(r['seed'],c)
        s.log=RecoveryLog(r['seed'],steps=[PredatorInput(tuple(v)) for v in r['steps']],
            spawns=[SpawnEvent(v['step'],v['kind'],tuple(v['position'])) for v in r['spawns']],response_config=c.response,candidate=c,intervention=r['intervention'])
        if any(v.step<0 or v.step>len(s.log.steps) for v in s.log.spawns):raise ValueError('spawn endpoint mismatch')
        s.intervention=r['intervention'];s.replay();return s
