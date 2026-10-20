"""Spikes choose a side; IMU supplies engineered flight stabilization only."""
from dataclasses import dataclass
import math
import random
import numpy as np
from flybrain.flight3d.contracts import CONTROL_DT, FlightCommand, FlightObservation
from flybrain.retreat3d.controller import BehaviorController, KinematicEstimate, stabilize
from flybrain.retreat3d.threat import ThreatMeasurement
from flybrain.retreat3d.neural import wing_indices


@dataclass(frozen=True)
class WingResponse:
    """Signed wing scores and norm of the six declared steering increments."""
    left: float = 0.
    right: float = 0.
    strength: float = 0.
    bilateral: bool = False


class WingScores:
    """Signed increments: i1/b3 reduce same-side amplitude, b1 increases it.

    Equal per-spike weights are an engineering readout, not a muscle model.
    Only the six existing i1/b3/b1 steering endpoints enter this calculation.
    """
    def __init__(self, manifest: dict):
        self.entries=tuple((j,n['metadata']['side'],n['metadata']['cell_type'])
                           for j,i in enumerate(wing_indices(manifest))
                           for n in (manifest['nodes'][i],)
                           if n['metadata']['cell_type'] in ('i1','b3','b1'))
        if len(self.entries)!=6:
            raise ValueError('six bilateral i1/b3/b1 endpoints required')

    def __call__(self, vector: np.ndarray) -> WingResponse:
        values=np.asarray(vector,dtype=float)
        if values.shape!=(48,) or not np.isfinite(values).all():
            raise ValueError('finite manifest-ordered signed 48-wing vector required')
        scores={'left':0.,'right':0.}; norms={'left':0.,'right':0.}
        for j,side,cell in self.entries:
            other='right' if side=='left' else 'left'
            scores[other if cell=='b1' else side]+=float(values[j])
            norms[side]+=float(values[j])**2
        return WingResponse(scores['left'],scores['right'],
                            math.sqrt(sum(norms.values())),all(v>0 for v in norms.values()))


class EscapePolicy:
    """90° turn, .12 m/s sustained flight, weak-threat braking; no RGB bearing."""
    def __init__(self, seed: int):
        self.seed=seed; self.reset()

    def reset(self) -> None:
        self.rng=random.Random(self.seed); self.state='idle'; self.side=None
        self.target_heading=0.; self.tick=0; self.pending_until=-1
        self.symmetry_breaking=False; self.home_guard=False; self.recovered=False
        self.trigger_count=0; self.last_response=WingResponse()
        self.desired_velocity=np.zeros(3)

    def command(self, estimate: KinematicEstimate, observation: FlightObservation,
                response: WingResponse, threat: ThreatMeasurement) -> FlightCommand | None:
        """No physical pose/target object enters this boundary."""
        self.tick+=1; self.recovered=False; self.last_response=response
        if threat.strong:
            self.pending_until=self.tick+25
        if self.state=='idle' and self.tick<=self.pending_until and response.strength>=1:
            difference=response.right-response.left
            if difference!=0 or response.bilateral:
                self.symmetry_breaking=difference==0
                self.side=('right' if difference>0 else 'left') if difference else self.rng.choice(('left','right'))
                self.target_heading=float(estimate.orientation[2])+(math.pi/2 if self.side=='right' else -math.pi/2)
                self.state='turning'; self.trigger_count+=1; self.pending_until=-1
        if self.state=='idle':
            return None
        e=estimate
        radius=float(np.linalg.norm(e.displacement[:2]))
        if radius>.22: self.home_guard=True
        elif radius<.18: self.home_guard=False
        error=self.target_heading-float(e.orientation[2])
        if self.state=='turning' and abs(error)<=math.radians(3):
            self.state='fleeing'
        if threat.weak_continuous:
            self.state='braking'
        if self.state=='braking' and np.linalg.norm(e.velocity[:2])<.02 and abs(e.velocity[2])<.02:
            self.state='idle'; self.recovered=True; self.pending_until=-1
            self.desired_velocity[:]=0.
            return None
        desired=np.zeros(3)
        desired[2]=float(np.clip(2*(.1-e.altitude),-.06,.06))
        heading=self.target_heading
        if self.state=='fleeing':
            if self.home_guard:
                radial=e.displacement[:2]/max(radius,1e-12)
                # Right tangent at +X points -Y under clockwise heading.
                tangent=np.array((radial[1],-radial[0]))*(1 if self.side=='right' else -1)
                direction=tangent-.4*radial
                direction/=np.linalg.norm(direction)
                heading=math.atan2(float(direction[0]),float(direction[1]))
                desired[:2]=.12*direction
            else:
                desired[:2]=.12*np.array((math.sin(heading),math.cos(heading)))
        elif self.state=='braking':
            desired[2]=0.
        yaw_error=(heading-e.orientation[2]+math.pi)%math.tau-math.pi
        yaw=0. if self.state=='braking' else float(np.clip(yaw_error*2/math.pi,-1,1))
        self.desired_velocity=desired.copy()
        return stabilize(e,desired,yaw,1. if observation.foot_contact else 0.)


class EscapeBehavior(BehaviorController):
    """Retained idle/flower behavior, with an explicitly enabled escape policy."""
    def reset(self) -> None:
        super().reset(); self.escape=EscapePolicy(self.seed)

    @property
    def state(self) -> str:
        return self.escape.state if self.escape.state!='idle' else super().state

    def encounter(self, kind: str, observation: FlightObservation | None = None) -> None:
        super().encounter(kind,observation); self.escape.reset()

    def calibrated_start(self, state: str) -> None:
        """Declared initial-state assumption, never an actual world-state handoff.

        Upright/heading-zero rest is the original calibration. Hover/cruise
        start at nominal .1 m; cruise has nominal .06 m/s forward velocity.
        The evaluator initializes physics independently to the same constants.
        """
        if state not in ('resting','hovering','cruising'):
            raise ValueError('named calibrated start required')
        self.idle.estimate=KinematicEstimate()
        if state!='resting':
            self.idle.estimate.altitude=.1
            if state=='cruising': self.idle.estimate.velocity[1]=.06
            self.last_contact=False; self.idle.transition('cruising')

    def respond(self, observation: FlightObservation, response: WingResponse,
                threat: ThreatMeasurement) -> FlightCommand:
        if self.kind!='predator':
            return super().command(observation)
        self.idle.estimate.update(observation); self.last_contact=observation.foot_contact
        command=self.escape.command(self.idle.estimate,observation,response,threat)
        if command is not None:
            return command
        if self.escape.recovered:
            self.idle.heading=float(self.idle.estimate.orientation[2]); self.idle.transition('cruising')
        return self.idle.command(observation,update=False)
