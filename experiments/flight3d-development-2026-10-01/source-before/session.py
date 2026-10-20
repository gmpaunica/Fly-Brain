"""Headless session scheduling and passive simulation-time activity traces."""

from dataclasses import replace
import numpy as np
from .contracts import FlightCommand, TrialSpec
from .world import FlightWorld, make_trial
from .teacher import FlightTeacher


class ActivityTrace:
    """Read-only display memory; neither camera nor render time affects fading."""

    def __init__(self,ids: tuple[str,...],decay_seconds: float=.2):
        self.ids=ids; self.decay_seconds=decay_seconds
        self.last=np.full(len(ids),-np.inf); self.time=-1.

    def update(self,time: float,counts) -> None:
        if time<self.time: self.last[:]=-np.inf
        counts=np.asarray(counts)
        if counts.shape!=(len(self.ids),): raise ValueError('activity count/ID mismatch')
        self.last[counts>0]=time; self.time=time

    def values(self,time: float) -> np.ndarray:
        return np.exp(-np.maximum(0,time-self.last)/self.decay_seconds)


class FlightSession:
    """Controller/world orchestration, callable identically with or without UI."""

    def __init__(self,case: TrialSpec,mode: str='teacher',checkpoint=None,simulate_brain: bool=True):
        from .anatomy import load_manifest, CACHE, digest
        from .neural import FlightBrain, FlightController, load_readout
        self.world=FlightWorld(case); self.mode=mode; self.checkpoint=checkpoint
        self.manifest=load_manifest() if simulate_brain else None
        if mode=='teacher': self.controller=FlightTeacher()
        elif mode=='manual': self.controller=None
        elif mode in ('fixed','trained','untrained'):
            if self.manifest is None: raise ValueError('neural mode requires a manifest')
            if mode!='fixed' and checkpoint is None: raise ValueError('checkpoint required; no substitute controller')
            model=load_readout(checkpoint,digest(CACHE/'circuit.json')) if checkpoint else None
            self.controller=FlightController(self.manifest,model)
        else: raise ValueError('unknown control mode')
        self.brain=(self.controller.brain if hasattr(self.controller,'brain') else FlightBrain(self.manifest)) if simulate_brain else None
        self.snapshot=self._snapshot()

    def _snapshot(self):
        return replace(self.world.snapshot(),mode=self.mode,
                       neuron_ids=self.brain.ids if self.brain else (),
                       spike_counts=tuple(int(v) for v in self.brain.last_counts) if self.brain else ())

    def advance(self,manual: FlightCommand | None=None):
        obs=self.world.observe()
        command=self.controller.command(obs) if self.controller is not None else (manual or FlightCommand())
        if self.brain is not None and not hasattr(self.controller,'brain'): self.brain.advance(obs)
        self.world.step(command); self.snapshot=self._snapshot(); return self.snapshot

    def replay(self) -> None:
        self.world.reset()
        if self.controller is not None: self.controller.reset()
        if self.brain is not None: self.brain.reset()
        self.snapshot=self._snapshot()

    def new_trial(self) -> None:
        self.world=FlightWorld(make_trial(self.world.case.seed+1)); self.replay()
