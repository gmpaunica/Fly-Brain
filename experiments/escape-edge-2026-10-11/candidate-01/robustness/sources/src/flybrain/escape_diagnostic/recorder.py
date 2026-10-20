"""Full 1 ms recording. Observations never enter the original state updates."""
from collections import deque
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from flybrain.encounters3d.diagnostics import DiagnosticInterval
from flybrain.flight3d.contracts import FlightObservation
from flybrain.encounters3d.neural import threat_contrast
from flybrain.retreat3d.neural import MotionBrain, MotionVisionBrain, wing_indices
from flybrain.turnflee.neural import ResponsePair, ResponseConfig, apply_gains

ENGINEERING = Path('artifacts/escape-diagnostic-2026-10-07/engineering.json')


def engineering_reserve(label: str, steps: int) -> None:
    """Persist cumulative individual synthetic kernel calls before execution."""
    ENGINEERING.parent.mkdir(parents=True, exist_ok=True)
    value=json.loads(ENGINEERING.read_text()) if ENGINEERING.exists() else dict(steps=0,calls=[],limit=2000)
    if steps<=0 or value['steps']+steps>2000: raise ValueError('engineering budget exceeded')
    value['steps']+=steps; value['calls'].append(dict(label=label,steps=steps))
    temp=ENGINEERING.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2))
    os.replace(temp,ENGINEERING)


@torch.jit.script
def advance_recorded(voltage: torch.Tensor, previous: torch.Tensor, phase: torch.Tensor,
                     rates: torch.Tensor, pre: torch.Tensor, post: torch.Tensor,
                     weights: torch.Tensor, inputs: torch.Tensor, live: torch.Tensor,
                     steps: int):
    counts = torch.zeros_like(voltage, dtype=torch.int64)
    delivered = torch.zeros_like(weights)
    effective = (~inputs[post]) & live[post]
    current_sum = torch.zeros_like(voltage)
    current_peak = torch.zeros_like(voltage)
    discarded = torch.zeros_like(voltage)
    peak_voltage = torch.full_like(voltage, -float('inf'))
    peak_phase = torch.zeros_like(voltage)
    first = torch.full_like(counts, -1)
    pre_v=torch.empty((steps,voltage.numel()),dtype=voltage.dtype)
    post_v=torch.empty_like(pre_v)
    spikes_trace=torch.empty((steps,voltage.numel()),dtype=torch.bool)
    pre_ph=torch.empty_like(pre_v); post_ph=torch.empty_like(pre_v)
    edges=torch.empty((steps,weights.numel()),dtype=weights.dtype)
    for step in range(steps):
        # This block retains _advance_measured's operation order verbatim.
        voltage = voltage * 0.9900498337491681
        contribution = weights * previous[pre]
        voltage.index_add_(0, post, contribution)
        delivered = delivered + contribution * effective
        phase = phase + rates * .001
        incoming = torch.zeros_like(voltage)
        incoming.index_add_(0, post, contribution * effective)
        lost = torch.zeros_like(voltage)
        lost.index_add_(0, post, contribution * (~effective))
        current_sum = current_sum + incoming
        current_peak = torch.maximum(current_peak, incoming)
        discarded = discarded + lost
        peak_voltage = torch.maximum(peak_voltage, torch.where(
            inputs | ~live, torch.zeros_like(voltage), voltage))
        peak_phase = torch.maximum(peak_phase, torch.where(
            inputs & live, phase, torch.zeros_like(phase)))
        pre_v[step]=voltage; pre_ph[step]=phase; edges[step]=contribution
        forced = phase >= 1.
        phase = torch.where(forced, phase - 1., phase)
        spikes = torch.where(inputs, forced, voltage >= 1.) & live
        voltage = torch.where(inputs | ~live, torch.zeros_like(voltage),
                              voltage - spikes.to(voltage.dtype))
        previous = spikes.to(voltage.dtype)
        counts = counts + spikes.to(torch.int64)
        first = torch.where(spikes & (first < 0), torch.full_like(first, step + 1), first)
        post_v[step]=voltage; post_ph[step]=phase; spikes_trace[step]=spikes
    return (voltage,previous,phase,counts,delivered,current_sum,current_peak,discarded,
            peak_voltage,peak_phase,first,pre_v,post_v,spikes_trace,pre_ph,post_ph,edges)


class PassiveMixin:
    """Copies of full state and original diagnostics; bounded to one interval."""
    def reset(self) -> None:
        super().reset()
        self.trace=None; self.initial=None; self.recorded_rates=None
        self.diagnostic=None; self.diagnostic_intervals=[]
        self._first_spike=[None]*len(self.ids)
        self.salience=None; self.growth=None; self.patch=None

    @torch.no_grad()
    def advance_rates(self, rates: np.ndarray, steps: int=20) -> np.ndarray:
        """Record both brains with original float64 updates and history formula."""
        values=np.asarray(rates,dtype=float)
        if values.shape!=(len(self.ids),) or not np.isfinite(values).all() or np.any(values<0) or np.any(values>240):
            raise ValueError('one finite 0..240 Hz input rate per neuron required')
        if type(steps) is not int or steps<=0: raise ValueError('positive integer neural steps required')
        self.initial=tuple(t.numpy().copy() for t in (self.voltage,self.previous,self.phase))
        self.recorded_rates=values.copy()
        live=torch.tensor([i not in self.silenced for i in self.ids])
        r=advance_recorded(self.voltage,self.previous,self.phase,torch.from_numpy(values),
                           self.pre,self.post,self.weights,self.inputs,live,steps)
        self.voltage,self.previous,self.phase,counts,delivered=r[:5]
        start=self.steps*.001; self.steps+=steps; elapsed=steps*.001
        for k,tau in enumerate((.1,.5)):
            alpha=-math.expm1(-elapsed/tau)
            self.history[k]=(1-alpha)*self.history[k]+alpha*counts.double()/elapsed
            self.history[k,~live]=0.
        self.last_counts=counts.numpy().copy()
        totals=np.zeros(len(self.edge_order)); totals[self.dynamic_edges]=delivered.numpy()
        self.edge_transmissions=tuple(float(v) for v in totals)
        first=tuple((int(s)+round(start/.001))*.001 if s>=0 else None for s in r[10].tolist())
        for i,t in enumerate(first):
            if self._first_spike[i] is None: self._first_spike[i]=t
        self.diagnostic=DiagnosticInterval(start,self.steps*.001,
            *(tuple(float(v) for v in t.tolist()) for t in r[5:10]),
            tuple(int(v) for v in self.last_counts),first,tuple(self._first_spike),tuple(sorted(self.silenced)))
        self.diagnostic_intervals=[self.diagnostic]
        self.trace=tuple(t.numpy().copy() for t in r[11:])
        return self.last_counts.copy()


class RecordedActual(PassiveMixin, MotionVisionBrain):
    """Original vision-plus-motion encoder with copies of patch measurements."""
    def sensory_rates(self, observation: FlightObservation) -> np.ndarray:
        salience=threat_contrast(observation.eyes)
        growth=np.zeros_like(salience) if self.previous_eyes is None else np.maximum(salience-self.previous_eyes,0.)
        # The original encoder, rather than this observer, computes its inputs.
        rates=super().sensory_rates(observation)
        self.patch=np.zeros((len(self.ids),3),dtype=np.float64)
        for side,eye in (('left',0),('right',1)):
            group=[int(i) for i in self.roles.get('vision',()) if self.nodes[i]['metadata'].get('side')==side]
            for rank,i in enumerate(group):
                row=(rank%6)*4; col=(rank//6%8)*6
                p=salience[eye,row:row+4,col:col+6]; g=growth[eye,row:row+4,col:col+6]
                self.patch[i]=[float(p.max()),float(p.mean()),float(g.max())]
        self.salience=salience.copy(); self.growth=growth.copy()
        return rates


class RecordedReference(PassiveMixin, MotionBrain):
    """Independent original motion-only encoder and recorded neuron state."""


class RecordedPair(ResponsePair):
    """Reuse original paired windows and guards; replace only passive observers."""
    def __init__(self, manifest: dict, config: ResponseConfig):
        config.verify(manifest); self.config=config
        self.actual=RecordedActual(manifest); self.reference=RecordedReference(manifest)
        apply_gains(self.actual,config); apply_gains(self.reference,config)
        self.indices=wing_indices(manifest); self.reset()


def chunk_arrays(pair: RecordedPair, rgb: np.ndarray) -> dict[str,np.ndarray]:
    """Self-contained complete interval, integer end-of-step IDs, no pickle."""
    arrays=dict(rgb=rgb.copy(),step_index=np.arange(pair.actual.steps-19,pair.actual.steps+1,dtype=np.int64),
                salience=pair.actual.salience.astype(np.float64),growth=pair.actual.growth.astype(np.float64),patch=pair.actual.patch)
    names=('pre_voltage','post_voltage','spikes','pre_phase','post_phase','contribution')
    for label,b in (('actual',pair.actual),('reference',pair.reference)):
        arrays.update({f'{label}_{k}':v for k,v in zip(names,b.trace)})
        arrays.update({f'{label}_initial_{k}':v for k,v in zip(('voltage','previous','phase'),b.initial)})
        arrays[f'{label}_rates']=b.recorded_rates.copy()
        arrays[f'{label}_history']=b.history.numpy().copy()
    return arrays
