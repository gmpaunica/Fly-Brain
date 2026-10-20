"""Cumulative paired motor evidence with positive, repeated directional margin."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Decision:
    """Immutable control calculation, never additional neural activity."""
    cumulative: tuple[int,...]=(0,0,0,0,0,0)
    left: int=0
    right: int=0
    strength: float=0.
    candidate: str|None=None
    streak: int=0
    side: str|None=None
    committed: bool=False
    first_strong_tick: int|None=None
    deadline: int|None=None
    timed_out: bool=False


class TemporalReadout:
    """Only six manifest-selected spike counts and sensory threat flags enter."""
    def __init__(self,manifest: dict):
        self.indices=tuple(next(i for i,n in enumerate(manifest['nodes'])
            if n['metadata'].get('side')==side and n['metadata'].get('cell_type')==cell)
            for side in ('left','right') for cell in ('i1','b3','b1'))
        self.size=len(manifest['nodes']);self.reset()

    def reset(self) -> None:
        """Clear complete episode state; topology and assumptions remain fixed."""
        self.tick=0;self.cumulative=np.zeros(6,dtype=np.int64);self.first_strong=None
        self.candidate=None;self.streak=0;self.side=None;self.hold_weak=False;self.last=Decision()

    def recovered(self) -> None:
        """Rearm only through weak continuity; never reset neurons mid-flight."""
        self.cumulative[:]=0;self.first_strong=None;self.candidate=None;self.streak=0
        self.side=None;self.hold_weak=True

    def update(self,actual: np.ndarray,reference: np.ndarray,strong: bool,weak: bool,active: bool) -> Decision:
        """Advance one 20 ms readout observation, without coordinates or RNG."""
        a=np.asarray(actual);b=np.asarray(reference)
        if a.shape!=b.shape or a.shape!=(self.size,) or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu' or np.any(a<0) or np.any(b<0):
            raise ValueError('complete nonnegative integer count vectors required')
        self.tick+=1;committed=False
        if self.hold_weak and weak:
            self.cumulative[:]=0
        else:
            self.hold_weak=False
            self.cumulative+=a[list(self.indices)].astype(np.int64)-b[list(self.indices)].astype(np.int64)
        c=self.cumulative;left=int(c[0]+c[1]+c[5]);right=int(c[3]+c[4]+c[2]);difference=right-left
        if strong and self.first_strong is None and not self.hold_weak:self.first_strong=self.tick
        deadline=None if self.first_strong is None else self.first_strong+25
        timeout=deadline is not None and self.tick>deadline
        if not active and self.side is None and deadline is not None and not timeout and not self.hold_weak:
            candidate=('right' if difference>=1 else 'left' if difference<=-1 else None)
            if candidate and (right if candidate=='right' else left)<1:candidate=None
            if candidate is None:self.candidate=None;self.streak=0
            elif candidate==self.candidate:self.streak+=1
            else:self.candidate=candidate;self.streak=1
            if self.streak>=2:self.side=self.candidate;committed=True
        elif timeout and self.side is None:self.candidate=None;self.streak=0
        self.last=Decision(tuple(int(v) for v in c),left,right,math.sqrt(float(c@c)),
            self.candidate,self.streak,self.side,committed,self.first_strong,deadline,timeout)
        return self.last
