"""Effective gains only; original count graph and neuron rules are preserved."""
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from flybrain.flight3d.anatomy import CACHE, digest
from flybrain.flight3d.contracts import FlightObservation
from flybrain.flight3d.neural import FlightBrain
from flybrain.retreat3d.neural import MotionBrain, MotionVisionBrain, wing_indices

GAIN_PAIRS = ((2,1), (4,2), (8,4))


def source_hashes() -> tuple[tuple[str,str], ...]:
    """Bind all executed circuit, sensor, world and candidate source bytes."""
    return tuple((p.as_posix(),digest(p)) for folder in
                 ('flight3d','encounters3d','retreat3d','turnflee')
                 for p in sorted(Path('src/flybrain',folder).glob('*.py')))


@dataclass(frozen=True)
class ResponseConfig:
    """Immutable engineering gains; a receipt is evidence, never physiology."""
    gains: tuple[int,int]
    incoming_edges: tuple[tuple[str,str], ...]
    outgoing_edges: tuple[tuple[str,str], ...]
    circuit_sha256: str
    source_sha256: tuple[tuple[str,str], ...]
    verification_receipt: str = 'development-unverified'

    def __post_init__(self) -> None:
        if self.gains not in (*GAIN_PAIRS,(1,1)):
            raise ValueError('only three declared candidates or parity gain one')
        if not self.incoming_edges or not self.outgoing_edges:
            raise ValueError('retained visual/DN edges required')

    @classmethod
    def create(cls, manifest: dict, gains: tuple[int,int],
               receipt: str = 'development-unverified') -> 'ResponseConfig':
        types={n['id']:n['metadata'].get('cell_type') for n in manifest['nodes']}
        dns={i for i,t in types.items() if t in ('DNp03','DNp26')}
        if len(dns)!=4:
            raise ValueError('exactly four retained descending cells required')
        edges=tuple((e['pre'],e['post']) for e in manifest['edges'] if e['pre']!=e['post'])
        return cls(tuple(gains), tuple(e for e in edges if types[e[0]]=='LPLC4' and e[1] in dns),
                   tuple(e for e in edges if e[0] in dns), digest(CACHE/'circuit.json'),
                   source_hashes(), receipt)

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def verify(self, manifest: dict) -> None:
        expected=self.create(manifest,self.gains,self.verification_receipt)
        if self!=expected:
            raise ValueError('response edge/source/config binding mismatch')
        if self.verification_receipt!='development-unverified':
            name,sha=self.verification_receipt.rsplit('#',1)
            path=Path(name)
            if digest(path)!=sha:
                raise ValueError('calibration verification receipt mismatch')
            rows=json.loads(path.read_text(encoding='utf-8'))
            eligible=[tuple(r['gains']) for r in rows if r['eligible']]
            if not eligible or self.gains!=eligible[0]:
                raise ValueError('configuration is not lowest eligible calibrated pair')

    @classmethod
    def from_record(cls, value: dict) -> 'ResponseConfig':
        return cls(tuple(value['gains']),tuple(map(tuple,value['incoming_edges'])),
                   tuple(map(tuple,value['outgoing_edges'])),value['circuit_sha256'],
                   tuple(map(tuple,value['source_sha256'])),value['verification_receipt'])


def apply_gains(brain: FlightBrain, config: ResponseConfig) -> None:
    """Multiply effective weights without renormalizing or editing anatomy."""
    incoming=set(config.incoming_edges); outgoing=set(config.outgoing_edges)
    for j,k in enumerate(brain.dynamic_edges):
        edge=brain.edge_order[k]
        if edge in incoming:
            brain.weights[j]*=config.gains[0]
        if edge in outgoing:
            brain.weights[j]*=config.gains[1]


class ResponsePair:
    """Rolling signed 48-wing counts every 20 ms, with paired stability audit."""
    def __init__(self, manifest: dict, config: ResponseConfig):
        config.verify(manifest); self.config=config
        self.actual=MotionVisionBrain(manifest); self.reference=MotionBrain(manifest)
        apply_gains(self.actual,config); apply_gains(self.reference,config)
        self.indices=wing_indices(manifest)
        self.reset()

    def reset(self) -> None:
        self.actual.reset(); self.reference.reset()
        self.window=deque(maxlen=10); self.count_window=deque(maxlen=10)
        self.last_vector=np.zeros(len(self.indices),dtype=np.int64)
        self.max_downstream_hz=0.; self.finite=True; self.stable=True

    def record_counts(self, actual: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """One complete interval; tests can exercise exact expiration/count limits."""
        self.window.append((actual-reference)[list(self.indices)].copy())
        self.last_vector=np.sum(self.window,axis=0)
        self.count_window.append(np.stack((actual,reference)))
        if len(self.count_window)==10:
            downstream=~self.actual.inputs.numpy()
            hz=float(np.max(np.sum(self.count_window,axis=0)[:,downstream]))/.2
            self.max_downstream_hz=max(self.max_downstream_hz,hz)
            self.stable=self.stable and hz<=240
        return self.last_vector.copy()

    def advance(self, observation: FlightObservation) -> np.ndarray:
        actual=self.actual.advance(observation); reference=self.reference.advance(observation)
        self.finite=self.finite and all(bool(torch.isfinite(getattr(b,f)).all())
                                       for b in (self.actual,self.reference)
                                       for f in ('voltage','phase','history'))
        return self.record_counts(actual,reference)
