"""Immutable fixed readout, calibrated gains and complete source binding."""
from dataclasses import dataclass,asdict
import hashlib
import json
from pathlib import Path
from flybrain.flight3d.anatomy import load_manifest,digest
from flybrain.turnflee.neural import ResponseConfig

POLICY=dict(sample_ms=20,winning_score_min=1,margin_min=1,agreement_samples=2,deadline_ticks=25,
            cumulative=True,random_ties=False,turn_tolerance_degrees=3,takeoff_height_m=.005,
            takeoff_ticks=3,separation_gain_m=.02,recovery_ticks=15)


@dataclass(frozen=True)
class CandidateConfig:
    """Engineered decoder identity, not a learned parameter checkpoint."""
    response: ResponseConfig
    source_sha256: tuple[tuple[str,str],...]
    policy: str=json.dumps(POLICY,sort_keys=True)

    @classmethod
    def create(cls) -> 'CandidateConfig':
        response=ResponseConfig.from_record(json.loads(Path('experiments/turnflee-2026-10-06/response.json').read_text()))
        response.verify(load_manifest())
        sources={**dict(response.source_sha256),**{p.as_posix():digest(p) for p in Path('src/flybrain/escape_recovery').glob('*.py')}}
        p=Path('src/flybrain/escape_diagnostic_journal.py');sources[p.as_posix()]=digest(p)
        return cls(response,tuple(sorted(sources.items())))

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()

    def verify(self) -> None:
        """No overrides or bypass: current and frozen physiology/config required."""
        self.response.verify(load_manifest())
        if self!=self.create() or self.response.gains!=(4,2):raise ValueError('candidate source/parameter binding mismatch')

    @classmethod
    def from_record(cls,r:dict) -> 'CandidateConfig':
        c=cls(ResponseConfig.from_record(r['response']),tuple(map(tuple,r['source_sha256'])),r['policy']);c.verify();return c
