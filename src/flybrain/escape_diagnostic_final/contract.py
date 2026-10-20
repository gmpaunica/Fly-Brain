"""Exact five-case coverage and three-source aggregation of complete episodes."""
import json
from pathlib import Path
from flybrain.escape_diagnostic_remaining.contract import ALL_SEEDS,sha

FINAL_SEEDS=ALL_SEEDS[13:]


def final_cases(cases: list[dict]) -> list[dict]:
    """Reject reordered/duplicate coverage and return only the accepted five."""
    if tuple(c['seed'] for c in cases)!=ALL_SEEDS: raise ValueError('full case order mismatch')
    return [c.copy() for c in cases[13:]]


def additional_budget(cases: list[dict]) -> tuple[float,int]:
    """Five 1.5 s episodes, two brains, one millisecond per kernel step."""
    if tuple(c['seed'] for c in cases)!=FINAL_SEEDS: raise ValueError('final budget/order mismatch')
    return 7.5,15000


def aggregate_sources(first: Path, second: Path, fresh: Path, cases: list[dict]) -> list[tuple[Path,int,dict]]:
    """Select 7 +6 +5 complete episodes; exclude every interrupted prefix."""
    final_cases(cases)
    return [(first,k,c.copy()) if k<7 else (second,k-7,c.copy()) if k<13
            else (fresh,k-13,c.copy()) for k,c in enumerate(cases)]


def verify_failed_prior(path: Path, expected_hash: str) -> dict:
    """Bind the closed I/O failure and distinguish attempted from executed steps."""
    if sha(path)!=expected_hash: raise ValueError('failed prior receipt hash mismatch')
    r=json.loads(path.read_text())
    if (r['status'],r['completed_cases'],r['recorded_intervals'],r['reserved_kernel_steps'])!=('failed',6,505,20240) or 'WinError 5' not in r['error']:
        raise ValueError('failed prior coverage mismatch')
    return r
