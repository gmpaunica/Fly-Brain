"""Fixed case coverage and honest accounting across consumed receipts."""
import hashlib
import json
from pathlib import Path

ALL_SEEDS=(1100000,1100004,1100001,1100003,1100002,1100005,1100010,
           1100008,1100012,1100014,1100013,1100015,1100017,1100016,1100032,1100033,1100036,1100037)
REMAINING_SEEDS=ALL_SEEDS[7:]


def sha(path: Path) -> str:
    """Stream a saved-byte hash without loading large archives."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def remaining_cases(cases: list[dict]) -> list[dict]:
    """Reject reordering/duplication; repeat only the explicitly accepted start."""
    if tuple(c['seed'] for c in cases)!=ALL_SEEDS: raise ValueError('aggregate case order mismatch')
    return [c.copy() for c in cases[7:]]


def additional_budget(cases: list[dict]) -> tuple[float,int]:
    """Eleven complete 1.5 s cases, two brains, 1 ms kernel steps."""
    if tuple(c['seed'] for c in cases)!=REMAINING_SEEDS: raise ValueError('remaining budget/order mismatch')
    return 16.5,33000


def aggregate_sources(prior: Path, fresh: Path, cases: list[dict]) -> list[tuple[Path,int,dict]]:
    """Exclude interrupted prefix; source only seven complete plus eleven fresh."""
    remaining_cases(cases)
    return [(prior,k,c.copy()) if k<7 else (fresh,k-7,c.copy()) for k,c in enumerate(cases)]


def verify_prior_receipt(path: Path, expected_hash: str) -> dict:
    """Require the exact consumed interruption and its recorded budget."""
    if sha(path)!=expected_hash: raise ValueError('prior receipt hash mismatch')
    r=json.loads(path.read_text())
    if (r['status'],r['completed_cases'],r['complete_neural_chunks'],r['recorded_intervals'],r['reserved_kernel_steps'])!=('interrupted',7,553,552,22120):
        raise ValueError('prior receipt coverage mismatch')
    return r
