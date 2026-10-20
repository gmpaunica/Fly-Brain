"""Load only a pinned, behavior-gated approach-and-stop checkpoint."""

import hashlib
import json
from pathlib import Path

import torch

from .controller import NeuralController
from .food_readout import load_food_checkpoint, make_food_controller


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PINNED_FILES = {
    'experiments/food-stop-fit-2026-09-28/protocol.json':
        '3771cd4bd2b3304f40421fb19f7666d01fa3f8a91894a96a3c410352ad1e7452',
    'experiments/food-stop-fit-2026-09-28/results.json':
        '0d55d9661d0a02b8690a39a0915cccfa90f879ce8b8ddf7463f3ae2e5b2c3237',
    'experiments/food-stop-fit-2026-09-28/trained.pt':
        '53526140b8b7f5e48b7f4a97f9db380171fad220275d339912638aadfaa8ee1b',
    'experiments/food-stop-fit-2026-09-28/untrained.pt':
        '1d036901efd72d18977b2c8382e1b23121be271b89b2259db7cde54c6ff15b48',
    'experiments/food-stop-fresh-2026-09-28/protocol.json':
        '825c590d6e8c65b170c7e0363bb287503a4693e1c4f7034eb30d5190a52a0731',
    'experiments/food-stop-fresh-2026-09-28/validation.json':
        'e7f4bc8eeb25a8899e4bf819f7ac58cdf3f31626aed18a2d9a6e0c2e18024ac5',
    'experiments/food-stop-fresh-2026-09-28/interruption.json':
        'aa5f2a19dbea6df9a6d25b425a99a85b542c57c099b39143ec863fe87ddfc6c5',
    'experiments/food-stop-contact-diagnostic-2026-09-29/protocol.json':
        '66bfb9bd3c2e082f373c38b4914a17219b0d4eb68be26a0ae9bd5ac9f7748750',
    'experiments/food-stop-contact-diagnostic-2026-09-29/results.json':
        'fb1f19131e33e29d7ccc64248e75d7f02569ed75eb7584340d73325949a8f762',
    'docs/plans/FOOD-STOP-RECOVERY-CASES.json':
        '1941e4436fa3b857faf26746560c43874df1cea352a8fe55e0ffa91072ac1d32',
    'experiments/food-stop-recovery-2026-09-29/protocol.json':
        '40cc1a9e178efa543a25177194a605f21b81db0a24ec7ae6296b8fb4fcce471d',
    'experiments/food-stop-recovery-2026-09-29/execution.json':
        '4458b0c9040d08dc7d1fd1b0fec0bd3ddd8baddfaf4997287e211552b2b9f7a3',
    'experiments/food-stop-recovery-2026-09-29/episodes.jsonl':
        'ecbe45c7d068283c1f709f077f3aac823ae4fa84b2c227e11d7edfa2ef509d51',
    'experiments/food-stop-recovery-2026-09-29/final.json':
        '9d37308c9b0a1ecfb40a8e0a388f1039f61395eba470191e346f4056479dab9d',
    'experiments/food-stop-recovery-2026-09-29/final.csv':
        'ef45485255c2374e5d2d0e95c5e3d7f45b8801bdb2032da25b321e1f1ac0b2a0',
}
FINAL_RESULT_SHA256 = 'd824a82685e1a3141f2abd4c400d3c31038b8753ae124d19bb58a652699e17a7'
RECOVERY = 'experiments/food-stop-recovery-2026-09-29'


def _verify(path: Path, digest: str) -> None:
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'accepted food evidence/source differs or is missing: {path}')


def load_accepted_controller(*, project_root: Path = PROJECT_ROOT) -> NeuralController:
    """Verify the complete final recovery evidence and return frozen inference."""
    root = Path(project_root)
    for name, digest in PINNED_FILES.items():
        _verify(root / name, digest)
    result_path = root / RECOVERY / 'results.json'
    _verify(result_path, FINAL_RESULT_SHA256)
    fit_protocol = json.loads((root / 'experiments/food-stop-fit-2026-09-28/protocol.json').read_text())
    fit_result = json.loads((root / 'experiments/food-stop-fit-2026-09-28/results.json').read_text())
    fresh_protocol = json.loads((root / 'experiments/food-stop-fresh-2026-09-28/protocol.json').read_text())
    interruption = json.loads((root / 'experiments/food-stop-fresh-2026-09-28/interruption.json').read_text())
    diagnostic = json.loads((root / 'experiments/food-stop-contact-diagnostic-2026-09-29/results.json').read_text())
    recovery_protocol = json.loads((root / RECOVERY / 'protocol.json').read_text())
    execution = json.loads((root / RECOVERY / 'execution.json').read_text())
    result = json.loads(result_path.read_text())
    recovery_protocol_sha256 = PINNED_FILES[f'{RECOVERY}/protocol.json']
    if (fit_result.get('training_gate_passed') is not True
            or fit_result.get('numerical_gate_passed') is not True
            or fit_result.get('checkpoint_reload_exact') is not True
            or fit_result.get('checkpoint_sha256') != PINNED_FILES[
                'experiments/food-stop-fit-2026-09-28/trained.pt']
            or interruption.get('status') != 'interrupted_after_validation'
            or interruption.get('validation_successes') != {
                'trained': 16, 'untrained': 0, 'fixed': 0}
            or interruption.get('scientific_acceptance') is not False
            or diagnostic.get('changes_stop') is not True
            or diagnostic.get('parameters_unchanged') is not True
            or execution.get('status') != 'consumed'
            or execution.get('protocol_sha256') != recovery_protocol_sha256
            or recovery_protocol.get('cases_manifest_sha256') != PINNED_FILES[
                'docs/plans/FOOD-STOP-RECOVERY-CASES.json']
            or result.get('protocol_sha256') != recovery_protocol_sha256
            or result.get('case_manifest_sha256') != recovery_protocol['cases_manifest_sha256']
            or result.get('checkpoint_sha256') != fit_result['checkpoint_sha256']
            or result.get('journal_sha256') != PINNED_FILES[f'{RECOVERY}/episodes.jsonl']
            or result.get('final_json_sha256') != PINNED_FILES[f'{RECOVERY}/final.json']
            or result.get('final_csv_sha256') != PINNED_FILES[f'{RECOVERY}/final.csv']
            or result.get('contact_intervention_sha256') != PINNED_FILES[
                'experiments/food-stop-contact-diagnostic-2026-09-29/results.json']
            or result.get('contact_intervention_changes_stop') is not True
            or result.get('validation_successes') != {'trained': 16, 'untrained': 0, 'fixed': 0}
            or result.get('validation_gate_passed') is not True
            or result.get('final_gates', {}).get('passed') is not True
            or result['final_gates'].get('trained_successes', 0) < 58
            or any(value < 7 for value in result['final_gates']['sector_successes'].values())
            or any(pair.get('gain', 0) < 8 or pair.get('one_sided_exact_p', 1) > .025
                   for pair in result['final_gates']['paired'].values())
            or result.get('parameters_unchanged') is not True
            or result.get('optimizer_steps') != 0
            or result.get('actual_episodes') != 192
            or result.get('actual_world_ticks') != 192000
            or result.get('actual_neural_steps') != 3840000
            or result.get('passed') is not True):
        raise ValueError('complete passing food-stop evidence required')
    if (root / 'experiments/food-stop-fresh-2026-09-28/results.json').exists():
        raise ValueError('interrupted original final attempt must remain without a result')
    for protocol in (fit_protocol, fresh_protocol, recovery_protocol):
        for name, digest in protocol['source_sha256'].items():
            _verify(root / name, digest)
            if root.resolve() != PROJECT_ROOT.resolve():
                _verify(PROJECT_ROOT / name, digest)
    normal = fit_result['normalization']
    model = load_food_checkpoint(root / 'experiments/food-stop-fit-2026-09-28/trained.pt',
                                 protocol_sha256=PINNED_FILES[
                                     'experiments/food-stop-fit-2026-09-28/protocol.json'],
                                 mean=torch.tensor(normal['mean']),
                                 scale=torch.tensor(normal['scale']))
    model.requires_grad_(False)
    return make_food_controller(model)
