"""Load the verified 750-update FB-10 head for interactive forward simulation only."""

import hashlib
import json
from pathlib import Path, PureWindowsPath

import torch

from .banc_projection import load_checkpoint, make_projection_controller
from .controller import NeuralController


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PINNED_FILES = {
    'experiments/fb10-extra-training/protocol.json':
        '60dc3b242fb1a50c713b7e381222094488a631a80707c95be7316e7dafbdbd01',
    'experiments/fb10-extra-training/results.json':
        '06ea206efdaa5026f4f1e19167653c22b5d90e6380a819b4167c3b4b4dce345d',
    'experiments/fb10-extra-training/trained.pt':
        '6c31470d5fe7be608272e1267ec6aa75d9655750b350c1e906eb9a079dd1e1e5',
    'experiments/fb10-extra-evaluation/protocol.json':
        '2c98fe0b0d4bc78b2e164ca0adb510e8acc87728fe4faec144af917b29ad44e3',
    'experiments/fb10-extra-evaluation/results.json':
        '6c590f52d57d1a55778577fca3196fa8ee12c875b249f290af4d0ce81610787d',
}


def _verify_file(path: Path, digest: str) -> None:
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'frozen evidence/source differs or is missing: {path}')


def load_frozen_controller(*, project_root: Path = PROJECT_ROOT) -> NeuralController:
    """Verify pinned records, installed source and strict contract; take zero steps.

    This separate candidate passed both training eligibility and its fresh
    assay. Historical failures retain their original files and loader.
    This loader consumes neither training data nor experiment receipts.
    The repository's saved artifacts are required; no fallback model is used.
    """
    root = Path(project_root)
    for relative, digest in PINNED_FILES.items():
        _verify_file(root / relative, digest)
    assay = json.loads((root / 'experiments/fb10-extra-evaluation/protocol.json').read_text())
    # Historical paths are Windows absolute paths. Resolve their relative names
    # locally without changing a single byte of the original protocol.
    historical_root = PureWindowsPath(assay['output_dir']).parents[1]
    for group in ('source_code_sha256', 'accepted_bundle_and_subset_sha256'):
        for recorded, digest in assay['provenance'][group].items():
            relative = Path(*PureWindowsPath(recorded).relative_to(historical_root).parts)
            _verify_file(root / relative, digest)
            if root.resolve() != PROJECT_ROOT.resolve():
                _verify_file(PROJECT_ROOT / relative, digest)
    protocol_name = 'experiments/fb10-extra-training/protocol.json'
    protocol = json.loads((root / protocol_name).read_text())
    model = load_checkpoint(
        root / 'experiments/fb10-extra-training/trained.pt',
        protocol_sha256=PINNED_FILES[protocol_name],
        mean=torch.tensor(protocol['normalization']['mean']),
        scale=torch.tensor(protocol['normalization']['scale']),
    )
    model.requires_grad_(False)
    return make_projection_controller(model)
