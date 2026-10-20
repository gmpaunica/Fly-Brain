"""Load the exact accepted FB-10 head for interactive forward simulation only."""

import hashlib
import json
from pathlib import Path, PureWindowsPath

import torch

from .banc_projection import load_checkpoint, make_projection_controller
from .controller import NeuralController


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PINNED_FILES = {
    'experiments/fb10-projection-readout/protocol.json':
        '769db7336a0adfb10e28d395dd2c64eb2a8da3aef091011e478c827a3fa91290',
    'experiments/fb10-projection-readout/results.json':
        '762a04fef20685c30ec5d4cb39968b5fe0a9c61419cfcb79523bfbd458bb860e',
    'experiments/fb10-projection-readout/trained.pt':
        '117f7e8b759c8916b33cd66a814e471a11cfc7aac941f71c51495dabeeddafa1',
    'experiments/fb10-projection-evaluation/protocol.json':
        '57c42d1b55cc3edb05a352354a843990603ff2945fc6df1b430866eb2cc6faa3',
    'experiments/fb10-projection-evaluation/results.json':
        'cc55062044687c8a271c7fbab1e5a453eff8cdceb149a25134728849fa8ed697',
}


def _verify_file(path: Path, digest: str) -> None:
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'frozen evidence/source differs or is missing: {path}')


def load_frozen_controller(*, project_root: Path = PROJECT_ROOT) -> NeuralController:
    """Verify pinned records, installed source and strict contract; take zero steps.

    The assay accepted original eligibility while retaining the covered-fit
    failure. This loader consumes neither training data nor experiment receipts.
    The repository's saved artifacts are required; no fallback model is used.
    """
    root = Path(project_root)
    for relative, digest in PINNED_FILES.items():
        _verify_file(root / relative, digest)
    assay = json.loads((root / 'experiments/fb10-projection-evaluation/protocol.json').read_text())
    # Historical paths are Windows absolute paths. Resolve their relative names
    # locally without changing a single byte of the original protocol.
    historical_root = PureWindowsPath(assay['output_dir']).parents[1]
    for group in ('source_code_sha256', 'accepted_bundle_and_subset_sha256'):
        for recorded, digest in assay['provenance'][group].items():
            relative = Path(*PureWindowsPath(recorded).relative_to(historical_root).parts)
            _verify_file(root / relative, digest)
            if root.resolve() != PROJECT_ROOT.resolve():
                _verify_file(PROJECT_ROOT / relative, digest)
    protocol_name = 'experiments/fb10-projection-readout/protocol.json'
    protocol = json.loads((root / protocol_name).read_text())
    model = load_checkpoint(
        root / 'experiments/fb10-projection-readout/trained.pt',
        protocol_sha256=PINNED_FILES[protocol_name],
        mean=torch.tensor(protocol['normalization']['mean']),
        scale=torch.tensor(protocol['normalization']['scale']),
    )
    model.requires_grad_(False)
    return make_projection_controller(model)
