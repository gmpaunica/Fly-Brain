"""Read failed-run frozen templates for diagnostics; never authorize movement."""
import json
from pathlib import Path
from flybrain.flight3d.anatomy import CACHE, digest
from .neural import FrozenDecoder, wing_indices

EXPERIMENT = Path(__file__).resolve().parents[3]/'experiments/retreat3d-motion-repair-2026-10-06'


def diagnostic_decoder(manifest: dict, directory: Path = EXPERIMENT) -> FrozenDecoder | None:
    """Require closed failed-gate/source identity and intact recorded templates."""
    if not (directory/'receipt.json').exists():
        return None
    receipt = json.loads((directory/'receipt.json').read_text(encoding='utf-8'))
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    result = json.loads((directory/'results.json').read_text(encoding='utf-8'))
    if (receipt['status'] != 'closed-validation-failed' or result['validation_passed'] or
            result['retreat_delivered'] or config['circuit_sha256'] != digest(CACHE/'circuit.json')):
        raise ValueError('diagnostic-only decoder must belong to the closed failed experiment')
    for name in ('templates.json', 'config.json', 'results.json'):
        if digest(directory/name) != receipt['sha256'][name]:
            raise ValueError('frozen diagnostic record changed')
    ids = tuple(manifest['nodes'][i]['id'] for i in wing_indices(manifest))
    return FrozenDecoder.load(directory/'templates.json', ids)
