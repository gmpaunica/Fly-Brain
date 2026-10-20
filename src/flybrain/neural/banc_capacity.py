"""One larger motor head; unchanged endpoint preprocessing and spiking circuit."""

from pathlib import Path

import torch
from torch import Tensor,nn

from .banc_memory import MemoryBancReadout,memory_contract
from .banc_readout import _valid_digest


class WideMemoryReadout(MemoryBancReadout):
    """16->64->2 /1,218 learned scalars; inherit the exact fixed feature math."""

    def __init__(self,mean: Tensor,scale: Tensor,*,seed: int=1010) -> None:
        super().__init__(mean,scale,seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden=nn.Linear(16,64,device='cpu',dtype=torch.float32)
            self.output=nn.Linear(64,2,device='cpu',dtype=torch.float32)


def capacity_contract() -> dict:
    """Keep the temporal source contract and bind the changed head dimensions."""
    original=memory_contract()
    return {**original,'schema':'banc_capacity_readout_v1',
        'architecture':{**original['architecture'],'hidden':64}}


def save_checkpoint(path: Path,model: WideMemoryReadout,*,protocol_sha256: str) -> None:
    """Exclusively save finite head state with its architecture/source contract."""
    if not _valid_digest(protocol_sha256) or any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('expected finite state and protocol SHA256')
    with Path(path).open('xb') as stream:
        torch.save({'contract':capacity_contract(),'seed':model.seed,
                    'protocol_sha256':protocol_sha256,'state_dict':model.state_dict()},stream)


def load_checkpoint(path: Path,*,protocol_sha256: str,mean: Tensor,scale: Tensor) -> WideMemoryReadout:
    """Reject incompatible architecture, preprocessing, core, norm or protocol."""
    payload=torch.load(Path(path),map_location='cpu',weights_only=True)
    if (not isinstance(payload,dict) or payload.get('contract')!=capacity_contract()
            or not _valid_digest(protocol_sha256) or payload.get('protocol_sha256')!=protocol_sha256):
        raise ValueError('capacity checkpoint contract differs')
    try:
        model=WideMemoryReadout(mean,scale,seed=payload['seed']);model.load_state_dict(payload['state_dict'],strict=True)
        if (not torch.equal(model.mean,mean.float()) or not torch.equal(model.scale,scale.float())
                or any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in payload['state_dict'].values())):
            raise ValueError('checkpoint normalization or numerical state differs')
    except (KeyError,TypeError,RuntimeError) as error:raise ValueError('invalid capacity checkpoint state') from error
    model.eval();return model
