"""Immutable cached source morphology; activity is recorded separately."""
from pathlib import Path
import threading
import numpy as np
from .anatomy import CACHE


class MorphologyCache:
    """Reuse immutable source segments; never invent or relocate coordinates."""
    def __init__(self,directory: Path=CACHE/'skeletons'):
        self.directory=directory
        self._coarse: dict[str,np.ndarray]={}
        self._full: dict[str,np.ndarray]={}
        self._lock=threading.Lock()

    def _load(self,id: str,full: bool) -> np.ndarray:
        storage=self._full if full else self._coarse
        with self._lock:
            if id in storage:return storage[id]
        with np.load(self.directory/f'{id}.npz') as data:
            lines=data['xyz'][data['edges']] if full else data['coarse']
        # Bytes backing prevents callers from making this array writeable again.
        lines=np.frombuffer(lines.tobytes(),dtype=lines.dtype).reshape(lines.shape)
        with self._lock:return storage.setdefault(id,lines)

    def coarse(self,id: str) -> np.ndarray:
        """Degree-two chains simplified in the existing audited source cache."""
        return self._load(id,False)

    def full(self,id: str) -> np.ndarray:
        """Exact source SWC segments, reserved for explicit selection."""
        return self._load(id,True)


