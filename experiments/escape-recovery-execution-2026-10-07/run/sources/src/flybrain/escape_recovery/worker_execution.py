"""Detached corrected entrypoint; old zero-step receipt remains consumed."""
import sys
from .worker import supervise
from .execution import RUN,ART

if __name__=='__main__':
    ART.mkdir(parents=True,exist_ok=True)
    raise SystemExit(supervise([sys.executable,'-m','flybrain.escape_recovery.execution'],RUN,ART/'experiment.log',ART/'worker.json'))
