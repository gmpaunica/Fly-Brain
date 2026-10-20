"""Corrected entrypoint after zero-step import failure; same fixed candidate."""
import json
import math
from pathlib import Path
from . import experiment

RUN=Path('experiments/escape-recovery-execution-2026-10-07/run')
ART=Path('artifacts/escape-recovery-execution-2026-10-07')
# Supply the omitted standard-library dependency without changing frozen bytes.
experiment.math=math
experiment.ART=ART

if __name__=='__main__':
    r=experiment.run(RUN)
    print(json.dumps({k:r[k] for k in ('status','cases','reserved_kernel_steps','validation_successes','intact_successes','silent_triggers')},indent=2))
