"""Final whole-suite check after the entrypoint-only engineering correction."""
import os
from pathlib import Path
import subprocess
import sys

ART=Path('artifacts/escape-recovery-execution-2026-10-07')
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
os.environ['FLYBRAIN_TEST_LEDGERS']=str(ART/'engineering-ledgers-final')
with (ART/'full.log').open('x',encoding='utf-8') as f:
    r=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','-p','flybrain.escape_recovery.pytest_plugin',
                      '--junitxml='+str(ART/'full.xml')],stdout=f,stderr=subprocess.STDOUT)
(ART/'full-exit.txt').write_text(str(r.returncode))
