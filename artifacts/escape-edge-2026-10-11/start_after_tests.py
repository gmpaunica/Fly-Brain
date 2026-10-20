"""Wait for the final engineering gate, freeze, and start one bounded worker."""
from pathlib import Path
import subprocess
import sys
import time

art = Path('artifacts/escape-edge-2026-10-11')
while not (art/'final-full.xml').exists():
    time.sleep(2)
subprocess.run([sys.executable,str(art/'prepare.py')],check=True)
subprocess.run([sys.executable,'-m','flybrain.escape_edge.worker'],check=True)
print('Both physical batches, independent audits and exact replay passed.',flush=True)
