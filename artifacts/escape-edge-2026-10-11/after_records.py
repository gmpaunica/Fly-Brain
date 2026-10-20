"""Generate saved-only review evidence after both batches seal successfully."""
import json
import subprocess
import sys
import time
from flybrain.escape_edge.validation import ART,BASE

while not (BASE/'robustness/receipt.json').exists():
    time.sleep(2)
for phase in ('development','robustness'):
    assert json.loads((BASE/phase/'receipt.json').read_text())['status']=='passed'
subprocess.run([sys.executable,str(ART/'check_sustained.py')],check=True)
subprocess.run([sys.executable,str(ART/'plot_results.py')],check=True)
