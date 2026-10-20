"""Remaining-case preflight: forbid all neural/sensory stepping."""
from collections import Counter
import json
from pathlib import Path
import sys

calls=Counter()
def guard(frame,event,arg):
    if event!='call': return
    file=frame.f_code.co_filename.replace('\\','/')
    if '/src/flybrain/' not in file: return
    name=frame.f_code.co_name; key=file.split('/src/flybrain/')[1]+':'+name; calls[key]+=1
    if name in ('advance','advance_rates','sensory_rates','encounter_eyes','_advance','_advance_measured','advance_recorded'):
        raise RuntimeError('Forbidden neural/sensory call: '+key)


if __name__=='__main__':
    sys.setprofile(guard)
    import pytest
    code=pytest.main(['tests/test_escape_diagnostic_recovery.py','tests/test_escape_diagnostic_remaining.py','tests/test_escape_diagnostic_journal.py','-q','-p','no:cacheprovider',
        '--junitxml=artifacts/escape-diagnostic-remaining-2026-10-07/journal-guarded-tests.xml'])
    sys.setprofile(None)
    Path('artifacts/escape-diagnostic-remaining-2026-10-07/journal-call-guard.json').write_text(
        json.dumps(dict(exit_code=int(code),added_kernel_steps=0,sensory_presentations=0,project_calls=dict(calls)),indent=2),encoding='utf-8')
    raise SystemExit(code)
