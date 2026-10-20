"""Audit the new saved run while forbidding simulation/sensor calls."""
from collections import Counter
import json
from pathlib import Path
import sys

calls=Counter()
def guard(frame,event,arg):
    if event!='call': return
    file=frame.f_code.co_filename.replace('\\','/')
    if '/src/flybrain/' not in file: return
    key=file.split('/src/flybrain/')[1]+':'+frame.f_code.co_name; calls[key]+=1
    if not file.endswith('/escape_diagnostic_recovery/audit.py') and frame.f_code.co_name!='<module>':
        raise RuntimeError('Forbidden simulation/sensor call: '+key)


if __name__=='__main__':
    sys.setprofile(guard)
    from flybrain.escape_diagnostic_recovery.audit import audit
    r=audit(Path('experiments/escape-diagnostic-recovery-2026-10-07'),Path('artifacts/escape-diagnostic-recovery-2026-10-07/analysis'))
    sys.setprofile(None)
    Path('artifacts/escape-diagnostic-recovery-2026-10-07/audit-call-guard.json').write_text(
        json.dumps(dict(status=r['status'],classification=r.get('classification'),brain_sensor_simulation_calls=0,project_calls=dict(calls)),indent=2),encoding='utf-8')
    print(json.dumps({k:r[k] for k in ('status','classification','cases','intervals','recorded_kernel_steps','audit_kernel_steps') if k in r},indent=2))
