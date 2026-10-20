"""Run the saved-only audit and reject any simulation/sensor function call."""
from collections import Counter
import json
from pathlib import Path
import sys

calls=Counter()
def guard(frame,event,arg):
    """Allow only the independent numerical record audit inside flybrain."""
    if event!='call': return
    name=frame.f_code.co_filename.replace('\\','/')
    if '/src/flybrain/' not in name: return
    key=name.split('/src/flybrain/')[1]+':'+frame.f_code.co_name
    calls[key]+=1
    if not name.endswith('/escape_diagnostic/audit.py') and frame.f_code.co_name!='<module>':
        raise RuntimeError('Forbidden simulation/sensor call: '+key)

if __name__=='__main__':
    sys.setprofile(guard)
    from flybrain.escape_diagnostic.audit import audit
    result=audit(Path('experiments/escape-diagnostic-2026-10-07'),
                 Path('artifacts/escape-diagnostic-2026-10-07/analysis'))
    sys.setprofile(None)
    Path('artifacts/escape-diagnostic-2026-10-07/audit-call-guard.json').write_text(
        json.dumps(dict(status=result['status'],classification=result.get('classification'),
                        brain_sensor_simulation_calls=0,project_calls=dict(calls)),indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','classification','cases','intervals','recorded_kernel_steps','audit_kernel_steps') if k in result},indent=2))
