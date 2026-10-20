"""Isolated serial navigation measurements with hundreds of real pulses."""
import json
from pathlib import Path
import subprocess
import sys
import time

out=Path(__file__).resolve().parent
prefix=sys.argv[1] if len(sys.argv)>1 else ''
records=[]
for cord in ('hidden','shown'):
    command=[sys.executable,'scripts/benchmark_flight3d_viewer.py','--view','repaired',
             '--cord',cord,'--frames','120','--output',str(out/f'{prefix}timing-{cord}.json')]
    start=time.perf_counter()
    with (out/f'{prefix}timing-{cord}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    records.append({'cord':cord,'command':command,'returncode':result.returncode,
                    'seconds':time.perf_counter()-start})
    (out/f'{prefix}timing-exits.json').write_text(json.dumps(records,indent=2))
    print(cord,result.returncode,flush=True)
    if result.returncode:sys.exit(result.returncode)
    data=json.loads((out/f'{prefix}timing-{cord}.json').read_text())
    print(json.dumps({'p95_ms':{k:round(v['p95_ms'],2) for k,v in data['conditions'].items()},
                      'min_pulses':{k:v['visible_pulses_min'] for k,v in data['conditions'].items()},
                      'gates':data['gates']}),flush=True)
    if not all(data['gates'].values()):sys.exit('Navigation timing/setup gate failed; saved evidence retained.')
