"""Run fresh serial suites/isolated timings with durable commands and exits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

out=Path(__file__).resolve().parent
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
mode=sys.argv[1]
prefix=sys.argv[2] if len(sys.argv)>2 else ''
if mode=='tests':
    commands=[('targeted',[sys.executable,'-m','pytest','-q','tests','-k','flight3d']),
              ('full',[sys.executable,'-m','pytest','-q'])]
else:
    commands=[(f'timing-{activity}-{cord}',[sys.executable,'scripts/benchmark_flight3d_viewer.py',
               '--view','repaired','--frames','120','--activity',activity,'--cord',cord,
               '--output',str(out/f'timing-{activity}-{cord}.json')])
              for activity in ('all','odor') for cord in ('hidden','shown')]
records=[]
for name,command in commands:
    if mode!='tests':
        command[-1]=str(out/f'{prefix}{name}.json')
    name=prefix+name
    start=time.perf_counter()
    runenv=env.copy()
    if mode=='tests':runenv['PYTEST_ADDOPTS']=f'--junitxml={out.as_posix()}/{name}.xml'
    with (out/f'{name}.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=runenv)
    records.append({'name':name,'command':command,'exit':result.returncode,'wall_seconds':time.perf_counter()-start})
    (out/f'{prefix}{mode}-exits.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records[-1]),flush=True)
    if result.returncode:sys.exit(result.returncode)
