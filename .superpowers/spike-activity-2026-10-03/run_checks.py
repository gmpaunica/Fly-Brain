"""Record fresh checks sequentially; no fitting or evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

out=Path(__file__).resolve().parent
prefix=sys.argv[1] if len(sys.argv)>1 else ''
records=[]
for label,args in [('targeted',['tests','-k','flight3d']),('full',[])]:
    command=[sys.executable,'-m','pytest','-q',*args]
    name=prefix+label
    env=os.environ.copy();env['PYTEST_ADDOPTS']=f'--junitxml={out / (name+".xml")}'
    if prefix:
        env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    start=time.perf_counter()
    with (out/(name+'.log')).open('w',encoding='utf-8') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
    records.append({'label':label,'command':command,'junit':str(out/(name+'.xml')),
                    'math_threads_environment':{key:env.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS')},
                    'returncode':result.returncode,'seconds':time.perf_counter()-start})
    (out/(prefix+'test-exits.json')).write_text(json.dumps(records,indent=2))
    print(label,result.returncode,round(records[-1]['seconds'],2),flush=True)
    if result.returncode:sys.exit(result.returncode)
