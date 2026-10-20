"""Run targeted then full suite with isolated historical engineering ledgers."""
import json
from pathlib import Path
import subprocess
import sys
import os

ART=Path('artifacts/escape-recovery-2026-10-07')
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
commands=[('targeted',['tests/test_escape_recovery.py','tests/test_turnflee_neural.py','tests/test_turnflee_control.py','tests/test_turnflee_session.py']),('full',[])]
results=[]
for name,args in commands:
    os.environ['FLYBRAIN_TEST_LEDGERS']=str(ART/'engineering-ledgers'/name)
    with (ART/f'{name}-final.log').open('x',encoding='utf-8') as f:
        command=[sys.executable,'-m','pytest',*args,'-q','-p','no:cacheprovider','-p','flybrain.escape_recovery.pytest_plugin',f'--junitxml={ART/name}-final.xml']
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    results.append(dict(name=name,exit_code=result.returncode,command=command))
    if result.returncode:break
(ART/'preflight-final-result.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
