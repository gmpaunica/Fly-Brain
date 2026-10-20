"""Record reopening, preliminary direct-rate evidence and executed-source audit."""
import json
from pathlib import Path
import shutil
from flybrain.flight3d.anatomy import digest

run=Path('experiments/encounters3d-characterization-2026-10-05')
rows=[]
for label,counts in [('left',[14,0]),('right',[0,4]),('front',[3,0]),('rear',[9,0])]:
    for condition in ('intact','DNp03-silent'):
        rows.append({'stimulus':label,'intervention':condition,'neural_steps':1000,
                     'DNp03_counts':counts if condition=='intact' else [0,0],
                     'motor_counts':[0]*48})
rows.append({'stimulus':'all-visual-240-Hz','intervention':'intact','neural_steps':1000,
    'DNp03_counts':[198,195],
    'motor_counts':[56,42,38,35,64,42,27,51,46,9,62,52,43,60,5,54,38,55,56,56,50,54,41,27,
                    21,8,50,39,28,27,31,43,48,56,31,74,43,73,48,43,60,0,41,65,61,54,71,56]})
record={'schema':'encounters3d-preliminary-direct-rate-probe-v1','date':'2026-10-05',
    'objective':'Source-path scouting before image characterization and motor selection.',
    'seed':None,'device':'cpu','commit':None,'network_cells':589,'network_edges':4139,
    'learning_rate':None,'checkpoint':None,'optimizer_updates':0,
    'neural_dt_seconds':.001,'steps':9000,
    'stimulus':'240 Hz for patch ranks with row indices 2/3 and lateral columns 3/4, front L7/R0, rear L0/R7; then all visual cells at 240 Hz.',
    'evidence':'Transcribed from observed preliminary tool output; no rerun after the image gate failure.',
    'source':'preliminary_probe.py','circuit_sha256':digest(Path('data/flight3d/circuit.json')),
    'rows':rows}
with (run/'preliminary_direct_rates.json').open('x') as stream: json.dump(record,stream,indent=2)
shutil.copyfile('.superpowers/encounters3d/probe.py',run/'preliminary_probe.py')

config=json.loads((run/'config.json').read_text())
checks=[]
for path,expected in config['source_sha256'].items():
    assert digest(Path(path))==expected
    assert digest(run/'executed_sources'/path)==expected
    checks.append(path)
with (run/'source_copy_audit.json').open('x') as stream:
    json.dump({'static_config_sources_match_current_and_archived':checks,
               'preliminary_neural_steps':9000,'formal_recorded_neural_steps':171000,
               'optimizer_updates':0},stream,indent=2)

path=Path('PROJECT_STATUS.md'); text=path.read_text(encoding='utf-8')
text=text.replace('launcher PID 30116','reopened launcher PID 24840 / window PID 6692',1)
path.write_text(text,encoding='utf-8')
note='''

### Same-session preview reopening

The user said they accidentally closed the preview and requested reopening and
the review question again. Reopened the same seed-980000 encounter preview:
launcher PID 24840 / native window PID 6692. Both processes were observed;
foreground activation warned, without a launch traceback. The spawn/reset/pause/
camera/display review question was asked again; live acceptance remains pending.
No product/test changes followed the final suites. The prior PID 30116 is closed.
One scratch documentation script hit the Windows default cp1252 decoder; its
remaining plan/log work completed with explicit UTF-8 without repeating prior
append operations or scientific runs. Preliminary direct-rate scouting is also
preserved in the experiment directory: 9 presentations / 9,000 neural steps,
before the formal image gate (171,000 formal steps). No optimizer updates.
'''
for target in ('PROJECT_LOG.md','docs/verification/encounters3d.md'):
    with Path(target).open('a',encoding='utf-8') as stream: stream.write(note)
with Path('docs/experiments.md').open('a',encoding='utf-8') as stream:
    stream.write('\nPreliminary direct-rate source scouting before the image gate is preserved in\n'
                 '`preliminary_direct_rates.json` and `preliminary_probe.py`: nine one-second\n'
                 'presentations / 9,000 neural steps / zero updates. These additional exploratory\n'
                 'steps are separate from the 171,000 formal recorded steps above.\n')
print('static executed source copies verified:',len(checks))
