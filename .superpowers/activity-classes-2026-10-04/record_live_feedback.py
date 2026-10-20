"""Record the user's actual class-switch review without overclaiming other checks."""
import json
from pathlib import Path

out=Path(__file__).resolve().parent
answer='Yes, the switches work now'
for name in ('live-preview.json','final-summary.json'):
    path=out/name
    record=json.loads(path.read_text(encoding='utf-8-sig'))
    record['live_review']='class switches confirmed; other native interactions pending'
    record['user_feedback']=answer
    path.write_text(json.dumps(record,indent=2),encoding='utf-8')
status=Path('PROJECT_STATUS.md');text=status.read_text(encoding='utf-8')
start=text.index('Separate native live review remains pending.')
end=text.index('\n## Actual modeled firing',start)
text=text[:start]+'''The user confirmed **"Yes, the switches work now"** in the revised native preview:
independent class color switching is live-verified. Other native interactions
(Sensory parent, selected scope/cyan, pause/replay/new trial, cord/navigation)
remain unconfirmed. Teacher preview seed 970000, launcher PID 27060 / window PID
25144, starts paused at two simulated seconds with Display settings open. Its
foreground warning did not prevent the user review; the native window is responding.
Automated/saved-frame evidence does not substitute for remaining live checks.
'''+text[end:]
text=text.replace('Final native review of this revised layout remains pending.',
                  'The user now confirms class switching works; other native checks remain pending.')
text=text.replace('native computer automation is disabled and user review is pending.',
                  'native computer automation is disabled. The user has confirmed class switches;\n  other native interactions remain pending.')
text=text.replace('Review switches and Sensory parent, class colors/legend, selected scope/cyan,',
                  'Class switches are user-confirmed. Review Sensory parent, legend, selected scope/cyan,')
status.write_text(text,encoding='utf-8')
queue=Path('docs/BUILD_QUEUE.md');text=queue.read_text(encoding='utf-8')
text=text.replace('difficulty finding them. Revised live review remains pending.',
                  'difficulty finding them. The user confirms the class switches now work;\n  other native interactions remain pending.')
queue.write_text(text,encoding='utf-8')
report=Path('docs/verification/flight3d-activity-classes.md');text=report.read_text(encoding='utf-8')
text=text.replace('A separate live review remains pending;',
                  'The user confirms revised native class switches work; other live checks remain pending;')
text=text.replace('acceptance is still open\nuntil the revised native preview is reviewed.',
                  'the revised class switches are now user-confirmed. Other native interactions\nremain unconfirmed.')
text+='''\n\n### User confirmation\n\nAfter the discoverability repair and native relaunch, the user answered\n**"Yes, the switches work now"** when asked whether another class color can be\nturned on/off with the visible controls. This closes class-switch discoverability\nand live color-toggle review. It does not establish that the reviewer also tested\nSensory parent, legend comprehension, selected scope/cyan, pause/replay/new trial,\ncord or camera navigation. Those remain the one next live-review task / Sol Medium.\nThe responding native window is PID 25144 (venv launcher PID 27060); the recorded\nforeground warning did not prevent this user review.\n'''
report.write_text(text,encoding='utf-8')
with Path('PROJECT_LOG.md').open('a',encoding='utf-8') as stream:
    stream.write('''\n\n### Same-day live-review confirmation\n\nThe user answered **"Yes, the switches work now"** after the revised two-column\ncontrols were reopened. Class switching/color-toggle discoverability is now\nlive-verified. Other native interactions remain unconfirmed; do not relabel this\nanswer as a complete pause/reset/selection/cord/navigation review. Preview window\nPID 25144 is responding; launcher PID 27060. No product/test change followed the\nfinal suites/timings. One next task / Sol Medium: finish those remaining native\nchecks. Scientific sources, cached geometry and interrupted fit remain unchanged.\n''')
print('Recorded user-confirmed class switches; remaining native checks are pending.')
