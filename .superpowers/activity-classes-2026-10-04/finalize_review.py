"""Update final handoff after the user's discoverability feedback."""
import json
from pathlib import Path
import re

out=Path(__file__).resolve().parent
exits=json.loads((out/'review-tests-exits.json').read_text())
assert len(exits)==2 and all(r['exit']==0 for r in exits)
test_times=[]
for name,count in (('targeted',83),('full',650)):
    match=re.search(rf'{count} passed[^\n]* in ([\d.]+)s',(out/f'review-{name}.log').read_text())
    assert match;test_times.append(float(match[1]))
timings=[]
for activity in ('all','odor'):
    for cord in ('hidden','shown'):
        record=json.loads((out/f'review-timing-{activity}-{cord}.json').read_text())
        assert record['gates']['p95_below_33_ms']
        timings.append((activity,cord,record))
maximum=max(v['p95_ms'] for _,_,r in timings for v in r['conditions'].values())
fps=[v['paced_frames_per_wall_second'] for _,_,r in timings for v in r['conditions'].values()]
ratios=[r['conditions']['running']['simulation_seconds_per_wall_second'] for _,_,r in timings]
rows=[]
for activity,cord,r in timings:
    conditions=list(r['conditions'].values())
    rows.append('| '+activity.title()+' / '+cord+' | '+' | '.join(f'{v["p95_ms"]:.2f}' for v in conditions)+f' | {conditions[0]["visible_pulses_min"]} |')
preview=json.loads((out/'live-preview.json').read_text(encoding='utf-8-sig'))
pid=preview['pid']
report=Path('docs/verification/flight3d-activity-classes.md')
text=report.read_text(encoding='utf-8')
text=text.replace('46.09 s',f'{test_times[0]:.2f} s').replace('157.21 s',f'{test_times[1]:.2f} s')
text=text.replace('54.78 s and 165.54 s',f'{exits[0]["wall_seconds"]:.2f} s and {exits[1]["wall_seconds"]:.2f} s')
for name in ('targeted.log','full.log','tests-exits.json','timing-all-hidden.json','timing-all-shown.json','timing-odor-hidden.json','timing-odor-shown.json','timings-exits.json'):
    text=text.replace('/'+name,'/review-'+name)
start=text.index('| All / hidden |');end=text.index('\n\n',start)
text=text[:start]+'\n'.join(rows)+text[end:]
text=text.replace('maximum 15.19 ms',f'maximum {maximum:.2f} ms')
text=text.replace('56.76–58.51',f'{min(fps):.2f}–{max(fps):.2f}')
text=text.replace('1.012–1.014',f'{min(ratios):.3f}–{max(ratios):.3f}')
text=text.replace('PID 7476',f'PID {pid}')
text += '''\n\n### Live-review feedback and discoverability repair\n\nThe user reported difficulty locating switches for the other colors. Initial\nsettings placed them beneath introductory text and shortcuts, outside the initial\nviewport. The compact two-column class grid now precedes shortcuts/scope and\nuses visible button backgrounds. Every class button is initially visible at both\nrequested sizes; an actual rendered-layout regression failed against the old\narrangement and passed after repair. The first test version inspected GUI transforms\nbefore the layout task ran; it was corrected to step/render the real GUI before\nchecking viewport membership. User feedback is preserved; acceptance is still open\nuntil the revised native preview is reviewed.\n\nRequired targeted/full suites and four isolated navigation processes were rerun\nafter that final product/test change. The fresh evidence linked above uses the\n`review-` prefix; initial passing logs/timings and pre-repair frames are retained\nunder their original names and `frames-before-discoverability/`. Final screenshots\ncover expanded/collapsed settings, the top class grid and remaining controls.\n'''
report.write_text(text,encoding='utf-8')
status=Path('PROJECT_STATUS.md');text=status.read_text(encoding='utf-8')
text=text.replace('**15.19 ms**',f'**{maximum:.2f} ms**').replace('at most 15.19 ms',f'at most {maximum:.2f} ms')
text=text.replace('56.76–58.51',f'{min(fps):.2f}–{max(fps):.2f}').replace('1.012–1.014',f'{min(ratios):.3f}–{max(ratios):.3f}')
text=text.replace('46.09/157.21',f'{test_times[0]:.2f}/{test_times[1]:.2f}')
text=text.replace('/targeted.*','/review-targeted.*').replace('and `full.*`','and `review-full.*`')
text=text.replace('PID 7476',f'PID {pid}')
status.write_text(text,encoding='utf-8')
queue=Path('docs/BUILD_QUEUE.md');text=queue.read_text(encoding='utf-8').replace('maximum 15.19 ms',f'maximum {maximum:.2f} ms')
text=text.replace('teacher preview launched, user feedback requested.', 'teacher preview reopened with all ten switches at the top after the user reported\n  difficulty finding them. Revised live review remains pending.')
queue.write_text(text,encoding='utf-8')
entry=f'''

## 2026-10-04 — Activity-switch live feedback: controls discoverability

### Goal

Address the user's report: "i cant figure out how to swtich on the other color neurons".

### Changes Made

Moved all ten class buttons to a compact two-column grid at the top of Display
settings, ahead of shortcuts/scope. Added button backgrounds and concise click
instructions. Every switch fits the initial viewport at 1440×900 and 1000×700.
Updated run guidance, architecture, decision/proposal records and verification.

### Decisions

This is a usability repair inside the authorized display task. Class membership,
scope, palette, recorder and neural rules are unchanged. Separate native acceptance
remains pending for the revised layout.

### Tests / Verification

Rendered-layout red/green regression and focused controls: 4 passed. Fresh final
`python -m pytest -q tests -k flight3d`: 83 passed, 567 deselected, 2 warnings,
{test_times[0]:.2f} s, exit 0. `python -m pytest -q`: 650 passed, zero skips,
2 warnings, {test_times[1]:.2f} s, exit 0. Final evidence has `review-` prefix;
initial passing runs are retained. No subsequent product/test change.

Four isolated final All/odor hidden/shown timing runs again pass every p95 <33 ms,
maximum {maximum:.2f} ms. Actual visible pulses remain 359/527 All, 69/69 odor.
Final saved-frame QA covers the revised controls at both sizes. All 2,368
protected files remain unchanged. No scientific fit/continuation/assay.

### Problems / Failed Attempts

The first viewport assertion read GUI transforms before layout publication. It
was corrected to step/render the GUI, then failed for the intended reason: the
original Odor switch was already below the panel viewport. Grid repair passed.
The first targeted rerun shared GPU time with the old live preview; it was closed
before final isolated timing. Earlier timings/screenshots/logs remain preserved.

### Current State

Automated/saved-frame/timing verification complete. Updated teacher preview
reopened with seed 970000, PID {pid}; revised native review requested/pending.

### Next Step

One task / Sol Medium: review the now-visible class buttons and remaining native
interactions, then record acceptance or concrete issues. No new learning work.

### Notes

Click Display settings above the brain; click a class button marked off to enable
its color. Scroll the right panel for whole/selected scope and other settings.
Glow switches change display strength, not recorded spikes or neural activity.
'''
with Path('PROJECT_LOG.md').open('a',encoding='utf-8') as stream:stream.write(entry)
(out/'final-summary.json').write_text(json.dumps({'test_times':test_times,'maximum_p95_ms':maximum,'preview_pid':pid,'live_review':'pending'},indent=2))
print(json.dumps({'test_times':test_times,'maximum_p95_ms':maximum,'preview_pid':pid}))
