"""Record final fade verification and preserve historical display handoffs."""
import json
from pathlib import Path
import re

out=Path(__file__).resolve().parent
exits=json.loads((out/'final-tests-exits.json').read_text(encoding='utf-8'))
assert len(exits)==2 and all(item['exit']==0 for item in exits)
times=[]
for name,count in (('targeted',87),('full',654)):
    log=(out/f'final-{name}.log').read_text(encoding='utf-8')
    match=re.search(rf'{count} passed[^\n]* in ([\d.]+)s',log)
    assert match;times.append(float(match[1]))
audit=json.loads((out/'preservation-final.json').read_text(encoding='utf-8'))
assert not audit['unexpected_changes']
preview=json.loads((out/'live-preview.json').read_text(encoding='utf-8-sig'))
pid=preview['pid']
report=f'''# Smooth after-spike color fade — verified 2026-10-05

The user requested a visible fade back to cold blue after each neuron finishes
firing. The old exponential envelope faded quickly, then switched its remaining
4.9787% strength off at 600 ms. The new display envelope eases class RGBA to idle
over the same **600 ms** window, without that visible cutoff. Each neuron's own
last spike refreshes its pulse; pause freezes it. Explicit class-off/scope choices
remain immediate. The 200 ms rate average and neural calculations are unchanged.

## Calculation and boundaries

For `u=clip(age/.600,0,1)`, strength is `(1-u)^2*(1+2u)` (one minus cubic
smoothstep). It is 1 at 0 ms, .740741 at 200 ms, .5 at 300 ms, .259259 at
400 ms and 0 at 600 ms. Both ends have zero slope. The original 1e-12-second
expiry tolerance removes floating-point tails, so a mathematically expired owner
does not remain in the colored-pulse count. Never-fired/reset cells stay idle.

At 300 ms, an odor branch interpolates halfway between soft idle
(.24,.30,.55,.22) and amber (.902,.706,.314,.43), giving RGBA approximately
(.571,.503,.432,.325). Its GPU pixel over black is approximately
(.186,.163,.140). These are display calculations, separate from PyTorch neuron
dynamics or optimizer learning. A faded color does not erase recorded spikes.

## Fresh verification

One agent, local cached geometry; no experiment fitting or reserved assay. Venv
Python 3.12.10, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, serial required commands:

| Command | Result | Pytest duration | Exit |
|---|---|---:|---:|
| `python -m pytest -q tests -k flight3d` | 87 passed, 567 deselected, 2 warnings | {times[0]:.2f} s | 0 |
| `python -m pytest -q` | 654 passed, zero skips, 2 warnings | {times[1]:.2f} s | 0 |

Warnings are the existing Pygame `pkg_resources` and Torch `jit.script`
deprecations. Wrapper durations {exits[0]['wall_seconds']:.2f}/{exits[1]['wall_seconds']:.2f} s.
Logs, XML and exact commands/exits:
[targeted](../../.superpowers/activity-fade-2026-10-04/final-targeted.log),
[full](../../.superpowers/activity-fade-2026-10-04/final-full.log),
[exits](../../.superpowers/activity-fade-2026-10-04/final-tests-exits.json).

Four new behavioral checks cover independent last-spike ages, a visible intermediate
fade, negligible endpoint jump, floating-point expiry, refreshed firing, pause,
class mask off/on and reset. Real GPU pixels verify intermediate class color and
alpha plus the final return to idle. Complete history/palette outputs at shared
simulation times agree for 30/60/120 display Hz during silence. Existing
recorded/unrecorded teacher/fixed/disposable frozen-readout snapshot, neural state,
parameter and RNG checks pass in the required suites. No product/test edits
followed the final checks. Local code review completed without subagents.

Audit: {audit['existing_files']:,} pre-existing source/data/experiment files;
**{audit['protected_unchanged']:,} protected files byte-identical**. Only display pulse
math and help text changed in existing product sources. Recorder, scientific
snapshot, neural/sensory/controller/physics, class index, static geometry/camera,
cached sources and interrupted 1,050-update fit remain unchanged. No Git repository.

## Visual evidence and limits

Actual GPU one-spike fade frames, a contact sheet and looping GIF were generated
and inspected. This explicitly labeled **synthetic display fixture** checks
rendering; it is not a neural experiment or a biological result. The real session
does not advance while rendering it. See
[contact sheet](../../.superpowers/activity-fade-2026-10-04/fade-contact-sheet.png),
[GIF](../../.superpowers/activity-fade-2026-10-04/fade-demo.gif) and
[samples](../../.superpowers/activity-fade-2026-10-04/fade-samples.json).

Native computer automation is disabled. The updated teacher viewer was reopened
with seed 970000, launcher PID {pid}; user confirmation of the new fade is pending.
Earlier class-switch live acceptance is retained. This change adds no new navigation
timing claim; the prior isolated timing evidence remains historical.

## Problems and next step

Test-first numerical and GPU checks failed against the old envelope, then passed
after repair. A boundary check found a residual 8.34e-26 strength from floating
time subtraction; restoring the expiry tolerance fixed the count boundary without
a visible cutoff. The first broad run passed 87 targeted tests, but its full-suite
process was absent on resumption with only a partial log and no final result.
Its termination cause is unknown; it is not passing evidence. Final suites were
run fresh after the boundary fix and are the results above. Initial logs remain
preserved, not overwritten.

**One next task / Sol Medium:** review the fade in the native viewer together with
remaining selected-scope, pause/reset, cord and navigation checks. No training or
new scientific milestone begins here.
'''
Path('docs/verification/flight3d-activity-fade.md').write_text(report,encoding='utf-8')
status=Path('PROJECT_STATUS.md');text=status.read_text(encoding='utf-8')
intro=f'''## Smooth after-spike fade — verified 2026-10-05

Each neuron now eases its class color and alpha back to idle blue/violet over
600 ms after its last spike. The former fast exponential envelope/hard cutoff is
replaced by cubic smoothstep: full color initially, half strength at 300 ms and
zero at 600 ms, with smooth endpoints. New spikes refresh color; pause freezes
the fade. Explicit switch/scope exclusions remain immediate display choices.
The 200 ms rate average, complete spike recording and neural computation stay
unchanged. [Evidence](docs/verification/flight3d-activity-fade.md).

Fresh **87 targeted / 654 full-suite passed**, zero skips, two existing warnings.
Numerical/GPU fade checks and 30/60/120 Hz equality pass; rendered fade frames/GIF
were reviewed. All **{audit['protected_unchanged']:,} protected files** are unchanged,
including cached geometry/scientific sources and the interrupted fit. No experiment
training or reserved assay ran. Updated teacher preview launcher PID {pid}, seed
970000; native fade confirmation remains pending. Class-switch live acceptance
from October 4 is preserved. Work is complete within this display task; no next
milestone or experiment is started.

'''
text=text.replace('# FlyBrain Current Status\n\n','# FlyBrain Current Status\n\n'+intro,1)
text=text.replace('  implemented and automatically/saved-frame verified.','  implemented and automatically/saved-frame verified, including the smooth\n  after-spike fade. Class switches are also user-confirmed.')
text=text.replace('## What Works\n\n','## What Works\n\n- Continuous 600 ms class-color fade after each last spike, independent per neuron,\n  with smooth endpoints, refresh on firing, frozen pause and clean reset. Real GPU\n  intermediate/tail checks and refresh-rate equality are verified.\n',1)
text=text.replace('  other native interactions remain pending.','  other native interactions, including the revised fade, remain pending.',1)
text=text.replace('## Important Decisions\n\n','## Important Decisions\n\n- The later fade request replaces the initial 200 ms exponential display envelope\n  with a 600 ms cubic eased envelope. Recorded rate averaging remains 200 ms;\n  neural/simulation time and data are unchanged. The existing expiry roundoff\n  tolerance prevents invisible tails from being counted as colored pulses.\n',1)
text=text.replace('Class switches are user-confirmed. Review Sensory parent, legend, selected scope/cyan,',
                  'Class switches are user-confirmed. Review the smooth after-spike fade,\nSensory parent, legend, selected scope/cyan,')
text=text.replace('Verified 2026-10-04: **83 targeted passed / 650 full-suite passed**, zero',
                  'Verified 2026-10-05: **87 targeted passed / 654 full-suite passed**, zero')
text=text.replace('80.49/144.48 s',f'{times[0]:.2f}/{times[1]:.2f} s')
text=text.replace('`.superpowers/activity-classes-2026-10-04/review-targeted.*` and `review-full.*`',
                  '`.superpowers/activity-fade-2026-10-04/final-targeted.*` and `final-full.*`')
status.write_text(text,encoding='utf-8')
queue=Path('docs/BUILD_QUEUE.md');text=queue.read_text(encoding='utf-8')
queue_intro=f'''## Current display task — smooth fade, verified 2026-10-05

- **Implemented and automatically/saved-frame verified:** each neuron's class RGBA
  fades smoothly to idle over 600 ms after its last spike. Retrigger, pause, reset,
  immediate explicit masks, GPU intermediate/tail colors and refresh-rate equality
  are verified. Fresh **87 targeted / 654 full-suite passed**, zero skips, two
  existing warnings. All {audit['protected_unchanged']:,} protected files unchanged;
  no experiment training or assay. [Evidence](verification/flight3d-activity-fade.md).
- **One next task / Sol Medium:** native fade and remaining selection/pause/reset/
  cord/navigation review. Prior class-switch live acceptance is retained. No
  additional milestone or learning method is started.

'''
text=text.replace('# FlyBrain task queue\n\n','# FlyBrain task queue\n\n'+queue_intro,1)
text=text.replace('## Current 3D task - independent activity classes, 2026-10-04','## Previous verified display task - independent activity classes, 2026-10-04',1)
queue.write_text(text,encoding='utf-8')
entry=f'''

## 2026-10-05 — Smooth color fade after each last spike (began October 4)

### Goal

Make each firing neuron's color fade visibly toward inactive blue instead of
switching abruptly when its pulse ends. Finish this bounded display request.

### Changes Made

Replaced the fast exponential display pulse/hard 600 ms cutoff with a 600 ms
cubic eased envelope using each neuron's existing last-active time. It has full
strength at firing, .5 at 300 ms and smooth zero-slope idle arrival. Retained
expiry roundoff tolerance, retrigger, pause, mask, reset, class/alpha endpoints and
static batch/upload caching. Help text now describes the visible fade. Added four
behavior/GPU checks and updated architecture, decisions, guide, proposal, status,
queue and verification. Saved actual GPU fixture frames, contact sheet and GIF.

### Decisions

This supersedes the earlier exponential display-envelope specification only.
Recorded firing-rate averaging remains 200 ms. Scientific snapshots/counts,
neural/sensory/controller/physics rules, palette, geometry and camera remain.
Explicit switch-off/scope exclusions stay immediate. The display fixture is
synthetic and labeled; it makes no biological or learning claim.

### Tests / Verification

Fresh serial venv commands with OMP_NUM_THREADS=1/MKL_NUM_THREADS=1:

- `python -m pytest -q tests -k flight3d`: **87 passed, 567 deselected, 2 warnings**,
  {times[0]:.2f} s pytest / {exits[0]['wall_seconds']:.2f} s wrapper, exit 0.
- `python -m pytest -q`: **654 passed, zero skips, 2 warnings**,
  {times[1]:.2f} s pytest / {exits[1]['wall_seconds']:.2f} s wrapper, exit 0.

Existing Pygame/Torch warnings. Checks cover visible half/tail fade, independent
last spikes, full refresh on firing, pause, mask, reset, never-fired owners,
floating-point expiry and 30/60/120 Hz equality in silence. Actual GPU pixels
verify intermediate RGBA and negligible final-step jump. Source/cache/experiment
audit: **{audit['protected_unchanged']:,} protected files unchanged** among
{audit['existing_files']:,} pre-existing files, only pulse math/help sources edited.
Scientific snapshots/states/RNG remain verified by the existing full regressions.
Actual GPU fixture/contact sheet reviewed; native fade review remains separate.
Local code/acceptance review completed with one agent. No product/test edits
followed final verification. No new timing claim or broad scientific assay.

### Problems / Failed Attempts

Tests first failed against the old numerical/GPU decay. The first eased version
left an 8.34e-26 floating-point tail at the expiry boundary; a failing strict-zero
test prompted restoration of the original 1e-12 expiry tolerance. Focused four
checks then passed. Initial broad targeted run passed 87 tests, but on resumption
its full-suite process was absent and the log was partial. No final result or
cause is known; fresh final suites supersede it and all initial logs remain.
A contextual documentation patch was rejected without edits; a narrower patch
updated the intended passages. The interrupted scientific fit was not touched.

### Current State

Smooth display fade is implemented and automatically/rendered verified. Updated
teacher preview launched with seed 970000, PID {pid}; native fade confirmation
pending. Prior class switches remain user-confirmed. No training, experiment
continuation, commit/push or new milestone was begun.

### Next Step

One task / Sol Medium: native fade and remaining selection/pause/reset/cord/
navigation review. Stop this implementation session after its handoff.

### Notes

At 300 ms, display strength is .5: idle and amber RGBA are averaged. This NumPy/
GPU afterglow is a visualization of recorded spike history, separate from PyTorch
neural activity and learning. Evidence: `docs/verification/flight3d-activity-fade.md`
and `.superpowers/activity-fade-2026-10-04/` (work spans October 4–5).
'''
with Path('PROJECT_LOG.md').open('a',encoding='utf-8') as stream:stream.write(entry)
(out/'final-summary.json').write_text(json.dumps({'date':'2026-10-05','targeted_passed':87,'full_passed':654,'times':times,'preview_pid':pid,'native_fade_review':'pending','protected_unchanged':audit['protected_unchanged']},indent=2),encoding='utf-8')
print(json.dumps({'targeted':87,'full':654,'times':times,'preview_pid':pid}))
