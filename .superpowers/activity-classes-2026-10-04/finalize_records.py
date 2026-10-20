"""Finish the dated handoff from recorded verification results."""
import json
from pathlib import Path

out=Path(__file__).resolve().parent
preview=json.loads((out/'live-preview.json').read_text(encoding='utf-8-sig'))
pid=preview['pid']
for name in ('PROJECT_STATUS.md','docs/verification/flight3d-activity-classes.md'):
    path=Path(name)
    text=path.read_text(encoding='utf-8-sig').replace('PID 26580',f'PID {pid}')
    path.write_text(text,encoding='utf-8')

entry=f'''

## 2026-10-04 — Switchable activity classes, colors and legend

### Goal

Implement the user's supplied class-switch specification, superseding exclusive
group presets. Preserve complete recording, accepted cached anatomy, camera and
the interrupted experiment. One agent; no experiment training.

### Changes Made

Added display-only cached role classification and direct adjacency. All 589 cells
map disjointly to ten classes. Independent frozenset switches intersect whole or
selected/direct-partner scope through an immutable tuple mask. Odor inputs is the
fresh-launch default. All activity enables every class and whole scope; Activity
off retains counters/scope. Sensory has expandable children and off/on/mixed parent
behavior. Runtime counts, exact class RGB endpoints, two-column swatch legend,
idle/cyan explanations and separate full/enabled/visible activity counts are shown.

Extended the existing static owned-batch RGBA table with optional palette/mask and
same-frame cache invalidation. Original no-option warm behavior remains. Added
cord-only/missing geometry explanations and explicit Show nerve cord inspector
action. Replay/new trial retain choices and clear history through the existing
reset. Updated guide, architecture, decisions, proposal, status, queue and evidence.

### Decisions

These categories describe existing model roles, not new biological classifications.
Selection/direct partners are defined by fixed non-self topology, independent of
spiking, edge strength and geometry. Clearing/background-only replacement restores
whole scope while retaining switches. Cyan selection remains independent. Active
alpha .43, 200 ms decay and 600 ms expiry are retained. No preference persistence;
no recorder, scientific snapshot, neural, sensor, training or cache changes.

### Tests / Verification

Fresh serial venv commands, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1, Python 3.12.10:

- `python -m pytest -q tests -k flight3d`: **83 passed, 567 deselected, 2 warnings**,
  46.09 s pytest / 54.78 s wrapper, exit 0.
- `python -m pytest -q`: **650 passed, zero skips, 2 warnings**, 157.21 s pytest /
  165.54 s wrapper, exit 0. Existing Pygame/Torch deprecations.

Tests cover classification, parent/independent/all/off behavior, class/scope
intersections, no-outgoing cells, silence, missing/cord-only and mixed/deferred
branches, pause/replay/reset, exact scientific snapshot/voltage invariance, and
30/60/120 display-Hz equality with a nonempty firing mask. Real GPU pixels verify
all ten class endpoints, alpha/channel ordering, paused recoloring and legend
swatches. Inspected saved layouts at 1440×900 and 1000×700, including settings,
all/odor, selected/cyan, activity off and shown cord. Local acceptance/code review
performed; no subagents launched.

Four isolated serial 1440×900 teacher timing processes, 12 warmup/120 measured
frames per condition. Paused/running/orbit/selection p95 ms:

- All, hidden: **4.66/9.07/12.03/15.19**, 359 visible pulses.
- All, shown: **5.62/8.95/5.00/11.54**, 527 pulses.
- Odor, hidden: **6.70/8.70/4.55/10.93**, 69 pulses.
- Odor, shown: **5.19/9.41/5.11/14.50**, 69 pulses.

All sixteen p95 values pass <33 ms. At the initial paused frame each mode still
has 462 latest firing / 530 fired since reset; enabled firing is 462 All /69 odor.
Paced display 56.76–58.51 FPS; running simulation 1.012–1.014 s/wall s. Historical
timing variability remains a limit. Audit: **2,368 protected existing files
unchanged**; only three intended display files changed among 2,371 inventoried
source/data/experiment files. Cached geometry, scientific sources and incomplete
1,050-update checkpoints/journals remain intact. No reserved assay or fit ran.

### Problems / Failed Attempts

Initial viewer fixture assumed a no-outgoing cell would have brain-visible
branches; cache/manifest evidence places those cells in the cord. The fixture now
uses full model membership. Saved-frame QA found invisible legend swatches:
DirectFrame no-relief omits its fill. Real-pixel regression failed, then passed
with flat relief. An initial documentation patch attempted conflicting operations
on one path; it was rejected without edits and replaced with a contextual update.
Native automation is disabled. The first preview logged a foreground warning and
was launched with hidden startup flags; that agent-owned process was replaced by
an explicitly visible interactive launch. Logs and prior PID are retained.

### Current State

Implementation, automated suites, saved visual QA and timing gates complete.
Separate live review pending: user feedback requested for switches/parent, colors,
legend, selection/scope, pause/reset, cord and navigation. Teacher preview launched
with seed 970000, PID {pid}; foreground activation may still depend on Windows.
The interruption remains incomplete at 1,050 updates; no learning continuation.

### Next Step

**One task / Sol Medium:** obtain and record the separate native live review.
Do not start another milestone, fit or reserved assay.

### Notes

Switching glow off is a display mask, not neuronal silence. At 200 ms after a
spike, exp(-1)≈.368 interpolates soft idle (.24,.30,.55) toward odor
(.902,.706,.314), yielding approximately (.484,.449,.463). This NumPy/GPU display
calculation is separate from PyTorch neuron state and from optimizer learning.
Evidence: `docs/verification/flight3d-activity-classes.md` and
`.superpowers/activity-classes-2026-10-04/`. No Git repository; no commit/push.
'''
log=Path('PROJECT_LOG.md')
with log.open('a',encoding='utf-8') as stream:stream.write(entry)
print(f'Records updated for preview PID {pid}')
