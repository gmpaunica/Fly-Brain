"""Record observed final verification and one concrete next task."""
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

for name in ('targeted','full'):
    tree=ET.parse(f'.superpowers/encounters3d/{name}.xml')
    suite=tree.getroot().find('testsuite')
    assert int(suite.attrib['failures'])==0 and int(suite.attrib['errors'])==0
    assert int(suite.attrib['skipped'])==0
    print(name,suite.attrib)

run=Path('experiments/encounters3d-characterization-2026-10-05')
config=json.loads((run/'config.json').read_text())
sources=run/'executed_sources'; sources.mkdir(exist_ok=False)
for path in config['source_sha256']:
    source=Path(path); target=sources/source
    target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
for name in ('controller.py','session.py','evaluate.py'):
    source=Path('src/flybrain/encounters3d')/name
    target=sources/source; target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)

verification='''# Exclusive 3D encounter verification — 2026-10-05

The encounter infrastructure is verified. The **directional neural escape gate
failed** and the preview is explicitly hover-only. Retreat/retrigger remain
unavailable. This is the authorized negative-gate outcome, not completed avoidance.

## Commands and observed results

Serial venv commands, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1:

- `python -m pytest -q tests -k "encounters3d or flight3d" --junitxml=.superpowers/encounters3d/targeted.xml`:
  **100 passed, 567 deselected, zero skips, two existing warnings, 49.22 s**, exit 0.
- `python -m pytest -q --junitxml=.superpowers/encounters3d/full.xml`:
  **667 passed, zero skips, two existing warnings, 152.09 s**, exit 0.
- `python -m flybrain.encounters3d --headless --kind flower --seed 970000 --steps 120`:
  teacher / flower / 2.40 simulated seconds, exit 0.
- `python -m flybrain.encounters3d --headless --kind predator --seed 980000 --steps 250`:
  circuit probe / avoidance unavailable / 5.00 seconds, exit 0.
- `python -m flybrain.encounters3d --seed 980000 --capture .superpowers/encounters3d/frames-final`:
  saved initial/flower/other-reset/predator frames, exit 0. Final flower and cyan
  predator frames reviewed; the 1000x700 renderer/event test also passed.
- `python .superpowers/encounters3d/integrity.py verify`:
  **2,465 historical source/test/data/experiment paths unchanged**. The original
  launchers, scientific assets, checkpoints and interrupted fit are preserved.

Logs and XML are under `.superpowers/encounters3d/`. Existing Pygame pkg_resources
and Torch JIT deprecation warnings remain. No tests were skipped/blocked; the
separate native interaction review is blocked by unavailable helper access.

## Behavior/numerical evidence

Thirteen new tests exercise exclusive object inventory, RGB-only observation,
left/right/front/rear visibility, ground occlusion, expansion, flower/background
rejection, swept predator collision, exact teacher/brain parity, reset/replay,
unchanged mode selection, IMU hover, pause/publication barriers, passive draw,
retained camera/display preferences, actual spawn-button callbacks and keyboard
R/Backspace. Actual framebuffer sampling verifies the cyan predator matches its
RGB eye color. No mocks replace physics, spikes or render-color checks.

Actual DNp03 numerical checks verify exp(-.001/.1) leak, threshold 1, subtractive
reset, episode clearing, delayed LPLC4 input and directed DNp03-to-wing current.
Matched saturation checks verify visual dependence and remaining responses after
DNp03 silence. The saved scientific records are inspected by tests without
rerunning historical experiments.

The predeclared static run completed 51 one-second presentations: eight directions
at two sizes plus a synthetic symmetric full field, with three matched conditions.
Every intact physical presentation had zero motor spikes; maximum sampled motor
voltage .229664. Full field gave **1,838 intact / 1,476 DNp03-silent / zero
visual-silent** wing spikes. The latter is a model contribution diagnostic, not
biological necessity or successful escape. The current spike-history motor gate
closed before selecting a decoder.

The bounded evaluator completed **24 five-second** physical cases: eight
directions, three conditions. All held hover with **zero separation change,
zero collisions, zero final speed and no escape/steering response delay**.
Absent response is JSON null, not zero latency. Front/rear test symmetric images.
These hover cases do not demonstrate retreat, recovery or threat avoidance.
Total measured steps: 171,000 neural / 24,000 closed-loop physics; zero optimizer
updates. No checkpoint, learning rate, reserved evaluation or interrupted-run
continuation. Scientific sources/config/software/device/cases/JSON/CSV and executed
source copies are recorded in the fresh experiment directory.

## Rendered review and separate live attempt

Saved offscreen frames were reviewed for readable labels/buttons, exclusive
objects, teacher behavior presentation and actual visual-circuit activity. The
first sphere was yellow despite cyan CPU vertex values; a framebuffer reproduction
isolated a packed vertex-color R/B swap on this WGL path. Explicit predator surface
color fixes that path and preserves every historical renderer byte. The new
framebuffer color assertion failed before and passed after this fix.

Computer-use skill initialization succeeded, but `sky.list_windows()` failed:
“Computer Use native pipe is unavailable ... The system cannot find the file
specified. (os error 2).” No live screenshot/click/keyboard result is claimed.
An ordinary native encounter preview was launched for user review with seed
980000, launcher PID 30116, title “FlyBrain | Exclusive encounters | escape gate
failed”. User feedback on spawn/reset/replay/pause/preferences remains pending.
No custom native automation fallback was used.

## Local acceptance review

One agent reviewed all new source/tests against the specification. Geometry is
restricted to environment/evaluator/display; controller inference receives only
FlightObservation. Original graph/dynamics and flower teacher remain unchanged.
Threat values derive from one recorded DisplayFrame. Spawn resets are worker
barriers; render methods do not advance neural history or use scene RNG.
Scientific avoidance requirements are explicitly unmet after the closed gate.
This is an author self-review, as the user requested one agent; no independent
reviewer, commit, push, training or follow-on source extension was started.

Failed attempts: initial missing-module RED tests; contrast .02 incorrectly
accepted sky roundoff and stem contrast .03 (threshold fixed at .05 after RGB
evidence); peak-only fields lost static expansion (added occupancy term); first
record setup mixed relative/absolute paths (zero neural steps, fixed before the
one run); two contextual doc patches rejected without changes; the yellow
packed-color regression above. No scientific retry followed the failed gate.

Next task / **Sol High**: specify the bounded sourced/physiological recovery in
[the guide](../ENCOUNTERS3D.md). Its DNp26/DNp57 proposal is not implemented.
'''
Path('docs/verification/encounters3d.md').write_text(verification,encoding='utf-8')

status=Path('PROJECT_STATUS.md'); text=status.read_text(encoding='utf-8')
text=text.replace('Fresh verification and live-review limits are recorded in',
'''Fresh **100 targeted / 667 full-suite passed**, zero skips, two existing warnings.
All **2,465 historical protected files** are unchanged. Offscreen frames/GPU color
and injected controls pass; native helper access failed. A native preview is open
for user review (seed 980000, launcher PID 30116); live acceptance is pending.
Verification and live-review limits are recorded in''',1)
text=text.replace('## How To Test\n','''## How To Test

Latest encounter verification: **100 targeted / 667 full-suite passed**, zero
skips, two existing warnings. Commands: `python -m pytest -q tests -k
"encounters3d or flight3d"` then `python -m pytest -q` (venv; OMP/MKL threads 1).
Logs/XML: `.superpowers/encounters3d/targeted.*`, `full.*`. Native review remains
separate; no avoidance success is claimed.
''',1)
status.write_text(text,encoding='utf-8')

queue=Path('docs/BUILD_QUEUE.md'); text=queue.read_text(encoding='utf-8-sig')
text=text.replace('# FlyBrain task queue\n','''# FlyBrain task queue

## Exclusive encounters — infrastructure verified, escape gate closed 2026-10-05

- Additive spawn buttons, exclusive scenes, exact teacher parity, RGB predator,
  actual neural activity, reset/replay/pause/preferences are automatically and
  rendered verified. **100 targeted / 667 full-suite passed**, zero skips; all
  **2,465 historical protected files unchanged**. Native encounter preview is
  open; helper could not connect, so live feedback remains pending.
- **Directional escape unavailable:** all 16 intact physical static stimuli
  produced zero wing spikes. Synthetic full field shows a DNp03 contribution
  and other paths. All 24 closed-loop cases hovered without retreat. No decoder,
  retrigger, training, optimizer updates or interrupted-fit continuation.
- **One next task / Sol High:** specify a bounded sourced/physiological escape
  recovery using the concrete DNp26/DNp57 network proposal. No implementation
  or experiment is automatically started. [Guide](ENCOUNTERS3D.md),
  [plan](plans/ENCOUNTERS3D.md), [verification](verification/encounters3d.md).

''',1)
text=text.replace('## Current display task','## Previous display task',1)
queue.write_text(text,encoding='utf-8')

plan=Path('docs/plans/ENCOUNTERS3D.md'); text=plan.read_text(encoding='utf-8')
text=text.replace('- [ ]','- [x]')
text+='''
## Completion ledger

Characterization complete: one physical/saturation run, negative wing spike gate.
Scene/neural numerical/session/viewer implementation and 13 new tests complete.
Motor-decoder/retreat/retrigger implementation stopped under the user-specified
failure condition; no hidden steering or training substitute.
24 matched closed-loop hover measurements complete, with no retreat claim.
100 targeted / 667 full tests passed; 2,465 historical files preserved.
Rendered UI reviewed; native access attempted and unavailable, user review pending.
Local acceptance self-review complete with one agent. No Git repository/commit.
Routine rulings: supplied spec authorizes implementation without extra skill gates;
All activity is initial only in the new launcher; use existing source-preserving
viewer/session subclasses; failed scientific gate leaves hover explicitly labeled.
Next: bounded sourced/physiological recovery specification / Sol High.
'''
plan.write_text(text,encoding='utf-8')

log='''## 2026-10-05 — Exclusive flower/predator trials; negative escape gate

### Goal

Implement the user-specified additive exclusive encounter lab, characterize actual
downloaded escape circuitry before motor decoding, and preserve all prior work.

### Changes Made

Added `flybrain.encounters3d` scene/vision/physics, EscapeBrain RGB encoder,
source-preserving session/viewer subclasses, Spawn flower / Spawn predator,
R/new selected kind, Backspace/exact replay, immutable publication and recorded
DNp03/wing UI signals. Flower mechanics/teacher/brain calculations match exactly.
Predator is a cyan stationary spherical proxy; one object only. Both spawning
and replay clear all body/controller/neural/display history and retain pause,
camera, class/scope, selection and display choices. Added 13 numerical/behavior/
real-render/record tests. Wrote new source-bound JSON/CSV and executed source copies.
Updated guide, plan, status, architecture, decisions, queue and experiment summary.

### Decisions

Controllers receive only FlightObservation; scene identity/XYZ stays environment-
only. Cyan recognition, assumed ID-ranked retinotopy and peak/occupancy/growth
rate drive are engineered. Keep 589 source cells / 4,139 edges and old LIF dynamics;
no additions or fitting. One predeclared static gate closed negative: zero wing
spikes in every physical intact stimulus. No escape decoder, retreat/retrigger
policy or hidden sensory steering substitute was selected. The preview explicitly
labels unavailable avoidance; IMU hover is the sole motor scaffolding. Only this
launcher initially enables All activity. Proposed sourced DNp26/DNp57 saccade
network recovery is a separate specification, not work begun here.

### Tests / Verification

Serial venv, OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1:

- `python -m pytest -q tests -k "encounters3d or flight3d"`: **100 passed**, 567
  deselected, zero skips, two existing warnings, 49.22 s, exit 0.
- `python -m pytest -q`: **667 passed**, zero skips, two warnings, 152.09 s, exit 0.
- Flower 120-step seed-970000 smoke: 2.40 s, exit 0; predator 250-step seed-980000
  smoke: 5.00 s, exit 0. Exact neural/teacher parity checked over 100 steps.
- 51 static matched presentations: 51,000 neural steps, no updates. Sixteen
  physical intact cases each produced zero motor spikes; maximum sampled wing
  voltage .229664. Synthetic full field: 1,838 intact / 1,476 DNp03-silent / zero
  visual-silent wing spikes. DNp03 contributes but is not the only route.
- 24 matched five-second closed-loop cases: 120,000 neural / 24,000 physics steps,
  zero separation change, collisions or escape responses; hover scaffolding only.
- Final offscreen 1440x900 frames visually reviewed; actual framebuffer cyan test
  and 1000x700 UI event checks pass. **2,465 historical source/test/data/experiment
  paths byte-identical**. No historical checkpoint/fit/assay modified or resumed.

### Problems / Failed Attempts

Missing-module tests first failed RED. Color threshold .02 accepted sky roundoff
and stem .03; fixed after inspecting actual RGB values. Peak-only fields lost
static expansion, so occupancy weighting was added. Initial record setup mixed
relative/absolute source paths; failed before config/neural steps and was repaired
in its empty directory before the single run. Actual WGL packed vertex colors
rendered a cyan sphere yellow despite correct CPU vertex RGBA; a framebuffer RED
test reproduced it, explicit surface color fixed it, GREEN passed. Contextual
doc patches twice rejected without edits. No scientific retry after gate closure.
Computer-use skill imported, but `sky.list_windows()` failed because the native
pipe was unavailable (os error 2). No live UI acceptance claimed.

### Current State

Encounter infrastructure and transparent neural probe are automatically/rendered
verified. Neural avoidance, retreat and renewed-growth/reappearance retrigger are
unavailable after the failed gate. Native preview launched seed 980000, launcher
PID 30116, separate encounter title; user review pending. One-agent local review
complete; no commit, push, training, optimizer update or interrupted-run continuation.

### Next Step

One task / Sol High: specify a bounded sourced/physiological escape-path recovery
using the DNp26/DNp57 proposal and a predeclared directional motor gate. No extra
FB-07 experiment, model switch or learning run is automatically started.

### Notes

Recorded spiking activity can exist without sufficient motor drive. PyTorch
advances tensor voltage/spike/phase state through source edges; assumed neuron
rules and RGB tuning are hand-written. Forward simulation is not learning.
Concrete phase/weight/leak example and sourced recovery references are in
`docs/ENCOUNTERS3D.md`; evidence in `docs/verification/encounters3d.md` and
`experiments/encounters3d-characterization-2026-10-05/`. These development cases
are now historical. This negative result concerns the current motor spike code,
not an impossibility claim about real source anatomy or subthreshold decoding.
'''
with Path('PROJECT_LOG.md').open('a',encoding='utf-8') as stream: stream.write('\n\n'+log)
