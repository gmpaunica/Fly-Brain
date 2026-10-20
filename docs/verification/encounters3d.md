# Exclusive 3D encounter verification — 2026-10-05

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


## 2026-10-05 — User-confirmed live encounter controls

The user reports: “all controlls work, but the fly doesnt react. its interesting
that the brain reacts differently depending on where the predator spawns”.

This confirms the requested live spawn exclusivity, R/Backspace, pause/reset and
camera/display retention checks. Placement-dependent brain appearance and absent
fly response are user observations, consistent with the recorded sensory activity
and failed motor-spike gate. No directional decoding or neural escape success is
inferred. Current predator flight is controlled solely by the explicit hover
stabilizer. Historical activity-view acceptance remains separate. No code or
scientific result changed; no tests rerun, training or recovery experiment begun.
