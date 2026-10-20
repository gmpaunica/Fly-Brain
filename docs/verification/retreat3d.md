# Idle encounter lab verification — 2026-10-06

Partial infrastructure delivery; neural retreat was not delivered. The directional
gate failed 9/96 and closed the experiment before all 48 closed-loop cases.

## Commands and evidence

All Python commands use `.venv/Scripts/python.exe`; OMP/MKL threads are set to 1.

- Initial targeted encounter/3D/retreat subset: 75 passed, zero skips, one upstream
  TorchScript warning; 95.23 s. XML: `artifacts/retreat3d-2026-10-06/targeted.xml`.
- Final broader targeted command: PowerShell `rg --files tests -g
  'test_retreat3d*.py' -g 'test_flight3d*.py' -g 'test_encounters3d*.py' -g
  'test_pathway*.py'` supplies the file array to `python -m pytest -q $retreatTests
  --junitxml=artifacts/retreat3d-2026-10-06/targeted-final.xml`: **153 passed**,
  zero skips, one upstream warning; 124.50 s.
- Full suite: `python -m pytest -q
  --junitxml=artifacts/retreat3d-2026-10-06/full.xml`: initial **719 passed,
  one failed**, zero skips, three upstream warnings; 202.06 s. Failure:
  `test_banc_projection_fit.py::test_canonicalization_preserves_original_values_and_pn_values_without_steps`.
  Its broad source-provenance guard caught a concurrent final HUD edit to
  `retreat3d/viewer.py`; saved protocol hash comparison identifies that sole
  mismatch. This was a verification orchestration error, not a historical source
  edit or canonicalization defect. Source files are now frozen. Isolated regression
  **passes (1 test, 9.44 s)**. Final frozen-source retry:
  `python -m pytest -q --junitxml=artifacts/retreat3d-2026-10-06/full-final.xml`:
  **720 passed**, zero failures/skips, three upstream warning instances, 188.49 s.
- `python scripts/audit_retreat_records.py
  experiments/retreat3d-motion-repair-2026-10-06`: passes source/receipt hashes,
  frozen calibration pooling, signed vectors, recorded case classifications,
  10,800 JSON/CSV intervals, all motion-only parity and exact budget. Zero new
  neural steps. 9/96 validation, no closed-loop cases, no optimizer/checkpoint.
- `python scripts/audit_retreat_preservation.py check`: **2,433 unchanged**, no
  changed/missing original source/data/experiment files.
- `python -m flybrain.retreat3d --headless --seed 1000000 --steps 600 --record
  artifacts/retreat3d-2026-10-06/headless-actions.json`: **12.00 s, resting,
  no object, crashed=False**. `--headless --replay-log` reproduces that endpoint.
- `python -m flybrain.retreat3d --seed 1000000 --capture
  artifacts/retreat3d-2026-10-06/final-frames`: saved frames visually reviewed.
  Final two-line HUD review: `--capture artifacts/retreat3d-2026-10-06/review-frames`.
  Predator placement and visible size/growth/behavior/failed confidence inspected.
  Initial ground, idle cruise and replay endpoint frames are separate evidence.
- Additional engineering-only idle check: seeds 100–109, 35 s each, local IMU/contact
  observations with blank eyes (idle ignores RGB), zero neural steps. All ten avoid
  crashes; maximum measured home displacement .227280 m. This is idle mechanics
  verification, not an extra neural/closed-loop escape presentation.

## Local acceptance review

Single-agent source review against the supplied spec. Additive modules keep world,
input recording, IMU/controller, neuron/decoder, sensing and rendering separate.
Historical implementations are hash-protected and are numerical parity oracles.
Actual/reference brains have identical fixed graph/rules; only actual cells are
displayed. Decoder input is exclusively manifest-ordered signed wing counts.
No object pose, range or RGB-derived steering enters the controller/decoder.
The shared estimate transfers IMU state into the retained teacher; no world pose
initializes it. Spawn events transfer only mode and current local contact.

Behavior tests cover safe/unsafe ground contact, takeoff/rest, repeated seeded
cycles/home envelope, above-ground original flower physics/eye parity, diagonal
movement, arena limits, relative crossing and resting collisions. Lifecycle tests
cover continuous spawning, removing flower support, local neural-clock origins,
reset, exact saved/reloaded actions, endpoint spawns, reconstructed display history,
worker pause, key/focus release and renderer independence. Real-renderer tests
project the published moving predator position into the framebuffer and check
cyan pixels; camera/classes/cord/settings persist through spawn and replay.

## Failures and corrections

Missing-module RED tests preceded each additive subsystem. One angular test
incorrectly expected constant new-appearance status; changed to assert a transient
onset and no persistent retrigger. A float equality test used exact `.08` instead
of numerical tolerance; the measured diagonal norm was .08000000000000002.
Local review reproduced stale support after removing a flower and previous-tick
contact on teacher handoff; regression tests failed first and passed after fixes.
Saved-frame review found moving predator geometry had a baked center plus a node
translation; the framebuffer regression failed, then passed after origin-centered
geometry with a single published-position translation. Offline audit JSON output
needed conversion of NumPy int64 counters to Python ints; no scientific rerun.
The initial full suite correctly rejected a source mutation during its provenance
check. All 719 other tests passed. Retry only freezes source during tests; no neural
model fix, historical assay or training rerun is involved.

## Separate live-review limits

Supported Computer Use initialization succeeds; `sky.list_windows()` fails with
`Computer Use native pipe is unavailable ... The system cannot find the file
specified. (os error 2)`. No native input/screenshot review was possible.
A responding paused preview opened at idle cruise (seed 1000000, 2.8 s), then
was closed during final verification to avoid rendering contention. The final
preview is reopened after successful checks. Saved-frame and injected-event tests do not substitute
for the requested live idle/rest/predator-control review. Live escape review cannot
be completed because escape failed its scientific prerequisite.
Final preview launcher PID 7172 / window PID 10076, seed 1000000, paused at 2.8 s
cruise. Panda reports `SetForegroundWindow() failed`; open the window from the
taskbar if it is behind another app. This is not a live acceptance claim.
