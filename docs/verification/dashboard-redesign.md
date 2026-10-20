# Dashboard redesign verification — 2026-10-12

All commands use the project venv from the project root with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `PYTHONUNBUFFERED=1`.
No training experiment, new scientific assay, receipt rewriting, commit or
publishing was performed. Previous presentation files are archived under
`artifacts/dashboard-redesign-2026-10-12/before/`.

## Red/green checks and fixes

Initial `python -m pytest tests/test_dashboard_ui.py -q --tb=short` failed all
five cases: missing pixel layout/filter table, checkbox sidebar and isolation
methods. Pure checks then passed after fixing their synthetic snapshot's ID
ordering to match the actual immutable frame.

Windows font startup reproduced: native `C:/...` font string cannot load;
FontPool rejects a Filename object with TypeError; a normalized path string
loads successfully. The production loader now uses that string and a portable
fallback. Those constructor failures caused subsequent multiple-ShowBase errors
in the same failed test process. No scientific run was started or retried.

After implementation, five functional checks passed. Added a real GPU overlap
check: a hidden nearer green branch must not occlude a visible red branch;
all-hidden pixels must be black, while the spike frame still contains both
spikes. It passes using the actual shader and orthographic render buffer.

Compact visual review found description text extended to y690 in a sidebar
ending at y680. Added the failing pixel-bounds test, then adjusted spacing.
Local acceptance review also added a failing case where the selected full
morphology extends beyond coarse branch bounds; Fit now includes the full
visible selection and excludes it when filtered out.

Final focused command:

```powershell
$env:FLYBRAIN_TEST_LEDGERS='artifacts/dashboard-redesign-2026-10-12/targeted-ledgers'
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_ui.py tests/test_escape_edge.py -q --tb=short --junitxml=artifacts/dashboard-redesign-2026-10-12/targeted-final.xml -p flybrain.escape_recovery.pytest_plugin
```

Observed **44 passed**, zero skips, two upstream TorchScript warning instances,
175.71 s. The six new dashboard cases include both sizes, genuine DirectCheckButton
states, idempotent explicit callbacks, independent/bulk filters, retained tabs,
checkbox synchronization, hidden selection/picking, visible camera bounds, empty
view, input routing, spawn/reset and exact three-interval interaction replay.
Repeated GUI changes/capture preserve snapshot, actual/reference voltage,
publication identity, camera and zero neural/world time. Spike history is
unchanged by alpha filtering. The original 38 edge regressions also pass.

## Visual review and preservation

```powershell
.\.venv\Scripts\python.exe scripts/review_escape_ui.py artifacts/dashboard-redesign-2026-10-12/reviewed-frames
```

Ten frames captured: dashboard, Vision-only filter, readable inspection, Help
and empty view at 1000x700 and 1440x900. Normal/filtered/Inspect/Help/empty layouts
reviewed before the full suite. All captures use zero simulation seconds and
zero actual/reference neural steps. Header-pixel checks pass. Original source
preservation checks pass for **3,010 protected files**, and the original
source-bound verified candidate remains valid. Raw capture JSON and image hashes
are recorded; `review.json` binds the reviewed code/tests/frames.

The preview tool can omit text in some multi-image presentations; direct pixel
checks and full-resolution inspection complement that review. No image-tool
artifact is claimed as a repaired simulation/rendering fault.

Native mouse/keyboard acceptance is not performed: tools expose no native app
control. Real GUI callbacks, real GPU pixels, saved-frame review, native window
launch and user review are separate evidence, reported as such.

## Full regression suite

Started after the final layout, local code review and focused checks:

```powershell
$env:FLYBRAIN_TEST_LEDGERS='artifacts/dashboard-redesign-2026-10-12/full-ledgers'
.\.venv\Scripts\python.exe -m pytest -q tests artifacts/escape-flight-validation-2026-10-08/test_serialized_audit.py --junitxml=artifacts/dashboard-redesign-2026-10-12/full.xml -p flybrain.escape_recovery.pytest_plugin
```

The first process stopped at 93% without a summary or XML and was absent on
2026-10-13. Its `full.log` is preserved; the interruption cause is unknown, and
it is not counted as a passing run. Reviewed source/frame hashes remained valid.

Fresh completion run, 2026-10-13:

```powershell
$env:FLYBRAIN_TEST_LEDGERS='artifacts/dashboard-redesign-2026-10-12/full-final-ledgers'
.\.venv\Scripts\python.exe -m pytest -q tests artifacts/escape-flight-validation-2026-10-08/test_serialized_audit.py --junitxml=artifacts/dashboard-redesign-2026-10-12/full-final.xml -p flybrain.escape_recovery.pytest_plugin
```

Observed **928 passed**, zero skips/failures/errors, three upstream warning
instances (Pygame pkg_resources and TorchScript deprecation), 480.87 s /8:00,
exit 0. No further implementation change occurred after reviewed code/focused
checks. Original protected source hashes and source-bound candidate remain valid.

## Native launch — 2026-10-13

Opened the actual dashboard paused for user review using
`artifacts/dashboard-redesign-2026-10-12/open_review.py`. Native child PID 26440
has title `FlyBrain | Explore the brain`, nonzero window handle and Responding
true. Windows reports SetForegroundWindow failed; the user can switch through
Alt+Tab or the taskbar. Launch logs are `launch-2026-10-13.*.log`.
This verifies window creation/responsiveness, not native mouse/keyboard acceptance.
