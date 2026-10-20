# Relay display color — 2026-10-13

User reports the dashboard works well, asks what Other relays means and requests
a less overwhelming color. Dashboard-only steel-blue #5B8FB9 replaces white in
the structural group's firing palette and checkbox swatch. Descriptions now say
these cells were retained on sensor-to-movement connection paths. This is a
structural selection category, not one biological type or proof of function.

Prior presentation files are archived under `artifacts/relay-color-2026-10-13/before`.
Original source-bound classification/palette, neurons, controller, pulse timing,
opacity, topology and recorded results remain unchanged. No training or new
scientific evaluation was performed. No redundant color-constant unit test added.

## Focused checks

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_ui.py -q --tb=short --junitxml=artifacts/relay-color-2026-10-13/targeted.xml
```

Observed **6 passed**, zero skips, one upstream TorchScript warning, 15.83s.
Existing tests exercise real controls, geometry filtering/picking, GPU pixels,
state preservation, camera bounds and exact short replay.

Actual app palette probe confirms exactly the 241 structural owners differ from
the immutable original palette. All other entries remain equal, and that
original palette is unchanged. The checkbox swatch uses the same color helper as
the branch palette. Paused PNG reviewed at 1000x700; snapshot and actual/reference
neural steps remain unchanged/zero. `palette.json` records this check. All 3,010
protected original source/experiment files and original verified candidate pass.
Saved-frame/programmatic review is separate from native mouse/keyboard review.

## Full suite

OMP/MKL threads one, PYTHONUNBUFFERED one; historical diagnostic ledgers isolated:

```powershell
$env:FLYBRAIN_TEST_LEDGERS='artifacts/relay-color-2026-10-13/full-ledgers'
.\.venv\Scripts\python.exe -m pytest -q tests artifacts/escape-flight-validation-2026-10-08/test_serialized_audit.py --junitxml=artifacts/relay-color-2026-10-13/full.xml -p flybrain.escape_recovery.pytest_plugin
```

Observed **928 passed**, zero failures/errors/skips, three upstream warnings,
414.38s /6:54, exit 0. Logs/XML under the new relay-color artifact directory.
Final source-bound delivery/protected-file checks pass. Saved screenshot reviewed;
no new native input acceptance claim is made. Reopen the app to load the palette.
