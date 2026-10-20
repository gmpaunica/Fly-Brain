# Sustained edge escape verification — 2026-10-11

All commands run from the project root with the venv, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. Historical diagnostic ledgers are isolated with
`-p flybrain.escape_recovery.pytest_plugin` and separate
`FLYBRAIN_TEST_LEDGERS` directories under `artifacts/escape-edge-2026-10-11/`.

## Engineering

- Red: `python -m pytest tests/test_escape_edge.py -q --tb=short`: 33 failed
  before implementation (`.escape-edge-red.log`). First implemented pass:
  31 passed/2 failed in 263.21 s; failures were absent delivery and a forged
  stop inserted inside the explicit startup grace. Corrected that fixture
  timing without changing the gate. Second run: 31 passed/2 failed, 74.65 s,
  before corrected fixture/delivery were loaded. Third run: 36 passed, two
  upstream warning instances, 51.57 s (`.escape-edge-green3.log`).
- Motor-only tests use actual physics/IMU/contact and synthetic impulses.
  Replaced irrelevant repeated retina rendering with a blank retinal image
  in motor tests; full-neural evaluation retains actual retinal rendering.
- Targeted: `python -m pytest tests -k "escape or turnflee or retreat3d" -q
  --junitxml=artifacts/escape-edge-2026-10-11/targeted.xml
  -p flybrain.escape_recovery.pytest_plugin`: 239 passed, 675 deselected,
  zero skips, three warning instances, 342.29 s.
- Additional edge-safe viewer placement: `python -m pytest
  tests/test_escape_edge.py -k viewer_predator -q
  --junitxml=artifacts/escape-edge-2026-10-11/spawn.xml`: red then one passed,
  36 deselected, zero skips, two warning instances, 13.03 s. Added after broad
  collection; its XML is a separately bound engineering proof.
- Full: `python -m pytest -q tests
  artifacts/escape-flight-validation-2026-10-08/test_serialized_audit.py
  --junitxml=artifacts/escape-edge-2026-10-11/full.xml
  -p flybrain.escape_recovery.pytest_plugin`: 920 passed, zero skips, three
  warning instances, 609.45 s. Fresh final run including both additional
  viewer regressions uses `final-full.xml`: **922 passed**, zero skips,
  three warning instances, 362.48 s. Latest focused `edge.xml`: **38 passed**,
  zero skips, two warning instances, 24.07 s.
- UI: `python artifacts/escape-edge-2026-10-11/check_viewer.py`: passed at zero
  simulation/neural steps; initial and updated frames preserved separately.
  Direct repeated graphics rendering lost text; normal frame tasks reproduce
  correct text rendering at zero steps. Absolute Windows screenshot paths
  require `Filename.fromOsSpecific`; regression failed then passed in the
  focused/final full runs. Reproduction and failed images/logs are retained.
  Native Windows Computer Use helper fails with missing native pipe on both
  discovery attempts. No native input or captured live UI review is claimed.

## Measured evaluation

`python artifacts/escape-edge-2026-10-11/prepare.py` froze the candidate after
all engineering proofs passed and 3,010 protected files verified unchanged.
`python -m flybrain.escape_edge.worker` completed under the hidden detached
pipeline: **24/24 intact in each batch**, zero silent triggers, 48 cases per
batch, exactly 1,152,000 reserved individual steps each. Both saved-only audits
pass. `python -m flybrain.escape_edge.replay`: **four exact replays pass**,
96,000 separately reserved steps, zero new scientific cases. Receipt hashes
and terminal statuses are in `development-audit.json`, `robustness-audit.json`,
`replay.json` and `worker.json`.

`python artifacts/escape-edge-2026-10-11/check_sustained.py`: all 48 intact cases
pass full commanded flee speed and cumulative progress; maximum pre-hover
retreat **0 m**, zero new steps. `plot_results.py` generated two saved-only
figures, both reviewed. `finish_delivery.py` activated the source-bound launcher
after those reviews, both audits and exact replay. No consumed source changes.

`pythonw artifacts/escape-edge-2026-10-11/live_review.py`: actual native viewer
API review passed at 12 s, 24,000 separate engineering steps, zero new scientific
cases. Turning/fleeing/hover frames saved at 1/3/6/12 s; 3/12 s native frames
visually inspected. Window process 21420 is responding and left open/paused.
Native mouse/keyboard input automation remains blocked by missing pipe; some
saved native frames lose labels transiently. Neither is counted as passing UI
input review. The numerical motion checks and real viewer API review pass.

Reproducibility/source/protection checks and a zero-step public CLI activation
smoke are recorded in the final handoff log; tests were not needlessly repeated
after the unchanged-source scientific runs.
Science and replay are separately reserved in append-only journals. Never rerun
a consumed batch/replay or overwrite a failed record. The approved task permits
new declared development candidates after diagnosis; it does not change gates.
