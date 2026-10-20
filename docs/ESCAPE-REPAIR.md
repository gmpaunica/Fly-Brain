# Escape completion repair - 2026-10-08

Status: implementation and targeted checks passed; full-suite and full-neural
evaluation pending. The original failed physical experiment remains unchanged.

## Why the old escape failed

Three trajectories recovered and then hit a wall. Recovery handed control back
to idle cruise, which immediately requested 60 mm/s along its outward heading.
Its home correction rotated that heading at at most 0.9 rad/s, too slowly to
turn before the nearby wall. The escape-only home guard stopped applying when
the state returned to idle. A literal recovery fixture reproduced the collision.

Two cruising trajectories stayed in fleeing through the six-second endpoint.
Their apparent threat size had fallen below 10 degrees, but small silhouette
changes during turning produced positive frame differences. The monitor averaged
only upward differences, discarding the downward ones, so sampling jitter looked
like sustained expansion and repeatedly reset the 300 ms weak timer. A real RGB
patch fixture stays below 10 degrees yet reports 44.85 deg/s positive growth in
the original monitor. It never clears. This reproduces the numerical mechanism
without running a neural circuit or supplying a steering answer.

## Repair

`escape_repair.threat.ClearanceMonitor` retains the original size calculation,
strong-onset detection and positive-growth diagnostic exactly. Weak clearance
uses 15 continuously small samples and their net smoothed-size change over
0.28 seconds. A steadily growing image still prevents clearance; alternating
pixel-edge jitter cancels instead of accumulating. No timer or distance override
forces escape to finish.

`escape_repair.control.BoundedIdleFlight` retains seeded cruise, landing and rest.
Outside 180 mm estimated displacement, it limits outward velocity directly,
rather than waiting for heading to turn. The radial limit is
`clip(2 * (0.22 - radius), -0.06, 0.06)` m/s. Beyond 220 mm it requests inward
translation while preserving the tangential component and altitude servo.
Displacement comes from the existing integrated IMU estimate. World coordinates,
predator position, distance and evaluator geometry never enter the controller.
This home envelope assumes the existing calibrated, centered arena start.

The original 589-cell/4,139-edge circuit, (4,2) gains, neuron rules, visual neural
encoding and accepted temporal direction readout are unchanged. The historical
94/96 result and its two known 86.25-degree failures remain historical limitations.
This repair adds no autograd, optimizer updates, training or learned behaviour.

## Verification protocol

See [the predeclared plan](plans/ESCAPE-REPAIR.md). Require all 24 intact historical
escapes and all 24 additional-seed escapes to pass the original measured physical
criteria, with zero escape triggers in their 48 visual-silent controls. Same
eight angles and three starting states; six seconds each. New seeds test seed
robustness, not unseen geometry. Budget: 1,152,000 individual neural steps.
Four predetermined exact action replays add 48,000 verification steps.

Motor-only tests use explicit synthetic wing impulses with real sensory images
and physics. They isolate the repaired motor policy and do not establish neural
escape. Full-neural measurements remain the acceptance evidence. Separate
20-second motor tests check recovery beyond the six-second evaluation endpoint.

The new session, logs, auditor and launcher are adjacent to frozen historical
sources. Earlier launchers and receipts retain their original results. The new
launcher will require passing scientific, independent-audit, replay, visual and
source-preservation evidence before activation.

## Commands and evidence

After activation, double-click `Launch Escape.cmd` from the project folder, or:

```powershell
.\.venv\Scripts\python.exe -m flybrain.escape_repair
```

Targeted checks:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:FLYBRAIN_TEST_LEDGERS='artifacts/escape-repair-2026-10-08/engineering-ledgers/targeted'
.\.venv\Scripts\python.exe -m pytest tests -k "escape_repair or escape_flight_validation or escape_recovery or turnflee or retreat3d" -q -p flybrain.escape_recovery.pytest_plugin --junitxml=artifacts/escape-repair-2026-10-08/targeted.xml
```

Observed: 154 passed, 724 deselected, zero skips, three upstream warning
instances, 194.00 seconds. The full suite command includes the six existing
saved-CSV adapter tests:

```powershell
$env:FLYBRAIN_TEST_LEDGERS='artifacts/escape-repair-2026-10-08/engineering-ledgers/full'
.\.venv\Scripts\python.exe -m pytest -q tests artifacts/escape-flight-validation-2026-10-08/test_serialized_audit.py -p flybrain.escape_recovery.pytest_plugin --junitxml=artifacts/escape-repair-2026-10-08/full.xml
```

Use fresh ledger/log/XML paths on any later engineering run. Scientific worker
and replay commands consume their predeclared budgets and are not rerun commands.

Artifacts live in `artifacts/escape-repair-2026-10-08/`; scientific records in
`experiments/escape-repair-2026-10-08/run/`. Initial red tests and the test-file
placement mistake are retained in logs. The corrected integration tests pass.
A clean viewer import initially exposed Windows DLL ordering; initializing
PyTorch before Panda3D fixed it. The offscreen viewer screenshot was reviewed
at zero world/brain steps. This is separate from a live interactive review.
