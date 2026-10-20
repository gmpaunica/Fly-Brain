# 3D visual repair and grouped network implementation plan

**Goal:** Execute the user-supplied repair, presentation and single-experiment request.
**Spec:** `experiments/flight3d-development-2026-10-01/REQUEST.md` (verbatim).
**Architecture:** Keep environment ray intersections separate from shared RGB
processing; instrument the existing delayed-spike calculation without changing
dynamics. Use deterministic source groups and passive transmission displays.
Freeze an independently authorized protocol only after development coverage passes.
**Stack:** Existing Python, NumPy, PyTorch and Panda3D; one agent, inline execution.

## Constraints and review focus

Preserve historical source/results/checkpoints; no commits or automatic reruns.
Controllers receive immutable local observations only. Preserve all behavioral
gates and reserved identities. Never interpret interrupted execution as success.
Review image boundaries/occlusion, temporary visual loss, missed render intervals,
small windows/panel routing, and renamed experiment directories explicitly.

## Tasks

- [x] Inventory hashes and copy pre-edit 3D source; preserve request.
- [ ] Add failing numerical sensor/search regressions, implement 2x4 sampling and
  shared contrast, run the regressions. Record causes and separate phase evidence.
- [ ] Record full 30-second results for 970000–970047 and historical 960000;
  require 49/49 landing/dwell successes before freezing.
- [ ] Test and implement immutable edge transmission telemetry, deterministic
  approximately 96 source groups, bundle accounting and 300 ms passive smoothing.
- [ ] Test and implement default Network/alternative Anatomy, group/member
  inspection, smooth bounded cameras, panning, scrolling and responsive controls.
- [ ] Make the runner consume frozen cases/configuration, enforce overlap and
  one-use identity checks, durable update/stage journals and reload reconciliation.
- [ ] Freeze/run exactly one experiment with seed 940404 and the requested
  980000/981000/982000 ranges, only if development prerequisites pass.
- [ ] Run targeted and full suites, inspect saved front/side/top frames, obtain
  live interaction evidence when available, reconcile preservation/results.
- [ ] Update status/log/guide/architecture/decisions/experiment summary/queue;
  leave one evidence-based next task and a concrete learning example.

Commands: `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests -k flight3d`
and `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`.

The user's detailed written design and explicit single-agent execution instruction
authorize routine implementation choices; no duplicate design approval is needed.


## Historical outcome reconciled during anatomical repair

Sensory/search repair, 49/49 development checks, transmission telemetry,
protocol/receipt/journal controls and one frozen experiment were implemented.
The grouped default was rejected by the user and superseded by
[the anatomical plan](FLIGHT3D-ANATOMICAL-REPAIR.md). Its experiment passed 48/48
teacher cases and interrupted after 1,050 updates; no scientific completion or
reserved evaluation is claimed. Original checkpoints/journals remain unchanged.
This ledger is historical and does not authorize resuming unchecked run steps.
