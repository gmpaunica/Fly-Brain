

## 2026-10-02 - Anatomical visual and responsiveness repair

### Goal

Implement the user's October 1 approved anatomical repair: recognizable registered
fly-brain silhouette, subdued branches and sparse warm activity, responsive UI,
preserved neural/sensory behavior and reconciliation of the interrupted fit.

### Changes Made

- Added anatomical_display.py: immutable shared cached source arrays and capped
  neuron-level activity ranking from existing snapshot current rates.
- Added anatomical_viewer.py: source surfaces/reference context, batched coarse
  branches, full cached cyan selection, paginated browsing, selected-only C
  connection diagrams, progressive loading and deferred hidden cord buffers.
  The default launches this viewer; historical viewers are preserved for comparison.
- Added simulation_pump.py: one serial worker with bounded wall-time debt, unchanged
  fixed numerical steps and pause/replay/new/exit barriers. UI reads immutable
  snapshots. No source neural/sensor/world/teacher/session calculations changed.
- Saved/inspected real geometry front/side/top before activity polish, then activity,
  cord and selected full morphology. Added silhouette margin and restrained lighting.
- Added three-view performance benchmark and updated guide, status, architecture,
  decisions, experiments summary, task queue and repair ledgers.

### Decisions

Keep cached source coordinates/branch junctions. Do not draw centroid bundles by
default or invent missing source geometry. Warm whole-branch light indicates modeled
neuron activity, without measured conduction location/speed. Teacher remains the
preview controller. One agent/local review, no commit or further scientific run.

### Tests / Verification

- Targeted: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests -k
  flight3d --junitxml=.superpowers/anatomical-repair-2026-10-01/targeted.xml`:
  50 passed, 567 deselected, two existing upstream warnings, 58.32 seconds.
- Full-suite outcome is recorded below after completion. The same 617 collected
  tests are run with OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 (preview/benchmark CPU
  configuration), without preloading PyTorch before pytest.
- Behavior checks include source picking, panel scrolling, mode/cord/presets,
  replay/new/exit, snapshot-only activity, serial/background exact snapshots and
  voltages, and camera/detail/render-frequency independence. UI pixels separately
  checked; native mouse/resize/Shift-drag still need user observation.
- Sequential benchmark, Intel Iris Xe, Python 3.12.10/PyTorch 2.14.0+cpu, 1440x900,
  120 frames per condition: prior Anatomy p95 paused/running/orbit/select=
  25.72/65.47/30.25/32.54 ms; Network=90.51/134.91/129.26/118.49 ms;
  repaired=5.10/14.64/6.35/13.70 ms. Repaired paced 55.8-58.8 FPS; steady 60
  not verified. Simulation 0.57 simulated seconds/wall second, unchanged steps.
  Max running 41.19 ms/select 34.68 ms disclosed. These are offscreen samples.
- SHA256 recheck: 594 existing source/experiment files, 593 unchanged. Sole existing
  source edit: 3D launcher import. All previous viewers, neural/sensor/session source,
  2D source and experiment/checkpoint/journal bytes unchanged.
- Read-only interruption reconciliation: 48/48 teacher cases, 1,050 completed updates,
  positive gradients, 12 changed tensors; candidate/interrupted tensors match.
  Original parameter-change file stops at 1,049; actual deltas recorded externally.
  No reload gate, behavioral gate or reserved validation/final was performed.

### Problems / Failed Attempts

- First full-suite process ended across the session interruption without an XML
  result; not counted as passing. A later run had an early CLI failure and was
  stopped because CPU execution was unusually slow; its failure details were not
  finalized. The isolated affected neural CLI passed in 9.05 seconds.
- Preloading Torch/set_num_threads before pytest produced a native Windows exception
  during existing Pygame initialization, with no complete report. Narrowed the
  environment: all 11 existing app tests passed with environment-only CPU thread
  limits in 33.91 seconds. The final full command uses those limits and normal imports.
- The screenshot display tool omitted unchanged static UI visually on some images;
  actual opaque PNG pixels contain identical complete headers across all eight.
- Initial serial repaired running frames exceeded the target; profiling found
  roughly 26-32 ms simulation steps. Background scheduling improved UI latency without
  altering their numerical calculations. Simulation remains slower than wall time.

### Current State

Anatomical repair implemented and saved-view QA completed. The p95 navigation gate
passes, while measured pacing is below steady 60 and live interaction review remains
pending. A native preview was opened for the user on seed 970000; review requested
through the async question. Scientific landing acceptance remains open; October 1
fit is incomplete at 1,050 and its authorization is consumed.

### Next Step

One next task / Sol Medium: finish live anatomical interaction review and record
any remaining pacing/control issue. No fitting or reserved evaluation in that task.

### Notes

Whole-neuron illumination is a display summary of existing modeled delivered
current. Rendering it does not create signals, learn weights or establish biological
signal travel. New code is additive; all historical scientific evidence remains.
Detailed reports/previews: `.superpowers/anatomical-repair-2026-10-01/` and
`docs/verification/flight3d-anatomical-repair.md`. No Git repository or commit.
