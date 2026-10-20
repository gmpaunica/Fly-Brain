# Restore anatomical shape and viewer responsiveness

User-approved implementation plan, 2026-10-01. One agent, existing cached source
geometry, teacher preview. No commits, new experiment, fitting, or reserved evaluation.

## Design

Use the recognizable fly-brain anatomy from the second user reference with the
restrained lighting of the first: black, faint blue/violet source branches and
registered surface, sparse warm activity. Restore actual source skeletons, preserve
the side lobes and central brain, and remove centroid bundles from default drawing.
Display at most 12 active modeled neurons. Whole-branch illumination represents
neuron-level activity, not measured signal location or conduction speed.

Full source morphology/evidence loads on selection. Connections appear only through
explicit selected-neuron inspection. Preserve independent cameras, scrolling,
reset, nerve cord, pause/new/replay/exit. Start brain-focused with cord hidden.

## Implementation ledger

- [x] Preserve hashes, reconcile the interrupted experiment (1,050 updates),
  measure prior Anatomy and Network under identical paused/running/orbit/selection conditions.
- [x] Test and build batched source geometry with bounded progressive loading;
  inspect front/side/top silhouette previews before activity polish.
- [x] Test sparse snapshot-driven activity and selection, camera/detail changes,
  pause/reset/render independence; preserve underlying neural calculations.
- [x] Benchmark repaired view at 1440x900: target 60 FPS, p95 <33 ms.
  Report rendering and simulation throughput separately and actual limits honestly.
- [ ] Run focused/full tests, saved-view QA, obtain live review where available,
  audit protected hashes, update status/log/guide/decisions/queue/experiment summary.

## Performance rules

Batch static source segments instead of thousands of objects. Cache arrays and
display computations. Update activity once per new snapshot, text only on change.
Load progressively within a bounded frame budget. Coarse overview, full selected
morphology, no expensive glow effects before measured basic responsiveness.

## Session decisions

- Ruling: use an additive viewer implementation and preserve prior view modules
  for fair benchmarking. No Git repository exists, so hashes/source copies replace
  worktree/commit bookkeeping. Local review follows the user's single-agent limit.
- Ruling: keep interruption reconciliation outside the frozen experiment directory;
  neither its journal nor its checkpoints will be rewritten or resumed.


## Current measured outcome

Sequential 1440x900 measurements: old Network running p95=134.91 ms, previous
Anatomy=65.47 ms, repaired=14.64 ms. Repaired paused/orbit/select p95=5.10/6.35/
13.70 ms. The <33 ms navigation p95 target passes. Paced FPS=55.8-58.8; steady
60 is not verified. Simulation throughput=0.57 simulated seconds/wall second.
Fixed numerical steps remain unchanged; every UI frame reads a published snapshot.
No expensive glow effects were added. [Detailed evidence](../verification/flight3d-anatomical-repair.md).

Targeted 3D tests: 50 passed, 567 deselected, 2 existing upstream warnings.
Preservation inventory: 593/594 unchanged; the sole existing source change is the
3D entry-point import. All experiments and previous scientific sources intact.
Live user interaction review and full-suite results are separate remaining checks.


## October 2 controls follow-up

The accepted appearance is retained and the user's explicit interaction fixes
are implemented: continuous quaternion/pixel drag, 60 ms response without inertia,
0.03 mm zoom with adaptive near, uncoupled pan, camera-preserving cord visibility,
explicit Fit/Reset, passive inspector and simplified responsive toolbar/settings.
Fresh **61 targeted / 628 full-suite passed** and isolated p95 **3.91-9.05 ms**.
All prior scientific/cache/experiment files are unchanged. See
[commands and limitations](../verification/flight3d-controls.md).

- [x] Fresh targeted/full suite, isolated navigation timing, saved-view review,
  protected hashes and updated status/log/guide/decisions/queue.
- [ ] Short native user review, including resize/focus; preview is open and
  observations have been requested. No acceptance inferred from offscreen tests.
