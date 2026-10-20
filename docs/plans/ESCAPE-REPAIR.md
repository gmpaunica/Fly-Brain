# Escape completion repair - 2026-10-08

The user explicitly requested diagnosis and complete repair, including ordinary
implementation and verification choices. One agent; no commit or training.
This supersedes the prior proposal-only next task, but preserves every consumed
experiment, source binding and original failed result. Additive `escape_repair`
code retains the accepted 94/96 readout and original neuron/gain/sensor encoding.

Observed causes: positive-only frame differences rectify silhouette sampling
noise into apparent threat growth; the weak timer consequently never finishes.
After recovery, idle cruise commands outward translation while its slowly
changing heading turns home, ignoring the escape policy's radius guard.

Repair: retain strong/onset detection exactly; require 300 ms continuously small
apparent size and measure net smoothed-size growth over that window for weak
clearance. Limit idle outward radial velocity from the existing integrated IMU
displacement, preserving tangential movement and altitude control. No world or
predator coordinates enter control. This is engineered control, not learning.

Before any scientific execution: reproduce both bugs, exercise physical/sensory
regressions, integration/replay/reset, source binding and failure handling; run
targeted and full tests. Freeze the one candidate before evaluating it.

Hypothesis: these two causal repairs complete turn/takeoff/flee/brake/recovery
without collision in every declared intact test while silent controls do not
trigger. One evaluation: 48 historical benchmark cases plus 48 cases at the
same angles/states with seeds offset by 200000. Six seconds each: 576 simulated
seconds, 1,152,000 individual paired neural steps, 28,800 control intervals.
The second set tests seed robustness, not unseen geometry. Required: 24/24
intact successes in each set, zero silent escape triggers, original physical
criteria unchanged. Stop on missing/nonfinite measurement, >240 Hz downstream,
source/hash mismatch, I/O failure or budget exhaustion; seal any terminal run.
Do not automatically retry a failed scientific experiment or change its gates.

Verification replay: four predetermined complete cases (historical 0-degree
rest intact/silent, historical 0-degree cruise intact, new-seed 180-degree
cruise intact), 48,000 additional individual steps, no new scientific cases.
Engineering tests use short neural episodes (under 10,000 added paired steps
per suite); motor-only tests use real physics/sensory observations with explicit
synthetic wing impulses and no neurons. Keep those separate from measured
full-neural success. Long motor-only checks cover recovery past the 6 s endpoint.

Deliver a separately receipt-gated repair launcher after passing audit/replay,
and retain the old disabled launchers and failed receipts. Review saved movement
plots and an offscreen application frame separately from any live UI check.
