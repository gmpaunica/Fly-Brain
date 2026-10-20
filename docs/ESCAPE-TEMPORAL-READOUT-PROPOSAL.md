# One fixed spike-only temporal readout - approval requested

The unchanged-model diagnostic supports timing redistribution and rolling
expiration in all five deficit-only failures. It also confirms three seeded
ties and the separate mixed unequal failure. No new decoder has been evaluated.
This proposal changes only the engineered side readout; the sourced circuit,
assumed neuron rules and frozen gains remain intact.

## Exact algorithm and fixed parameters

Read only the existing paired actual/reference counts for the six endpoints,
ordered **[i1L,b3L,b1L,i1R,b3R,b1R]**. Every 20 ms, add the interval's signed
increment to an int64 six-vector:

`C <- C + actual_six_counts - reference_six_counts`.

Use the original unit score mapping on this cumulative vector:

`L = C0 + C1 + C5`, `R = C3 + C4 + C2`, `D = R - L`.

Candidate right if D >=1, left if D <=-1. It is eligible only if its winning
score is **>=1**. Exact ties and solely deficit-based preferences are ineligible.
There is no RNG tie choice, clamping, fitted weight or alternative pathway.

Latch the first strong-threat interval as tick `t0`. Starting with that tick,
require the **same eligible side on two consecutive 20 ms observations**.
First eligible observation starts a streak of one; the next eligible observation
with the same side commits it. A tie, nonpositive winner or sign change clears
the old streak (a new eligible sign starts a streak of one). Pre-strong samples
accumulate C but never count toward the streak or produce movement. Commit only
at ticks <=`t0+25` (**500 ms**); repeated strong samples do not extend this first
deadline. Missing evidence by the deadline yields no side/escape response for
that threat event, rather than a random fallback.

The chosen side stays fixed through the existing 90-degree turn, flee, weak
braking and recovery state machine. Do not change its gains, speed/altitude,
home guard, weak thresholds or motion/IMU scaffolding in this candidate.

Reset C, streak, first-strong/deadline and selected-side state on every original
episode/reset/spawn/replay barrier, alongside the existing paired-brain reset.
During an active escape do not collect another side decision. Once the existing
policy reports recovered/idle under weak-continuous threat, clear readout state
and keep C zero while weak-continuous remains true. Begin accumulating again
when that weak condition clears; the next strong event gets a new fixed deadline.
This rearming clears stale accumulated evidence without resetting neurons in
mid-flight. Episode barriers still clear every neural state exactly as before.

Fixed values: **20 ms sampling, unit existing weights, cumulative int64 counts,
winning score >=1, absolute margin >=1, two consecutive eligible samples,
500 ms first-strong deadline**. Two samples are the smallest temporal agreement
test beyond a single instantaneous decision; one count is the native integer
resolution. These are engineering choices, not physiological thresholds. No
timing/threshold value was selected by applying a candidate to validation
performance; there was no search or offline replacement decoding.

Tiny illustrative calculation, not a measured new response: C begins at zero.
An interval increment [0,1,0,0,0,0] gives L=1,R=0,D=-1. At a strong event this
starts a left streak. A following zero increment leaves C unchanged; another
eligible left observation commits left 20 ms later. PyTorch generates spikes;
this cumulative arithmetic and gate are hand-written scaffolding, not training.

## Observed failures targeted and expected latency

The five sole b3 deficits arise when an earlier actual spike leaves the rolling
window. C has no expiration, and the positive-winner condition forbids selecting
right solely because left is negative while right is zero. This addresses those
mechanisms structurally; it does not establish improved case outcomes.

For seeds 1100010, 1100017 and 1100032, remove seeded symmetry breaking: a tie
waits for unequal, positive neural evidence with temporal agreement. Persistent
ties time out and count as failed directional responses. The mixed failure
1100002 is tested under the same integration/agreement rule, but its successful
repair is unproven. The same applies to every historical success: a change can
also introduce regressions and the full directional gate must catch them.

Minimum decision delay is **20 ms after the first eligible strong-gated sample**.
Transient signs can extend latency; maximum allowed response latency is 500 ms
from first strong onset. Retain the original **1.5 s presentation endpoint**:
no response by that endpoint is a failure even where part of the 500 ms deadline
would fall later. Do not extend exposure after seeing results. Cumulative memory
can preserve a mistaken early bias; the positive/streak gate cannot guarantee
correctness and must pass the fixed assay.

## Preservation, replay and activation

Implement a new additive recovery package. Preserve all production modules,
historical launchers, frozen experiments and failed/partial diagnostic archives.
Keep **589 cells, 4,139 anatomical edges, gains (4,2)**, original field assignment,
RGB/motion encoding, precision, delay/leak/threshold/reset and all source checks.
Only actual/reference neural count signals and existing sensory-derived
strong/weak flags reach side selection. No target coordinates, azimuth/elevation,
bearing, distance, world object or precomputed steering answer enters it.

Bind source, configuration, circuit, prior calibrated gain receipt, algorithm
parameters and outputs in a fresh parent-scoped consumed experiment receipt.
Use tested append-only reservation/heartbeat journals and detached supervision;
no resumed child or scientific retry after failure/interruption. Reset/replay
must reproduce cumulative/streak/deadline state and the exact endpoint, recorded
environmental inputs, trajectories and responses. Add a versioned recovery action
log/launcher; original replay and launchers remain byte-preserved.

Escape remains disabled during implementation/testing. Activation requires all
behavioral gates, audit/replay/preservation checks **and separate activation
authorization**; then enable only the new receipt-bound recovery launcher.
Historical defaults/launchers are not promoted implicitly.

## Tests and one scientific budget

Before scientific consumption, behavior tests cover cumulative counts that do
not expire, positive-winner rejection of a sole negative deficit, exact ties,
two-sample agreement/sign reversal, pre-strong silence, first-strong deadline,
timeout/rearming, reset/replay and no coordinates in the side-selection boundary.
Use actual neuron behavior tests for leak/threshold/subtractive reset/directed
delay, same-gain model parity and unchanged baseline; tests must not replace
numerical checks with mocks. Verify finite state and downstream rolling-rate
guard. Run targeted tests and the small full suite after implementation.

Historical diagnostic tests currently write a consumed engineering ledger.
Use a test-only isolated-ledger adapter for those tests so the full suite never
changes archived proof bytes; do not skip the tests or disable source checks.
Log new engineering verification separately. The closed diagnosis remains at
1,747 steps; its 2,000-step cap is not permission to mutate its receipt.

Predeclare one hypothesis: this fixed cumulative/positive/agreement readout
produces sufficiently accurate timely direction choices and supports the
retained turn/flee/brake/recovery policy, without changing neural dynamics.
One source-locked candidate, **no gain/threshold/window/streak search, optimizer,
training, ablation or automatic retry**.

Directional gate: run exactly the original **96 historical validation cases**
once, 1.5 s each including the 0.5 s baseline, in their original order. Require
**>=90/96 correct and >=5/6 at every azimuth, within 500 ms**. Compare unchanged
RGB/count/diagnostic records with historical originals; new readout/side/timing
are the changed outputs. Historical cases remain historical, never unseen.
If this gate fails, close the experiment: no extra cases or closed-loop run.

Only after directional passage, run the prescribed **48 six-second closed-loop
cases**: eight directions x resting/hovering/cruising x intact/visual-silent.
Require **>=22/24 intact successes, >=7/8 per initial state, zero silent triggers**.
Success includes correctly signed actual turn, takeoff where applicable,
collision-free flight, separation, weak-threat braking and stable recovery,
verified from recorded trajectories under the existing protocol. A side score
alone is insufficient.

Scientific maximum: **144 cases /432 simulated seconds /864,000 individual
kernel steps** (96 x1.5 s plus 48 x6 s, two brains). If directional validation
fails, stop at **96 /144 s /288,000 steps**. Retained calibration (4,2) is reused;
no new gain calibration. Stop immediately on source/receipt mismatch, missing
measurement, nonfinite state, downstream rate >240 Hz, I/O failure or budget
breach. Otherwise stop at the first completed failed behavioral gate, or after
all required cases. Seal success, failure or interruption; no retry.

Independently audit saved records, source preservation and replay. Review saved
visuals separately from live interaction; mark unavailable live review pending.
Update status/log/decisions/experiment documentation/queue with the measured
result, keeping one next task. Passing tests or directional choice alone cannot
establish finalized physical escape or learned behavior.

**Approval requested:** implement this exact fixed readout and execute only its
single gated experiment after preflight. No parameter search, training or
activation is included in that approval.
