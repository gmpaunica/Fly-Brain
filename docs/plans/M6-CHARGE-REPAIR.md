# M6 neural charge readout repair

Date: 2026-09-22. The user explicitly requested a renewed repair with Astra
and fixes for other broken project behavior after the bounded recovery failed.
The prior 500-update experiment remains closed and preserved. This document
declares one new candidate before its training run; it is not an FB-07 letter
or an authorization to resume the reward-learning proposal.

## Evidence and hypothesis

Astra's train-only investigation found no sign, LIF integration or autograd
bug. Endpoint voltage is a spike-phase measurement: with left odor 0.008 and
right odor 0.010 held fixed, the teacher turns right (+0.8889), but endpoint
right-minus-left voltage is negative for three successive ticks. Spikes reset
voltage, discarding overshoot. EMA counts and endpoint voltages together are
poor coordinates for a linear fit to instantaneous normalized odor contrast.

Hypothesis: replacing the two endpoint voltages with independently transformed
neural charge measurements will expose enough sensory information for the
same learned linear/tanh motor map to fit and navigate the training starts.

## One correction and its limits

Retain seven spike EMA features. Replace only the two sensory voltage features
with per-side log(Q / (2.5 - Q)), using explicit finite floors/clamps. Measure Q
from actual simulated state changes, accumulated leak loss and reset loss over
20 neural substeps. Divide by 20 times the LIF input fraction. Accumulate the
ledger in float64 using the actual rounded membrane states so absent input
cancels even when old voltage is decaying.

The decoder receives neural activity only. It receives no raw odor, pose,
food location, bearing, teacher command or precomputed bilateral contrast.
The learned weights must determine how the independent left/right values
affect turn. LIF dynamics, graph, teacher, motor gain and world stay fixed.
Keep old activity/checkpoint behavior available; the optional charge recorder
and schema-3 checkpoint have a distinct feature contract.

Charge balance at these sensory neurons algebraically reconstructs their
drive. The independent log transform compensates the rational encoder. This
is an engineered readout representation; success cannot establish a benefit
from the downstream synthetic topology or learning inside spiking neurons.
Zero-odor exploration and biological plausibility are outside this repair.

## Predeclared budget and gates

- Reuse the original eight fixed-controller trajectories, 200 ticks each,
  and exactly the same 1,600 sensory teacher targets.
- One zero-initialized nine-input/two-output readout. Reuse the previous
  train-only standardization and Adam learning rate 0.05; at most 500
  full-batch updates. No sweeps, restarts, extra data or second candidate.
- Stop immediately on nonfinite calculations. At the final update require
  turn MSE <= 0.016752 and <= 10% of the train-target-mean predictor's turn
  MSE. Record forward and turn separately, gradient/parameter evidence,
  fixed topology, feature/data hashes and save/reload equality.
- Evaluate eight training starts at unchanged gain 1 and 600 ticks. Require
  at least 7/8 arrivals. Compare the trained candidate, the same candidate
  with its initial zero weights, and the original fixed controller. The
  already recorded unchanged teacher's 8/8 train result is a reference.
- If either gate fails, close this candidate and ask the user before another
  approach. If both pass, save the candidate and prepare one fresh matched
  validation/final protocol for review. Do not run that protocol or promote
  the checkpoint to the live app automatically. Retire old final seeds 200-207.

## Independent project repair

Older experiment runners and checkpoint writers can overwrite existing
records. Add early refusal of occupied run destinations and exclusive
checkpoint creation, with regression tests using temporary directories.
This changes persistence safeguards, not historical results or model behavior.

## Verification

Test the numerical charge identity, input changes, zero-input residual state,
reset, feature order/shape, missing activity rejection, old dynamics equality,
gradients, separate checkpoint semantics and reload. Run targeted tests and
the full suite, then update current project records. Report headless smoke
checks separately from any actual visual review.

## Recorded outcome

The one 500-update run on 2026-09-22 passed both training gates. Turn MSE was
**0.001761889**, a **98.95%** reduction from constant 0.167515755; forward
MSE was zero at recorded float32 precision. All 20 readout scalars had
nonzero first gradients and changed. The graph remained fixed, saved/reloaded
outputs matched exactly and the first training episode replayed exactly.

The trained candidate reached **8/8** training starts, averaging 2.4975
seconds; its frozen-untrained version and the original fixed controller each
reached **0/8**. See [results](../../experiments/m6-charge-repair/results.json),
[curve](../../experiments/m6-charge-repair/curve.csv), and
[all 24 episode rows](../../experiments/m6-charge-repair/train_episodes.csv).
This is a training calibration, not a held-out learning claim.

The [fresh protocol](../../experiments/m6-charge-repair/heldout_protocol.json)
locks 16 validation starts (seeds 31000-31015), 32 final starts (41000-41031),
the two candidate checkpoints, three matched controllers and failure gates.
It is prepared for review only. Its maximum budget is 144 episodes / 86,400
world ticks with zero optimizer updates, and final opens only after validation.
No held-out case or app promotion was executed in this repair.

Verification: 261 full-suite tests passed, 12 charge feature/training tests
passed, 30 preservation/evaluation/legacy-training tests passed and Astra's
72 focused neural tests passed. Manual and neural headless app checks each
completed 120 ticks. No live visual review. All 34 historical JSON/CSV/PT files
have unchanged aggregate SHA-256
`55b446e166252e656a499840f0c75831c741c6b0f0fbd9a5277d081d0338bcd9`.

## Approved fresh evaluation (2026-09-23)

The user approved execution of the prepared protocol. Its original JSON and
SHA-256 remain unchanged; the preparation-state label is historical. Execution
is recorded separately under `experiments/m6-charge-evaluation/` and by a
receipt keyed to the approved protocol hash in the checkpoint directory.
Copying or renaming the protocol cannot reset that receipt.

`flybrain.training.charge_evaluation` verifies all frozen source/checkpoint
hashes and the software/device environment before evaluation. It loads both
candidate checkpoints with gradients disabled and records each episode before
proceeding. No optimizer exists in this evaluation path. Validation requires
at least 13/16 trained arrivals and more than each control; final requires
at least 26/32, at least eight more than each control and paired one-sided
exact binomial p <= .025 for each comparison. A failed validation gate leaves
final unevaluated. Any failure ends this run without tuning or replacing cases.

Runner tests use separate synthetic temporary cases, never the approved
cases. Targeted evaluation tests passed 11/11; the full suite passed 266/266.
Astra reviewed the gates and found the filename-based repeat loophole. A
regression reproduced it; the receipt now uses the protocol content hash.

### Evaluation outcome

The approved run passed both gates: trained 16/16 validation and 32/32 final
arrivals, versus 1/16 and 1/32 for each matched untrained and original fixed
control. Each final comparison had 31 trained-only successes and zero
control-only successes (one-sided exact p = 2^-31 = 4.656612873e-10).
The 144 episodes used 62,463 world ticks and zero optimizer updates. Frozen
source/checkpoints and loaded weights were unchanged; journal, JSON and CSV
case rows agree. See [results](../../experiments/m6-charge-evaluation/results.json).

This meets the M6 navigation-improvement gate within the frozen arena, odor
source, 60-180-unit annular start distribution, graph seed 7 and recorded
software/device. It establishes a learned motor-readout benefit, with the
engineered sensory reconstruction limitation above. It does not establish
zero-odor exploration, behavior after arrival, other environments or a
downstream topology benefit. No app promotion occurred. These 48 cases are
now historical and cannot be reused for another unseen-case claim.
