# FB-07D plan: sensory observability diagnosis

## Question and controlled intervention

Measure how often distinct bilateral odor observations produce the same seven
filtered spike features on the locked train and validation fixed-controller
trajectories. Count exact feature collisions with different raw observations
and teacher turns, and report saturation and feature variety by case. Replay
each recorded observation sequence through fresh networks at encoder scales
0.025 and 0.05 so movement does not confound the sensory comparison. A small
constant-input probe checks current and spike discriminability separately.

The single intervention doubles the concentration saturation scale from 0.025
to 0.05, retaining the current ceiling of 2. The choice uses the observed
train/validation concentration ranges and is fixed before embodied results.
Train the same 7-to-2 readout with the same eight train starts, 200 samples per
case, sensory teacher, 80 Adam steps and learning rate 0.05. Keep the 100-neuron
weights and evaluation starts/horizon unchanged. Compare validation cases with
the recorded FB-07 learned result and FB-06 fixed baseline. Do not evaluate
final cases in this diagnosis; keep them sealed for a later selected learning
attempt. Changed encoder scale is hand-written scaffolding, not learned biology.

## Implementation and verification

1. Test scale validation and sensory response with literal paired observations.
2. Test collision metrics on hand-checked feature/observation fixtures and
   paired replay on real short locked trajectories.
3. Extend the existing training/evaluation path with an explicit scale and
   checkpoint contract. Test rejection of mismatched scale and the short
   validation-only diagnostic artifact.
4. Run one full diagnostic to `experiments/fb07d-observability/`, audit its
   JSON/CSV/checkpoint and case-level results, then run focused and full tests.
5. Record the measured outcome and next learning decision in status, log,
   experiment summary, decision record and build queue.

The controller continues to receive only two raw concentrations. Probe replay
does not move the fly, and coordinate-based scoring stays in the evaluator.

## Verification record

On 2026-09-21, 46 focused tests and the 205-case full suite passed, with one
existing Pygame deprecation warning; `compileall` passed. The artifact audit
matched four validation CSV/JSON rows, 81 loss rows, checkpoint hash, controlled
spike probe, topology flag and exact reload replay. Train aliased ticks fell
1,030 to 452, but validation aliased ticks rose 1,446 to 1,589. The validation
arrival result remained 0/4; no new final cases were run. The conclusion is to
test weak-signal coding before another readout training attempt. Full measures
and limits are in [experiments](../experiments.md).
