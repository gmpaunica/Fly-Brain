# FB-07H plan: matched voltage-readout learning test

## Goal and fixed protocol

Test whether FB-07G's two voltage features produce better embodied navigation,
not only better exact feature discrimination. Keep the original fixed-controller
trajectories, rational encoder `2.5*c/(c+0.005)`, seeded 100-neuron graph,
0.02-second world tick, locked poses and 600-tick evaluation horizon fixed.

Replay the eight train cases for 200 ticks each. Train **two** zero-initialized
linear readouts on the same 1,600 observations and the existing sensory-only
teacher: seven spike EMA features versus those seven plus two post-interval
sensory voltage means. Both use one 80-step full-batch Adam run at LR 0.05 and
MSE command loss. Only readout parameters receive gradients (16 versus 20
scalars). This is a feature comparison, not a capacity-matched ablation or a
claim about biological wiring. No hyperparameter sweep or seed selection.

Checkpoint both models with their exact feature, encoder and topology
contracts. Reload before evaluating the four validation cases. Include the
unchanged FB-06 fixed-controller baseline on the same starts. Proceed to the
eight final cases **once** only if the nine-feature model has more validation
arrivals than both the seven-feature model and fixed controller, and higher
mean progress than fixed. Evaluate both learned models and the recorded fixed
baseline on final if this gate passes. If it fails, stop and record all
validation failures. M6 improvement requires higher final arrival than the
fixed controller; a small 4/8-case split limits inference even then.

The controller receives only two raw concentrations. The teacher sees only
those concentrations during training. World/source coordinates remain in
scoring and fixed-trajectory generation, never in the learned decoder.

## Implementation and verification

1. Test nine-value online feature shape/order, post-interval state use, reset
   and read-only extraction; preserve the seven-value default contract.
2. Test paired train-only collection and numerical autograd updates for the
   nine-input readout. Reject validation/final cases as training samples.
3. Test checkpoint feature/topology validation and exact embodied replay after
   reload. Test that a failed validation gate leaves final starts unevaluated.
4. Run one full bounded experiment, audit curve/checkpoint/CSV/JSON consistency,
   run focused and full tests, then update experiment, decision, queue, status
   and log records. Do not change the live app's default controller.

## Verification record

On 2026-09-22, the matched 1,600-sample run produced two 81-point curves,
checkpoints, and four validation rows. Both readouts had nonzero gradients
and changes to all intended scalars (16/16 and 20/20), fixed topology and
exact checkpoint replay. The nine-feature training loss fell from 0.104135
to 0.079091. Validation arrival was 0/4 for fixed, seven and nine. Nine's
mean progress was -1.770 versus fixed -1.688, so the gate failed; no new
final cases were evaluated. The artifact audit matched curves, checkpoint
hashes and fixed rows. Twenty focused tests and the 227-case full suite
passed; `compileall` passed. See [experiments](../experiments.md).
