# FB-07 plan: first trainable readout experiment

## Goal and method

Keep the seeded 100-neuron spiking graph, sensory encoder and world fixed. Train
only a 7-input, 2-output PyTorch linear motor readout with bias (16 scalar
parameters). Its inputs are EMA-filtered group mean spike counts from the seven
existing neuron groups. Sigmoid bounds forward output to [0,1], and tanh bounds
clockwise turn to [-1,1]. Each episode resets the EMA and network state.

Use a deliberately simple **supervised sensory teacher** as the first gradient
path. For each recorded train tick, its target forward command is 0.7 and its
turn command is `clip(8*(right-left)/max(right+left,1e-9),-1,1)`. This target
uses only the two raw antenna concentrations; it never enters the controller.
The training objective is mean squared command error. This is imitation of a
hand-written rule, not reward learning or biological synaptic learning. The
spiking threshold and graph receive no autograd gradient. Train samples come
from fixed-controller trajectories on the eight locked train starts, capped at
200 ticks each. The optimizer is full-batch Adam on the readout only.

Run one predeclared configuration and checkpoint. Record the full loss curve,
gradient and parameter-change evidence, topology equality, and metadata.
Reload the checkpoint into a fresh readout before validation/final evaluation.
Use the same locked 600-tick evaluator and compare each case with the saved
FB-06 fixed baseline. Use validation only for diagnosis; read final once after
the trainer and configuration are fixed. No best-of-many selection.

The input probe on 2026-09-21 found that all 12 train/validation initial poses
give the same seven group counts on the first 20 ms tick. Concentrations differ,
but the sensory current saturates at concentration 0.025. The experiment may
therefore fail to improve navigation. Log that result rather than silently
changing the encoder or topology.

## Implementation and verification

1. Add behavior tests for feature filtering/reset, command bounds, actual finite
   nonzero gradients and optimizer changes, and unchanged graph weights.
2. Add a trainer with train-only sample collection, fixed step budget,
   checkpoint save/load, curve export, and renderer-free evaluation. Test
   checkpoint replay and separation of train/final cases.
3. Run the bounded training once and write config, curve, checkpoint, and
   validation/final episode JSON/CSV under `experiments/fb07-readout/`.
4. Review measured results, run focused and full tests, and update experiment,
   decision, queue, status and log records. Mark M6 complete only if the held-out
   behavior improves under the locked protocol.

No source coordinates, heading, target bearing or distance are passed to the
controller. Coordinate-based success scoring remains in the evaluator.

## Verification record

On 2026-09-21, 31 focused tests and the 199-case full suite passed. The
bounded run produced 1,600 train samples, 81 recorded loss points, a saved
checkpoint and 12 held-out episode rows. All 16 gradients and all 16 net
parameter changes were nonzero, topology hash stayed fixed, and the checkpoint
replayed the first validation episode exactly. Validation arrival tied at 0/4;
final arrival fell to 0/8 from the fixed 1/8. FB-07 is a verified negative
navigation experiment, so M6 remains open. See [results](../experiments.md).
