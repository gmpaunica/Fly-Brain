# Long-term demo vision and learning explanation

User clarification: 2026-09-21. This records future requirements and live-demo
feedback; it does not begin 3D implementation, training, or full-brain import.

## User's target

- Show a fly moving in a 3D world, approaching attractive stimuli and moving
  away from aversive stimuli.
- Include exploratory behavior when no relevant stimulus is detected; constant
  forward drive alone does not satisfy that product goal.
- Show the simulated neural network simultaneously in a 360-degree rotatable
  3D view, with active neurons highlighted in a different colour and fading
  driven by the same simulation clock as the body.
- Use a dark background, fine connections, purple/pink structural colouring,
  and orange/yellow highlights inspired by the supplied front/side/top image.
- Retain the long-term ambition of a larger, biologically sourced network.
  Scaling and behavioral fidelity remain separate feasibility questions.

Reference: [user-supplied image](fly-brain-3d-visual-reference.png), copied
unchanged from `C:\Users\Lenovo\Desktop\fy brain reference.png`. This is a
visual direction only. Its anatomical accuracy, neuron identities, connectivity,
provenance, and reuse rights have not been established. Do not import its drawn
edges or positions as biological data or publish it without resolving rights.

## Feedback on the current prototype

The user observed circling rather than reaching the odor source. This is useful
behavioral feedback, not acceptance of navigation, pulse/fade readability, or
reset behavior. FB-05 displays a working sensory-neural-motor loop; successful
navigation has not been established.

Code inspection and a short sensory-only diagnostic confirm a fixed tonic
forward input, a fixed right-minus-left decoder, and no search or arrival rule.
After 20 decisions (0.4 simulated seconds), zero odor gives forward about 0.714
and turn 0; constant left-only input gives turn about -1, and right-only about
+1. These probes demonstrate drive and steering responses. They do not replay
the reported trajectory or establish its specific cause, and they are not
performance-evaluation results.

## Timing and scope

- FB-06 remains next: define success, horizon, disjoint evaluation cases, and
  fixed-controller baselines before tuning. Document the no-odor behavior.
- FB-07 is the first measured learning attempt, using the small synthetic graph.
  Navigation need not wait for a whole-brain dataset.
- FB-09 remains a sourced, relevant connectome subset with audited interfaces.
  It is not a promise to connect a complete brain and obtain realistic behavior.
- The 3D neural viewer can be built around a small graph independently of graph
  size. Anatomical layout requires sourced positions or morphology plus a
  documented mapping from biological IDs to simulated neurons.
- 3D body presentation and true 3D locomotion are separate changes. The eventual
  locomotion stage needs vertical state, sensing, actuator, and boundary
  contracts; a 3D camera around a planar trajectory does not satisfy that stage.
- Select a renderer, hardware budget, and detailed milestones when this future
  work becomes active. Continue to keep rendering read-only.

## How machine learning fits

Current execution changes voltages and spikes, but no trainable parameters are
updated. PyTorch currently performs tensor arithmetic for rules written by us.

The first candidate learning experiment keeps the spiking graph fixed and
trains a small motor readout: a mapping from recent neural activity to movement.
For illustration, two normalized activity features `[0.8, 0.2]` with turn
weights `[-0.5, 0.5]` give `turn = -0.5*0.8 + 0.5*0.2 = -0.3` before optional
smoothing/clamping. Hand-selecting these weights is programming. Updating them
from a defined training signal is learning. These numbers are an illustration,
not an implemented controller or measured training result.

With all 100 neurons, a candidate feature vector has shape `(100,)`; a readout
matrix of shape `(2, 100)` has output rows and input-neuron columns, producing
two movement outputs before their bounds are applied. Training must specify an
objective and a valid way to update parameters; rewards and hard thresholding
do not automatically give usable gradients. A supervised example is imitation;
reward-based learning needs its own documented estimator. Neither method has
been selected yet.

Measure the fixed controller, train only selected parameters, verify actual
updates, then compare on unseen starts and save/reload the result. The runtime
controller still receives simulated sensory signals only. Evaluation may know
the source position to score success but must not supply steering answers.

Biological connectivity supplies structural constraints. Neuron dynamics,
sensory encoding, and coupling outputs to the body still need explicit models.
For a primary example, Shiu et al. combine connectivity and transmitter
predictions with assumed neuron dynamics and targeted sensory stimulation:
[Nature, 2024](https://www.nature.com/articles/s41586-024-07763-9).
Their model also distinguishes complete feeding pathways from pathways with
incomplete descending neurons; a brain dataset does not guarantee complete
locomotor machinery. Our simulated activation must be labeled as model output,
not a recording from a living fly.
