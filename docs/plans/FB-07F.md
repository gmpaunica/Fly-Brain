# FB-07F plan: locate residual feature collisions

## Diagnostic question

On the same FB-07E fixed-controller observations, where do distinct readings
with different sensory-teacher turns become indistinguishable to the readout?
Replay original, wider and rational encoders on eight train cases for 200 ticks
and four validation cases for 600 ticks. Reset network and feature EMA per case.
Keep topology, world, observations, timing and final starts fixed.

Record exact representations at each 20 ms decision: two encoded sensory
currents, 20 sensory-neuron spike counts, all 100 neuron spike counts, seven
unfiltered group means, and seven EMA features. The 100-count stage distinguishes
loss from group averaging from loss already present in sensory counts. For each
stage, report unique vectors, ticks shared by distinct raw readings, and the
bucket-optimal teacher-turn MSE floor. For pairs with different raw readings
and target turns but identical final EMA features, label the latest earlier
representation that still differs: group means, all counts, sensory counts,
current, or no earlier difference. This is a snapshot comparison, not a proof
of causation: voltage and prior spikes carry unrecorded temporal state.

Focus the case table on train-2 and validation-101, while preserving all cases.
Choose the stage with the strongest measured evidence for a single later
intervention. Do not change the controller, train, run embodied evaluations,
or open final cases in FB-07F.

## Implementation and verification

1. Test stage-collision and final-pair attribution on hand-checked vectors.
2. Test short locked-case replay, exact reproduction of FB-07E final-feature
   collision metrics, topology equality and rejection of final cases.
3. Record JSON plus per-case CSV under `experiments/fb07f-collision-stages/`,
   with config, seed, software/device, stage metrics and attribution counts.
4. Audit the artifact, run focused and full suites, and update experiment,
   decision, queue, status and log records with one next task.

The neural controller continues to receive only two raw concentrations.
This task observes forward PyTorch tensors; it invokes no autograd or optimizer.

## Verification record

On 2026-09-22, 30 focused tests and the 216-case full suite passed;
`compileall` passed, with one existing upstream Pygame deprecation warning.
The JSON/CSV audit matched 36 case/encoder rows, all FB-07E EMA metrics,
pair-category totals, topology equality and the absence of training/final
evaluation. In the rational candidate, all 3,402 train and 4,392 validation
final conflicting tick pairs had different encoded currents but equal all-100
spike counts. See [experiments](../experiments.md) for case-level results and
the next voltage-feature probe decision.
