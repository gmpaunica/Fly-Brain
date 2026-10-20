# FB-07G plan: sensory voltage feature probe

## Question and fixed conditions

On the locked FB-07E fixed-controller train and validation observations, do
two post-interval sensory membrane-voltage summaries recover information lost
when 20 ms spikes become counts? Replay the rational candidate encoder only.
Keep the 100-neuron graph, seven-group EMA, world trajectories, teacher turns,
20-substep timing and final starts fixed. Reset network and EMA at each case.

Take the mean post-interval voltage over the ten left sensory neurons and the
ten right sensory neurons. Append these two float32 values after the existing
seven EMA spike features, yielding a nine-value diagnostic vector. Report
exact aliased ticks, unique vectors and bucket-optimal teacher-turn MSE floors
for seven versus nine values, plus the fraction of FB-07F conflicting EMA tick
pairs separated by voltage. These are observed-tick information measures, not
a prediction of held-out navigation or a biological feature claim.

The information gate requires strictly fewer aliased ticks, a strictly lower
turn MSE floor and at least one resolved conflicting pair on **each** split.
No readout training, changed controller, or final-case evaluation occurs in
this section, even if the gate passes. A later learning experiment needs its
own bounded task and matched held-out evaluation.

## Implementation and verification

1. Test two voltage values against the post-interval snapshot, including
   episode reset and detached tensor shape.
2. Test collision counts and turn floor on a hand-checked pair with identical
   spike EMA and distinct voltages.
3. Test short locked replay, rejection of final cases, deterministic reset,
   unchanged topology and 12-case JSON/CSV export.
4. Run the full 200/600-tick probe, audit old seven-feature metrics against
   FB-07F, inspect all 12 case rows, run focused and full tests, and record
   the measured outcome without claiming navigation learning.

## Verification record

The 2026-09-22 full replay passed the split-wise information gate. Train
aliases fell from 627/1,600 to 18/1,600 and the turn floor from 0.006354624
to 0.000549582. Validation aliases fell from 992/2,400 to 46/2,400 and
the turn floor from 0.034047927 to 0.002997121. Voltage separated 3,389 of
3,402 train and 4,259 of 4,392 validation conflicting EMA pairs. Seven-EMA
metrics matched FB-07F exactly. This is an information result; no readout or
embodied learning result was produced. See [experiments](../experiments.md).
