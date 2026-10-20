# FB-07J plan: bounded turn-gain intervention

## Question and fixed conditions

FB-07I found that all validation controllers retreated before first wall
contact despite a directionally consistent sensory teacher. Test whether a
single, fixed actuator change improves embodied navigation: multiply the
decoded clockwise turn command by **2.0**, clamp to `[-1,1]`, and leave
forward command untouched. This is hand-written motor scaling, not a learned
or biological parameter. Apply it to both the unchanged fixed controller and
the saved nine-feature readout. Do not alter encoder, graph, checkpoint,
teacher, world or app defaults; do not train.

Evaluate original gain 1.0 and candidate gain 2.0 on eight locked train starts
for the full 600-tick horizon. The original fixed and nine readout use their
existing motor rules; the same wrapper adds gain 2.0 after decoding. Evaluate
gain-2 fixed and nine controllers once on four locked validation starts, and
compare with recorded gain-1 FB-06/FB-07H validation rows. Do not tune the
gain after observing train results. The validation gate for opening final
starts requires gain-2 nine to exceed both gain-1 nine and gain-2 fixed in
arrivals, and to exceed both in mean progress. If the gate passes, evaluate
gain-2 nine and gain-2 fixed once on eight final starts; compare both with
recorded FB-06 gain-1 fixed. Otherwise leave final starts sealed.

An improvement caused by gain alone cannot establish a benefit from learning.
The matched gain-2 fixed controller is necessary for that comparison. Four
validation and eight final starts are too few for a reliable population
estimate even if a gate passes.

## Implementation and verification

1. Test scaling/clamping, reset delegation and preservation of forward
   command; test gain 1.0 reproduces the unchanged controller exactly.
2. Test short matched train/validation runs, case identity, checkpoint/hash
   checks, failed-gate final seal and the gate rule.
3. Record config, seed, software/device, checkpoint and topology hashes,
   per-case JSON/CSV, split summaries and explicit no-training/final flags.
4. Run the one bounded full experiment, audit matched rows, run focused/full
   tests and update experiment, decision, queue, status and log records.

## Verification record

On 2026-09-22, gain 2.0 passed the validation gate: the saved nine-feature
readout reached 1/4 validation starts versus 0/4 for its gain-1 version and
the gain-2 fixed controller. One final evaluation followed. Final arrival
was 1/8 for gain-2 nine, gain-2 fixed and original fixed. Nine's final mean
progress improved (-0.895 versus -2.312 for gain-2 fixed), but the primary
arrival objective tied; M6 remains open. No retraining occurred. The artifact
audit matched 8/4/8 per-case CSV/JSON rows, recorded gain-1 controls and the
single gated final run. Fifteen focused and 236 full-suite tests passed;
`compileall` passed. See [experiments](../experiments.md).
