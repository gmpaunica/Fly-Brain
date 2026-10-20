# FB-07I plan: embodied navigation failure diagnosis

## Question and fixed conditions

FB-07H's nine-feature readout learned its sensory-only teacher numerically but
had no validation arrival. Replay the saved seven- and nine-feature checkpoints
and the unchanged fixed controller on the eight locked train starts (200 ticks)
and four validation starts (600 ticks). Reset controller/world state per case.
Do not train, tune, change the controller, or evaluate final starts.

At each tick, record the **pre-movement** bilateral concentrations, sensory
teacher turn, actual motor command, source-bearing error from pre-movement
body pose, change in center distance, and wall contact. Source position and
bearing are read only by this scoring diagnostic; they never enter a controller
or teacher. For sign comparisons, use ticks with bearing magnitude above 0.1
radians and turn magnitude above 0.05, and report denominators. Bearing sign
is a geometric diagnostic, not an oracle action target or proof of an optimal
motor policy. Report wall contacts and progress before/after the first contact,
plus per-case and split summaries; preserve timeouts and successes.

## Implementation and verification

1. Test clockwise bearing-error convention and signed distance change on a
   small hand-checked pose/tick example.
2. Test that short traces reproduce the evaluator's end pose/tick count,
   reset per case, use only locked train/validation starts and never send
   coordinates through the controller call boundary.
3. Record per-tick CSV, per-case/split JSON, config, seeds, checkpoint hashes,
   software/device and explicit no-training/no-final flags.
4. Audit trace rows against FB-07H validation outcomes, run focused/full tests,
   then document one evidence-based next intervention. Do not use final starts
   to choose it.

## Verification record

On 2026-09-22, the read-only replay recorded 11,827 tick rows across 36
case/controller runs (4,627 train; 7,200 validation). All validation end
states matched FB-07H exactly. Every one of the 12 validation runs moved
farther from the source before first wall contact. On nine-feature validation
trajectories, the sensory teacher's sign matched source bearing on 2,091/2,091
eligible ticks, while the learned command matched on 1,564/1,915. The learned
model contacted a wall on 1,112/2,400 validation ticks and arrived on 0/4
cases. These are descriptive associations, not causal attributions. Five new
focused tests, 18 combined focused tests and the 232-case full suite passed;
`compileall` passed. No training or final evaluation occurred. See
[experiments](../experiments.md).
