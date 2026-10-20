# FB-09 single wide-range encoder repair

2026-09-25. The user approved the diagnostic and continued repair to full FB-10
readiness. The 4,500-step diagnostic is complete; earlier failed runs stay closed.

## Evidence and hypothesis

Astra's read-only analysis recommends changing only the engineered sensory
sensitivity range. With existing scales .005-.05 at odor .25, all ORNs receive
currents 2.083-2.451 and fire with periods only 11-14 ms. Many share the same
timing. The diagnostic proved equal/right-strong PN spike trains identical,
with distinct input dissipated through leak/reset. Incoming directional charge
change itself is only about 3.01 Hz-equivalent, so preserving overshoot alone
is unlikely to meet the 6.64 Hz directional gate and risks deep-chain amplification.

Hypothesis: widening ORN scales to .005-.375 retains a range of sensitivity and
spike periods throughout the declared band and produces transient directional
endpoint activity. The upper bound is derived, not swept: `2.5c/(c+s)=1`
when `s=1.5c`, hence .375 puts threshold at the tested high center .25.

## Frozen candidate and budget

- Exactly one candidate: row budget 3; log-spaced per-side ORN scales .005-.375
  assigned by sorted source root ID. Current gain remains 2.5.
- Raw source, all 145 effective edges including all cross-side edges, hard-reset
  LIF equations, time constants, endpoint features and decoder remain unchanged.
- Exactly the original 16 pulse cases and timing: 24,000 neural substeps.
- All original finite/zero-input/propagation/contrast/rate/washout gates apply.
  In particular, keep low-.025 unilateral propagation; fewer active ORNs are a
  known risk, not grounds to weaken the gate afterward.
- Only if every dynamics gate passes, run the three declared causal cuts:
  another 4,500 steps. Maximum 28,500, zero optimizer updates.
- Write protocol and code/artifact hashes before measurement. Preserve every
  previous file, retain failed outputs, and refuse an existing output directory.
- Failure closes this candidate; no automatic gain/scale sweep or default.

## Implementation and completion

1. Test the new explicit encoding variant without changing historical defaults.
2. Extend the existing runner with an explicit single-candidate mode, reusing
   original pulse/contrast assessments; record its distinct protocol identity.
3. Run once and inspect evidence. If successful, bundle the actual verified
   evidence and run the already specified teacher/feature readiness assay.
4. Finish real BANC closed-loop, motor-gate/reset/render tests, app smoke and
   visual review; complete source/experiment integrity, full suite and review.
5. Update status/log/queue/decisions and leave FB-10 as the next training task.
   Do not start its optimizer run as part of FB-09 recovery.

No Git repository exists, so no worktree or commits. The source graph and old
experiment records are the preserved baseline. Implementation is test-first.

## Ledger

- Diagnostic prerequisite: exact replay, no changed parameters, 4,500 steps.
- Astra recommendation: one wider sensory range; no new learning method or
  biological-dynamics claim. All changes are explicit engineering scaffolding.
- Implementation ruling: reuse the existing runner behind `--wide-encoder`
  instead of duplicating its evaluator; historical default parameters remain
  unchanged. Every run records the exact full candidate configurations.

## Closed result

The candidate completed exactly24,000 steps and zero updates. All pulse gates
passed, including low-.025 unilateral propagation. Contrast passed at .025
but failed at .1 (shifts2.667/2.667Hz, required3.787/3.84) and .25
(rightshift1.333Hz, required5.547; leftshift10.667Hz passed). No causal controls
or app receipt followed. See
[recorded result](../../experiments/fb09-wide-encoder-repair/results.json).
No additional scale values were tried. A debugging reset isolates PN temporal
integration in [one separate fixed test](FB-09-PN-FILTER.md).
