# FlyBrain escape recovery and finalization handoff

## Goal and execution rules

Work in `C:\Users\Lenovo\Documents\Gabriel\Projects\Fly-Brain`, using **one agent** and **gpt-5.6-sol, High selected in the client**.

The goal is reliable threat-triggered turn-and-flee, followed by braking and recovery. Start by verifying the recorded spike-timing explanation. **Complete the diagnostic stage below, then present one concrete repair proposal for approval.** Do not invent a repair before examining the measurements.

Read `AGENTS.md`, `PROJECT_STATUS.md`, the latest relevant log entry, and `docs/TURNFLEE-WRONG-SIDE-REVIEW.md`. Use `superpowers:executing-plans` for implementation.

Current state:

- Frozen gains `(4,2)` passed calibration but failed validation: **87/96**.
- Nine wrong-side responses: three seeded ties and six unequal scores.
- Five unequal failures contain only a −1 left b3 rolling increment.
- All nine later favor left in the saved timelines.
- The experiment is closed; escape remains disabled.
- Later correct-looking activity has not established a successful replacement decoder.

Preserve all existing production code, launchers, defaults, frozen experiments and source archives. Use additive modules and fresh output directories. No subagents, commits, training, gain search or automatic retries.

## Stage 1 — Build passive instrumentation

Create an isolated `flybrain.escape_diagnostic` package with a recorder, single-use runner and saved-record audit. Do not change the frozen `turnflee` implementation or bypass its source verification.

Keep exactly:

- 589 cells, 4,139 anatomical edges and frozen effective gains `(4,2)`.
- Original visual-field assignment, motion inputs, numerical precision and neuron rules.
- Independent actual and motion-only reference brains.
- Original 1 ms neural steps, 20 ms observation intervals, threat detector and escape readout.

Add passive recording to both brains. Preserve the original numerical operation order; recorded values must never feed back into simulation.

Record:

- Raw RGB and its hash every 20 ms.
- LPLC4 patch salience, positive growth, sensory rates and phase.
- Complete 1 ms spike events, pre-reset voltage and post-reset voltage.
- Edge-resolved contributions, including their effective/discarded status.
- Original interval diagnostics, paired counts, rolling windows, scores, strong-event times and first selected side.
- Exact window entry/expiration and cumulative actual-minus-reference counts.

Use integer step indices for timing and float64 for neural measurements. Record all non-self edges and all modeled neurons to avoid selecting a pathway after seeing results. Stream compressed interval chunks rather than retaining the complete experiment in memory.

Bind configuration, original sources, added instrumentation sources, circuit, software/device and outputs with hashes and receipts. Refuse overwriting or resuming a consumed run. An interrupted run remains consumed.

### Preflight acceptance

Before scientific execution:

- Run the existing read-only record and preservation audits.
- Confirm the complete case list, original response configuration and protected hashes.
- Test leak, threshold, subtractive reset, directed delay and recorded edge contributions.
- Compare instrumented and original kernels on identical synthetic tensors: voltages, spikes, phases, counts and transmissions must be bit-identical.
- Test reset, timing, chunk serialization, tamper rejection and budget enforcement.

Allow at most **2,000 individual kernel steps across all synthetic parity checks**, logged separately as engineering verification. No additional rendered threat presentations are permitted. Do not run the full simulation suite during this diagnostic stage; reserve it for the subsequently approved repair.

## Stage 2 — Run one unchanged-model diagnostic

**Hypothesis:** vision shifts tonic left b3 spike timing through the retained right-DN pathway; rolling-window expiration then produces the temporary negative increment at the strong-event gate in the five deficit-only failures.

Run each following **historical validation case once**, using its exact saved azimuth, elevation, profile and seed:

| Failure seed | Successful comparison seed |
| --- | --- |
| 1100000 | 1100004 |
| 1100001 | 1100003 |
| 1100002 | 1100005 |
| 1100010 | 1100008 |
| 1100012 | 1100014 |
| 1100013 | 1100015 |
| 1100017 | 1100016 |
| 1100032 | 1100033 |
| 1100036 | 1100037 |

Run pairs in table order, failure first. These comparisons have elevation/profile differences documented in the review; retain those labels. They provide context rather than controlled interventions.

**Scientific budget: 18 presentations × 1.5 seconds = 27 simulated seconds /54,000 paired neural steps**, including each 0.5-second baseline. No shadow brains or extra presentations.

At every interval, compare with the original record:

- RGB hash.
- Actual/reference counts and original actual diagnostics.
- Rolling vectors, scores and threat events.
- Trigger timing and selected side.

Stop immediately on any mismatch, changed protected hash, missing measurement, nonfinite state, downstream rolling rate above 240 Hz or budget breach. Otherwise stop after exactly 18 presentations. Seal the receipt on success, failure or interruption; do not retry.

The hypothesis receives observational support only if **all five deficit-only failures** reproduce their original trigger and show:

1. Actual/reference spike-timing redistribution before the trigger.
2. Window expiration accounting exactly for the −1 b3 increment.
3. Positive final cumulative b3 increment.
4. Recorded right-DN delivery preceding the changed b3 timing.

Temporal order does not establish causal necessity. Classify the result as supported, unsupported or unresolved.

## Stage 3 — Produce the repair decision

Deliver a concise diagnostic report, reproducible tables and timelines. Distinguish measured evidence, supported inference and unresolved causes.

Then prepare **one concrete repair proposal** containing:

- The exact algorithm and fixed parameter values.
- Which observed failures it addresses, including the three ties.
- Expected latency and failure behavior.
- Tests, experiment budget and stop conditions.
- Preservation, replay and activation behavior.

If the timing hypothesis is supported, prioritize a minimal **spike-only temporal readout** proposal that addresses transient reversals and ties. Do not choose its timing parameters by searching validation performance.

If the hypothesis fails or remains unresolved, report that outcome. Do not automatically change learning method, modify sensory fields, add neurons or launch another experiment.

**Stop for approval of the concrete repair proposal.** This handoff does not authorize implementing or evaluating an unspecified repair.

## Finalization requirements for the later approved repair

Carry these requirements into the repair proposal:

- Controller side selection uses simulated sensory/neural signals only. No target coordinates, bearing, distance, world objects or precomputed steering answer.
- Historical cases remain historical; a new experiment name does not make them unseen.
- Use one source-locked candidate and stop at the first failed gate.
- Require **at least 90/96 correct directional responses and at least 5/6 at every azimuth**, within the existing 500 ms latency requirement.
- Only after directional validation passes, run the prescribed **48 six-second closed-loop cases**: eight directions × resting/hovering/cruising × intact/visual-silent.
- Require **at least 22/24 intact successes**, **at least 7/8 in each initial state**, and **zero visual-silent triggers**.
- Closed-loop success includes correctly signed response, collision-free flight, weak-threat braking and stable recovery under the existing protocol.
- Verify actual turning, takeoff where applicable, separation, braking and recovery through recorded trajectories; a side-selection pass alone is insufficient.
- Run targeted tests, the small full suite, independent record audits, preservation checks and replay checks after implementation.
- Review saved visuals separately from live interaction. If live review is unavailable, report it as pending.
- Activate only the new receipt-bound recovery launcher after all required gates pass and activation is authorized. Preserve historical launchers and failed records.

Update status, log, decisions, experiment documentation and queue after each completed stage. Leave one concrete next task.

Do not describe escape as finalized until its required behavioral gates pass. This plan provides the diagnosis-first route to that outcome; it does not guarantee an experimental success.
