# FB-10 frozen-candidate acceptance and completed assay

## Update — accepted continuation, 2026-09-28

Following the explicit original-gate-versus-stop question, the user replied
"conitnue". The assistant acknowledged proceeding with this proposed frozen
assay under the original gates while preserving the extra recovery failure.
The exact response and interpretation are recorded in
[FB-10-FROZEN-ACCEPTANCE.json](FB-10-FROZEN-ACCEPTANCE.json).
This supersedes the pending status below; the historical proposal is retained.

The additive `flybrain.training.banc_frozen_evaluation` runner independently
recomputes original and covered errors, verifies the exact initialization,
all nine integrity checks, original episode summaries and complete provenance,
and carries the failed covered gate into its protocol/result. It scans JSON,
JSONL and CSV history, consumes a central receipt, and records partial counters.
Astra's final code review found no remaining assay blocker after regression
fixes for missing provenance and journal-failure double counting. The assay passed: validation 15/16 versus 1/16 and 1/16; final 31/32 versus
4/32 and 3/32. All fresh gates and integrity checks passed, with zero updates.
See `experiments/fb10-projection-evaluation/results.json` and the current status.
The extra covered-training failure remains unchanged.

## Original proposal — 2026-09-27

2026-09-27. This is a reviewable proposal, not an executed evaluation or an
approved exception. The user's answer to the pending question is required.

The frozen candidate is `experiments/fb10-projection-readout/trained.pt`.
Checkpoint SHA256: `117f7e8b759c8916b33cd66a814e471a11cfc7aac941f71c51495dabeeddafa1`.
Training protocol SHA256: `769db7336a0adfb10e28d395dd2c64eb2a8da3aef091011e478c827a3fa91290`.
Its 500-update fit, source and data must remain unchanged. Original encoder,
83 cells, effective wiring, LIF rules and 20ms/20-step timing are unchanged.
Only the 370 motor-readout weights/biases were learned. Inference uses sixteen
endpoint histories and four measured PN incoming-spike histories; it cannot
receive odor columns, teacher labels or geometry.

## Recorded evidence

Original training turn MSE .010884924333201046 is below .02225273146561865.
Original trained arrivals8/8 strictly beat matched untrained0/8 and fixed0/8
in arrivals and all-case progress. All nine integrity/numerical checks pass,
including changed parameters and exact checkpoint/embodied replay. The covered
turn MSE .0235447046758735 is below25% of its train-only constant .3152332512,
but above the additional recovery ceiling .02225273146561865. The covered
fit therefore remains closed negative. Do not modify its gate, receipt or result.

The proposed acceptance exception uses the original FB-10 training gates for
eligibility and explicitly carries the additional failed gate into the new
protocol/result. It does not assert that the covered-data gate passed or that
an unseen case has already succeeded. A user decision is required because this
changes the recovery's scientific acceptance assumption after that fit.

## If the user chooses the frozen fresh assay

Add a separately tested runner/schema. Recompute original-data turn error and
all original training gates from the strict checkpoint. Freeze a new protocol
that records the user's acceptance decision, exact candidate/data/source bytes,
the failed covered gate, one hypothesis, budget and stop below. No new fit.
Verify fresh identities against every prior experiment, including the failed
620000 assay and all reserved earlier finals. Consume a central receipt in the
candidate directory before the first episode, even if later execution aborts.

Hypothesis: the frozen learned internal-spike readout improves fresh food
arrival over its identical untrained initialization and the fixed BANC decoder.
Use validation720000-720015 and final721000-721031, the reserved area-uniform
60-180 unit annulus and uniform independent heading, same world and600-tick
horizon, with trained/untrained/fixed paired on every case. All use the original
encoder and spiking core. No historical validation case enters fitting.

Budget: zero optimizer updates, maximum144episodes /86,400worldticks /
1,728,000neuralsteps. Validation needs>=13/16 and strict arrival/progress gain
over both controls. Stop before final if it fails. Final needs>=26/32,
paired arrival difference>=8, exact one-sided p<=.025 against each control,
strict all-case progress gain and unchanged parameters/core/provenance.
Keep partial failures. No tuning, retries, case selection or topology claim.

Audit actual JSON/CSV/counters, checkpoint hashes and all prior bytes; generate
and inspect plots from saved records. Update status/log/queue honestly. These
fresh cases become historical after the assay, regardless of its outcome.

## If the user retains the additional gate

Stop this recovery. Preserve all six completed fits, the first failed fresh
assay, the failed range-entry probe and every diagnosis. FB-10 remains unfinished.
Do not append an encoder/width sweep, more updates or another fresh assay.
The next task is a scientific scope decision, not an automatically launched run.
