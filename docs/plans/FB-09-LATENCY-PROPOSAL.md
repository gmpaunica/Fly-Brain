# FB-09 latency-aligned sustained-rate protocol — approved

2026-09-26. The user directed continuation through FB-09 after the measurement
choice was presented, and confirmed continuation after the approval interpretation
was stated. This changes the scientific measurement definition, not the model.
The original PN-charge experiment remains failed and its evidence is immutable.

## Measured reason

The user-approved 3,000-step [relay diagnostic](../../experiments/fb09-relay-diagnosis/results.json)
exactly replayed all83 full-pulse and steady counts on equal/right-strong mid-band
input. Every PN event traversed every corridor edge without loss or added events:
left PN-to-DNa02 delay4ms, right3ms. The equal-input fixed-window count difference
changes from PN -1 to DNa02 +1 due solely to boundary crossings; right-strong
remains +5. The resulting four-spike response fails the original4.02 threshold.

## One proposed replacement requirement

If the intended requirement is preservation of sustained directional rate
across the relay, count each steering endpoint over the same upstream PN
interval after its mechanically derived graph delay. Keep the original common
PN interval end-steps501..1250. Count left DNa02 at505..1254, right at504..1253.
Both windows remain750ms. No chosen offset, offset sweep or parameter change.

Keep the same16 input cases, row budget3, phase-rate encoder, PN-only tau25ms/
threshold1.1/subtractive reset, source edges, endpoint decoder,1Hz and2% contrast
thresholds. Preserve all other original zero/propagation/finite/rate/washout
checks and record the original simultaneous-window metrics alongside the new
ones. The simulation remains1,500steps/case; its existing250ms washout already
covers the last shifted measurement. Do not align the controller's live inputs
or output features, or give it future data. Only offline measurement changes.

One assay:24,000steps; only on full acceptance run the four existing causal
controls,6,000steps. Maximum30,000steps, zero optimizer updates. Preserve old
results and freeze the revised protocol/source before stepping. Stop after this
candidate; do not select a favorable window or append another parameter trial.

## Limits and alternative

This proposal is informed by historical outcomes, not independent confirmation.
It tests a different definition of sustained-rate transfer, and does not prove
robust simultaneous counts, navigation improvement or biological fidelity.
Conditional causal, teacher/feature and app/visual gates remain required before
FB-10. If simultaneous endpoint activity in the original interval is itself
the requirement, do not adopt this proposal: retain failure and reconsider the
scientific model rather than merely changing the test.

## Implementation outline

1. Add tests for fixed graph-derived endpoint delays and aligned counting
   without changing neural stepping; verify RED then GREEN.
2. Record both measurement definitions in a distinct protocol and result.
   Do not rewrite any old experiment or promote it retrospectively.
3. Execute the single capped assay and applicable causal controls; preserve
   all files and source hashes. On failure, close the protocol.
4. Only on complete dynamics/causal success finish the already planned
   teacher-feature, receipt and live-app verification. No FB-10 training here.

## Verified outcome — 2026-09-26

Completed exactly 30,000 neural steps and zero optimizer updates. All 16 pulse
cases and six aligned directional checks pass, as do the four conditional
causal controls. The 750 ms directional shifts (right/left) are 4/4 Hz at0.025,
8/8 Hz at0.1, and14.666667/12 Hz at0.25; every original1Hz/2% threshold passes.
All98 earlier files and six frozen core source files were preserved.

The result also records the original simultaneous contrasts, including the
mid-right5.333333Hz<5.36Hz failure. No old result was modified or promoted.
Original successful result/protocol bytes are bundled beside the verified
configuration. The installed loader accepts this bundle and rejects stale or
failed evidence. Endpoint windows alone changed; controller timing did not.

The readiness run then passed: teacher8/8,1,008 finite varying endpoint samples,
5,808 world ticks,0updates. Fixed untrained BANC0/8 remains a baseline. All106
preceding artifacts,45 source files, accepted bundle and actual weights stayed
unchanged. Source odor range0.023099-1.905076 exceeds the contrast-test band;
the encoded rate saturates above about0.29752, so broad steering/learning success
is not implied. Mean-command turn MSE0.08901093; exact-alias lower bound0.00238492.

Verification: 358 tests pass (22.40s), one upstream Pygame warning. Manual,
synthetic and BANC app modes each pass120 headless ticks. The actual120-tick
BANC frame and saved-data contrast plot were inspected. This is static visual
review, not manual live interaction testing. Astra's independent final code
review found no blocking/important issues. No deferred code findings remain.

FB-09 is complete under this contract. FB-10 is next and has not started.
Evidence: [dynamics/causal result](../../experiments/fb09-latency-aligned/results.json),
[readiness result](../../experiments/fb09-banc-readiness/results.json),
[inspected frame](../../experiments/fb09-app-verification/banc-frame.png).
