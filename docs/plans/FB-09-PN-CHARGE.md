# FB-09 controlled projection-neuron dynamics redesign

2026-09-26. The user explicitly approved changing only the two PNs' assumed
dynamics, preserving excess charge and documenting membrane parameters, while
keeping source wiring and every success threshold unchanged.

## Closed precursor and mechanism

The phase-rate candidate closed after24,000steps. All pulse gates and five of
six individual directional checks passed; high-left contrast was5.333Hz versus
required8.8Hz. No coefficient/exponent/cap/phase changes are made from that run.
The instrumented precursor had already measured input differences disappearing
into PN reset loss. A controlled PN-only redesign is now authorized explicitly.

## Exactly one fixed candidate

- Preserve phase-rate encoding `min(96,176*sqrt(c))`, phases `(j+.5)/N`, all145
  effective edges, row budget3 and eight endpoint-rate features.
- Only the two PNs use membrane tau25ms, threshold1.1, and subtractive reset:
  `V_next = V_candidate - spike*1.1`. One spike maximum per1ms step; residual
  above threshold can trigger a subsequent step, rather than being discarded.
- PN relaxation/input fraction match tau25ms. All other cells retain tau20ms,
  threshold1 and hard reset0. No PN synaptic filter and no new graph edges.
- These are explicit engineering assumptions, not BANC-measured physiology.
  The source stores structure and counts, not this membrane model.

## Analytical constraints and unchanged empirical gates

A post-spike subthreshold PN residual decays by exp(-250/25)=exp(-10) over the
250ms washout, below the existing1e-4 limit for a residual below1.1. Any larger
residual must drain through spikes during the existing20ms grace period; this
is tested, not assumed. Around52 jump units/s, the mean-voltage approximation
is1.3, above1.1, preserving weak-signal headroom. Around275 units/s, threshold1.1
and leak predict rates below250Hz. Actual discrete/synchronous behavior still
must pass the measured gate. No approximation counts as acceptance evidence.

## Budget, evidence and stop

One16-case pulse assay:24,000steps. Exactly the original low/mid/high patterns,
steady-window contrast threshold1Hz plus2%, finite/zero/propagation, <=250Hz,
20ms late-spike and250ms voltage washout gates. No weakened gate or narrowed band.

Only after every dynamics gate passes, four source-corrected causal controls
add6,000steps (all-synapses off, left/right DNa02 input cuts, downstream-only
cut preserving the ORN-to-ORN edge). Maximum30,000steps, zero optimizer updates.
Freeze code and write protocol before starting; protect old records and source.
If it fails, close it without a tau/threshold/encoding sweep. Do not publish a
verified default unless all gates and evidence-integrity checks pass.

## Verification / handoff

Test PN-only overshoot retention, exact tau/input fractions, unchanged other
cells, one-spike-per-step, washout, reset, CPU ownership and causal model reuse.
On success, bundle verified evidence; finish teacher/feature readiness and BANC
app/motor/reset/view checks, visual review, full suite and independent review.
Only then is FB-10 next. Do not begin its optimizer run in this task.

## Execution ledger

- User approval received; the fixed design above is unchanged.
- PN unit tests observed RED before implementation, then three passed. Added
  reset/rebuild integration and a compensating-ORN-count regression; five pass.
- Astra's precursor audit found group-total causal comparisons could conceal
  changes between ORNs. Controls now compare each ORN separately; only the
  actual sensory-edge recipient is exempted for the all-synapse cut. This
  strengthens measurement before any causal controls have run.
- No Git repository exists; preserve files in place, with no commit/worktree.
  The protocol freezes the six core source hashes before simulation.

## Closed result — 2026-09-26

Exactly 24,000 neural substeps, zero optimizer updates. All 16 pulse cases
pass propagation, finite/zero, rate and washout gates. Five of six directional
checks pass, including both high-band directions. Mid-right alone fails:
5.333333333 Hz versus 5.36 Hz. This is not a floating-point tolerance problem.
Equal DNa02 steady counts are L100/R101; right-strong L98/R103. Bias-corrected
contrast is four spikes, while 2% of 201 spikes is 4.02. All 88 earlier files
and six frozen core modules retained their hashes. No causal controls ran;
no accepted receipt, teacher-data collection or BANC training was enabled.

Corresponding PN steady contrast is six spikes and full-pulse DNa02 contrast
is seven. Those descriptive metrics do not replace the declared 750 ms window.
Saved counts alone cannot identify the precise relay/reset events. Astra's
read-only audit found no blocking implementation/gate bug and advised against
another parameter change based on these counts. A separately bounded,
unchanged-parameter event-timing diagnostic on these two cases is the next
decision; no such run is started automatically here.

Verification: six PN unit/integration tests; full suite 336 passed, one upstream
Pygame warning (42.60 s). Tests include current fraction, non-PN equivalence,
CPU ownership with a meta default, excess-charge accounting, one-spike limit,
reset, causal rebuild, individual sensory counts, and actual embodied timing,
motor gate and rendering independence. The saved contrast PNG was inspected;
this is not an accepted-controller live-app visual review.

Commands:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_banc_projection_dynamics.py -q -p no:cacheprovider
.venv/Scripts/python.exe -m flybrain.training.banc_recovery --pn-charge --output-dir experiments/fb09-pn-charge
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --junitxml=.fb09-verification.xml
.venv/Scripts/python.exe scripts/plot_banc_recovery.py experiments/fb09-pn-charge/results.json experiments/fb09-pn-charge/contrast_gates.png
```

The assay returned a failure exit status as designed. Its output directory is
closed/protected. Do not repeat the command as an automatic retry.

## Approved follow-up diagnostic — 2026-09-26

The user approved the proposed 3,000-step observation-only investigation and
asked to resume from saved state. This is not another dynamics candidate.

Hypothesis: exact PN-to-DNa02 event timing and actual charge accounting can
locate the six-to-four steady-window contrast difference. Replay only the
recorded 0.1 equal and right-strong cases, 1,500 steps each (250 baseline,
1,000 pulse,250 washout). Freeze all original parameters, gates and core source.
Record the 15 nonsensory cells' actual transitions and all83 spike timestamps.
Use one-based end-step numbering: original steady interval501..1250 inclusive.
Check each real corridor edge for exactly one-step transmission and compare
the complete path with its structural delay. Record boundary crossings as a
diagnostic explanation; never substitute aligned counts for the declared gate.

Implementation is confined to `training/banc_relay_diagnosis.py`, its tests and
documentation. Protocol and old-file hashes are written before neural stepping.
Required fidelity: exact recorded full-pulse and steady counts for all83cells,
unchanged core/source data and preserved prior evidence. Stop after3,000steps,
with zero optimizer updates. No automatic new candidate or training follows.

### Diagnostic closed

The approved run completed exactly3,000steps with zero updates. Both cases
exactly reproduced all83 full-pulse and steady counts; all94 earlier files,
six core source modules and the new diagnostic source were preserved.

Every PN event matches its DNa02 event after4ms left /3ms right; every individual
corridor edge also matches after1ms. There are no missing or extra relay events
in either case. The first hypothesis of relay spike loss is not supported.
The six-to-four contrast difference is fully explained by fixed-window edges:

| Case / side | PN steady spikes | DNa02 steady spikes | PN events crossing window boundary |
|---|---:|---:|---|
| Equal / left |101|100|1248 arrives1252, outside end1250|
| Equal / right |100|101|499 arrives502, inside start501|
| Right-strong / left |98|98|498 enters;1249 leaves, net0|
| Right-strong / right |103|103|499 enters;1250 leaves, net0|

Numbers are one-based end-step timestamps in milliseconds. Maximum charge
balance error is1.1921e-7, consistent with the actual float32 transition.
Astra independently reviewed observer code and confirmed the saved event
accounting. This explains the miss; it does not turn the failed gate into a pass.

Evidence: `experiments/fb09-relay-diagnosis/{protocol,results}.json` and
`relay_trace.csv` (45,000 central-neuron transitions). The next scientific
decision is the [unapproved latency-aligned proposal](FB-09-LATENCY-PROPOSAL.md).
Do not automatically rerun, retune or begin FB-10.
