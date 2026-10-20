# FB-09 PN input-filter mechanism test

2026-09-25. Within the user's continued repair authorization. The preceding
wide-encoder candidate is closed: all pulse gates passed but contrast failed
at centers .1 and .25. No further encoder/gain candidates are added.

## Single hypothesis

Spread incoming projection-neuron (PN) voltage jumps over an assumed 5 ms
exponential tail. This reduces coincident peaks discarded by hard reset while
retaining each pulse's complete-tail area. The measured PN input differences
otherwise vanish into leak/reset loss. This is an engineered temporal model,
not a BANC-sourced physiological parameter or a guaranteed contrast repair.

For each of the two PN rows only, `u = W @ previous_spikes`,
`z_next = exp(-dt/.005)*z + (1-exp(-dt/.005))*u`; use `z_next` instead of `u`
in the existing LIF step. Other rows remain instantaneous. Reset z per episode.
Do not leak coordinates or raw sensory values into PN/motor computations.

## Fixed configuration, gates and budget

One candidate: retained-input row budget3, ORN scales .005-.375, original LIF
rules/tau20ms and eight endpoint-rate features. Only PN input timing changes.
Keep all145 effective edges, original source metadata and counts untouched.

Exactly16 original pulse cases:24,000 neural steps. All original low/medium/high
contrast, unilateral propagation, silence, finite, <=250Hz and washout gates
remain unchanged. Only if they all pass, three causal cuts add4,500 steps.
Maximum28,500 steps, zero training updates, no tau/gain/range search.

Risks: low-band propagation may rely on coincident peaks; PN rates may rise;
filter tails may produce late spikes. Retain every failed gate. A failure closes
this mechanism test and requires a debugging/architecture reassessment, not
another filter time constant. No configuration receipt unless all gates pass.

## Verification and ledger

- Astra reviewed the proposed mechanism read-only and found it defensible
  within the repair scope, with explicit rate/low-band/washout risks.
- Test first: impulse area and one-step delay, exact exponential fraction,
  unaffected non-PN rows, reset and invalid-input state protection.
- The historical default LIF path remains unchanged. A protected synaptic-input
  hook permits one BANC subclass to filter only the diagnosed receiving rows.
- Predeclare/hash protocol before running. Preserve all prior evidence.

## Closed outcome — 2026-09-26 review

Exactly24,000 neural steps, zero optimizer updates, no selected config or causal
cases. Finiteness, rate and washout gates passed. Low-.025 left-only propagation
failed. Contrast passed at .025 but failed at .1 (right4Hz<4.507) and .25
(right5.333Hz<6.507, left4Hz<6.48). See
[result](../../experiments/fb09-pn-filter/results.json).
No other time constant is tried. This result confirms the stated weak-input
tradeoff and does not justify promoting the filter as a successful repair.
Further work must reconsider the representation, not append local parameter
variants. The raw source and all preceding records remain untouched.
