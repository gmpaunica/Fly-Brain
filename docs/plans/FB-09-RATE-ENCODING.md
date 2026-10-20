# FB-09 architectural repair: phase-distributed sensory rate coding

2026-09-26. User requested continued repair through full FB-10 readiness.
This follows the required debugging reset after the wider-current and PN-filter
repairs failed. Those experiments stay closed; no further scale/tau sweep.

## Representation problem and one hypothesis

Rational constant currents compress high odors and hard-reset ORNs quantize
their periods to whole milliseconds. Large cohorts then emit identical
volleys. The instrumented replay showed exact PN output spike-train identity
despite distinct inputs. Wider currents improved the operating range but not
all contrast gates. Smoothing PN inputs suppressed weak left-only propagation.

Replace only the sensory encoding with an explicit deterministic spike-rate
code. For each antenna independently, `r(c)=min(96,176*sqrt(c))` Hz. Uniformly
distribute initial phases at `(j+0.5)/N` across sorted source IDs on each side.
Each neural millisecond adds `r*dt`; crossing1 schedules a sensory current pulse
and subtracts1. The pulse triggers the existing ORN LIF neuron and is otherwise
zero. The decoder receives no rates, phases, currents or odor values: only the
eight downstream endpoint spike rates through the actual sourced edges.

The rate function preserves relative contrast rather than flattening inside
the original .025-.3 tested range. Fractional phases avoid forcing each ORN
into one integer interspike period; distributed phases avoid a single volley.
All aspects of this encoder are hand-written scaffolding, not sourced ORN
physiology, learned encoding, or biological reconstruction. Both sides use
the same formula. No geometry, steering labels, feedback or random draws.

## Analytical checks, not success claims

At .025, rate27.828Hz gives about50.61 normalized jump units/s into the left
PN under unilateral stimulation, close to the continuous mean-input LIF
threshold50. At high right-strong input, mean-input approximations place PN
rates near the250Hz ceiling; the exact discrete/jump dynamics may differ.
Astra identified these fragile margins. They are documented risks, not waived
gates. The96Hz sensory cap does not itself establish the downstream rate limit.

## Fixed implementation and budget

- One candidate: rate encoder above, original instantaneous hard-reset LIF,
  row budget3, no PN filtering, all145 effective source-derived edges retained.
- Maintain historical current-encoder defaults as separately named variants.
  `encode` may preview next drive but must never advance phase/neural state;
  only neural stepping commits phase. Reset restores every phase and LIF state.
- Test scheduled counts (per-cell error<1spike), distinct distributed phases,
  side independence, silent zero input, overflow safety and exact reset replay.
- Exactly16 original pulse cases:24,000 steps; all original contrast,
  low-unilateral propagation, finite, <=250Hz and washout gates unchanged.
- Four conditional causal cuts only on a complete dynamics pass:6,000 steps.
  Maximum30,000, zero updates. Preserve source/evidence hashes and old records.
- Failure closes this representation test. Do not tune the coefficient, cap,
  exponent or phase assignments from these outcomes or start an automatic sweep.

## Completion route

Only a complete pass can publish a verified config/evidence receipt. Then finish
teacher/feature readiness, real closed-loop/reset/motor-gate/render tests, app
smokes and visual review, final full suite and independent review. Training is
the subsequent FB-10 task, not this repair. No Git/commit/push actions.

## Source-induced preflight correction (before the assay)

The retained graph has one ipsilateral ORN-to-ORN edge:
`720575941577670489 -> 720575941478318641`, count5. The receiver may fire
additional synaptically triggered spikes beyond its external phase schedule.
Tests must measure the external schedule on afferent-only cells and separately
verify the actual retained edge, not remove it to satisfy the encoder test.

Consequently the old assertion that an all-synapse cut must leave every ORN
count unchanged is structurally incorrect. No prior causal assay had run.
The corrected all-synapse control instead requires byte-identical external
drive and silent downstream cells. Add one downstream-only cut that retains
sensory-to-sensory wiring: it must preserve all ORN counts and silence every
downstream cell. Keep both DNa02 cuts. All four must preserve external drive,
finite/rate/washout gates, and the applicable endpoint silence condition.
This is a source-based protocol correction made before measurement, not a
post-result relaxation. The additional control sets the fixed ceiling30,000.

## Closed result — 2026-09-26

Exactly 24,000 steps, zero optimizer updates. All 16 pulse gates and five of
six directional checks passed. High-left shift 5.333333 Hz was below 8.8 Hz;
the candidate failed, so no causal controls ran and no receipt was published.
All 82 earlier files were preserved. The user subsequently approved a separate
[PN-only charge redesign](FB-09-PN-CHARGE.md); this rate candidate is not rerun.
