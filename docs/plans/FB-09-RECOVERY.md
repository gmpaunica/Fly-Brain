# FB-09 recovery and FB-10 readiness

Date: 2026-09-25. Execution authorized by the user with Astra.

Current update (2026-09-26): the original experiment below remains failed.
The later explicitly approved [delay-aligned protocol](FB-09-LATENCY-PROPOSAL.md)
and readiness checks passed, completing FB-09. See [current status](../../PROJECT_STATUS.md).

## Agreed finish line

Finish FB-09 when the sourced subset drives the existing embodied interface,
has a correctly labelled live view, and passes dynamics, causality, reset and
timing checks. FB-10 will be the first bounded training experiment on this
BANC controller. No optimizer updates belong to this recovery.

The user selected an engineered normalized subset and readiness for the
existing food-approach setup, rather than sensing throughout the entire arena.
Preserve the original BANC artifact, all original IDs and source annotations.

## Fixed recovery hypothesis and budget

Hypothesis: masking self-edges and normalizing retained incoming connections,
with a fixed multiscale sensory encoding, permits transient endpoint activity
that distinguishes bilateral odor contrasts in the declared operating band.

- Simulation weights: zero the 31 self-edges; for every other edge use
  `W[post,pre] = budget * count / sum(nonself incoming retained counts)`.
  Empty rows remain zero. Preserve all 62 cross-side edges. The resulting
  145-edge graph is acyclic with maximum path length seven.
- ORN current is `2.5*c/(c+scale_i)`. Assign logarithmically spaced scales
  from 0.005 through 0.05 in sorted original-ID order, separately per side.
  This heterogeneous encoder and the normalization are synthetic assumptions.
- Test budgets 1.25, 2 and 3 in that order with the encoder fixed. Select the
  smallest passing budget. No gain/scale sweep may be appended after failure.
- Each independent case has 250 zero-input steps, 1,000 pulse steps and 250
  zero-input steps. Centers are 0.025, 0.1 and 0.25. At each center test
  left-only, right-only, equal bilateral and both swaps of `(0.8c,1.2c)`.
  Include one all-zero case: 16 cases x 3 candidates x 1,500 = 72,000 steps.
- Record a protocol before running, source/code hashes, software/device,
  per-case counts, selected candidate or failure, and plots from saved data.
  Preserve prior experiments and reject existing output destinations.

## Acceptance gates

- Finite state and exact reset replay; zero-only input generates no spikes.
- In the final 750 pulse steps, unilateral input reaches its PN, DNa02 and
  at least one same-side motor neuron. Equal bilateral reaches both sides.
- Contralateral activity is allowed. Define `D = right_DNa02_Hz-left_DNa02_Hz`.
  Both contrast swaps must shift D in the correct direction from equal-input
  D at that center, by at least 1 Hz and 2% of total steering activity.
- No spikes after a 20-step washout grace period. Final absolute voltage is
  below 1e-4 after 250 silent steps. Sustained per-neuron rates <=250 Hz.
- On the selected candidate, deleting all effective synapses preserves ORN
  responses but silences downstream endpoints; cutting a DNa02's incoming
  connections silences that endpoint and its motors. Do not renormalize cuts.
- If all candidates fail, close this experiment and report the specific failed
  gates. FB-09 remains open; do not label it ready merely because tests pass.

## Embodied interface and training preparation

- Use eight endpoint spike-rate features: two DNa02 and six motor cells in
  stable root-ID-labelled order. Update from full tick counts divided by
  elapsed seconds using EMA `alpha=1-exp(-dt/0.1)`; reset all feature state.
- Fixed diagnostic decoder: turn `(r_right-r_left)/(r_right+r_left+10)`;
  forward `mean_motor_rate/(mean_motor_rate+20)`. Rates use Hz. The motor
  reduction is an engineering proxy for the point-body simulator.
- Add `--control banc`, using only a packaged verified dynamics config. Show
  all 83 nodes in a functional ORN/PN/corridor/DNa02/motor layout, display
  effective edges, and label the source, assumed dynamics and untrained state.
- Reuse `NeuralController`, `ClosedLoopSimulation` and the evaluator through
  structural interfaces. Preserve manual and synthetic modes/checkpoints.
- Check teacher behavior on the eight existing training starts; collect the
  eight endpoint features and sensory-only teacher targets, report feature
  variation/aliasing and the train-only constant-command MSE baseline. Record
  actual concentration coverage. No held-out cases or optimizer updates.
- Define FB-10 as a separate bounded readout experiment using the fixed
  accepted topology, these endpoint features, save/reload checks and matched
  trained/untrained/fixed controls. Fresh evaluation requires a separate
  locked split; historical final cases remain historical.

## Work ledger

- [x] Astra: source validation, normalized dynamics, endpoint decoder, protocol
  runner and bounded probe. Probe FAILED; conditional causal checks did not run.
- [x] Root: structural typing, gated BANC app mode, functional brain view and
  app/view tests. Actual accepted-config smoke/visual check blocked by failed gate.
- [x] Root: teacher/feature readiness runner and FB-10 entry contract implemented.
  Readiness experiment did not run; no accepted dynamics config exists.
- [x] Final verification: 27 targeted and 319 full-suite tests pass; manual and
  synthetic 120-tick smoke pass; BANC refuses unverified config. Source artifact
  and 60 earlier files retain hashes. Saved contrast plot visually inspected.
  Independent Astra review identified one receipt-evidence issue, reproduced
  and fixed with six tests. Reviewer hit a usage limit before its full verdict;
  root completed local review. No full independent approval is claimed.

No Git repository exists; do not initialize, commit or delete research data as
part of this task. After two failed fixes to the same software failure, collect
a minimal reproduction before trying another change. Scientific failures close
the declared probe rather than starting a sequence of unplanned candidates.

## Closed outcome — 2026-09-25

All three declared candidates were measured once, exactly 72,000 neural steps
and zero optimizer updates. No candidate passed, no causal controls ran and no
verified app configuration was published. Budgets 2 and 3 passed all individual
pulse/finite/washout/rate gates. Budget 3 passed contrast centers 0.025 and 0.1
but failed 0.25: right-strong shift 0 Hz (required 6.64), left-strong shift
4 Hz (required 6.56). At budget 2 the high-band right-strong shift was reversed.
See [results](../../experiments/fb09-dynamics-recovery/results.json).

Do not reduce the band or change the threshold after observing these results.
Next requires explicit approval of one revised encoding hypothesis and budget;
the plateau's exact origin has not been causally isolated. FB-10 remains blocked.

## Final recovery handoff — 2026-09-26

The subsequent diagnostic isolated exact relay delay/window-boundary effects.
User-approved latency-aligned sustained-rate acceptance retained the same
model,750ms duration and1Hz/2% thresholds. It passed all16pulse cases, all six
directions and four source-aware causal controls,30,000steps and0updates.
Original simultaneous failure and all earlier failed experiments are preserved.

The packaged evidence/config enables BANC mode. Teacher readiness passes8/8
with1,008 endpoint-feature samples; untrained fixed BANC0/8 is recorded. All
358 tests, three120-tick app smokes, static BANC/plot visual inspection and
independent Astra final code review pass. Live manual usability is not claimed.
The agreed finish line is met; next task is FB-10's separate bounded training
protocol, not more FB-09 tuning. No optimizer was run in this recovery.
