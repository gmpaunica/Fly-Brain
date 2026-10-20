# FB-07K proposal: one bounded reward-based readout experiment

**Status:** Superseded by the [M6 recovery review](M6-RECOVERY.md) on
2026-09-22. Retained as historical context; no code or training run began.
Do not execute this proposal as the active task.

## Why this changes the method

FB-07H showed parameter updates and reduced imitation loss, but did not
establish useful reproduction of the sensory teacher's turns. FB-07J's fixed motor gain improved validation
and secondary final progress, yet primary final arrival tied fixed at 1/8.
The existing final seeds 200–207 have been read for this intervention. More
feature or gain tuning against them would compromise a held-out comparison.

The proposed next test replaces **imitation loss** with reward from actual
world progress and arrival. The graph, rational encoder, nine sensory/spike/
voltage features and 20-scalar readout remain fixed in structure. Initialize
readout weights from the FB-07H nine-feature checkpoint and retain the
predeclared gain 2.0; optimize only those 20 readout scalars. The controller
still receives only simulated sensory input. World coordinates appear only
in reward computation and evaluation.

## Bounded protocol for approval

- Lock new starts before training: train seeds 0–31, validation 300–307,
  final 400–415. Use the same pose generator, source, 600-tick horizon and
  20-unit arrival radius. The previous final seeds 200–207 are excluded.
- Run exactly **128 training episodes** (four passes through 32 train starts),
  each at most 600 world ticks. One locally seeded CPU generator supplies
  stochastic motor exploration; record its seed. No seed sweep, early best
  checkpoint selection, or hyperparameter search.
- At each tick, sample Gaussian latent forward/turn commands around the
  readout's two bounded outputs with fixed standard deviations 0.15 and 0.20.
  Clamp forward to `[0,1]`, turn to `[-1,1]`, then apply the fixed 2.0 turn
  gain. The sampled latents and their log probabilities remain available to
  the policy-gradient calculation. Deterministic evaluation uses the readout
  means with gain 2.0 and no exploration noise.
- Define tick reward as `(distance_before-distance_after) /
  (initial_distance-20) - 0.002*collision`, plus `1.0` on first arrival.
  Reward sees world coordinates; the policy does not. Use return-to-go minus
  a running, action-independent episode-return baseline in the score-function
  loss `-mean(log_prob * advantage)`. Use Adam LR 0.01 and gradient-norm cap
  1.0. These values are engineering assumptions, not biological parameters.
- Record episode rewards, arrival, progress, gradients and weight changes.
  Save the final checkpoint and verify reload. Compare deterministic learned,
  frozen pre-training nine-feature, and fixed gain-2 controllers on the same
  eight new validation starts. Open the 16 final starts once only if learned
  validation arrival and mean progress exceed **both** controls.
- Mark M6 achieved only if final arrival exceeds both matched controls, with
  all failures retained in CSV/JSON. Report uncertainty and do not claim a
  topology advantage. If validation or final fails, record a negative result
  and stop rather than choosing another seed or reward weight.

This is a small score-function policy-gradient experiment, not a large RL
framework. Its gradient comes from action log probabilities weighted by
reward; it does not differentiate through the hard spike threshold or Python
world. PyTorch documents `sample()` plus `log_prob()` for REINFORCE, and its
official example shows the same likelihood-ratio pattern:
[PyTorch distributions](https://docs.pytorch.org/docs/2.14/distributions.html),
[PyTorch REINFORCE example](https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py).
The method originates in [Williams (1992)](https://doi.org/10.1007/BF00992696).

## Verification before and after the one run

1. Test the local RNG, sensory-only controller boundary, bounded actions,
   tick reward, return-to-go and score-function gradient on hand-checked
   numerical examples. Verify fixed graph/encoder and intended parameters.
2. Test a short training run, deterministic reset/replay, checkpoint reload,
   disjoint cases and validation gate, including a failed gate that never
   evaluates final starts.
3. Record exact config, environment, seed, software/device, checkpoint,
   per-episode curve and matched evaluation rows under `experiments/`.
4. Run targeted and full tests, audit artifacts, then update project status,
   log, queue, decisions and experiment explanation. No live app default
   change or automatic push/commit.

**Run budget requested:** 128 episodes × at most 600 ticks = at most 76,800
training world ticks, plus the predeclared validation and conditional final
comparisons. No training run starts until the user confirms this new
scientific protocol and budget.
