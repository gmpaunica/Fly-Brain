# M6 recovery review and bounded next step

Date: 2026-09-22. Replaces FB-07K as the active direction.

## Scope of this review

The user requested a review and correction after FB-07 grew into repeated
diagnostics without improved navigation. Preserve the working simulation and
historical results. Review against the original brief's M6 acceptance and the
sensory-only contract. Use one agent. No new lettered FB-07 tasks or long run.

- [x] Compare original brief, architecture, queue, app and learning code.
- [x] Run the existing full test suite.
- [x] Record two missing controls in `experiments/m6-recovery-review/`:
  the unchanged sensory teacher driving training episodes, and a constant
  training-target-mean predictor versus the saved FB-07H readouts.
- [x] Correct status, queue, scientific claims and the active next step.
- [x] Verify artifacts and bounded manual/neural app smoke runs; report limits.

## Reproduction boundary

`experiments/m6-recovery-review/reproduce.py` will reuse the existing evaluator,
teacher, paired feature collector and checkpoints. No optimizer or final-case
evaluation. Teacher control is an explicitly labelled diagnostic bypass of
the neural decoder, never an app mode or a learned/biological controller.
All eight existing training starts use the original 600-tick horizon. Replay
one fixed training episode against FB-07J to check evaluator consistency.
Compare saved model MSE with the mean-target constant on the same 1,600
training samples, separately for forward and turn. Check checkpoint hashes
and the saved loss. These controls cannot prove generalization or isolate
optimization from representational capacity.

## Stop rule

Finish this review with one concrete, evidence-based next task. Do not turn a
negative result into another automatically appended letter. M6 remains open
until a matched learned-versus-frozen comparison shows improved navigation
on fresh held-out starts. A short review is complete even if M6 is not.

## Findings and corrections

1. **The embodied prototype works; learned navigation is unverified.** The
   [brief](../reference/PROJECT_BRIEF.md#20-definition-of-the-first-successful-prototype)
   permits poor navigation for the first prototype. Its M6 acceptance requires
   improved behavior. The full suite passed 236 tests; both 120-tick headless
   app modes passed. Neither is a new live visual review.
2. **The method pivot was premature.** The original
   [teacher](../../src/flybrain/training/readout_experiment.py:77) reached 8/8
   training starts in the existing evaluator, versus the recorded fixed 0/8
   and saved nine-feature 1/8 at gain 1 in
   [FB-07J](../../experiments/fb07j-turn-gain/results.json). This is a
   diagnostic bypass, not learned neural navigation or a held-out result.
3. **Loss reduction overstated useful steering.** On the same 1,600 train
   samples, the mean-command predictor's turn MSE is 0.167516. Seven and nine
   achieve 0.157916 and 0.157941, only 5.73% and 5.72% better. Nine has slightly
   worse turn fit despite lower combined loss. Its forward MSE is 0.000241;
   forward fitting accounts for about 79% of its reduction from the initial
   combined loss. [Recorded audit](../../experiments/m6-recovery-review/results.json).
   Parameter updates are real; accurate imitation has not been established.
4. **Information and fitting were conflated.** The
   [FB-07G information gate](FB-07G.md) measures exact feature collisions and
   a lookup-bucket error floor. It cannot establish what a linear readout can
   learn. The audit localizes an imitation gap without assigning it solely
   to optimizer, features, capacity or changing closed-loop observations.
5. **Final-set language was misleading.** The
   [first learning record](M6-LEARNING.md#verification-record) and
   [gain record](FB-07J.md) both evaluate seeds 200-207. Intermediate "sealed"
   statements meant no additional evaluation in those tasks. Those starts
   are now a historical benchmark, not a fresh confirmatory test.
6. **The roadmap lacked a stop boundary.** The queue expanded through H-J
   and proposed K while M6 stayed open. K is now superseded; no more lettered
   successors are to be invented automatically. Stale setup guidance in
   `SOL_WORKFLOW.md` and architecture status is corrected. Historical logs,
   checkpoints and experiment outputs remain intact.

## Next task: M6 readout recovery

Recommended effort: **Sol High**, one agent. One task, one candidate, no
reward-learning pivot. The review above is complete independently of this
future implementation task.

**Question:** can a numerically checked fit of the existing nine-feature
readout reproduce the unchanged sensory teacher well enough to navigate?

**Scope and budget:** reuse the frozen graph, rational encoder, nine features,
original teacher and 1,600 training samples. First inspect feature scales,
linear dependence and gradients on a small reproducible calculation. Use
that evidence to select at most one fitting correction, documented before
training. A maximum of 500 full-batch readout optimizer steps is the ceiling
for this task, not an instruction to run all 500 or sweep settings. Keep the
saved H checkpoint intact. No new neurons, reward function or turn-gain sweep.

**Training gate:** record forward and turn error separately and compare with
the train-target-mean predictor. Require at least 90% lower turn MSE than that
constant (MSE <= 0.016752), finite intended gradients/updates, fixed topology,
checkpoint replay, and at least 7/8 training arrivals at unchanged gain 1.
The numerical thresholds are conservative engineering calibration gates,
not biological claims or proof of generalization. If the fitting route cannot
meet them within the budget, record the limitation and stop for user input;
do not append another task or increase model capacity automatically.

**Evaluation:** only after this training gate passes, write a single protocol
with fresh validation/final starts, matched trained and frozen-untrained
versions of the same candidate, and the original fixed baseline. Lock cases,
metrics, budget and failure stopping rule before evaluating them. The old
validation set is development data; seeds 200-207 are retired from final use.
No held-out success is claimed from this review. Do not run a new evaluation
or promote a checkpoint into the app just because the fitting gate passed.

**Completion boundary:** a documented fit failure ends this task as a negative
result. A fit success ends it with the saved candidate and concrete held-out
protocol ready for review. M6 itself stays open until the subsequent matched
evaluation establishes improvement; FB-08/FB-09 remain downstream. This makes
the next decision explicit instead of silently extending the sequence.

## Verification record

- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  **236 passed**, one existing Pygame deprecation warning, 14.45 seconds.
- `.\.venv\Scripts\python.exe experiments/m6-recovery-review/reproduce.py`:
  eight teacher successes; source checkpoint hashes and saved losses match;
  fixed train-0 replay matches FB-07J exactly; reference command/reset/gate
  checks pass. Zero optimizer steps and no validation/final episodes.
- `.\.venv\Scripts\python.exe -m flybrain.app --headless --steps 120 --seed 7`
  and the same command with `--control neural`: **120 ticks each**, exit 0.
- This review adds a reproducible diagnostic and corrects project records.
  It changes no production controller, training algorithm or app behavior.
- Focused evaluation/readout/H checks: **15 passed in 3.30 seconds**. Audit
  CSV/JSON, local documentation links, required status sections, diagnostic
  syntax and refusal to overwrite the recorded output also passed. Review
  was local and single-agent, per project instructions; no Git repository
  exists and no commit was made.

## 2026-09-22 recovery fit predeclaration

The 1,600 training rows contain a full-rank nine-feature matrix plus bias,
but its raw condition ratio is about 73; the seven spike channels include
strongly similar left/right groups. The saved turn readout still has nonzero
gradients. An affine least-squares probe on these rows reaches turn MSE about
0.15539, far above the 0.016752 gate. This probe is diagnostic, not a bound
on the tanh readout or an optimizer run.

**Hypothesis:** raw feature offsets/scales impair the original 80-step Adam
fit. **One correction:** center and divide each of the nine training features
by its train-only standard deviation during Adam fitting, then algebraically
fold that transform into the same nine-input readout weights and bias. This
keeps the deployed architecture and sensory contract unchanged. The scaled
matrix condition ratio is about 11.5. Use the original Adam learning rate
0.05 and at most 500 full-batch updates; stop at 500 or immediately on a
nonfinite value. Record separate forward/turn errors on every step. Compare
the folded model with the train-target-mean predictor, then run exactly the
eight original training starts at gain 1. Require turn MSE <= 0.016752 and
at least seven arrivals, plus finite gradients/updates, topology and replay
checks. A failure closes this fit experiment and requires user input before
another approach. No validation or final starts are read in this task.

### Outcome of the bounded fit

The one 500-update run ended at turn MSE **0.155287** versus the constant
predictor's **0.167516** and reached **1/8** training starts. The required
turn and arrival gates both failed. The first gradients were finite and
nonzero for all 20 scalars; all 20 changed. The topology hash and exact
checkpoint replay checks passed. The saved FB-07H checkpoint was unchanged.
The [JSON, curve, case rows and candidate checkpoint](../../experiments/m6-readout-recovery/results.json)
record the result. No validation or final starts were evaluated. This closes
the authorized recovery fit as a negative result; wait for a user decision
before any new intervention.
