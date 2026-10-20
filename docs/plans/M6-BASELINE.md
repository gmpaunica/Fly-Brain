# FB-06 plan: evaluation and fixed baseline

## Goal and locked choices

Measure the current synthetic, fixed controller before changing its weights or
decoder. Use the same 800 x 600 arena, source at (600, 300), 0.02 s world tick,
and network seed 7 as the interactive neural demo. A fly succeeds when its
**center** is within 20 world units of the source center. Check the initial pose
and each resulting tick; stop on success or after 600 ticks (12 seconds).
Wall contact does not terminate an episode. These thresholds are engineering
choices, not biological measurements.

The primary objective is held-out food-arrival rate. The secondary measure is
mean normalized progress, `min(1, (initial_distance - final_distance) /
(initial_distance - success_radius))`, over **all** episodes. Report success
time and path efficiency only for successful episodes, with their denominator.
There is no training reward or optimizer in FB-06.

Use locally seeded starts in an annulus 60–180 units from the food, with
uniform angle and heading. Lock disjoint seed sets before tuning: train 0–7,
validation 100–103, final 200–207. The final set is a held-out audit set;
FB-07 development should use train and validation. Store generated poses and
the per-episode results so later algorithms use exactly the same cases.

## Implementation steps

1. Add `tests/test_evaluation.py` for seeded split integrity, start success,
   horizon timeout, distance/path/collision accounting, exact replay, and
   serialized result metadata. Observe the expected failures.
2. Implement the renderer-free evaluator in `training/evaluation.py` and a
   small command-line entry point for the unchanged fixed controller.
3. Run the locked validation and final baseline once, write JSON and CSV under
   `experiments/`, and summarize recorded results in `docs/experiments.md`.
4. Run targeted and full tests, update architecture decision, queue, status,
   and dated log. Review the resulting code against the FB-06 exit gate.

## Verification record

On 2026-09-21, 6 focused evaluation tests and 190 full-suite tests passed.
The fixed baseline completed 12 renderer-free validation/final episodes and
wrote JSON/CSV. A second full 12-episode run matched every recorded result
field exactly. The results are summarized in `docs/experiments.md`. The full
suite retains one upstream Pygame `pkg_resources` deprecation warning.

## Boundaries

Only the evaluator reads coordinates for scoring. `NeuralController.decide`
continues to receive `SensoryObservation` only. Evaluation creates fresh world,
network, and decoder state per episode. It does not use Pygame, train parameters,
or claim learning or reliable navigation.
