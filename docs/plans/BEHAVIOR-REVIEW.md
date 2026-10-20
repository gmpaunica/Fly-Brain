# Behavior reassessment and test foundation — 2026-09-28

Goal: make the user's random-heading, speed-control and stopping requirements
explicit, diagnose the current model, and make the app expose the missing behavior.
The user selected simulated food-contact sensing. Execute inline with one agent.

## Findings and corrected scope

- The historical assay used full-circle random headings (16/16 and 32/32 unique
  headings), recorded seeds and matching poses across controllers. Its arrival
  result remains valid for those cases. Randomness alone does not prove learning.
- Desktop reset reused the launch seed. A new trial must advance a local seed
  stream; replay must be a separate action. Explicit seeds preserve reproduction.
- Every one of the 15,864 training rows requests forward=0.7. The evaluator ends
  on first entry into a 20-unit radius. Neither train nor evaluation teaches or
  tests braking, contact, staying, or feeding.
- The saved speed output has a global lower bound of about 7.4 units/s from its
  bounded hidden activations and saved weights. It cannot command a true stop.
- Two historical starts continued for 20 seconds both arrived without settling.
  Seed 7 arrived at 4.76 s at 68.99 units/s and ended 101.61 units away.
  Evidence: `experiments/behavior-review-2026-09-28/` (2,000 world ticks,
  40,000 neural steps, zero optimizer updates; all prior source/artifacts preserved).
- Rate encoding saturates at antenna distance <=54.42 units in this fixed odor
  field. Contact sensing addresses contact; approach-speed observability still
  needs a preflight before training. The model has no current contact input.
- App starts at distance 200, outside the historical 60–180 start annulus.
  Food/layout/strength generalization and downstream topology benefit are unproven.

## Authorized implementation

Architecture: add independent world sampling and behavior measurement modules;
adapt the unpinned current app. Keep archived neural/physics/training source bytes
and evidence contracts intact. This implements the testing foundation; it does
not grant the old checkpoint a learned contact response or change its commands.

- [ ] Add `world/food.py`: a radius-10 patch centered on the existing odor source;
  body/patch contact at center separation <=10+fly.radius. Frozen sensory values
  contain only odor_left, odor_right, food_contact. Contact is a simulated local
  sense, with no coordinate/distance/action fields delivered to inference.
- [ ] Add `behavior.py`: locally seeded distinct episode headings and a replayable
  seed sequence; current contact, first contact, speed/displacement, and consecutive
  settled ticks. Settled means contact, commanded and actual speed <=1 unit/s,
  absolute angular speed <=0.05 rad/s for 100 consecutive 20 ms ticks (2 s).
  These are engineering acceptance values, not biological measurements.
- [ ] Add `foraging.py`: sample the richer sensory observation, run the original
  neural inference and world step, then observe behavior. No distance-based motor
  override, no automatic pause, and no ending on first contact. Full reset clears
  all controller, world, metric and visual state.
- [ ] Update current `verified_app.py`: fresh launch seed when omitted; `--seed`
  reproduces the episode sequence; R starts a new orientation; Backspace replays
  the current episode. Show seed/episode, speed, contact, hold duration and the
  arrival-only scope. Apply to current trained and fixed BANC comparison modes.
  The historical manual/neural launchers retain their old debugging controls.
- [ ] Add meaningful numerical tests for RNG/replay, sensor locality and boundary,
  speed range, pass-through failure, zero-command dwell, turning/creep/wall failures,
  controller observation delivery, timing/reset and app key events. Run focused and
  full suites. A saved renderer frame is distinct from a live desktop review.
- [ ] Update status, queue, decisions, architecture, README, experiments and log.

## Next scientific task: approach-and-stop preflight / Sol High

Implement and verify one sensory-only teacher plus a candidate contact encoding
and motor readout that can issue zero. First show teacher speed variation,
slowing and stationary contact across full-circle headings on training starts;
check near-food sensory information and positive/negative contact examples.
Describe the added contact neural pathway as engineered, not sourced BANC anatomy.
No automatic training run is authorized by this reassessment.

Then predeclare one bounded fit, its data, train-only normalization, seed, loss,
update budget and failure stop. Evaluate a frozen candidate and its identical
untrained and fixed controls on the same new recorded cases, using disjoint
train/validation/final identities and full-circle headings. Report entry, contact,
approach speed, overshoot, stop latency and two-second stationary dwell separately.
Do not reuse the historical final cases as unseen. Keep failed episodes and do not
stop observation at first arrival. Set all final gates before inspecting outcomes.

Project acceptance remains open until learned speed/stop behavior and a live
desktop review pass. The old arrival score and software test count do not close it.
