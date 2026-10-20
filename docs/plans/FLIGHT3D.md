# 3D flower landing and anatomical flight circuit — 2026-09-30

User-approved implementation scope: true 3D motion; one modeled flower at a
random position/height per trial; fixed fly start; odor, vision, motion and
contact sensing; sourced BANC flight/landing pathways; an anatomical viewer
styled after `docs/reference/fly-brain-3d-visual-reference.png`.

## Contracts

- Separate `flybrain.flight3d` package and launcher. Preserve historical 2D
  source and evidence. Python/PyTorch simulation, Panda3D display.
- SI units, 0.6 x 0.6 x 0.3 m arena, Z upward, initial body (0.3,0.3,0.1),
  yaw zero facing +Y. Neural/physics/control intervals: 1/5/20 ms.
- Body forces, gravity, drag, tilt, steering and real surface contact;
  flower stem/petals/nectar surface. No coordinate-based controller override.
- `FlightObservation`: odor pair, eye image pair, local motion/foot/food
  sensing only. `FlightCommand`: power, steering, leg deployment.
  `FlightSnapshot`: immutable render data. `TrialSpec`: world/evaluation only.
- Root IDs are strings; source connections are pre->post, matrices [post,pre].
  Identity, geometry and physiology evidence are separate. No invented sourced
  links or anatomical positions. Freeze circuit manifest before any fitting.
- Whole-brain/project-only visibility changes drawing only. Actual BANC
  morphology/positions, purple/pink structures, orange/yellow activity on
  black; orbit/zoom/front/side/top; selectable nerve cord, neuron details,
  activity decay in simulation time. Missing anatomy coverage is explicit.
- R new flower, Backspace same trial; reset every state and RNG stream.

## Ordered deliverables

1. Tested world, immutable interfaces, eye/odor/contact sensors and manual mode.
2. Audited registry, flight circuit manifest, source geometry cache and atlas.
3. Neural loop and endpoint-constrained readouts; propagation, discriminability,
   motor/circuit intervention checks; sensory-only teacher preflight.
4. Single predeclared imitation experiment and honest recorded outcome.
5. World/anatomical viewer, visual inspection, full-suite verification, handoff.

## Experiment (do not weaken after outcomes)

48 train / 24 validation / 96 final randomized flowers, balanced horizontal
sectors and lower/equal/higher heights. All episodes 30 seconds, same initial
fly pose. Safe landing and >=2 s stationary food contact retained to the end.
Teacher 48/48 before fitting; trained >=44/48 plus better per-continuous-channel
error than the train-only mean-command predictor before validation. Validation
>=20/24; final >=82/96 and >=26/32 per height; exact paired one-sided p<=.025
against identical untrained and fixed neural decoders. Fixed graph/dynamics;
readouts only, 2,000 Adam updates, lr .001, batch 512, recorded seed and train-only
normalization. Collection/fitting wall ceiling 2 h. A failed prerequisite/gate
closes that experiment, with no automatic sweep or final reuse. Incomplete is
distinct from failed. Preserve trajectories, curves, gradients, parameter
changes, checkpoints, source/config/software hashes and durable progress.

## Verification

Numerical physics and collision tests, deterministic resets/replays, sensor-only
boundary, above/below/behind/occluded stimuli, source IDs and edge orientation,
causal propagation/ablations, save/reload, camera/render independence, complete
existing suite. Saved-image checks and live desktop review reported separately.

## Scope decisions

The user explicitly supersedes the old deferral of 3D/vision. They selected
sourced anatomy with documented modeled flight, one flower per trial, and
visibility toggles rather than full-brain dynamic simulation. This work uses
one agent, no commits/pushes. Passing learning is an empirical outcome, not a
condition to fabricate or silently relax.
