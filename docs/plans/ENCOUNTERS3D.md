# Exclusive flower / predator encounters

User specification: 2026-10-05, “Flower or predator: exclusive 3D trials”.
Execution: one agent, local acceptance review; no commits, training, optimizer,
interrupted-run continuation, historical experiment or checkpoint edits.

## Design and characterization gate

Add `flybrain.encounters3d` alongside the historical lab. An environment-only
`SceneSpec(seed, kind, position)` configures exactly one flower or stationary
predator. Only the unchanged `FlightObservation` crosses into controllers.
Flower delegates to the existing world/teacher. Predator uses a distinct cyan
surface, nearest-hit RGB sensing, existing LPLC4 cells and source connectivity.
RGB color recognition and visual receptive fields are engineered assumptions.

Hypothesis: existing visually driven wing endpoints retain enough directional
information for a fixed activity-only escape decoder. Budget: one characterization
of eight azimuths plus bilateral symmetry, intact / visual-silent / DNp03-silent,
at most 1 s per static presentation; then at most 5 s per closed-loop case.
Stop: if endpoint responses cannot distinguish directions reliably, record that
gate failure, leave avoidance unavailable and propose a specific sourced extension.
Do not add cells, choose a hidden visual steering policy, fit weights or train.
Characterize before fixing any decoder. A flight stabilizer may integrate only
body-local motion senses to maintain lift. Retreat/rearm/hover are explicitly
engineered policies, and any threat display comes from recorded neural activity.

## Implementation plan

- [x] Characterize existing graph and visual-to-motor responses, store config,
  software/device/source binding and JSON/CSV under a fresh `experiments/` path.
- [x] Add scene/world/sensors with exclusive objects and nearest-hit occlusion;
  behavior tests for visibility, expansion and flower/background rejection.
- [x] Add neural encoder/controller subject to the characterization gate;
  verify numerical dynamics and matched silencing, including symmetric stimuli.
- [x] Add session and viewer subclass reusing anatomy, publication and reset
  barriers; spawn buttons, R/new selected type, Backspace/exact replay, labels.
- [x] Measure delay/separation/collisions using evaluator-only geometry; verify
  reset/pause/render independence and source preservation.
- [x] Run encounter/3D tests, full suite, rendered review and separate live UI
  attempt; update status/log/architecture/decisions/queue and hand off limits.

Review focus: rear fields and eye boundary symmetry; persistent visible stimuli
must not drive endless retreats; silent motor endpoints cannot produce escape;
worker resets must atomically clear history; camera/display choices must persist.

The supplied specification authorizes implementation choices and one-agent
execution. Extra skill design/plan approval gates and automatic commits do not
apply under the user's project instructions. There is no Git repository.

## Completion ledger

Characterization complete: one physical/saturation run, negative wing spike gate.
Scene/neural numerical/session/viewer implementation and 13 new tests complete.
Motor-decoder/retreat/retrigger implementation stopped under the user-specified
failure condition; no hidden steering or training substitute.
24 matched closed-loop hover measurements complete, with no retreat claim.
100 targeted / 667 full tests passed; 2,465 historical files preserved.
Rendered UI reviewed; native access attempted and unavailable, user review pending.
Local acceptance self-review complete with one agent. No Git repository/commit.
Routine rulings: supplied spec authorizes implementation without extra skill gates;
All activity is initial only in the new launcher; use existing source-preserving
viewer/session subclasses; failed scientific gate leaves hover explicitly labeled.
Next: bounded sourced/physiological recovery specification / Sol High.


## 2026-10-05 — User-confirmed live encounter controls

The user reports: “all controlls work, but the fly doesnt react. its interesting
that the brain reacts differently depending on where the predator spawns”.

This confirms the requested live spawn exclusivity, R/Backspace, pause/reset and
camera/display retention checks. Placement-dependent brain appearance and absent
fly response are user observations, consistent with the recorded sensory activity
and failed motor-spike gate. No directional decoding or neural escape success is
inferred. Current predator flight is controlled solely by the explicit hover
stabilizer. Historical activity-view acceptance remains separate. No code or
scientific result changed; no tests rerun, training or recovery experiment begun.
