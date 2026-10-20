# FlyBrain Project Log

This file is the chronological development record for the FlyBrain project.

Do not delete old entries. Append new entries after meaningful work.

---

## Initial Project Setup

### Goal
Define the project before implementation begins.

### Changes Made
- Created the project specification in `AGENTS.md`.
- Defined the 2D embodied fly concept.
- Defined the PyTorch spiking-network architecture.
- Defined the eventual connectome-constrained machine-learning goal.
- Defined staged milestones to prevent premature full-connectome complexity.

### Decisions
- Use a 2D world initially.
- Use Pygame for the first visual implementation.
- Use PyTorch for neural computation and later learning.
- Begin with a small toy neural network before integrating real Drosophila connectivity.
- Treat the live brain visualization as a core project feature.
- Keep biological structure, assumed dynamics, trainable parameters, and hand-written scaffolding explicitly separated.

### Tests / Verification
No implementation exists yet.

### Problems / Failed Attempts
None yet.

### Current State
Project specification is ready. Coding has not begun.

### Next Step
Complete Milestone 0 and Milestone 1: create the repository skeleton and a deterministic 2D fly/world loop.

### Notes
The first ML work should not begin until basic world movement and odor sensing are clean and tested.

---

## 2026-09-18 — Sol build preparation and project analysis

### Goal
Prepare a reliable, economical workflow so subsequent project implementation can use GPT-5.6 Sol Medium/High instead of Astra.

### Changes Made
- Preserved the complete original `AGENTS.md` unchanged at `docs/reference/PROJECT_BRIEF.md` and replaced the active guide with concise operating/scientific rules.
- Added `SOL_WORKFLOW.md` with model routing, first-build/continuation/review prompts, focused reading, and recovery guidance.
- Added `docs/architecture.md` with project assessment, module boundaries, geometry, timing, sensory restrictions, neural synchronization, and learning/evaluation risks.
- Added `docs/BUILD_QUEUE.md` with FB-01 through FB-09, dependencies, recommended effort and measurable exit gates.
- Added `docs/plans/M0-M1.md` with exact files/interfaces, numerical acceptance examples, environment setup, fixed-step replay, and separate headless/visible checks.
- Added `docs/decisions.md`, refreshed `PROJECT_STATUS.md`, and added `scripts/Start-Sol.ps1` for Medium/High CLI launch without persistent configuration changes.

### Decisions
- One Sol agent; Medium for ordinary implementation, High for neural math, learning, scientific evaluation and focused difficult debugging. No automatic Astra fallback or delegation.
- Implement only M0-M1 in FB-01. Keep future-stage plans scoped rather than implementing the simulator during this preparation session.
- Use a pure Python world and Pygame adapter, screen-coordinate orientation, radius-aware clamping and 0.02-second physics ticks; later use 20 neural substeps of 0.001 seconds.
- Treat valid optimization and held-out behavioral evaluation as explicit gates. A small trainable readout is the first proposed investigation; a final learning algorithm is not yet selected.

### Tests / Verification
- `git status --short`: confirmed this folder is not currently a Git repository.
- `python --version` and `py --list`: Python 3.14.2 was the sole discovered interpreter.
- `codex --version` / `codex --help`: CLI 0.154.0; checked model, config override and working-directory flags. Local model metadata listed Sol Medium and High support.
- `Get-FileHash` and an inline Python SHA-256 check: original brief preserved exactly, hash DB381E5FF8CDF6B6AC7D9B4B26D7BCBF85A18022A5E0D6677618C9FC1E3EA554.
- Inline Python document checks: 9 Markdown files, 6 local links, balanced fenced blocks, required status headings, all 9 task IDs, and no unresolved drafting placeholders passed.
- PowerShell parser plus `scripts/Start-Sol.ps1 -Preview` / `-Effort high -Preview`: passed syntax and argument checks; unsupported effort rejected.
- In-process PowerShell invocation with a stubbed `codex`: verified no launch on preview, correct project/model/effort/prompt forwarding, and surfaced nonzero exit. No authenticated model request was made.
- Active AGENTS.md reduced from 24,835 to 5,728 bytes (76.9% smaller). This measures instruction-file size, not tokens billed or future model performance.
- Reviewed task interfaces, coordinate/turn signs, time ratios, milestone order and status for consistency. Official OpenAI and PyTorch documentation links are included near relevant guidance.

### Problems / Failed Attempts
- Codex help/version emitted access-denied warnings for temporary PATH-alias/stale-dir housekeeping in the sandbox but returned the requested output. A live interactive model session was not launched or tested.
- One combined patch attempted both deletion and addition of PROJECT_STATUS.md; the patch tool rejected it atomically. Reapplied the new plan and updated status separately, then verified the files.
- Stable PyTorch documentation URLs initially returned redirects; followed them to the actual versioned documentation.
- Pygame installation, GUI operation and simulator tests are not yet available. FB-01 must verify compatibility with the chosen Python interpreter.

### Current State
Preparation documents and launch helper are in place and checked. No simulator source, Python environment, neural controller, training, biological dataset or recorded experiment exists yet. User-wide Codex settings are unchanged.

### Next Step
Select Sol Medium and implement FB-01 from `docs/plans/M0-M1.md`, completing Milestones 0 and 1 only.

### Notes
For the next session read AGENTS.md, current status, this entry, and the active plan; load only the relevant architecture sections. The full original brief is an on-demand reference. Existing historical log content is retained. Documentation cannot change the model of an already running conversation.

---

## 2026-09-18 — FB-01 deterministic world and manual adapter

### Goal
Implement Milestones 0 and 1 from `docs/plans/M0-M1.md` with one agent; stop before odor sensing.

### Changes Made
- Added setuptools src package metadata, delegated requirements, ignore rules, local environment and README.
- Implemented immutable validated motor commands, turn-before-translation body motion and radius-aware arena bounds.
- Added local seeded center spawning, explicit physics ticks, frame accumulation and episode reset.
- Added Pygame world/HUD rendering, manual key mapping, seed/reset/exit controls, positive physics-tick limits and bounded headless mode with restored SDL environment variables.
- Added 51 behavior/integration cases and a resolved environment snapshot. Updated status, queue and decisions; preserved preparation files and historical log entries.

### Decisions
- Followed existing geometry/speed/time contracts; no neural, odor, food or brain placeholders.
- Installed Python 3.12.10 through approved user-scope winget after binary-only Pygame install failed on Python 3.14.2. Retained old `.venv-py314/`; working `.venv/` uses 3.12.
- Interactive frame delays cap at 0.25 s. Accumulator guard is 1e-12 tick; physical state is not rounded.
- No Git repository was present. No commit, push, subagent or training run.

### Tests / Verification
- `python --version`, `py --list`: initially Python 3.14.2 only; `py -3.12 --version` after installation: Python 3.12.10.
- `python -m venv .venv`: sandboxed ensurepip failed; approved retry succeeded. New 3.12 environment created with `py -3.12 -m venv .venv` after retaining old environment.
- `.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -e '.[dev]'`: passed on 3.12; Pygame 2.6.1, pytest 9.1.1. Full resolved versions: `docs/environment.md`.
- Test-first missing-module runs confirmed absent fly/arena/simulation/app behavior. Early numerical tests used `$env:PYTHONPATH='src'` and `.venv-py314` while 3.12 setup ran; 33 fly/arena and then 44 world/simulation cases passed with `-p no:cacheprovider`.
- `.\.venv\Scripts\python.exe -m pytest -q`: 51 passed on 3.12. Tests include four walls/corner/contact, seeded global-RNG isolation, validation, frame partitions, full-pose replay, reset, CLI rejection, key aliases/opposition, SDL restoration, read-only drawing and shutdown.
- `.\.venv\Scripts\python.exe -m flybrain.app --headless --steps 120 --seed 7`: exit 0, reported exactly `ticks=120`; SDL 2.28.4. This is wiring/drawing smoke, not visible controls verification.
- Absolute venv interpreter from `C:\Users\Lenovo\Documents` with `-c "import flybrain, sys; print(flybrain.__file__); print('pygame imported:', 'pygame' in sys.modules)"`: installed package imported, Pygame not imported.
- `.\.venv\Scripts\python.exe -m flybrain.app --seed 7`: visible arena, fly/body heading and readable MANUAL DEBUG HUD inspected through Windows screenshots. UI input attempts did not establish key behavior; live visual controls/wall/reset acceptance remains pending. Process later returned exit 0 after window-close attempt.
- Locally reviewed implementation against architecture and acceptance requirements: pure world imports, local RNG, no rendering mutation, explicit timing, correct turn sign and scoped dependencies.
- Final `.\.venv\Scripts\python.exe -m flybrain.app --seed 7 --steps 5`: bounded interactive loop exited 0. Final `.\.venv\Scripts\python.exe -m pytest -q`: 51 passed in 1.82 s. This bounded launch does not verify keyboard-driven exit.

### Problems / Failed Attempts
- Initial venv pip bootstrap failed under sandbox; retry used normal approval. Sandboxed pytest cache creation also hit access denied; final suite used approved execution.
- Python 3.14 binary-only dependency install returned no matching Pygame distribution; no SDL compilation or graphics-library substitution attempted.
- App tests before implementation failed for absent app/Pygame, as expected. Missing-module collection errors were recorded rather than claimed numerical failures.
- Computer-use bindings did not persist across calls; reinitialized/selecting fresh returned windows allowed screenshots. R twice and Escape produced unchanged visible pose/tick progression, even after canvas focus. Input delivery versus app event handling remains uncertain; no speculative implementation fix applied. Full visual acceptance is pending.
- A combined record patch using delete/add for the same status path was rejected atomically; reapplied with an update operation.

### Current State
M0 verified. M1 code and numerical/headless checks pass; FB-01 is not fully verified until visible controls/wall/reset acceptance passes. No odor sensing exists. Runtime environment is documented.

### Next Step
FB-01 visual acceptance / Sol Medium: check held/released movement and turn keys, aliases/opposing turns, wall contact, seed reset and clean quit. Then mark FB-01 verified and set FB-02 / Sol Medium as next.

### Notes
Learning takeaway: fixed physics ticks and an explicit screen-coordinate convention make motion independently testable; successful dummy-driver rendering cannot validate a human-facing control loop.

---

## 2026-09-18 — FB-01 user acceptance and FB-02 antenna sensing

### Goal
Record the user's acceptance of FB-01 visible checks and implement M2 / FB-02;
defer visual polish and stop before neural control.

### Changes Made
- Added `docs/plans/M2.md` and executed its scoped steps inline, with one agent.
- Added immutable `OdorSource`, finite softened scalar field and validated raw
  `SensoryObservation` with only left/right concentrations.
- Added left/right antenna world geometry and independent sampling. Simulation
  initializes/updates/resets current-pose readings while preserving source state.
- App now displays one static food odor source, colored paired antennae and a
  raw current-pose sensor HUD. Rendering remains read-only.
- Added 34 numerical/integration cases (85 total). Updated README, status,
  queue, architecture and decisions; retained historical records.

### Decisions
- User's “all good we can move on” closes FB-01's manual acceptance gate by user
  observation. Their visual-style concern is tracked as deferred polish.
- Source at (600,300), strength 1000, softening 20; offsets 10 forward/5 lateral.
  These are scaffolding in arbitrary units, not biological measurements.
- Field extends outside walls; antennae do not clamp. Hypot-based field evaluation
  avoids squared-distance overflow; nonfinite peak configurations are rejected.
- Readings published after movement describe current pose; future M4 inference
  must sample before movement separately. No normalization, coordinate-bearing
  observation, steering answer, reward, source consumption or neural dependency.

### Tests / Verification
- `git status --short`: no repository present, as before.
- `.\.venv\Scripts\python.exe -m pytest tests/test_odor.py tests/test_sensing.py -q`:
  first produced expected missing-module collection errors; implementation then
  passed 30 cases. Includes expected values, monotonicity, radial symmetry,
  validation, large finite inputs, rotation at four headings, left/right sources,
  unclamped points and observation immutability.
- `.\.venv\Scripts\python.exe -m pytest tests/test_simulation.py -q`: initially
  3 failed/11 passed for missing source argument/observation lifecycle.
- `.\.venv\Scripts\python.exe -m pytest tests/test_odor.py tests/test_sensing.py tests/test_simulation.py tests/test_fly.py tests/test_arena.py -q`:
  77 passed after publication implementation; motion changes readings, reset and
  replay reproduce them, frame partitions agree and no source gives zero input.
- `.\.venv\Scripts\python.exe -m pytest tests/test_world_view.py -q`: initially
  failed because the source/antenna markers were absent.
- `.\.venv\Scripts\python.exe -m pytest tests/test_world_view.py tests/test_app.py -q`:
  8 passed after drawing implementation. Additional draw calls preserve pose,
  observations, global RNG and subsequent trajectories; source/sensors/HUD draw.
- `.\.venv\Scripts\python.exe -m pytest -q`: 85 passed in 2.19 s.
- `.\.venv\Scripts\python.exe -m flybrain.app --headless --steps 120 --seed 7`:
  exit 0; exactly 120 ticks. Environment remains Python 3.12.10/Pygame 2.6.1/SDL 2.28.4.
- `.\.venv\Scripts\python.exe -m flybrain.app --seed 7`: screenshot review showed
  the green labeled source, blue/pink paired antennae and distinct readings
  L=2.4675e-02/R=2.2674e-02 at centered seed-7 pose. No new automated manual-key
  verification claimed; prior manual controls accepted by the user. App remains
  running for inspection.
- Local acceptance review: geometry sign, softened formula, independent sampling,
  reset, phase labeling, read-only renderer, minimal dependencies and scope agree.

### Problems / Failed Attempts
Only expected test-first failures occurred. No speculative fixes to earlier UI
input delivery were attempted; user observation supplied the FB-01 acceptance.
No unrelated simulator changes or new packages were needed.

### Current State
FB-01 and FB-02 verified. M2 sensory HUD is live; movement remains MANUAL DEBUG.
Visual polish is deferred. No neural controller or learning exists.

### Next Step
FB-03 / Sol High: specify exact LIF dynamics and test leak, threshold equality,
reset and tiny directed propagation before building a synthetic 50-200-neuron graph.

### Notes
Learning takeaway: two local concentrations can carry directional information
without handing a controller source coordinates. The field/geometry are simplified
engineering choices, not a biological reconstruction. No commit/push/training.

---

## 2026-09-18 — FB-03 tensor LIF and synthetic topology

### Goal
Implement and verify M3 / FB-03: precise LIF dynamics and a tiny directed
propagation example before a standalone 100-neuron synthetic graph. Stop before
embodied motor control, brain UI or learning.

### Changes Made
- Recorded equations, units, group sizes, current scale and acceptance fixtures
  in `docs/plans/M3.md`, executing inline with one agent.
- Added CPU float32/float64 `LIFConfig`/`LIFPopulation`, synchronous previous-spike
  recurrence, validated dense tensors, fixed owned weights/state, atomic input
  and transition validation, and full episode reset.
- Added complete unique labels/groups, locally seeded fixed same-side topology,
  independent saturated sensory currents and separately documented tonic
  forward drive. Neural code imports no world geometry or Pygame.
- Added `SyntheticNetwork.advance`, detached copied snapshots and int64 counts
  across every neural substep, with completed-step interval/time metadata.
- Added 56 numerical/behavior cases (141 total). Installed CPU PyTorch and the
  optional `neural` extra; documented setup and resolved software/device.
- Updated README, architecture, decisions, queue/status and plan checklist.

### Decisions
- Time seconds; other quantities normalized arbitrary model units. Exact passive
  integration plus end-of-step synaptic jumps, `W[post,pre] @ prior_spikes`.
  Default dt 0.001 s/tau 0.020 s/rest=reset=0/threshold=1. Equality fires,
  overshoot is discarded, no refractory counter, at most one spike/neuron/step.
- Groups: sensory 10 left/10 right, interneurons 25 left/25 right, motors 10
  left/10 forward/10 right. Same-side pathway jumps in [0.096,0.144), no other
  edges; local CPU generator seed. Independent current
  `2*min(concentration/0.025,1)`, forward tonic 1.3. All are hand-written
  scaffolding, with no biological, navigation, learning or topology-benefit claim.
- CPU allocations explicit even inside ambient device contexts. Snapshots and
  public weight/state access return copies. Inference is not an autograd route.
- PyTorch/NumPy are an optional extra; manual app/world imports remain separate.
  No new UI; no visual review required or claimed for this milestone.

### Tests / Verification
- `git status --short`: no repository, consistent with prior sessions.
- Initial `.\.venv\Scripts\python.exe -m pytest tests/test_lif.py -q` stopped
  at missing torch and emitted sandbox cache-write warnings. After torch install,
  `.\.venv\Scripts\python.exe -m pytest tests/test_lif.py tests/test_neural_network.py -q -p no:cacheprovider`
  stopped at the absent neural package, as expected. These were collection
  errors, not numerical test results.
- `.\.venv\Scripts\python.exe -m pytest tests/test_lif.py -q -p no:cacheprovider`:
  33 passed before larger graph implementation. Includes half-life/rest offset,
  exact equality/just-below/overshoot, configured reset, consecutive firing,
  A→B next-step propagation/no reverse edge, pending-spike reset and replay.
- Network tests initially stopped at missing `flybrain.neural.network`; after
  implementation, 51 neural cases passed. Added inhibition, overflow atomicity,
  dtype scalar-rounding, non-CPU input and ambient-device checks. Ambient-device
  case failed with a meta weight tensor; minimal reproduction showed ambient
  `torch.zeros(1)`→meta versus explicit `device='cpu'`→CPU. Explicit CPU
  allocations fixed the cause; 56 cases passed.
- Final `.\.venv\Scripts\python.exe -m pytest tests/test_lif.py tests/test_neural_network.py -q -p no:cacheprovider`:
  **56 passed in 7.12 s, no warnings** after installing NumPy. Covers complete
  labels/directed fixed edges, local RNG isolation, independent scale/saturation,
  finite extreme concentrations, both same-side sensory→interneuron→motor paths,
  silent opposite paths, tonic-only baseline, all-substep counts/partition
  equivalence, reset replay, snapshot independence and dtype/device contracts.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  **141 passed in 16.37 s**, one upstream Pygame `pkg_resources` deprecation
  warning. Cache disabled to avoid the existing sandbox permission issue.
- `.\.venv\Scripts\python.exe -m flybrain.app --headless --steps 120 --seed 7`:
  exit 0, exactly 120 ticks. This checks existing manual app compatibility,
  not neural closed-loop control or visible behavior.
- Separate Python import check (`import flybrain.app; assert 'torch' not in
  sys.modules`) passed. `.\.venv\Scripts\python.exe -m pip check`: no broken
  requirements. `pip freeze` recorded in `docs/environment.md`.
- Local acceptance review: equations/threshold/delay/reset match the numerical
  fixtures; sensory-only boundary has no world imports; group coverage, fixed
  topology, copied snapshots, count timing, explicit CPU and milestone scope agree.

### Problems / Failed Attempts
- Sandboxed CPU torch install reported no matching distribution and could not
  clean temporary pip directories. Approved retry installed torch 2.14.0+cpu;
  copying the wheel took several minutes. No system-wide changes made.
- Initial torch import warned about absent NumPy. Added the documented prerequisite
  to the optional extra and installed numpy 2.5.3; neural warnings cleared.
- Ambient default device exposed implicit allocation dependence; one focused
  reproduction and one fix resolved it. No repeated speculative fixes.
- Pygame retains a setuptools deprecation warning; no rendering rewrite or
  dependency pin was introduced for an upstream warning.

### Current State
FB-03 verified. Standalone fixed synthetic spiking network exists, while the app
remains MANUAL DEBUG. No neural motor decoder, live brain view, learning,
checkpoint or experiment results exist. World/renderer code was preserved.

### Next Step
FB-04 / Sol High: causal sensory-neural-motor integration. Sample pre-movement
concentrations, hold current for 20 substeps, decode full motor counts, move once,
and verify sign/range, motor silencing, sensory-only inference and reset/replay.

### Notes
Learning takeaway: a last-substep spike vector can be silent even when activity
propagated earlier in the same world tick. Aggregate all 1 ms spikes before
motor decoding or rendering. No commit, push, publishing or training run.

---

## 2026-09-18 — ongoing PyTorch learning preference

### Goal
Support the user's understanding of what PyTorch is doing throughout the build.

### Changes Made
Recorded teaching expectations in AGENTS.md and PROJECT_STATUS.md: explain
concrete calculations, tensor shapes, state changes and small numerical examples.

### Decisions
Teach incrementally alongside authorized work. Distinguish tensor computation,
our assumed neuron rules, forward simulation and later actual optimization.

### Tests / Verification
Documentation-only change; inspected the updated instructions and status.
No runtime changes or additional numerical checks required.

### Problems / Failed Attempts
None.

### Current State
FB-03 remains verified. Future neural work includes accessible explanations.

### Next Step
FB-04 / Sol High remains next: causal sensory-neural-motor integration, with a
worked explanation of how population spike counts become motor commands.

### Notes
Learning takeaway: PyTorch executes the arithmetic; we specify the model and
must separately implement and verify a learning procedure.

---

## 2026-09-20 — FB-04 causal sensory-neural-motor integration

### Goal
Connect the verified synthetic spiking network to the fly with sensory-only,
pre-movement inference, exact 20-substep timing and bounded causal motor output.

### Changes Made
- Added a full-count motor decoder with documented left/right sign, saturation,
  0.5 exponential smoothing, atomic validation and reset.
- Added a `NeuralController` whose per-tick API accepts only
  `SensoryObservation`, plus a diagnostic motor-output gate.
- Added a closed-loop orchestrator that samples before movement, advances 20
  one-millisecond neural steps, moves once for 0.02 seconds, and publishes an
  immutable timing/activity/resulting-body snapshot.
- Added explicit `--control manual|neural` app selection. Neural imports are
  lazy, held movement keys are excluded from neural control, and the HUD/caption
  identify the active mode.
- Added motor, closed-loop, CLI/import and dual-mode rendering tests; wrote and
  completed `docs/plans/M4.md`; updated architecture, decisions, README, queue
  and current status.

### Decisions
- One spike per neuron across a 20 ms tick (50 Hz mean population activity) is
  the arbitrary full-scale motor reference: `min(group_sum/10,1)`.
- Positive turn is `motor_right - motor_left`; alpha-0.5 smoothing is cleared on
  reset. These are hand-written actuator assumptions, not learned or biological.
- The output gate is applied after real neural/decoder state advances, so
  silencing proves physical causality without replacing the network with a mock.

### Tests / Verification
- TDD RED states were observed for missing `neural.motor`, missing closed-loop
  modules, absent `--control`, and absent renderer mode support.
- `pytest tests/test_motor.py`: **15 passed**.
- Motor/closed-loop/simulation/network focus: **63 passed**.
- Final motor/closed-loop/app/view focus: **43 passed**, one existing upstream
  Pygame `pkg_resources` deprecation warning.
- Final branch-finish suite: **176 passed in 18.45 s**, the same single warning.
- Exact 120-tick manual and neural headless smokes both exited 0.
- Clean subprocess import confirmed `flybrain.app` does not import Torch.
- Seed-7 neural smoke advanced 2,400 neural steps over 120 world ticks. Pose
  changed from `(400.0,300.0,2.0347)` to `(371.6091,451.7880,1.3151)` and the
  raw observation changed, establishing a closed future-input loop without a
  claim that the fly found food.
- A left-only numerical fixture produced motor sums `(10,0,0)` and command
  `(forward=0, turn=-0.5)`, verifying the documented tensor-to-command example.

### Problems / Failed Attempts
- Expected RED collection/behavior failures occurred before each implementation.
- One documentation/code patch missed an existing Unicode dash context; it made
  no file change and was reapplied with narrower context. No repeated code fix or
  unresolved numerical failure occurred.

### Current State
FB-04 is verified. The fly can run under explicit neural closed-loop control;
manual mode remains the default and Torch-free on import. There is no live brain
panel, learning, reward, checkpoint or measured navigation performance.

### Next Step
FB-05 / Sol Medium: render actual accumulated spike activity and motor output in
a live brain panel, with simulation-time fading and read-only/reset tests, then
perform a visible two-panel review.

### Notes
Learning takeaway: PyTorch produced the `(100,)` integer spike-count tensor and
group sums. The population normalization, right-minus-left sign, smoothing and
motor command are ordinary hand-written rules. Forward execution is not learning;
learning will require a defined objective, gradients and optimizer updates later.

---

## 2026-09-20 — FB-05 live synthetic-brain panel

### Goal
Display the verified synthetic network beside the world using real accumulated
spikes, simulation-time fading, limited deterministic edges, labeled groups,
timing, and motor output without altering the simulation.

### Changes Made
- Added a read-only `BrainActivityModel` and `BrainView` for all 100 labeled
  neurons, seven group colours, 80 group-balanced edges, motor traces, decoded
  output, and causal timing.
- Added `ClosedLoopSimulation.advance_snapshots` so every tick completed during
  a display frame reaches the visual model; retained the existing integer-return
  `advance` API.
- Expanded neural mode to a 1200 x 600 world-plus-brain layout while keeping
  manual mode 800 x 600 and free of Torch imports.
- Wired real snapshots and reset into headless and interactive app paths.
- Added focused activity, fade, reset, edge, rendering, trajectory, aggregation,
  and two-panel integration tests; documented the M5 contract and limitations.

### Decisions
- The panel is a deterministic functional diagram of the hand-built graph, not
  anatomical fly-brain geometry. It says this in the panel and documentation.
- A spike raises its trace and the trace fades linearly over 0.2 seconds of
  neural simulation time. Draw frequency has no effect on visual state.
- Selected edges preserve `W[post,pre]` direction and are balanced across the
  four populated group-to-group pathways.

### Tests / Verification
- TDD RED states were observed for the missing brain-view module, missing
  all-snapshots advance API, missing two-panel app builder, and colour-erasing
  white spike highlights.
- Brain-view/closed-loop/app/world-view focus: **36 passed**.
- Full suite: **184 passed in 23.37 s**, with the existing upstream Pygame
  `pkg_resources` deprecation warning.
- `compileall -q src tests` exited 0.
- Exact 120-tick manual and neural headless smokes both exited 0.
- A clean app import printed `False` for Torch presence.
- The exact-size 1200 x 600 frame at `artifacts/fb05-brain.png` was visually
  inspected after preserving group colour during flashes.

### Problems / Failed Attempts
- The first documentation patch missed a Unicode heading context and made no
  changes; it was split into narrower patches.
- Initial spike highlights blended every active group to white. A failing pixel
  test reproduced the loss of group identity before the colour-preserving fix.
- Live motion was not inspected interactively; the saved static frame and
  automated timing/fade tests cover separate visual and behavioral checks.

### Current State
FB-05 is verified. Neural mode presents the world and live synthetic graph
together, and rendering remains causally read-only. There is still no learning,
reward, checkpoint, measured baseline, or downloaded connectome.

### Next Step
FB-06 / Sol High: define the evaluation contract and record deterministic fixed
controller baselines before selecting any trainable parameters.

### Notes
Learning takeaway: PyTorch supplies the per-neuron spike-count tensor, while the
view is ordinary presentation state. Feeding every causal snapshot and fading by
simulation time makes the picture a faithful display of the run rather than a
second clock that can change what the user sees.

---

## 2026-09-21 — live feedback, 3D demo vision, and learning clarification

### Goal
Record the user's circling observation and future 3D demo reference, and clarify
the roles of current neural simulation, future learning, and sourced connectivity.

### Changes Made
- Copied the supplied image unchanged into `docs/reference/` and added
  `DEMO_VISION.md` with the requested 3D body and rotatable neural-view goals.
- Recorded exploration and stimulus approach/avoidance as behavioral goals.
- Updated current status and queue pointers while retaining FB-06 as next.
- Explained the candidate learned motor readout with a small numerical example.

### Decisions
- Recording future requirements does not start 3D implementation or full-brain
  import. The image is a visual reference, not sourced anatomical/connectivity data.
- FB-09 still starts with a biological subset. Navigation and learning are
  separate from importing a graph and should be measured with the toy system.
- Distinguish a 3D renderer from actual 3D locomotion and simulated activity from
  experimentally recorded neural activity.

### Tests / Verification
- Inspected current neural drive, encoding, motor decoding, and movement code.
- A read-only diagnostic ran 20 decisions each for zero, constant left-only,
  and constant right-only odor. At 0.4 s, forward was about 0.714 in all three;
  turn was respectively 0, about -1, and about +1. Process exited 0.
- Source/copy SHA256 hashes matched for the reference image.
- Checked a primary whole-brain modeling paper for the distinction between
  connectivity, neuron dynamics, and complete sensorimotor pathways.
- No runtime source changes or training. The full suite was not rerun for this
  documentation/reference update; the previous recorded 184-test run remains.

### Problems / Failed Attempts
- No repository exists. The user's circling trajectory has not been replayed,
  so the diagnostic does not establish its exact cause or navigation performance.

### Current State
The user has observed the running prototype; navigation is unsatisfactory.
Pulse/fade readability and reset acceptance are still unconfirmed. Future demo
requirements are preserved, with no additional implementation or training.

### Next Step
FB-06 / Sol High: define evaluation and record fixed baselines, including the
current navigation limitation, before tuning or training a controller.

### Notes
Changing neuron activity during a run is simulation. Learning requires a defined
training signal, actual parameter updates, and improvement on separate cases.

---

## 2026-09-21 — FB-06 fixed navigation evaluation

### Goal

Lock the evaluation protocol and record the unchanged fixed controller before
any optimizer or learned parameter is introduced.

### Changes Made

- Added renderer-free episode evaluation, seeded train/validation/final cases,
  success/timeout scoring, distance/path/wall/spike metrics, and JSON/CSV export.
- Recorded 12 validation/final episodes with exact initial poses, config,
  network and case seeds, software/device details, and nullable checkpoint and
  learning-rate fields under `experiments/fb06-fixed-baseline/`.
- Added the FB-06 plan, experiment summary, architecture decision, README
  command, evaluation tests, and updated queue/status pointers.

### Decisions

- Success means fly-center distance <=20 from (600,300), including the initial
  pose and the final allowed tick. Timeout is 600 x 0.02-second ticks.
- Wall contact counts but does not end an episode. The primary objective is
  held-out arrival rate; all-case normalized progress is secondary. Successful
  times and path efficiencies have a success-only denominator.
- Lock train seeds 0-7, validation 100-103, and final 200-207, with poses
  generated by local random streams in a 60-180-unit annulus. Keep final cases
  out of FB-07 tuning. No reward function or optimizer was added in FB-06.

### Tests / Verification

- Initial focused test run failed as expected because `flybrain.training` was
  absent. A focused progress test then failed at 3.333 before progress was
  capped at 1 on success.
- Six focused evaluation tests passed; full suite **190 passed in 27.84 s**,
  with the existing Pygame `pkg_resources` deprecation warning.
- `compileall -q src tests` exited 0. Importing the evaluator in a clean
  process reported `pygame_imported=False`.
- The headless baseline wrote 12 episode rows. Validation arrival was 0/4,
  mean progress -1.688; final arrival was 1/8, mean progress -3.021.
- A second 12-episode run exactly matched the recorded per-episode result
  objects. JSON/CSV row IDs and outcomes matched, with 12 rows in each.
- Local acceptance review confirmed the evaluator has no display import,
  `ClosedLoopSimulation` still passes only `SensoryObservation` to the
  controller, and score-only coordinates remain outside that call boundary.

### Problems / Failed Attempts

- The first success test assumed movement on the first neural interval, but
  the delayed forward pathway produces no first-tick displacement. The test
  now checks success on the second and final allowed tick.
- The wall-contact test assumed two collisions in two ticks. The first tick
  has no outward movement, so the observed and asserted count is one.
- Baseline cases mostly ended against walls. Their total absolute turn was
  below 1.4 radians each, so they do not reproduce the user's sustained
  live-circling observation. No source-free exploration mechanism exists.

### Current State

FB-06 is verified. The fixed synthetic controller performs poorly on this
small locked set. There is no learning, checkpoint, or improvement claim.

### Next Step

FB-07 / Sol High: verify a trainable readout's gradient and update, then run
bounded training and compare the saved/reloaded model to this fixed baseline
on identical held-out cases. Mark M6 complete only with measured improvement.

### Notes

PyTorch generated the 100-neuron `int64` spike-count tensor per world tick;
the evaluator summed those counts for reporting. It did not call autograd or
change weights. A simulated tensor calculation is not learning.

---

## 2026-09-21 — FB-07 first trainable readout experiment

### Goal

Verify a real gradient/update path through a small motor readout, save/reload
it, and measure one bounded training attempt against the locked FB-06 cases.

### Changes Made

- Added EMA-filtered seven-group spike features, a 16-scalar PyTorch linear
  motor readout and a learned decoder above the unchanged spiking network.
- Added train-only fixed-trajectory sample collection, a sensory-only teacher,
  full-batch Adam training, checkpoint topology validation, and renderer-free
  learned evaluation with matched fixed per-case comparison.
- Recorded config, software/device, starts, gradients, parameter changes,
  topology hash, checkpoint, 81-point loss CSV/SVG and 12 episode rows under
  `experiments/fb07-readout/`.
- Added numerical/integration tests, the FB-07 plan, PyTorch explanation,
  experiment summary, decision, queue and status updates.

### Decisions

- Freeze graph and encoder. Train only the readout from filtered spike counts;
  hard spike thresholds receive no surrogate gradient. The teacher uses raw
  antenna contrast for a target, outside the controller decision path.
- Use 200 fixed-controller ticks from each of eight locked train starts, then
  80 full-batch Adam steps at learning rate 0.05. Do one configuration and keep
  final cases out of hyperparameter selection. This is supervised imitation,
  not reinforcement or biological synaptic learning.
- Mark FB-07 as a verified negative attempt, while keeping M6 open because the
  primary held-out arrival metric did not improve.

### Tests / Verification

- First focused run failed on the missing readout module; after implementation,
  3 readout tests passed. The experiment test first failed on its missing
  module; 4 passed after implementation. The artifact-aliasing metadata test
  then failed on its missing JSON field and passed after it was added.
- Final focused command: `python -m pytest tests/test_readout.py
  tests/test_readout_experiment.py tests/test_evaluation.py
  tests/test_closed_loop.py -q -p no:cacheprovider`: **31 passed**.
- Full command: `python -m pytest -q -p no:cacheprovider`: **199 passed in
  11.71 s**, one existing Pygame `pkg_resources` deprecation warning.
  `python -m compileall -q src tests` exited 0.
- Artifact audit checked 12 matching JSON/CSV episode IDs, 81 matching curve
  rows, checkpoint SHA-256, topology/reload flags and eight-way teacher alias.
- Recorded 1,600 train samples; loss 0.104135 to 0.079003. All 16 first-step
  gradients were finite/nonzero (minimum absolute 0.01918); all 16 parameters
  changed (minimum absolute net change 0.000125). Saved/reloaded model exactly
  replayed the first validation episode. Topology hash stayed fixed.
- Validation arrival: learned 0/4, fixed 0/4; progress -1.286 versus -1.688.
  Final arrival: learned 0/8, fixed 1/8; progress -1.188 versus -3.021.

### Problems / Failed Attempts

- All eight train starts yielded one identical first-tick spike-feature vector
  despite eight distinct sensory-teacher turns. Current saturation at 0.025
  removes those differences from the readout. This establishes a specific
  information bottleneck, not a complete cause for every timeout.
- One fixed final success became a learned timeout. The second experiment run
  added aliasing metadata without changing configuration or case outcomes; no
  final-case tuning occurred. No new interactive visual review was performed.

### Current State

FB-07 is implemented and verified as a failed navigation experiment. Real
PyTorch learning occurred in the readout, but M6 embodied improvement has not
been shown. The interactive neural app still uses the fixed decoder.

### Next Step

FB-07D / Sol High: quantify feature aliasing over locked train/validation
trajectories and test one controlled encoder-scale intervention, reserving
final cases until a candidate is chosen.

### Notes

For a seven-element spike feature `x`, PyTorch computes `Wx+b` with readout
weight shape `(2,7)`. Autograd differentiates the teacher loss through that
matrix and Adam changes its weights. The 100-neuron graph produces `x` during
forward simulation but does not receive a training gradient. This experiment
does not establish a benefit from real connectome structure.

---

## 2026-09-21 — FB-07D sensory observability diagnosis

### Goal

Measure feature aliasing over locked train/validation trajectories and test one
encoder-scale intervention with the graph and evaluation protocol fixed.

### Changes Made

- Made the synthetic sensory saturation scale explicit with default 0.025;
  carried it through training, learned control and checkpoint validation.
- Added a controlled reversed-odor spike probe, paired same-observation replay,
  exact feature collision counts and a bucket-optimal teacher-turn MSE floor.
- Ran one doubled-scale (0.05) readout attempt and recorded checkpoint, loss
  curve, four matched validation case rows and full JSON metadata under
  `experiments/fb07d-observability/`.
- Added numerical/integration tests and updated plan, decisions, experiment
  summary, queue and status.

### Decisions

- Choose scale 0.05 from train/validation concentration ranges before the
  embodied result. Retain current ceiling 2, all 100 graph weights, seven
  features, 16 readout scalars, teacher, train/validation starts and budgets.
- Use train/validation only. Do not evaluate new final cases or continue
  readout optimization until a weak-signal encoder probe improves observability
  on both splits.

### Tests / Verification

- The new diagnostic tests first failed for missing scale/metric/probe
  behavior. Final focused command: `python -m pytest tests/test_fb07d_diagnosis.py
  tests/test_readout_experiment.py tests/test_readout.py tests/test_neural_network.py
  tests/test_sensing.py -q -p no:cacheprovider`: **46 passed**.
- Final full command: `python -m pytest -q -p no:cacheprovider`: **205 passed
  in 9.41 s**, one existing Pygame `pkg_resources` deprecation warning.
  `python -m compileall -q src tests` exited 0.
- Artifact audit checked four matching validation rows, 81 curve rows,
  checkpoint SHA-256, controlled spike totals, topology hash and exact reload
  replay. Six new focused diagnostic tests passed after the final change.
- On 1,600 train paired ticks, aliasing fell 1,030 to 452 and turn MSE floor
  0.134511 to 0.009461. On 2,400 validation ticks, aliasing rose 1,446 to
  1,589 while the floor changed 0.072211 to 0.070585. Both-antenna saturation
  fell on both splits. All 16 readout scalars had nonzero gradients and changed.
- Validation arrival was 0/4 for wider, original learned and fixed. Mean
  progress: -1.068, -1.286 and -1.688 respectively. No new final episodes.

### Problems / Failed Attempts

- The first focused run had two mistaken test expectations about saturation
  and the intended train probe length; the fixture and expected tick count
  were corrected. The explicit MSE-floor test first failed on a missing field.
- Wider scale helped train observability but not validation feature variety.
  Better progress on this tiny set cannot establish navigation learning.

### Current State

FB-07D is verified as a diagnosis and a negative arrival intervention. M6
navigation learning remains open. Original app and FB-07 behavior still use
scale 0.025. Final starts remain reserved from this intervention.

### Next Step

FB-07E / Sol High: probe one weak-signal sensory representation on the locked
train/validation observation sequences before another readout training run.

### Notes

The two antenna values fill a 100-element PyTorch current tensor. The frozen
`(100,100)` graph advances spikes and yields seven group features. The `(2,7)`
readout alone receives autograd gradients; changing the hand-written encoder
does not make the hard spiking graph trainable.

---

## 2026-09-21 — FB-07E weak-signal encoder probe

### Goal

Test one zero-preserving sensory rule for weak odor readings on the same locked
train and validation observations; permit another learning attempt only if
feature discriminability improves on both splits.

### Changes Made

- Added a rational sensory current option, `2.5*c/(c+0.005)`, while preserving
  the app's default linear 0.025 encoder and all 100 graph weights.
- Added three-way fixed-observation replay with per-case and split-level exact
  feature collisions, teacher-turn MSE floor and weak-input spike counts.
- Recorded config, cases, seed, software/device, gate and 36 case/encoder CSV
  rows under `experiments/fb07e-weak-signal/`.
- Added seven behavior tests, plan, experiment explanation, decision, queue and
  status updates.

### Decisions

- Use one predeclared rational rule with half-saturation 0.005 and current
  asymptote 2.5, both arbitrary synthetic assumptions. Replay 200 ticks per
  train case and 600 per validation case; reset neural and EMA state per case.
- Require fewer aliased ticks than both prior scales, a teacher-turn MSE floor
  no worse than either, and more weak-reading spikes on each split. No readout
  training or final cases in this task regardless of the probe outcome.

### Tests / Verification

- New tests first failed because the FB-07E probe module was absent. Final
  focused command: `python -m pytest tests/test_fb07e_probe.py
  tests/test_fb07d_diagnosis.py tests/test_neural_network.py
  tests/test_sensing.py -q -p no:cacheprovider`: **44 passed**.
- Final full command: `python -m pytest -q -p no:cacheprovider`: **212 passed
  in 10.75 s**, one existing Pygame `pkg_resources` deprecation warning.
  `python -m compileall -q src tests` exited 0.
- Artifact audit checked 36 CSV rows, train/validation case IDs, the recorded
  reversed weak-input spike probe, exact replay of FB-07D original/wider
  metrics, unchanged topology, the failed gate, and absence of training and
  new final evaluation.
- Weak-input spikes: candidate 1,379/1,382 train and 3,632/3,733 validation;
  original 875 and 1,090; wider 0 on both splits. Candidate aliasing was
  627/1,600 train versus best prior 452, and 992/2,400 validation versus
  best prior 1,446. Teacher-turn MSE floor fell to 0.006355 train and
  0.034048 validation.

### Problems / Failed Attempts

- The candidate passed the validation criteria and the train MSE/activity
  criteria but failed the predeclared train alias criterion. Per-case effects
  were mixed: validation-101 worsened and train-2 retained 157 aliased ticks.
  No second encoder was tried and no optimizer run followed.

### Current State

FB-07E is verified as a bounded diagnostic with a failed overall gate. The
default app remains on the original linear encoder. M6 navigation learning is
still unverified; final starts were not used in this probe.

### Next Step

FB-07F / Sol High: locate residual aliasing along encoded current, sensory
spike counts, group means and EMA features, then identify one stage for the
next representation change without touching final starts or the optimizer.

### Notes

The two concentrations become a 100-element current tensor. The fixed
`(100,100)` graph yields spike counts, then seven group features. The rational
formula changes only forward simulation. FB-07E invokes no autograd or Adam,
so it demonstrates sensory information changes, not learning.

---

## 2026-09-22 — FB-07F collision-stage diagnosis

### Goal

Locate where distinct sensory readings that share a final seven-feature
vector collapse along the fixed train/validation neural pathway.

### Changes Made

- Added exact per-tick traces for bilateral encoded current, 20 sensory
  counts, all 100 counts, seven group means and seven EMA features.
- Added per-stage collision and teacher-turn floor measurements plus a
  snapshot-pair classification for conflicting final features.
- Recorded config, starts, software/device, per-case metrics and 36 CSV rows
  under `experiments/fb07f-collision-stages/`.
- Added four numerical/integration tests, plan, experiment explanation,
  decision, queue and status updates.

### Decisions

- Replay only locked train/validation original-controller observations, using
  the same 200/600 tick lengths and three encoders as FB-07E. Include all 100
  spike counts to distinguish group averaging from earlier quantization.
- Interpret pair labels as the latest earlier differing snapshot, not as a
  causal proof; hidden voltage/prior-spike state can alter stage values.
- Target the 20 ms spike-count boundary next with a read-only two-voltage
  feature probe. No training, controller change or final-case evaluation here.

### Tests / Verification

- New tests first failed because the FB-07F module was absent. Final focused
  command: `python -m pytest tests/test_fb07f_collision.py
  tests/test_fb07e_probe.py tests/test_neural_network.py
  -q -p no:cacheprovider`: **30 passed**.
- Full command: `python -m pytest -q -p no:cacheprovider`: **216 passed in
  9.86 s**, one existing Pygame `pkg_resources` deprecation warning.
  `python -m compileall -q src tests` exited 0.
- Artifact audit checked 36 CSV rows, exact FB-07E EMA ticks/unique vectors/
  alias counts/turn floors for every split and encoder, pair totals, unchanged
  topology and absence of training/new final evaluation.
- Rational encoder: zero current aliases among 1,592 distinct train and 1,961
  distinct validation raw pairs. All 3,402 train and 4,392 validation final
  conflicting pairs had distinct current but identical all-neuron counts and
  group means. Train-2 had 157 EMA alias ticks versus 34 with wider linear;
  validation-101 had 416 versus 380.

### Problems / Failed Attempts

- Stage alias-tick counts were nonmonotonic: EMA history distinguished some
  ticks whose instantaneous count vectors matched. The interpretation was
  narrowed to final conflicting pairs, with temporal-state limits explicit.
- No encoder or feature intervention was attempted in this task, so there is
  no new arrival or learning result.

### Current State

FB-07F is verified as a bounded diagnosis. The measured rational-encoder
final collisions occur at whole-interval spike-count resolution. M6 navigation
learning remains unverified; default app behavior and final starts are intact.

### Next Step

FB-07G / Sol High: probe two sensory membrane-voltage summaries beside seven
EMA spike features on locked train/validation observations before training.

### Notes

PyTorch produces a 100-element current tensor and 100-neuron spike counts.
Seven group means and EMA features are hand-written readout summaries. This
diagnostic only observes the forward state; autograd and Adam are unused.

---

## 2026-09-22 — FB-07G sensory voltage feature probe

### Goal

Test whether two post-interval sensory membrane-voltage means preserve
readout information lost by 20 ms spike counts on locked train/validation
observations, before any new learning run.

### Changes Made

- Added left/right post-interval voltage means to diagnostic traces and a
  nine-value feature comparison beside the seven spike EMA values.
- Added exact collision, teacher-turn floor and conflicting-pair resolution
  measurements, a split-wise gate, 12-row CSV and JSON results.
- Added four behavior tests and updated plan, experiment, decision, queue and
  status records.

### Decisions

- Replay only the rational encoder on the same locked fixed-controller
  trajectories; use two ten-neuron sensory voltage means from the neural
  snapshot. Keep graph, default app controller, timing and final starts fixed.
- Require lower alias ticks, lower teacher-turn floor and at least one
  resolved EMA conflict on each split. Do not interpret an information-gate
  pass as navigation learning.

### Tests / Verification

- New test module initially failed collection because the FB-07G module was
  absent. Focused command: `python -m pytest tests/test_fb07g_voltage.py
  tests/test_fb07f_collision.py -q -p no:cacheprovider`: **8 passed**.
- Full command: `python -m pytest -q -p no:cacheprovider`: **220 passed in
  9.84 s**, with one existing Pygame `pkg_resources` deprecation warning.
  `python -m compileall -q src tests` exited 0.
- Artifact audit checked 12 CSV rows, exact FB-07F seven-EMA and conflicting
  pair metrics on both splits, fixed topology, passed gate and absence of
  training/new final evaluation.
- Train: aliases 627 → 18/1,600; turn floor 0.006354624 → 0.000549582;
  3,389/3,402 conflicting EMA pairs resolved by voltage. Validation: aliases
  992 → 46/2,400; floor 0.034047927 → 0.002997121; 4,259/4,392 resolved.

### Problems / Failed Attempts

- None beyond the expected missing-module test failure. Exact float32
  discrimination may be fragile and was not tested under noise or unseen
  trajectories.

### Current State

FB-07G's information gate passed. No readout was trained, no embodied
evaluation was run and M6 food-arrival improvement remains unverified.

### Next Step

Obtain the user's choice for a bounded nine-feature learning/evaluation
experiment before proceeding toward FB-08, whose M6 improvement gate is open.

### Notes

PyTorch's 100-element post-interval voltage tensor supplies two ten-neuron
means. These are concatenated to seven hand-filtered spike features in a
diagnostic; neither autograd nor an optimizer runs. The synthetic graph and
LIF rules remain hand-written assumptions.

---

## 2026-09-22 — FB-07H matched voltage readout learning test

### Goal

Determine whether the FB-07G information gain yields better held-out food
arrival with a small trainable readout before advancing to FB-08.

### Changes Made

- Added an online nine-value feature filter and generalized the linear readout
  to seven or nine inputs while preserving the original default.
- Added paired train-only sample collection, two bounded readout runs,
  feature-specific checkpoints, reload checks, validation gate and JSON/CSV/
  curve artifacts under `experiments/fb07h-matched-readout/`.
- Added six FB-07H behavior tests, one legacy-checkpoint regression test, plan,
  experiment explanation, decision, queue and status updates.

### Decisions

- Keep graph, rational encoder, original fixed-controller trajectories,
  sensory-only teacher, 80-step Adam budget, LR 0.05 and locked cases fixed.
- Train both seven- and nine-feature models as a paired feature comparison.
  Require nine to beat both controls in validation arrival and fixed in mean
  progress before opening final starts. The readouts differ in capacity.

### Tests / Verification

- Tests first failed because `HybridFeatureFilter` was absent. A symmetric
  test fixture cancelled one turn-bias gradient; a direct gradient inspection
  confirmed the cause, then the fixture was corrected.
- The original checkpoint test first showed it accepted a nine-value model
  under a seven-value contract; the save path now rejects that mismatch.
- Focused command: `python -m pytest tests/test_fb07h_training.py
  tests/test_readout_experiment.py tests/test_readout.py tests/test_fb07g_voltage.py
  -q -p no:cacheprovider`: **20 passed**.
- Full command: `python -m pytest -q -p no:cacheprovider`: **227 passed in
  10.13 s**, with one existing Pygame `pkg_resources` deprecation warning.
  `python -m compileall -q src tests` exited 0.
- Artifact audit checked two 81-point curves, checkpoint SHA-256 hashes,
  exact reload replays, fixed baseline validation rows, topology, 1,600 train
  samples and absence of new final evaluation. An initial exact PowerShell
  float comparison differed by 1.4e-17 after JSON parsing; a 1e-15 tolerance
  passed without changing the stored artifacts.
- All 16/20 readout scalars had nonzero first-step gradients and changed.
  Validation arrival: fixed 0/4, seven 0/4, nine 0/4. Nine mean progress
  -1.770 versus fixed -1.688; validation gate failed.

### Problems / Failed Attempts

- Extra voltage information and slightly lower imitation loss did not improve
  validation arrival under this teacher/readout. The result does not isolate
  teacher quality, motor policy, or wall trapping as the primary cause.

### Current State

FB-07H is a verified negative validation learning result. Default app behavior
and topology are unchanged; new final starts remain sealed. M6 improvement and
FB-08's prerequisite remain open.

### Next Step

FB-07I / Sol High: a read-only embodied failure diagnosis on locked
train/validation trajectories, without retraining or final evaluation.

### Notes

For each `(1600,9)` batch, PyTorch multiplies features by a `(2,9)` readout
matrix. Autograd and Adam affect that matrix and its two biases only. The
synthetic graph and LIF dynamics are fixed hand-written assumptions.

---

## 2026-09-22 — FB-07I embodied failure diagnosis

### Goal

Locate teacher, learned-command, early retreat and wall-contact patterns
behind FB-07H's zero validation arrivals before changing the learning rule.

### Changes Made

- Added scoring-only per-tick replay for fixed and saved seven/nine controllers
  on locked train/validation starts, with pre-movement sensory teacher turn,
  source-bearing error, command, distance change and wall contact.
- Recorded 11,827 tick CSV rows and 36 case/controller summaries in JSON.
- Added five behavior tests, plan, experiment explanation, decision, queue and
  status updates.

### Decisions

- Treat source-bearing sign as a scoring diagnostic, never a controller input
  or optimal-action label. Compare signs only for bearing magnitude >0.1 radian
  and turn magnitude >0.05; report eligible denominators.
- Replay train 200 ticks and validation 600 ticks per case without retraining,
  controller change or final-case evaluation.

### Tests / Verification

- New test module initially failed collection because the FB-07I module was
  absent. Focused command: `python -m pytest tests/test_fb07i_diagnosis.py
  tests/test_fb07h_training.py tests/test_readout_experiment.py
  -q -p no:cacheprovider`: **18 passed**.
- Full command: `python -m pytest -q -p no:cacheprovider`: **232 passed in
  11.34 s**, with one existing Pygame `pkg_resources` deprecation warning.
  `python -m compileall -q src tests` exited 0.
- Artifact audit checked 11,827 CSV rows, 36 case summaries, exact FB-07H
  validation endpoint replay, all 12 pre-wall retreats, and no training/final
  evaluation.
- Validation nine-feature teacher sign matched bearing on 2,091/2,091
  eligible ticks; learned command on 1,564/1,915. Wall contacts occurred on
  1,112/2,400 ticks; all 4/4 validation episodes timed out. The fixed and
  seven-feature controls also retreated before first wall contact on 4/4 each.

### Problems / Failed Attempts

- No implementation failure beyond the expected missing-module test failure.
  The observed sign agreements cannot identify causal responsibility for
  turning strength, forward motion or wall geometry.

### Current State

FB-07I is a verified descriptive diagnosis. It did not change the controller,
graph or checkpoint; M6 improvement and FB-08's prerequisite remain open.

### Next Step

FB-07J / Sol High: one matched, bounded motor turn-gain intervention on train
and validation using sensory-only commands, with a gate before final starts.

### Notes

Source coordinates appear only in diagnostic scoring after each controller
decision. PyTorch forward activity and saved readout weights are unchanged;
this task runs no autograd or optimizer.

---

## 2026-09-22 — FB-07J bounded motor turn-gain intervention

### Goal

Test whether one post-decoder turn gain improves navigation after FB-07I's
early-retreat finding, while comparing learned and fixed controllers fairly.

### Changes Made

- Added a sensory-only motor wrapper multiplying decoded turn by 2.0 and
  clamping it, preserving forward output and decoder reset.
- Ran matched original/gain-2 fixed and nine-feature controllers on train and
  validation; a predeclared validation gate opened one final comparison.
- Recorded 8/4/8 matched case rows and JSON summaries under
  `experiments/fb07j-turn-gain/`, plus four behavior tests, plan, experiment,
  decision, queue and status updates.

### Decisions

- One gain of 2.0 was chosen before evaluation. Both fixed and learned
  controllers received the same wrapper; graph, weights, teacher and app
  defaults stayed fixed. No optimizer or retraining ran.
- Final starts opened once only after gain-2 nine beat gain-1 nine and gain-2
  fixed in validation arrivals and mean progress. Primary M6 success remains
  higher held-out arrival than a matched fixed controller.

### Tests / Verification

- New tests first failed because the FB-07J module was absent. Focused command:
  `python -m pytest tests/test_fb07j_gain.py tests/test_fb07i_diagnosis.py
  tests/test_fb07h_training.py -q -p no:cacheprovider`: **15 passed**.
- Full command: `python -m pytest -q -p no:cacheprovider`: **236 passed in
  12.88 s**, with one existing Pygame `pkg_resources` deprecation warning.
  `python -m compileall -q src tests` exited 0.
- Artifact audit matched 8 train, 4 validation and 8 final CSV/JSON rows,
  original fixed and nine validation outcomes, recorded final fixed baseline,
  topology and absence of training.
- Validation arrival: gain-2 nine 1/4 versus gain-1 nine 0/4 and gain-2 fixed
  0/4. One final run: gain-2 nine 1/8, gain-2 fixed 1/8 and original fixed 1/8.
  Final mean progress: -0.895 versus -2.312 and -3.021 respectively.

### Problems / Failed Attempts

- The validation arrival gain did not produce a higher final arrival count.
  Further gain tuning on these now-read final starts would invalidate the
  intended held-out comparison.

### Current State

FB-07J is a verified bounded intervention, with a positive validation result
and a negative primary final result. M6 and FB-08's prerequisite remain open.

### Next Step

Choose a new bounded M6 learning/control objective and predeclare fresh
held-out starts before another intervention. A small reward-based readout is
the recommended direction; confirm its scientific scope and training budget.

### Notes

The 2.0 gain is hand-written actuator scaling after PyTorch's readout, not a
learned synapse or evidence of biological motor gain. It does not change
neural state advancement or consume simulation randomness.

---

## 2026-09-22 — FB-07K reward-based protocol proposal

### Goal

Prepare a concrete next M6 learning route after repeated sensory-teacher and
motor-gain experiments failed to improve primary final arrival.

### Changes Made

- Added `docs/plans/FB-07K-PROPOSAL.md` with one bounded reward-based readout
  method, a fresh disjoint split, exact training budget and evaluation gate.
- Linked the proposed task from queue and status. No code or training ran.

### Decisions

- Recommend score-function policy gradient on the frozen graph's nine-feature
  readout, initialized from FB-07H, rather than further tuning the used final
  split. Keep controller input sensory-only and coordinates in reward/scoring.
- Request user confirmation before the 128-episode run because this changes
  the scientific method and exceeds a routine short probe.

### Tests / Verification

- Reviewed the proposal against `docs/architecture.md` and the FB-07J result;
  linked primary PyTorch documentation and the original REINFORCE paper.
- No implementation test or experiment applies yet. The last FB-07J full
  suite remains **236 passed** with one existing Pygame warning.

### Problems / Failed Attempts

- None in the proposal. The proposed policy-gradient result is unmeasured;
  the single-run budget may still be insufficient for reliable arrival.

### Current State

FB-07K is proposed only. M6 and FB-08's improvement prerequisite remain open.

### Next Step

Wait for user confirmation of the concrete FB-07K protocol and 128-episode
maximum training budget before implementation or running it.

### Notes

The proposal uses action log probabilities for gradients and keeps the hard
spiking graph outside autograd. It does not claim biological learning.

---

## 2026-09-22 — Project recovery review after FB-07 scope drift

### Goal

Review the project against the original goals, correct the expanding FB-07
sequence, and restore one bounded next task as requested by the user.

### Changes Made

- Added `docs/plans/M6-RECOVERY.md` with the evidence, corrections, next-task
  budget and stopping conditions. Closed automatic letter expansion at J;
  superseded the unimplemented K reward-training proposal.
- Added a training-only reference reproduction and JSON/CSV results under
  `experiments/m6-recovery-review/`. Evaluated the unchanged sensory teacher
  and compared saved H readouts with a constant train-target-mean predictor.
- Replaced the long historical status recap with a current, compact handoff.
  Updated queue, README, architecture, decisions, experiment interpretation
  and stale workflow text that still said no simulator existed.
- Added operational rules against automatic experiment expansion and reused
  final-set claims, plus teacher, constant-fit and matched learning controls.
  Historical logs, checkpoints and prior experiment outputs are preserved.

### Decisions

- Preserve the working world and neural loop. There is no demonstrated need
  to change physics, add neurons or start reward learning in this review.
- The next task is one M6 readout fitting recovery: at most one evidence-based
  correction and 500 optimizer steps, with explicit fit/arrival gates. A
  failed gate ends that task and requires user input before another approach.
- Mark seeds 200-207 as a historical benchmark, since FB-07 and FB-07J both
  evaluated them. Future confirmation needs fresh cases and a frozen-untrained
  version of the same candidate alongside the original fixed baseline.

### Tests / Verification

- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  236 passed in 14.45 seconds; one existing Pygame deprecation warning.
- `.\.venv\Scripts\python.exe -m pytest tests/test_evaluation.py tests/test_readout.py tests/test_fb07h_training.py -q -p no:cacheprovider`:
  15 passed in 3.30 seconds.
- `.\.venv\Scripts\python.exe -m flybrain.app --headless --steps 120 --seed 7`
  and the same command with `--control neural`: 120 ticks each, exit 0.
  These are headless smoke checks; no live visual review was performed.
- `.\.venv\Scripts\python.exe experiments/m6-recovery-review/reproduce.py`:
  teacher reached 8/8 training cases, averaging 2.52 seconds. Fixed train-0
  replay and source checkpoint hashes matched; reference command/reset/output
  gate checks passed. No optimizer or validation/final episodes ran.
- Saved nine turn MSE 0.157941 versus constant 0.167516, a 5.72% reduction.
  Seven turn MSE 0.157916. About 79% of nine's combined loss reduction came
  from fitting constant forward speed. Combined losses reproduced within
  1e-7 (float32 reduction-order tolerance).
- Audit CSV/JSON values and case IDs agree. Syntax, overwrite protection,
  unchanged result hash, local documentation links and required status
  sections passed. Single-agent review against the original brief and code.

### Problems / Failed Attempts

- Found process drift: each unsuccessful intervention produced another task,
  information gates did not establish useful linear fitting, and the K
  proposal inferred a need to change objectives without testing the teacher.
- Intermediate "sealed" language overstated the novelty of old final cases.
  Current records now explicitly correct that interpretation.
- One text search used a Windows-incompatible glob path. Two documentation
  patches were rejected before mutation (a mismatched heading, then a
  duplicate delete/add path). Corrected the glob/patch approach and verified
  the resulting files; no production bug or experiment failure occurred.

### Current State

The review and roadmap correction are complete. The embodied prototype works;
M6 navigation improvement remains unverified. The teacher's train success
localizes an imitation gap but does not isolate optimizer, feature capacity
or closed-loop data shift, and it is not a learned or held-out success.
No production source, app behavior, model weights or neural topology changed.

### Next Step

M6 readout recovery / Sol High, one agent, following `docs/plans/M6-RECOVERY.md`.
Do not resume K, add another FB-07 letter, reuse old final cases for a fresh
claim, or automatically deploy a candidate to the app.

### Notes

A neural readout can lower loss mostly by learning a constant forward speed.
Separate turn error and a constant predictor expose whether steering is
actually being learned. New diagnostic control is explicitly a sensory-rule
bypass, retained only as a reference. No Git repository exists; no commit,
push, new dependency or long training run was performed.

---

## 2026-09-22 — Bounded M6 readout recovery fit

### Goal

Diagnose the existing nine-feature teacher-turn fit and make at most one
evidence-based correction within 500 full-batch updates, then apply the
declared training gates.

### Changes Made

- Predeclared train-only per-feature standardization and algebraic folding
  into the unchanged nine-input readout. Added its bounded runner and two
  numerical behavior tests.
- Recorded one run under `experiments/m6-readout-recovery/`: JSON, 501 curve
  points, eight training episode rows and a separate checkpoint. Preserved
  prior experiment files and the saved H checkpoint.
- Updated recovery, experiment, decision, queue and status records.

### Decisions

- Used exactly one fitting correction and the full 500-update ceiling at
  Adam learning rate 0.05. The unchanged teacher, rational encoder, fixed
  topology and gain-1 training starts remained the only scientific inputs.
- Both training gates failed. Closed this recovery fit and stopped before
  validation, final evaluation, another model change or app promotion.

### Tests / Verification

- The 1,600-row feature matrix plus bias had full rank; its condition ratio
  was about 73 raw and 11.5 scaled. Saved turn gradients were nonzero. An
  affine train-only diagnostic reached turn MSE about 0.15539; this is not a
  mathematical lower bound for the tanh readout.
- The run used exactly 500 optimizer updates. Forward MSE was 0; turn MSE
  was 0.155287 against constant 0.167516 and required 0.016752. Training
  arrivals were 1/8 against required 7/8. The gate failed.
- All 20 intended first-step gradients were nonzero and finite; all 20
  parameters changed. Frozen topology and exact reload replay checks passed.
  The recorded checkpoint hash matched its file; old H checkpoint hash still
  matched its earlier record. No validation/final cases were run.
- `.\.venv\Scripts\python.exe -m pytest tests/test_m6_recovery.py -q -p no:cacheprovider`:
  2 passed. Combined focused evaluation/readout/H/recovery command: 17 passed.
  `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`: 238 passed,
  one existing Pygame `pkg_resources` deprecation warning. No fresh app
  smoke or visual review; app behavior was not changed.

### Problems / Failed Attempts

The one numerical correction improved conditioning and turn error only
slightly; it did not approach either training gate. No second fitting attempt
was made. The result does not distinguish limited feature information from
readout capacity or closed-loop data shift.

### Current State

M6 learning remains unverified. The bounded recovery experiment is complete
as a negative result. The original fixed app controller and all historical
experiment artifacts remain intact. No Git repository exists.

### Next Step

Recovery decision / Sol High: ask the user whether to investigate the
existing feature/readout information gap or pause M6. Do not proceed on a new
approach without their response.

### Notes

PyTorch autograd found a gradient and Adam changed weights, but those updates
did not make the turn output imitate the teacher well enough for navigation.
The conditioning transform was folded into the same raw-input linear map;
the hand-written neuron rules and synthetic graph did not learn.

---

## 2026-09-23 — Astra neural charge repair and experiment preservation

### Goal

Respond to the user's renewed request to fix steering with Astra and repair
other broken project behavior. Preserve the failed recovery and historical
experiments; retain sensory-only control and stop before an unreviewed fresh
held-out evaluation.

### Changes Made

- Astra identified a reset-phase feature mismatch, then implemented optional
  LIF charge recording, a distinct nine-feature charge decoder and numerical
  tests. The parent reviewed/integrated the changes and built a bounded runner
  with schema-3 trained/untrained checkpoints and matched training controls.
- Kept the seven spike EMAs and replaced two endpoint voltages with independent
  log-charge features measured from actual neural state changes. Recorded the
  scientific limitation that this reconstructs sensory drive.
- On 2026-09-22, ran one newly authorized 500-update fit and saved JSON, curve,
  24 training episode rows and checkpoints under `experiments/m6-charge-repair/`.
- Added early refusal of occupied experiment destinations and exclusive
  checkpoint creation. Existing runners previously overwrote records.
- Prepared a frozen protocol containing 16 fresh validation and 32 fresh
  final cases, matched controllers, metrics, budget and failure stopping rules.
  Updated status, README, architecture, decisions, queue and experiment records.

### Decisions

- The old recovery remained closed. The user explicitly authorized Astra
  repair and broader bug fixes; one new feature correction was predeclared in
  `docs/plans/M6-CHARGE-REPAIR.md`, with no reward pivot or new FB-07 letter.
- Continue to distinguish engineered feature processing and learned motor
  weights from frozen spiking dynamics and biological/topology claims.
- Training gates passed. Leave fresh evaluation unexecuted until protocol
  review, and leave the live app on its original fixed decoder.

### Tests / Verification

- `.\.venv\Scripts\python.exe -m flybrain.training.charge_repair --output-dir experiments/m6-charge-repair`:
  exactly 500 optimizer updates, 1,600 original training samples, turn MSE
  0.001761889 versus constant 0.167515755 (98.95% lower), and forward MSE 0
  at float32 recording precision. Trained arrivals 8/8, untrained 0/8, fixed
  0/8. Trained mean arrival time 2.4975 seconds. Both gates passed.
- All 20 intended first-step gradients were finite/nonzero; all 20 parameters
  changed. Topology hash, exact checkpoint outputs and episode replay passed.
- `.\.venv\Scripts\python.exe -m pytest tests/test_charge_training.py tests/test_charge_readout.py -q -p no:cacheprovider`:
  12 passed in 3.51 seconds. Astra's focused LIF/network/readout/voltage command
  passed 72 tests in 3.58 seconds.
- `.\.venv\Scripts\python.exe -m pytest tests/test_experiment_preservation.py tests/test_evaluation.py tests/test_readout_experiment.py tests/test_fb07h_training.py -q -p no:cacheprovider`:
  30 passed in 3.96 seconds, after 11 new preservation tests reproduced failures.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  261 passed in 12.24 seconds, one existing Pygame deprecation warning.
- `.\.venv\Scripts\python.exe -m flybrain.app --headless --steps 120 --seed 7`
  and the same command with `--control neural`: 120 ticks each, exit 0.
  These are headless checks; no live visual review occurred.
- All 34 historical experiment JSON/CSV/PT files retain aggregate SHA-256
  `55b446e166252e656a499840f0c75831c741c6b0f0fbd9a5277d081d0338bcd9`.
  New candidate paths, data hashes, software/device, cases and checkpoints are
  recorded separately. No new or old held-out episode ran.
- Final audit: curve/checkpoint/JSON/CSV agree; all eight fixed training
  episodes match the historical FB-07J rows exactly. All 48 proposed fresh
  cases are unique, disjoint from the old splits and within arena bounds.
  The frozen evaluation protocol includes source-file hashes and SHA-256
  `3f3f7a9c32eecb7c591265420dfae1c3b0398bfbd2bf9733131524e1af576a71`.

### Problems / Failed Attempts

- The first test runs failed as expected before new APIs and preservation
  guards existed. An agent full-suite check reached the parent's unfinished
  training module import; the final integrated full suite passed.
- A Windows wildcard search and one documentation patch were rejected before
  changes; corrected the search syntax and patch context. No failed scientific
  candidate or second training attempt occurred in this repair.

### Current State

The steering training calibration is repaired and verified. M6 generalization
remains open. The charge representation recovers sensory drive from neural
state and does not establish benefit from downstream connectivity. The live
app and all historical experiments preserve their prior behavior/results.

### Next Step

Fresh matched evaluation / Sol High: review the prepared protocol, then
implement/run it with frozen weights after user approval. Stop on its failed
validation/final gate; no retraining or automatic app promotion.

### Notes

For odor left/right .008/.010, spike reset makes endpoint voltages look like
.97249/.08128. Accounting for charge lost through leak and resets yields
1.53846/1.66667; independent log features are .47000/.69315. The `[2,9]`
motor matrix learned strong opposite signed weights on these two features.
Autograd trained that matrix; it did not train the neuron dynamics or graph.
No Git repository exists, and no commit, push or new dependency was added.

---


## 2026-09-23 — approved fresh matched M6 evaluation

### Goal

Run the user-approved frozen charge-readout protocol, validation first and
final only after its gate passes. Preserve checkpoints and earlier experiments.

### Changes Made

- Added `training/charge_evaluation.py`: frozen protocol/source/checkpoint and
  software checks, matched controllers, per-episode journal, CSV/JSON results,
  exact paired tests, validation-before-final gate and one-use receipt.
- Added five behavior/statistical/integrity tests using temporary synthetic
  cases. Reused the explicitly requested Astra agent for read-only review.
- Ran the approved protocol once; updated status, README, architecture, queue,
  workflow, repair plan, decisions and experiment record.

### Decisions

- User approval resolved the prior evaluation pause. No fitting, gain changes,
  checkpoint edits or case replacements were permitted or performed.
- Receipt identity follows the approved protocol hash in its fixed checkpoint
  directory. Copying the same protocol does not allow another evaluation.
- Both gates passed; mark M6 navigation improvement verified only in the
  declared world/start/graph/software setup. Retire these 48 cases from future
  unseen-case claims. Keep the live app on its original fixed controller.

### Tests / Verification

- `.\.venv\Scripts\python.exe -m pytest tests/test_charge_evaluation.py tests/test_evaluation.py -q -p no:cacheprovider`:
  11 passed in 2.44 seconds.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  266 passed in 14.03 seconds; one existing Pygame deprecation warning.
- `.\.venv\Scripts\python.exe -m flybrain.training.charge_evaluation`:
  exit 0; 144 episodes, 62,463 ticks, zero optimizer steps. Validation trained
  16/16, untrained 1/16, fixed 1/16. Final trained 32/32, untrained 1/32,
  fixed 1/32. Both final paired p values 4.656612873e-10; both gates passed.
- Read-only artifact audit: JSON/CSV/journal agree, paired case order matches
  the approved protocol, tick budget matches rows, runner hash matches and all
  frozen source/checkpoint/protocol hashes pass. Loaded weights stayed fixed.
- No new GUI review or app smoke run; the frozen app source is unchanged.
  Prior 120-tick manual/neural headless checks remain historical evidence.

### Problems / Failed Attempts

The initial runner tests failed while its module was absent. Astra then found
that a receipt based on the protocol filename allowed copied-protocol reuse.
A temporary-case regression reproduced that failure; the hash-based receipt
fixed it. No approved case was run before these checks passed. No scientific
gate failed, no experiment was restarted, and no extra training occurred.

### Current State

M6's fresh navigation gate is verified. Final trained mean arrival time was
2.918125 seconds across all 32 successes; mean collision ticks were 5.03125
across all 32 cases. The result supports learned motor decoding on engineered
sensory features, with no downstream topology or biological inference.
Existing experiment results and checkpoints remain preserved.

### Next Step

Optional trained-checkpoint app mode / Sol Medium: load the saved schema-3
candidate through an explicit option; verify reset/timing and visible behavior.
Keep existing manual/fixed modes. This evaluation does not deploy it automatically.

### Notes

The original weakness was the voltage sample after a spike reset: stronger
input could produce a smaller endpoint voltage. Accounting for charge removed
by leak/reset gave the readout a stable pair of inputs. PyTorch trained the
20 output weights/biases; this evaluation only applied those saved values.
The prepared protocol JSON remains immutable, including its historical
preparation label; the separate execution receipt and results record its use.
No Git repository, commit, push, new dependency or reward-learning run.

---

## 2026-09-23 — FB-08 ablation and perturbation experiments

### Goal

Follow the user's request to proceed to the next FB section. Complete FB-08's
reproducible intervention/config/result/plot requirements after M6 passed.

### Changes Made

- Predeclared `docs/plans/FB-08.md`: one hypothesis, ten conditions, sixteen
  new matched starts, maximum 160 episodes / 96,000 ticks and zero updates.
- Added `training/fb08_experiments.py` using the existing evaluator and frozen
  checkpoint. Experimental copies support left/right/bilateral sensory-current
  ablation, zero synapses, sensory-pathway gains 0.5/2 and three constrained
  random wirings. Original production neural sources remain unchanged.
- Added five numerical/behavior test cases covering intact equivalence,
  targeted interventions, rewiring, local RNG/reset, stopped/successful runs
  and checkpoint/output preservation.
- Added optional `plots` dependency; installed Matplotlib 3.11.2 and recorded
  its environment. Generated PNG/SVG from saved results and inspected the PNG.
- Updated project status, queue, README, architecture, decisions, environment,
  workflow, plan and experiment records. Worked locally with one agent.

### Decisions

FB-08 needs the passed M6 gate, not optional app checkpoint integration.
Run the bounded experiment directly; retain the fixed app mode. Do not train,
retune, replace cases, reuse historical finals or infer biological/topology
benefit from unchanged success alone. Random controls preserve each incoming
row's weights/layer, but not source out-degree or lateral wiring blocks.

### Tests / Verification

- `.\.venv\Scripts\python.exe -m pytest tests/test_fb08_experiments.py tests/test_charge_readout.py tests/test_charge_training.py tests/test_evaluation.py -q -p no:cacheprovider`:
  23 passed in 4.52 seconds.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  271 passed in 13.63 seconds; one existing Pygame deprecation warning.
- `.\.venv\Scripts\python.exe -m pip check`: no broken requirements.
- `.\.venv\Scripts\python.exe -m flybrain.training.fb08_experiments`:
  exit 0; 160 episodes, 50,378 ticks, zero optimizer updates. Intact 15/16
  passed the >=13 assay check; each sensory ablation 0/16; all synaptic
  interventions and three random controls 15/16. Hypothesis supported on
  these cases. Case 51003 remained a timeout in intact/synaptic conditions.
- Artifact audit verified matched case order, journal/result agreement,
  CSV row count, tick accounting and all 47 prior experiment hashes. Frozen
  checkpoint/readout/source checks passed. Plot labels/values visually checked.
- No app changes, new app smoke test or live app visual review. No commit
  exists because this workspace has no Git repository.

### Problems / Failed Attempts

Initial tests failed before the module existed. A test then incorrectly
expected tonic forward neurons to fire in the first 20 ms. The LIF threshold
is reached later; the test now checks the measured tonic charge on that first
tick. No neuron dynamics were changed. No scientific run failed or restarted.

### Current State

FB-08 / M7 acceptance is complete. The frozen readout depends on bilateral
sensory drive; downstream wiring interventions changed activity and some
trajectories but caused no arrival losses on sixteen starts. This is not a
claim that connectivity never matters. The intact 15/16 result also records
a remaining failure rather than extending the earlier 32/32 M6 result to
all possible starts. Existing experiment artifacts remain unchanged.

### Next Step

FB-09 source/pathway feasibility decision / Sol High: verify accessible data,
license, neuron IDs, edge direction/signs and sensory/motor endpoint mapping.
Explain how a biological subset would contribute given this readout's weak
dependence on downstream wiring. No biological import has started.

### Notes

For a matrix `W[post, pre]`, halving a 0.12 sensory-to-interneuron weight
makes a prior sensory spike contribute 0.06 voltage units on the next neural
step. It does not halve that sensory neuron's measured charge. That separation
explains why changing synapses and silencing sensory current are different
experiments. PyTorch performed tensor simulation with saved readout weights;
no gradients or optimizer updates were used in FB-08.

---

## 2026-09-24 — FB-09 source and pathway feasibility decision

### Goal

Start FB-09 with the required pre-import decision: choose an accessible,
licensed Drosophila source and verify IDs, direction, structural weights,
sensory inputs and motor endpoints while accounting for FB-08's topology-bypass
result.

### Changes Made

- Added `docs/plans/FB-09.md` with the source comparison, pinned archive
  contract, exact neuron/path/motor IDs, edge semantics, simulator mapping,
  missing biological information and importer gates.
- Selected BANC v888 Harvard Dataverse release 3.0 under CC BY 4.0. Pinned the
  57.6 MB metadata and 305.3 MB v2 edge files by Dataverse file ID, size and
  MD5 rather than depending on the changing Codex view.
- Audited a deterministic 83-neuron subset: 68 bilateral VA2 ORNs, two
  VA2 adPNs, five corridor neurons, two DNa02 steering neurons and six named
  leg motor neurons. Its internal five-synapse-threshold graph has 176 edges,
  6,062 summed synapse counts and 31 retained self-edges.
- Updated status, queue, README, architecture, workflow and decision records.
  No package code, dependency, source dataset or experiment artifact changed.

### Decisions

Use source `pre -> post` as project `W[post,pre]`, preserve `root_888` strings
and provenance, and retain edges with at least five synapses. The first import
will be an unsigned structural baseline `connectome_gain * norm`; the gain and
all LIF dynamics remain explicit synthetic assumptions. Bilateral simulated
odor drives matching VA2 ORNs only. A future topology-dependent decoder must
exclude charge features, use DNa02 asymmetry for steering and label any
motor-neuron forward reduction as an engineered 2D proxy.

### Tests / Verification

- Source audit: both downloaded file sizes and MD5 hashes matched the pinned
  Dataverse records; metadata had 188,508 rows and the edge file 11,752,828.
- Recomputed from the pinned source: left input 31 cells / 1,133 synapses,
  right input 37 / 1,425; all seven declared corridor edges and six DNa02 motor
  edges matched; final subset 83 nodes / 176 edges / 6,062 counts / 31
  self-edges. The audit ended `FB09 AUDIT PASS`.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`: 271 passed in
  17.67 seconds; one existing Pygame `pkg_resources` deprecation warning.
- Research used temporary PyArrow 21.0 outside the project environment. It was
  not added to project dependencies.

### Problems / Failed Attempts

The first broad metadata print was too large and was truncated; focused Arrow
filters produced the auditable rows. The first long edge download session was
interrupted by a user continuation message, but the completed file existed and
its size/MD5 matched before use. No source or project file was corrupted.

### Current State

The source/pathway feasibility gate is complete. BANC provides a structural
route from the project's two sensory channels to experimentally supported
DNa02 steering cells and named motor neurons in one specimen. Functional signs,
membrane parameters, a calibrated odor transfer and a fly-gait decoder are not
in the dataset. No biological graph is loaded by FlyBrain yet.

### Next Step

FB-09 BANC loader/transform / Sol High: implement the pinned contract with
small checked-in fixtures, preserve source IDs and metadata, reproduce the
83-node/176-edge audit, and test orientation, reset, propagation and stability.
Stop before long training or fresh embodied evaluation.

### Notes

For one concrete path, left `VA2_adPN` root `720575941653217813` reaches left
DNa02 `720575941510475536` through edge counts 21, 11, 17 and 12. In a tensor,
the first of these becomes a value at the row for postsynaptic
`720575941539370133` and the column for the VA2 projection neuron. PyTorch will
perform that matrix-vector multiplication later; BANC supplies the directed
count, while our LIF scale and dynamics remain hand-written.

---

## 2026-09-24 — FB-09 pinned subset import and neural dynamics diagnostic

### Goal

Continue FB-09 by connecting the downloaded BANC subset to FlyBrain's PyTorch
neural system and establish a bounded gate before training.

### Changes Made

- Added an optional PyArrow extractor that checks both full-source sizes and
  MD5 hashes and regenerates the 90,011-byte packaged JSON subset. Its SHA-256
  is `76c254e22741230b3d97aeb21c9938405b2e98c07ca29b8cde9331f3c4e6d0ae`.
- Added the validated `flybrain.data.banc` loader/transform and `BancNetwork`.
  The latter uses the existing LIF population and bilateral sensory encoder.
  Added 17 behavior tests and package-data configuration.
- Corrected the earlier shortest-path wording. The artifact uses the exact
  already audited corridor IDs, whose directed edges are reverified.
- Recorded the two-candidate dynamics probe in
  `experiments/fb09-gain-probe/results.json` and summarized it in
  `docs/experiments.md`.

### Decisions

Retain source `count` and `norm` without renormalizing over selected cells.
The initial matrix is unsigned `gain * norm`, including 31 self-edges and 62
cross-side edges. No candidate gain was adopted for behavior. The live app
continues to use its synthetic graph and fixed decoder while the biological
dynamics gate remains open.

### Tests / Verification

- The extractor reproduced 83 nodes, 176 edges, 6,062 summed synapses and 31
  self-edges from the pinned full archive. Two regenerations gave the same
  artifact SHA-256.
- `tests/test_banc_import.py`: 17 passed; source orientation, malformed rows,
  real artifact audit, finite stepping and exact reset replay are covered.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`: 288 passed in
  13.83 seconds, with one existing Pygame `pkg_resources` deprecation warning.
  Both JSON files parse, all edited local links resolve, and the subset hash
  matches the regenerated output.
- Dynamics diagnostic: 100 steps of left odor 0.025 and 100 silent steps at
  gains 1 and 600. Gain 1 caused 217 left ORN spikes but zero projection or
  motor spikes. Gain 600 caused left and right DNa02 spikes (83 each) and
  motor spikes (246 each); silent-interval activity persisted. All voltages
  stayed finite. Both candidates failed selectivity/washout.

### Problems / Failed Attempts

The first extractor run failed its path assertion: unrestricted strongest-first
BFS found another four-hop left path with stronger source edges. Source rows
confirmed the discrepancy was in the previous selection description, not a
corrupt download. One correction pinned the already audited path explicitly;
the following extraction and tests passed. The gain probe itself failed its
scientific gate and was closed after the two declared candidates.

### Current State

The BANC source has been downloaded; the small subset is checked into the
package and runs in the neural simulator. The full 362.8 MB Feather files
remain in temporary research storage outside the repository. The biological
graph is not used by the live fly controller, and no biological-model training
has started.

### Next Step

FB-09 neural dynamics recovery / Sol High: predeclare a single bounded
mapping hypothesis, budget and stop condition to address cross-side and
persistent firing while preserving provenance. Require bilateral selective
propagation and washout, then build a topology-dependent motor readout and
check teacher behavior and a train-only constant-command control before a
new, separately approved training protocol.

### Notes

`W[post,pre] = gain * norm` preserves matrix direction but does not supply
physiological sign. A gain of 600 multiplies the largest retained self-edge
fraction 0.112903 into a 67.7418 voltage jump, enough to make recurrence a
plausible source of persistent firing. PyTorch performs the matrix-vector
calculation; all gains, currents and membrane rules are our assumptions.

---

## 2026-09-25 — Astra FB-09 normalized recovery, closed negative

### Goal

Repair FB-09 up to readiness for the first bounded FB-10 BANC training run.
The user requested Astra and selected normalized retained-input dynamics and
the existing food-approach operating setup. No training was authorized here.

### Changes Made

- Added strict source artifact/row consistency checks, normalized nonself
  weights and fixed multiscale ORN currents. Original 83-cell/176-edge artifact
  and its IDs/counts remain unchanged; the simulation has 145 effective edges.
- Added eight endpoint-rate features, resettable fixed motor decoding,
  structural interfaces, gated `--control banc`, and a labelled functional view.
- Implemented one capped recovery runner, train-only readiness scaffolding,
  tests, saved protocol/results/CSV/journal and a plot derived only from results.
- Added FB-10's blocked entry contract and updated status/queue/architecture.
- Astra's independent review found a receipt could claim success while pointing
  at the failed result. A regression reproduced this, then loader hardening
  required matching successful evidence, configuration and current-source hashes.

### Decisions

Keep all 62 cross-side edges; contralateral firing is not itself a defect.
Mask 31 self-edges only in simulation; use count-normalized retained-input rows.
Budgets are exactly 1.25, 2 and 3; encoder scales are fixed at 0.005-0.05.
Do not narrow the high-concentration gate or append a new sweep after failure.
No verified app config is published. Training waits for dynamics, causality,
teacher-feature readiness and accepted-controller app checks.

### Tests / Verification

- `.venv\Scripts\python.exe -m pytest tests/test_banc_recovery.py tests/test_banc_app.py tests/test_banc_readiness.py -q -p no:cacheprovider`:
  27 passed, one existing Pygame warning (7.78 s).
- `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  319 passed, one existing Pygame warning (61.15 s), after the receipt fix.
- `.venv\Scripts\python.exe -m flybrain.app --headless --steps 120 --seed 7 --control manual`
  and the same command with `--control neural`: both complete 120 ticks.
- `--headless --steps 1 --seed 7 --control banc`: deliberately refuses missing
  verified dynamics. No accepted-config BANC smoke or live visual review ran.
- `.venv\Scripts\python.exe -m flybrain.training.banc_recovery`:
  closed failed gate after exactly 72,000 neural substeps and zero optimizer
  updates. All three candidates ran once; no conditional causal cases ran.
- `.venv\Scripts\python.exe scripts/plot_banc_recovery.py`: generated the saved
  contrast PNG from JSON. The three panels, signed responses and thresholds
  were visually inspected; this was not an app animation review.
- Independently rehashed all 60 prior experiment files: zero mismatches.
  Source subset SHA-256 remains
  `76c254e22741230b3d97aeb21c9938405b2e98c07ca29b8cde9331f3c4e6d0ae`.

### Problems / Failed Attempts

Budget 1.25 failed low-band unilateral propagation. Budgets 2 and 3 passed all
individual pulse, rate, finite-state and washout gates but failed the full
contrast gate. Budget 3 passed centers 0.025 and 0.1, then gave zero right-strong
shift at center 0.25 (required 6.64 Hz) and left-strong shift 4 Hz (required
6.56 Hz). Budget 2 high-band right-strong shift was reversed at -6.667 Hz.
No accepted config, causal evidence, teacher record or BANC training followed.

Expected test-first failures identified missing view/app APIs, a sensory-field
name mismatch and the receipt bug; focused fixes passed. Astra's implementation
tests also caught ambient-device allocation and pinned CPU tensors explicitly.
The independent reviewer reported the receipt issue and checked saved contrast
arithmetic, but hit a usage limit before delivering a full verdict. Root
completed local review and regression verification; no full independent signoff
is claimed. No retrying reviewer loop was started.

### Current State

Downloaded BANC data is connected to PyTorch neural computation, not accepted
live behavior. The normalized experiment fixed persistent activity but did not
establish reliable high-band directional information. FB-09 remains open and
FB-10 blocked. All scientific records are preserved, no process remains active,
and no commit/push was made (this directory is not a Git repository).

### Next Step

Obtain approval for one revised high-concentration encoding hypothesis and
fixed budget, keeping the original concentration/contrast gates. First isolate
whether the observed plateau is caused by encoder compression or downstream
spike quantization. Do not rerun the closed candidates. Recommended effort:
Sol High; Astra only if requested. FB-10's entry contract is prepared.

### Notes

Two endpoint spikes in a 0.02-second tick give 100 Hz. Starting from zero,
`alpha = 1-exp(-0.02/0.1)` turns that into an 18.13 Hz filtered feature.
PyTorch performs those tensor calculations; this is forward simulation, not
learning. A future optimizer must actually change readout parameters and beat
matched controls before we can claim improvement. More spikes alone do not
guarantee more usable left/right information.

---

## 2026-09-25 — follow-up saved-count diagnosis

### Goal

Continue narrowing the remaining FB-09 failure without reopening the failed
candidate sweep or starting training.

### Changes Made

Analyzed existing per-group/per-neuron counts and source edge weights. Added
`docs/plans/FB-09-CONTRAST-DIAGNOSIS.md` and linked the evidence from status and
the experiment summary. No production code changed.

### Decisions

Do not assume an encoder-only fix. Right-strong high-band input changes ORN
counts but not either PN count; the first observed aggregate-count plateau is
at PN output. Propose one instrumented 4,500-step replay with unchanged
parameters to distinguish timing/leak/reset effects from compressed drive.
This diagnostic proposal is not a new accepted candidate and has not run.

### Tests / Verification

Read saved JSON with PowerShell, compare equal/right-strong/left-strong counts,
and calculate weighted presynaptic count sums from the 176-edge source
artifact with self-edges excluded. The result SHA-256 is
`8cfa725e30510469a38bf7b56d312967f0207ef3dd3a654bd07646f4980a5311`.
No new neural steps or optimizer updates. No tests rerun for this documentation
and saved-data analysis; the prior 319-test result remains the latest suite.

### Problems / Failed Attempts

One PowerShell formatting command had an invalid pipe after a foreach block;
collecting its output before piping corrected it. No file was changed by that
command. Aggregate counts do not reveal spike timing or prove the exact cause.

### Current State

FB-10 remains blocked. Source data remains downloaded and connected to the
neural core; live BANC behavior is gated and BANC training has not started.

### Next Step

Approve or decline the exact 4,500-step instrumented diagnostic proposal.
Stop after its three fixed cases; no automatic tuning or expanded sweep.

### Notes

Different sensory spike counts can still produce the same downstream counts.
This is a measured information bottleneck, not evidence of a corrupt download.

---

## 2026-09-26 — approved diagnostic and architectural recovery in progress

### Goal

The user approved continued repair through full FB-10 readiness, including the
4,500-step diagnostic. Keep the goal active until every entry gate is verified.

### Changes Made

Implemented/tested observation-only PN tracing, an explicit wider-current
encoder variant, a PN-only area-preserving temporal-filter variant, and a
phase-distributed sensory rate encoder. Historical defaults remain available;
no failed variant is enabled as a live app default. Added separate plans and
protected result directories for each bounded experiment.

### Decisions

The diagnostic exactly reproduced all83 counts and identical PN spike times
for equal/right-strong input, identifying loss through integration/reset with
compressed input. Astra recommended one wider scale range .005-.375; it failed
contrast. One5ms PN filter also failed, including weak-left propagation. Both
are closed without scale/tau searches. The next architecture-level hypothesis
uses independent rate=min(96,176*sqrt(c)) and distributed fractional phases,
with unchanged145-edge topology, hard-reset LIF and endpoint decoder.

A pre-run guard caught an actual ORN-to-ORN edge (count5) in the source. It is
preserved. The recipient can receive additional spikes beyond external input.
All-synapse ablation now requires byte-identical external drive and downstream
silence; an added downstream-only cut preserves ORN wiring/counts. This corrects
a structural precondition before any causal run, rather than changing a failed
result. The representation assay has24,000steps plus6,000conditional control
steps. No rate coefficient/cap/exponent/phase sweep is authorized by its plan.

### Tests / Verification

Observer tests:2pass; exact diagnostic count replay on all3cases/4,500steps.
Wider encoder/recovery tests:23pass; then full suite322pass. PN filter/LIF tests:
61pass; then full suite325pass (90.84s). Latest rate/recovery/filter tests:
28pass (10.19s), including real retained ORN-edge propagation, CPU ownership,
rate-count bounds, phase reset and pure previews. Existing Pygame warning only.
Scientific runs preserve code and previous-file hashes and use zero updates.

### Problems / Failed Attempts

Wider encoder:24,000steps, all pulse gates pass but .1/.25 contrast fails.
PN filter:24,000steps, .025-left propagation and .1/.25 contrast fail. No causal
controls ran for either. The first phase-code test exposed the overlooked
source ORN-edge precondition; tests and causal protocol were corrected to honor
the edge, with no source deletion. No representation assay had run at that point.

### Current State

The one phase-rate assay is running in `experiments/fb09-phase-rate-repair`.
No accepted receipt, BANC teacher-data run or training yet. All prior failed
experiments remain intact. No commit/push; no Git repository exists.

### Next Step

Collect this specific live run, inspect each gate and conditional intervention.
If it passes, finish readiness/app/visual/source verification and full review.
If it fails, close it without coefficient tuning or claiming FB-10 readiness.

### Notes

This rate encoder is explicitly hand-written sensory scaffolding. The graph
must still carry its spikes to downstream endpoints. Running tensor operations
or passing a dynamics test is not learning; optimizer training remains FB-10.

---

## 2026-09-26 — approved PN-only redesign closed; FB-10 still gated

### Goal

Continue the user's Astra-assisted recovery through FB-10 readiness, without
automatic tuning loops. The user explicitly approved PN-only charge retention.

### Changes Made

Implemented ProjectionChargePopulation with PN-only tau25ms, threshold1.1 and
subtractive reset, retaining one spike per millisecond. Added explicit config
and a separately capped --pn-charge runner mode; historical defaults remain.
Strengthened causal checks to compare individual ORNs. Added numeric/reset/CPU
and embodied motor/timing/render tests. Updated status, queue and evidence docs.

### Decisions

Kept source wiring, encoder, row budget, endpoint features and every gate
unchanged. The single assay is closed after failure; no accepted receipt or
training is enabled. Astra found no blocking implementation bug. The exact
mid-right miss is four bias-corrected spikes versus required4.02, not rounding.
Do not tune parameters or replace the declared750ms metric after seeing it.

### Tests / Verification

Six PN tests pass. Full command `.venv/Scripts/python.exe -m pytest -q -p
no:cacheprovider --junitxml=.fb09-verification.xml`:336passed, one upstream
Pygame warning,42.60s. Actual candidate embodied replay,20-substep timing,
motor-off neural continuation and extra-draw trajectory independence pass.

`.venv/Scripts/python.exe -m flybrain.training.banc_recovery --pn-charge
--output-dir experiments/fb09-pn-charge`:24,000steps,0updates,16/16pulse cases
pass,5/6directional checks pass, overall failure. All88prior files and six core
source hashes unchanged. Saved plot generated from JSON and visually inspected.
No causal controls, teacher-readiness run or accepted-config app visual test.

### Problems / Failed Attempts

The preceding phase-rate run closed on high-left5.333Hz<8.8Hz. PN retention
fixes both high-band directions but mid-right is5.333333Hz<5.36Hz. Focused test
fixtures initially omitted a required connectome_gain keyword; corrected the
fixtures, without changing production behavior. A full-suite process handle
was lost across incoming user messages; no scientific assay was restarted.
The final suite emitted a persistent JUnit report and its result was collected.

### Current State

Source subset downloaded and connected; software tests pass; FB-10 not ready.
All scientific runs are closed. No optimizer training, commit or push occurred.

### Next Step

Agree one observation-only PN-to-DNa02 event-timing diagnostic on the existing
equal/right-strong mid-band cases, keeping parameters and gates fixed. Saved
aggregate counts cannot isolate individual relay losses. No run started here.

### Notes

Retaining extra PN voltage can improve contrast without guaranteeing every
downstream spike-count margin. Software correctness and scientific acceptance
are separate. The exact failed gate remains visible rather than rounded away.

---

## 2026-09-26 — approved 3,000-step relay diagnostic complete

### Goal

The user approved the observation-only diagnostic and asked to continue from
saved project state. Explain the remaining miss without further tuning loops.

### Changes Made

Added `training/banc_relay_diagnosis.py` and four tests. The observer calls the
actual network once per step, preserving phase state, and records real mixed-PN
charge losses plus every cell's event timestamps. Added source-derived corridor
matching, fixed-window boundary accounting and previous-result/source checks.
No core neural, encoder, decoder or gate code changed. Updated handoff documents.

### Decisions

Exactly two cases,1,500steps each; no candidate or threshold changes. The common
steady window remains end-steps501..1250. Astra reviewed code before the run and
independently interpreted results. The original failed result remains failed.
No PN events were lost: path delays alone explain the count-window discrepancy.
A separate latency-aligned sustained-rate definition requires explicit approval;
it is not independent confirmation or proof of useful navigation.

### Tests / Verification

`.venv/Scripts/python.exe -m pytest tests/test_banc_relay_diagnosis.py -q -p no:cacheprovider`:
4passed. Three observer tests were observed RED before implementation, then GREEN;
the fourth verifies real path lengths and existing-output protection.

`.venv/Scripts/python.exe -m flybrain.training.banc_relay_diagnosis`:
exactly3,000steps,0updates, all83pulse/steady counts replayed on bothcases;
all94prior files and source/diagnostic hashes preserved. Every corridor edge
transmits with1ms delay; whole paths replay exactly atleft4ms/right3ms.
Charge-balance error <=1.1921e-7. Saved45,000central-neuron transition rows.

`.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --junitxml=.fb09-relay-verification.xml`:
340passed,1upstream Pygame warning,86.48s. No app visuals claimed this session.

### Problems / Failed Attempts

No scientific retry or parameter search. One documentation patch had an
unmatched context line and applied no edits; corrected the patch context.
The broader goal's product status reads usageLimited; it cannot be resumed
through update_goal. Work here followed the user's direct continuation request.

### Current State

Diagnostic complete; source subset remains downloaded/connected. FB-10 still
gated and training unstarted. The six-to-four contrast difference is explained
by specific boundary events, not missing transmission. No code/data deletion,
commit or push. No live process remains after collection of the final suite.

### Next Step

User decision on `docs/plans/FB-09-LATENCY-PROPOSAL.md`: align endpoint counting
windows mechanically to sourced delays, or retain simultaneous-window acceptance.
Do not start a new assay or change measurement definition without that approval.

### Notes

A spike arriving a few milliseconds later can fall outside a fixed counting
window even when transmission is perfect. That distinguishes measurement timing
from a biological or software signal-loss claim; it does not erase the old gate.

---

## 2026-09-26 — FB-09 verified; FB-10 ready and not started

### Goal

The user directed continuation until everything through FB-09 inclusive is
working, then confirmed continuation after the delay-aligned protocol approval
interpretation was stated. Finish this milestone without beginning FB-10 training.

### Changes Made

Added source-derived offline steering delays and fixed aligned counting windows
to the bounded recovery runner. Both simultaneous and aligned results remain
recorded; no neuron, encoder, graph, decoder or live timing changed in this phase.
Packaged the actual successful protocol/result beside a verified config receipt.
Astra hardened teacher-readiness provenance, actual tick/weight accounting,
exclusive artifacts and recorder reset. Added tests and a reproducible static
BANC frame capture script. Updated current status, README, queue, architecture,
decision/experiment records and FB-10 entry handoff.

### Decisions

Accept the explicitly approved sustained-rate contract, not the old simultaneous
window gate. Offline left/right windows follow the source delays4ms/3ms, retaining
750ms duration,1Hz/2% thresholds and all other acceptance gates. The historically
informed definition does not supply independent confirmation or simultaneous-
window robustness. All prior failed experiments remain unchanged and failed.

BANC neurons receive sensory signals only; live output remains the eight actual
endpoint-rate features. Source physiology, assumed dynamics and future learning
remain distinct. Teacher commands are data-collection scaffolding, not features.
Untrained BANC0/8 arrival is a learning baseline, not a completed navigation goal.
No held-out runs, parameter sweep, optimizer update, commit or push occurred.

### Tests / Verification

- Latency tests observed RED before implementation. Targeted command
  `.venv/Scripts/python.exe -m pytest tests/test_banc_latency.py tests/test_banc_recovery.py -q -p no:cacheprovider`:
  24passed. Astra readiness/app tests:23passed, one upstream Pygame warning.
- `.venv/Scripts/python.exe -m flybrain.training.banc_recovery --latency-aligned --output-dir experiments/fb09-latency-aligned`:
  exactly30,000neural steps,0updates; all16pulse cases,6directional checks and
  4causal cuts pass. All98previous files and six frozen core files preserved.
  Original mid-right simultaneous5.333333Hz<5.36Hz remains in the result.
- `.venv/Scripts/python.exe -m flybrain.training.banc_readiness`:
  passed; teacher8/8, fixed BANC0/8;1,008samples;5,808worldticks;0updates.
  All106previous artifacts,45Python source files, bundle and weights unchanged.
  Mean turn0.16428607, constant turn MSE0.08901093, exact-alias lower bound0.00238492.
- `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --junitxml=.fb09-final-verification.xml`:
  **358passed**, no failures/errors/skips, one upstream Pygame warning,22.40s.
- `.venv/Scripts/python.exe -m flybrain.app --headless --steps 120 --seed 7 --control manual`
  and the same command with `neural` and `banc`: each exit0, exactly120ticks.
- `.venv/Scripts/python.exe scripts/capture_banc_frame.py`:120worldticks,
  2,400neural steps,83nodes,80displayed of145effective edges. Inspected actual
  `experiments/fb09-app-verification/banc-frame.png`. Static review only, not
  a manual live interaction test. Saved-data contrast plot also inspected.
- Independent Astra final read-only review found no blocking or important
  findings. Its declined items (actual readiness, visual review, final tests)
  were checked by root afterward. No deferred code findings remain.
- Rechecked both original downloaded Feather archives: expected byte sizes and
  MD5 `6275eda42f98c49539d1ab513d979d09` /
  `394406f8a9bdf093c895f95aff4f6c49`. Source subset SHA256 unchanged.
- Fresh final provenance audit passed; accepted configuration loader succeeds.
  Consolidated record: `experiments/fb09-completion/verification.json`.

### Problems / Failed Attempts

No failed scientific run in this phase. One optional read used `rendering`
instead of the actual `visualization` folder and was corrected. An atomic
documentation patch rejected simultaneous delete/add of the status file; a
single update replaced it without losing history (historical entries remain
in this log). Astra corrected one targeted-test command naming a nonexistent
test file; the valid targeted suite and final full suite were then collected.

### Current State

FB-09 is complete under the approved contract; `--control banc` works with the
verified bundle. The sourced83-cell subset is downloaded and connected, not a
full reconstructed fly brain. No BANC training has started. Readiness odor
range0.023099-1.905076 exceeds the contrast band and the encoder saturates above
about0.29752; these remain explicit learning/generalization limits. The fixed
untrained decoder's poor arrival performance is recorded, not hidden.

### Next Step

FB-10 / Sol High: define one bounded endpoint-readout training protocol using
the preserved readiness data, frozen accepted graph/dynamics/features, train-only
normalization and constant-command baseline. Record exact optimizer budget,
checkpoint/replay gates and matched controls before fitting. Fresh evaluation
requires a separate locked split. Do not launch a long run automatically.

### Notes

A working sourced neural circuit is not a trained controller. PyTorch currently
calculates forward spiking state; FB-10 will introduce gradients and optimizer
updates only in the motor readout. The measured graph-path dependence does not
yet show a navigation advantage or biological fidelity.

---

## 2026-09-26 — FB-10 endpoint-readout protocol preparation

### Goal

Start FB-10's active preparation task: define and implement one bounded
endpoint-only readout attempt from the accepted FB-09 readiness evidence.
Finish the reviewable protocol and software checks before a scientific fit.

### Changes Made

Added `neural/banc_readout.py`: a CPU float32 8->16->2 readout with178 trainable
parameters, fixed train-only normalization, unchanged endpoint EMA composition,
and a separate source/config/feature/normalization/checkpoint contract.
Added `training/banc_protocol.py` to validate the1,008 original samples and
freeze one candidate, plus `training/banc_training.py` for separate single-use
execution, actual gradient/update accounting, matched training controls and
exact save/reload replay. Added38 numerical and integrity tests. Updated
architecture, decisions, task/queue, README, experiment summary and status.

Prepared `experiments/fb10-banc-readout/protocol.json`, its SHA256 sidecar and
`preparation.json`. Protocol SHA256:
`f25f9c528dcba29302aeb2372df6fea23f3ea4e0a8ecab8f881cc580d4c13d05`.
The actual execution receipt, checkpoints, curve and scientific result are
absent. Preserved all110 prior experiment files,45 existing Python sources and
the accepted bundle. Three new Python modules do not change the receipt-bound
spiking core or app decoder.

### Decisions

This session follows the active handoff's first task, protocol preparation.
The scientific fit remains a separately invoked task, and fresh evaluation
requires another locked protocol after passing all training gates.

One hypothesis and candidate: eight endpoint rates,16 tanh hidden units,
sigmoid-forward/tanh-turn output, seed 1010, train population mean/std with a1 Hz
floor, Adam lr 0.01 and exactly500 updates unless numerics fail. Turn error must
beat25% of constant baseline (ceiling0.02225273146561865), with>=7/8 training
arrivals and strict arrival/all-case-progress improvement over both matched
untrained and fixed BANC controls. Core, normalization, provenance and exact
checkpoint replay gates also apply. No automatic sweep/retry or held-out run.

The25-episode maximum includes24 matched training episodes and one replay:
15,000 worldticks /300,000 neuralsteps. Protocol destination binding plus an
exclusive pre-update receipt preserve partial attempts. No model switch or subagent was used.

### Tests / Verification

- Initial `.venv/Scripts/python.exe -m pytest tests/test_banc_readout.py tests/test_banc_training.py -q -p no:cacheprovider`:
 32 failed on missing modules, before implementation. First implementation:
 32 passed. Local-review boundary regressions:3 failed /16 passed before fixes.
- Final identical targeted command: **38 passed**,24.12s. Checks include a
 hand-calculated normalization/matrix example, sensory-only endpoint filtering,
 reset, exact checkpoint outputs, artificial-data gradients/parameter changes,
 nonfinite stop, corrupted readiness and protocol/case/normalization rejection.
 A real two-tick three-control plus replay check accounts for8ticks /160steps;
 it is software verification, not scientific navigation evidence.
- `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --junitxml=.fb10-final-verification.xml`:
 **396 passed**,102.75s, no skips/failures/errors; one existing upstream Pygame
 deprecation warning. No UI change or new visual inspection.
- `.venv/Scripts/python.exe -m flybrain.training.banc_protocol --output-dir experiments/fb10-banc-readout`:
 exit0, **0 experiment updates /0 worldticks**. Frozen protocol reload validates
 actual data, configuration, effective weights, implementation and prior hashes.
- Final preparation audit parsed the JUnit counts, loaded the frozen protocol,
 verified preservation against `.fb10-session-before.json`, and confirmed no
 execution receipt/trained checkpoint/curve/results. Record saved exclusively
 to `experiments/fb10-banc-readout/preparation.json`.

### Problems / Failed Attempts

No scientific experiment was attempted or failed. Local review reproduced and
fixed three software gaps: copied protocols lacked destination binding, a
single feature vector could broadcast as a sample batch, and checking only a
split label could accept relabeled historical validation cases. Each fix has
a test that failed first. Focused discovery initially used the wrong package
path/name; `rg` located the existing `src/` modules. One no-op documentation
patch was rejected without modifying its target; the real update then applied.

### Current State

FB-10 protocol preparation is complete and verified. The capped fit, trained
BANC checkpoint, behavioral gates and fresh evaluation remain unexecuted.
Software correctness does not establish steering improvement or representation
sufficiency. Existing saturation/generalization and biological limits remain.

### Next Step

FB-10 capped fit / Sol High: execute the one frozen protocol with
`.venv/Scripts/python.exe -m flybrain.training.banc_training --protocol experiments/fb10-banc-readout/protocol.json`.
Close the attempt after its predeclared gates, pass or fail. A failure returns
to an explicit recovery decision; a pass permits proposing a separately
locked fresh evaluation. Do not start a long run or another candidate automatically.

### Notes

For 30 Hz with training mean 20 Hz/std 10 Hz, normalization yields 1. A hidden weight 1
gives tanh(1)=0.761594; a turn weight 0.5 gives tanh(0.380797)=0.363399. Actual
inputs are [1008,8], with PyTorch weight matrices [16,8] and [2,16]. Forward
simulation calculates features/commands; autograd and Adam separately update
only178 motor-readout scalars. They do not reconstruct biology or train the
spiking source graph.

---

## 2026-09-27 — FB-10 single capped fit and matched training controls

### Goal

Continue the user's authorized next task: execute the existing frozen
500-update motor-readout fit with matched controls, record its evidence and
close the attempt under the original scientific gates. Do not treat FB-10
preparation or passing tests as a completed learning milestone.

### Changes Made

Consumed `experiments/fb10-banc-readout/protocol.json` and recorded
`execution.json`, `untrained.pt`, `trained.pt`, 501-row `curve.csv`, 25-row
`episodes.csv` and `results.json`. Added `verification.json` with independent
offline error/summary/gate, CSV/JSON, checkpoint and preservation checks, plus
`training_summary.png`/`.svg`; the PNG was visually inspected. Preserved the
original protocol/hash/preparation and all historical experiments.

Updated status, task/queue, experiment summary, decisions, architecture and
README. No application implementation, source or tests changed. Pre-run
snapshot: `.fb10-fit-before.json`; fresh JUnit reports are separate from the
preparation reports.

### Decisions

One existing hypothesis/candidate only: seed 1010, CPU float32, one execution
thread, eight endpoint-rate features, 16 tanh units, two motor outputs, fixed
train-only normalization, Adam lr 0.01 and exactly 500 updates. The 83-cell
spiking circuit stays fixed. Teacher commands remain labels, not live features.

Turn MSE 0.06678674714883089 exceeds the frozen ceiling 0.02225273146561865;
trained arrivals 2/8 miss 7/8. Strict matched arrival/progress improvements and
all integrity checks passed, but they cannot replace failed gates. Close the
attempt negative, preserving its checkpoints and consumed receipt. No retry,
extended fit, new learning method, held-out run or topology claim follows.

### Tests / Verification

- Preflight `load_protocol` verified the unchanged SHA256
  `f25f9c528dcba29302aeb2372df6fea23f3ea4e0a8ecab8f881cc580d4c13d05`,
  source/history/environment and original `[1008,8]` features /`[1008,2]` labels.
- `.venv/Scripts/python.exe -m flybrain.training.banc_training --protocol experiments/fb10-banc-readout/protocol.json`:
  **exit 1, scientific gate failure**, no numerical failure. Exactly 500 updates,
  25 episodes including replay, 14,131 world ticks, 282,620 neural steps.
  All 178 scalars had nonzero gradients and changed. All nine recorded
  numerical/integrity checks passed, including exact output and embodied
  save/reload replay, unchanged normalization/graph/source/history/protocol.
- Final forward/turn MSE: 0.00001754834288 /0.06678674714883089. Constant turn
  baseline: 0.0890109258624746. Trained/untrained/fixed arrivals: 2/8, 1/8, 0/8.
  All-case mean progress: 0.18489028, -0.10572687, -2.39173960.
- `.venv/Scripts/python.exe -m pytest tests/test_banc_readout.py tests/test_banc_training.py -q -p no:cacheprovider --junitxml=.fb10-fit-targeted-verification.xml`:
  **38 passed**, 5.00s, no skips.
- `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --junitxml=.fb10-fit-full-verification.xml`:
  **396 passed**, 20.32s, no skips/failures/errors, one existing upstream Pygame
  deprecation warning. Targeted/full test counts were parsed from fresh JUnit.
- Offline audit recomputed checkpoint MSE, all summaries/gates and changed
  scalars; matched every CSV row to JSON; confirmed budget/replay records and
  all 113 prior experiment files plus 48 Python sources stayed byte-identical.
  Audit/plot generation used **zero additional updates and zero world ticks**.
- Plot visually inspected for readable labels and agreement with metrics.
  No new headless app smoke or live UI review; earlier FB-09 checks remain
  historical. Local acceptance review used one agent, without a model switch.

### Problems / Failed Attempts

The scientific attempt failed both turn-fit and trained-arrival thresholds.
No optimization crash or source-integrity failure occurred. Its failure does
not determine whether representation, optimization or rollout mismatch is the
cause; do not replace uncertainty with an unmeasured explanation. One combined
documentation patch was rejected for mismatched context without changing any
file; the corrected patches applied. No scientific run was repeated.
The final documentation audit initially failed because PowerShell's stdin
encoding converted its literal em dash to `?` (observed codepoint 63). A
Unicode-codepoint header check confirmed exactly one correct log entry; the
corrected audit passed without modifying application code or experiment data.

### Current State

FB-10's authorized capped fit task is executed, audited and closed negative.
FB-10 overall is unfinished. Its checkpoint, curve and training comparisons
exist, while held-out evaluation remains unexecuted. The app continues to use
the fixed untrained BANC decoder. No Git repository, commit or push exists.

### Next Step

FB-10 train-only recovery review / Sol High: review preserved training evidence
and prepare one bounded diagnostic protocol separating turn-fit limitations
from embodied rollout failure. Predeclare its hypothesis, budget and stop rule
before new measurements. Preserve this failed attempt; any new candidate
requires an explicit recovery decision. Do not run another fit or fresh split
as part of protocol preparation.

### Notes

PyTorch forward calculations turn eight normalized rate features into motor
commands. Autograd and Adam separately changed all 178 readout weights/biases;
neither updated the spiking graph. Observed parameter change proves the fit
occurred, while the separate behavioral gate tests whether it navigates
reliably. Lower imitation error and passing software tests did not meet that
gate here. Reproducibility is limited to the recorded software/device setup.

---


## 2026-09-27 - FB-10 measured recovery and pending acceptance decision

### Goal

Act on the renewed user instruction to fix whatever is needed and finish FB-10,
starting from the preserved failed original 500-update fit. Diagnose measured
limitations, implement bounded readout repairs, run matched controls, and
complete honest verification/handoff without inventing a passed learning gate.

### Changes Made

Added separate conditioned, temporal-memory, covered-data, capacity and PN-input
readout/training/checkpoint modules; no original source was edited. Added tests
before implementation, observed missing-module failures, then numerical passes.
Collected all 32 teacher/32 frozen student training trajectories, preserving labels
and commands; added actual weighted PN incoming-spike histories while replaying
all historical training outcomes. Preserved a float64 exact-feature failure and
created a separately frozen zero-step canonicalization. Added an isolated fixed
range-encoder probe, closed negative before fitting; its software refuses new
collection/optimization. Added an independent offline audit and inspected plots.
Updated status, README, queue, architecture, decisions and experiment ledger.

### Decisions

One agent, no Git/model switch/commit/push. Every scientific attempt had its own
predeclared hypothesis/budget/stop and consumed receipt. Five additional fits
used500updates each, never extensions of the original fit. All 83 source cells,
weights, LIF rules, original encoder and timing stayed fixed in every fit. Only
motor heads learned; transforms, histories and training normalization stayed fixed.
PN incoming events move the readout upstream; downstream mediation and topology
benefit are not established. No historical validation row entered fitting.

Keep every failure. Original0.02225273 MSE/>=7/8arrival/strict matched progress
and integrity gates were retained; covered fits added a stricter aggregate
absolute ceiling. The latest candidate passes all original gates but misses
that extra ceiling. Request an explicit scientific acceptance decision before
admitting it to a new frozen fresh assay; do not silently drop a failed flag.
The reviewable proposal is `docs/plans/FB-10-FROZEN-CANDIDATE-ASSAY.md`.

### Tests / Verification

- Conditioning fit: 500 updates,turnMSE .06408428,arrivals 2/8 versus 1/8 and 0/8;
  13, 850 ticks/277, 000 steps; closed negative.
- Temporal fit: 500 updates,MSE .00837231, 8/8 versus 0/8 and 0/8;
  10, 775 ticks/215, 500 steps; training passed. First fresh validation 11/16 versus
  0/16 and 1/16 failed 13/16; 48 episodes/24, 519 ticks/490, 380 steps/zero updates;
  final not stepped. The central receipt remains consumed.
- Coverage data:teacher 32/32,student 22/32, 14, 856 ticks/297, 120 steps/zero updates.
  Covered306-scalar fit: 500 updates,aggregate .03579290,original .02694830,
  8/8 versus 0/8 and 0/8, 11, 009 ticks/220, 180 steps; error gates failed.
- Capacity1, 218-scalar fit: 500 updates,aggregate .03413672,original .02529960,
  8/8 versus 1/8 and 0/8, 10, 671 ticks/213, 420 steps; error gates failed.
- PN replay: 64body outcomes/labels exact, 14, 856 ticks/317, 280 steps/zero updates;
  original slow rounding mismatch<=1.27898e-13 Hz preserved as failed raw gate.
  Separate canonicalization used zero simulation/optimizer steps.
- PN370-scalar fit: 500 updates,aggregate .02354470,original .01088492,
  8/8 versus 0/8 and 0/8, 10, 805 ticks/216, 100 steps; all original gates and all 9
  integrity checks pass; additional aggregate absolute ceiling failed.
- Counting probe 9, 000 steps; original PN probe 20, 160 steps; historical rollout
  diagnosis3, 445 ticks/68, 900 steps; clipping reproduction600ticks/12, 000 steps;
  fixed range entry39, 000 steps, 26transient cases passed but contrast failed.
  All used zero optimizer updates and prescribed training/history only.
- Recovery totals excluding original fit/software tests: 312 worldepisodes,
  115, 386 worldticks, 2, 396, 040 neuralsteps, 2, 500readout updates, within individual
  budgets. No extended fit, encoder sweep or additional fresh assay.
- `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_banc_memory_evaluation.py tests/test_banc_projection.py tests/test_banc_projection_training.py tests/test_banc_projection_fit.py tests/test_banc_projection_evaluation.py tests/test_banc_range.py tests/test_banc_range_training.py --junitxml=.fb10-recovery-targeted.xml`: 33 passed, 28.19 s.
- `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --junitxml=.fb10-recovery-stable-final.xml`: 470 passed, 81.44 s,zero skips,one existing Pygame warning.
- `.venv/Scripts/python.exe scripts/audit_fb10_recovery.py`:exact checkpoint
  MSE,parameter changes,summaries,gates and all CSV rows for six fits, plus first
  fresh validation; all 48 original Python sources and 122 prior experiment files
  byte-identical. Audit/plot generation used zero simulation/optimizer steps.
  PNG visually inspected; SVG contains the same data. No new app/UI check.

### Problems / Failed Attempts

All scientific gate misses are preserved under their own protocols. A memory
reconstruction alias initially mutated a row during episode reset; numerical
reproduction fixed that before its fit. PN contract conditioning is a dictionary,
not a string; targeted tests caught a composition TypeError before collection.
The PN raw collection's original slow difference came from exp versus 1-expm1
float64 arithmetic, isolated and bounded, not changed trajectories or spikes.

An earlier concurrent full-suite fixture captured a growing journal; stable
rerun433passed without weakening scientific guards. The later full run initially
had466passes/3failures: consumed validation seeds were correctly historical in
new diagnosis records, while tests assumed permanent freshness. Preparation
fixtures now replay only their actual frozen pre-assay history; an added live
history rejection test proves reuse is blocked. Stable full470passed. These are
fixture fixes; no protocol-bound source or scientific integrity check changed.
One JavaScript orchestration call had a syntax error before executing its command;
it performed no experiment steps. One documentation patch failed context
verification without modifying files; the corrected patch succeeded. A file-only
PN calibration read exceeded the initial tool yield; resumed output completed
with zero simulation/updates and did not justify a candidate fit.

### Current State

FB-10 is unfinished. The original-encoder PN-input checkpoint is frozen and
reviewable, passes all original training gates, and retains its failed additional
covered-data gate. Original graph/encoder/app remain unchanged. All 470tests pass,
all preceding source/evidence is preserved, and no new run is active. The user's
scientific acceptance answer is pending; no exception has been applied and no
720000/721000 fresh assay has been consumed. No Git repository,commit or push.

### Next Step

FB-10 frozen-candidate acceptance decision / Sol High: choose the separately
locked zero-update fresh assay under the original gates, explicitly reporting
the extra failed gate, or retain that gate and stop. The next agent must honor
the answer, preserve all closed attempts and unchanged fresh 16/32case thresholds.
Do not append more optimizer updates or an encoder/width sweep while pending.

### Notes

The latest tensor batch is[15864, 20]; first PyTorch matrix[16, 20],then[2, 16].
An internal weighted event sum1 over20ms with fixed budget 3 means16.67 Hz.
Forward computes neural features and commands; autograd/Adam separately changed
only 370 readout scalars. This does not train the spiking graph or establish
held-out navigation. Reproducibility applies to recorded CPU/software setup.

---


## 2026-09-28 ? FB-10 completed: frozen original-gate assay with Astra review

### Goal

Fix and finish FB-10 using Astra, preserving scientific failures and prior
artifacts. Resolve the concrete acceptance choice, then verify one frozen
fresh assay under unchanged held-out gates.

### Changes Made

- Added `src/flybrain/training/banc_frozen_evaluation.py` and twelve targeted
  tests. Historical source modules were unchanged. The new path recomputes
  original/covered checkpoint errors, exact untrained initialization, all nine
  integrity flags, training identities/summaries and changed parameters.
- Added complete nonempty provenance validation, JSON/JSONL/CSV history checks,
  central single-use receipt protection, stop-before-final validation handling,
  per-episode counters and honest interrupted execution records.
- Recorded the user's "conitnue" response to the original-gate-versus-stop
  choice and the assistant's explicit interpretation in the acceptance JSON.
  Locked and consumed a new protocol for the same frozen 370-scalar candidate.
- Added `scripts/audit_fb10_frozen.py`; reconciled all JSON/CSV/journal records,
  protocol/checkpoint hashes, summaries/gates/counters and historical bytes.
  Saved and visually inspected scientific PNG/SVG plots.
- Updated status, queue, acceptance proposal, decisions and experiments.

### Decisions

Admit the exact preserved candidate using the original training gates while
explicitly carrying the failed additional covered-data ceiling into protocol
and result. Original MSE 0.0108849243 passes 0.0222527315; covered MSE
0.0235447047 still fails its additional ceiling. Historical fit result remains
`training_gate_passed=false`. No new training, lowered fresh threshold,
checkpoint change, retry or case selection. PN incoming-spike observation is
an explicit scope change from endpoint-only readout; no downstream mediation,
topology benefit or biological fidelity claim follows.

### Tests / Verification

- New regression tests initially failed because the additive module did not
  exist; first implementation passed six, then expanded ten tests. Astra found
  two further issues; the reproductions failed as intended (three failures:
  omitted provenance and journal write/flush double counting), then passed.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_banc_frozen_evaluation.py --junitxml=.fb10-frozen-targeted.xml`:
  **12 passed**, 33.54 s, zero skips/failures, recorded JUnit.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --junitxml=.fb10-frozen-full.xml`:
  **482 passed**, 56.92 s; one existing Pygame deprecation warning.
- `.\.venv\Scripts\python.exe -m flybrain.training.banc_frozen_evaluation --prepare --acceptance-file docs/plans/FB-10-FROZEN-ACCEPTANCE.json`:
  locked protocol SHA256 `57c42d1b55cc3edb05a352354a843990603ff2945fc6df1b430866eb2cc6faa3`,
  zero simulation/optimizer steps. This is a consumed execution record.
- `.\.venv\Scripts\python.exe -m flybrain.training.banc_frozen_evaluation`:
  **validation 15/16 trained, 1/16 untrained, 1/16 fixed; final 31/32, 4/32,
  3/32**. All arrival/progress/paired-effect/integrity gates passed. Final paired
  gains +27/+28; one-sided p=7.450580596923828e-9/3.725290298461914e-9.
  Actual **144 episodes, 63,599 world ticks, 1,271,980 neural steps, zero
  optimizer updates**, within the 144/86,400/1,728,000 declared ceiling.
- `.\.venv\Scripts\python.exe scripts/audit_fb10_frozen.py`:
  **23 audit checks passed**. All 63 pre-session Python source files, 219 prior
  experiment artifacts and four accepted bundle files remained byte-identical.
  Audit and plot used zero neural/world/optimizer steps. PNG visually inspected;
  SVG saved. This was not a live app/UI review.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --junitxml=.fb10-frozen-final.xml`:
  post-assay **482 passed**, 51.95 s, zero skips, one existing Pygame warning.
  This verifies fixtures and history safeguards after fresh cases became historical.
- Independent Astra reviews covered eligibility, runner implementation,
  regression fixes, audit helper and final saved scientific evidence. Final
  reviews reported no remaining completion blocker. No agent ran extra episodes.

### Problems / Failed Attempts

The first new implementation allowed omission of provenance entries from a
resealed protocol and could count completed neural work twice if journal I/O
failed. Regression reproductions confirmed both; complete inventory validation
and an explicit completed-accounting transition fixed them before preparation.
No scientific episode was lost, retried or repeated. All six historical fit
attempts, the earlier failed fresh assay and failed range probe remain unchanged.
A resumed tool-session handle expired between dated messages; the persisted
JUnit proved the earlier twelve-test completion. No scientific run was active
at that point. No blocked/skipped software tests remain.

### Current State

**FB-10 complete within the accepted frozen internal-PN-spike readout scope.**
The accepted fresh scientific gate passes, all software tests pass, and
results/plots/audit are under `experiments/fb10-projection-evaluation/`.
The additional covered-training failure is preserved. Both new splits are now
historical; the candidate's central receipt prevents another assay. Fixed BANC
app mode is unchanged. No new UI smoke or live manual review, Git repository,
commit, push or publication.

### Next Step

**Optional trained BANC checkpoint app mode / Sol Medium:** expose this saved
candidate through a distinct control option; verify checkpoint loading,
reset/timing, rendering independence and visible world/brain behavior. Keep
fixed BANC mode available. Do not retrain or start another fresh experiment.

### Notes

At inference, one sensory-driven interval produces twenty neural features.
PyTorch applies the saved `[16,20]` and `[2,16]` matrices to produce two motor
commands; the hand-written spiking dynamics and sourced graph remain fixed.
The earlier optimizer changed 370 readout scalars. This assay used forward
calculations only; its separate fresh paired outcomes establish measured
navigation improvement for the recorded CPU/software/task distribution.

## 2026-09-28 — Frozen trained BANC app integration

### Goal

Continue the recorded next task: expose the saved accepted BANC head in the app,
while preserving the extra covered-training threshold as a failure.

### Changes Made

- Added `src/flybrain/trained_app.py` with distinct `banc-trained` control,
  trained world/brain labels, exact-tick headless mode and reset/exit handling.
  Existing `manual`, `neural` and `banc` modes delegate to the original app.
- Added `neural/frozen_banc.py`: pin five saved training/assay artifacts,
  check historical source/data hashes, strict-load the original checkpoint
  and freeze all 370 parameters. No training data or runner is invoked.
- Added thirteen integration tests, inspected a saved 120-tick renderer frame,
  and updated status, queue, architecture, decisions, README and task handoff.

### Decisions

Use additive app/loader modules and HUD subclasses because original source
files are protocol-bound. Preserve the failed fit result and the successful
accepted assay as separate pinned records. Require the checkout's saved
evidence; this is not a standalone model-distribution task. No scientific
gate, threshold, checkpoint, protocol or historical result was edited.
One agent; local review against acceptance criteria; no commit or push.

### Tests / Verification

- Initial `tests/test_trained_app.py` run: twelve expected failures because
  the two new modules did not exist. First implementation: eleven passes and
  one test-fixture NameError; corrected the misplaced temporary-copy setup.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_trained_app.py --junitxml=.trained-app-targeted.xml`:
  final **13 passed**, 18.10 s, zero skips, one existing Pygame warning.
  Covers altered model/evidence/source rejection, global RNG preservation,
  zero-step loading, exact reset replay, 20:1 timing, unchanged model state,
  motor gate, rendering independence, three CLI modes and injected R/Esc events.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --junitxml=.trained-app-full.xml`:
  **494 passed**, 263.41 s, zero skips, one existing Pygame warning. The final
  reset/exit test was added after full-suite collection and passed separately
  and in the final targeted suite: 495 unique tests covered across these runs.
- `.\.venv\Scripts\python.exe -m flybrain.trained_app --control banc-trained --seed 7 --headless --steps 120`:
  exit0, exactly120 world ticks. Separate saved-frame render at120 ticks
  confirms2400 neural steps. Visually inspected
  `docs/verification/trained-app-frame.png`: world, source, fly, brain activity,
  outputs and frozen-readout labels readable with no clipping.
- Snapshot `.trained-app-before.json` and final SHA256 comparison: all300
  pre-session Python source/data/experiment files unchanged. Final evidence
  saved in `docs/verification/trained-app-checks.json`.
- Launched the normal app at seed7. Windows Computer Use `list_windows`
  failed with `native pipe ... The system cannot find the file specified
  (os error 2)`; one retry and kernel reset/reinitialization also failed.
  Terminated the session-owned demo. Live display and real keyboard behavior
  are **not verified**. Injected events and saved-frame review are separate.

### Problems / Failed Attempts

The test-fixture setup was initially inserted into the RNG test instead of
the temporary-artifact test; the traceback identified the error. After moving
it, corruption tests copy full source/data and assert the exact altered path
causes rejection. No product-code failure was observed. The desktop helper
failure blocks the live UI acceptance item; it is not hidden by the smoke test.

### Current State

Trained app implementation, automated checks and saved-frame review are done.
Live desktop review remains pending. All historic scientific failures,
especially covered MSE0.02354470 exceeding0.02225273, remain intact.
No optimizer updates, retraining, fresh assay, commit, push or publication.

### Next Step

**Finish trained-app live desktop review / Sol Medium:** when the native helper
is available, inspect movement and both panels, press R and Esc, compare fixed
mode labeling, and record the live outcome. No scientific run is required.

### Notes

One weighted PN event in20ms gives16.67Hz after the fixed budget3 normalization;
the first100ms EMA update gives3.02Hz. Twenty features feed a saved[16,20]
matrix and then[2,16]. PyTorch computes forward commands; neural state and
histories evolve, while saved parameters stay fixed. No autograd/optimizer
step occurs. See `docs/plans/TRAINED-BANC-APP.md` for the calculation.

## 2026-09-28 — One extra covered-data fit and fresh assay

### Goal

Continue the user's request through the extra training gate while preserving
the earlier 500-update failure and the accepted frozen assay.

### Changes Made

- Predeclared one same-method 750-update candidate, budget, unchanged gates
  and failure stop in `docs/plans/FB-10-EXTRA-TRAINING.md`.
- Added separate `banc_extra_training.py` and `banc_extra_evaluation.py`
  runners, one-use receipts, saved protocols, tests and records under
  `experiments/fb10-extra-training/` and `experiments/fb10-extra-evaluation/`.
- Added offline audit and saved-result PNG/SVG. Updated experiments, status,
  decisions, queue and README.

### Decisions

Test optimization time only: seed1010, original encoder/circuit/20-feature
readout and 15,864 canonical train-only rows, full-batch Adam at0.01,
forward-plus-turn MSE. Start from the same initialized model because the old
optimizer state was not saved. Require update500 curve identity before
embodied checks. One candidate, no hyperparameter sweep, no retraining on
fresh outcomes. A passing fit permits only the separately locked fresh assay
on new730000/731000 starts. Earlier failure and checkpoint remain untouched.

### Tests / Verification

- New training/evaluation tests failed first for missing modules, then passed.
  Before the fit, targeted training **5 passed** and full suite **500 passed**;
  before the fresh assay, evaluation **4 passed** and full suite **504 passed**.
- `.\.venv\Scripts\python.exe -m flybrain.training.banc_extra_training --prepare`: locked SHA256
  `60dc3b242fb1a50c713b7e381222094488a631a80707c95be7316e7dafbdbd01`,
  zero optimizer/world/neural steps. `.\.venv\Scripts\python.exe -m flybrain.training.banc_extra_training --fit` consumed one receipt and completed
  **750 updates**, 25 episodes, 10,778 ticks / 215,560 neural steps.
  Covered turn MSE **0.0213424152** passed the unchanged **0.0222527315**
  ceiling; original MSE **0.0100253166**. Train arrivals **8/8 vs0/8 vs0/8**;
  all nine integrity checks and training gates passed. All 370 parameters
  had nonzero gradient/change evidence; exact saved checkpoint replay passed.
- `.\.venv\Scripts\python.exe -m flybrain.training.banc_extra_evaluation --prepare`: locked SHA256
  `2c98fe0b0d4bc78b2e164ca0adb510e8acc87728fe4faec144af917b29ad44e3`,
  zero optimizer/world/neural steps. `.\.venv\Scripts\python.exe -m flybrain.training.banc_extra_evaluation` passed
  validation **16/16 vs0/16 vs2/16** and final **31/32 vs2/32 vs2/32**.
  Both final paired gains +29, exact one-sided p=1.862645149230957e-9 each.
  All-case progress, frozen model/core and budget gates passed. Actual
  **144 episodes, 64,106 ticks, 1,282,120 neural steps, zero updates**.
- `.\.venv\Scripts\python.exe scripts/audit_fb10_extra.py`: **25 checks passed**, zero simulation
  or optimizer steps. JSON/CSV/journal rows, counters, summaries, gates,
  protocol/checkpoint hashes and preservation reconciled. All 303 pretraining
  and 312 preevaluation files remain byte-identical. PNG visually inspected;
  SVG saved. This is plot review, not a live app review.
- Post-assay first targeted test run: six passed, three test-fixture failures
  because it treated consumed seeds/receipt as unused. Isolated those
  preparation fixtures from the completed assay; rerun targeted suite
  **9 passed** with `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_banc_extra_training.py tests/test_banc_extra_evaluation.py --junitxml=.fb10-extra-final-targeted.xml`; `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --junitxml=.fb10-extra-final-full.xml` **504 passed**, 93.55 s,
  zero skips, one existing upstream Pygame warning.

### Problems / Failed Attempts

The three post-assay test failures were fixture assumptions about freshness,
not product behavior; the corrected tests preserve the real consumed receipt.
A tool session handle expired after the final fixed-control cases, but the
completed `results.json`, 144-row journal and audit prove the assay finished;
no episode was retried or repeated. The live desktop inspection helper still
reports a missing native pipe, so real-window keyboard review remains open.

### Current State

The bounded extra-training experiment and fresh evaluation are complete and
pass all locked gates. All new case identities are historical; both receipts
are consumed. The previous 500-update covered-data failure is still recorded,
and the app still loads that earlier accepted frozen head. No downstream
topology benefit or biological fidelity claim, commit, push or publication.

### Next Step

**Finish trained-app live desktop review / Sol Medium.** When the helper is
available, inspect the world/brain window, movement, R reset and Esc exit,
then record the actual live outcome. No additional scientific training is
required. App exposure of the 750-update checkpoint is a separate optional
integration decision.

### Notes

For the training matrix `[15864,20]`, PyTorch applies the saved `[16,20]`
hidden matrix and `[2,16]` output matrix to produce two commands per row.
Autograd computes gradients for 370 readout scalars; Adam updates only those
scalars over 750 steps. The neural graph, LIF rules and feature histories have
no optimizer gradient path. Fresh evaluation executes forward calculations
only. The earlier 500-update result remains a failure despite the new pass.


## 2026-09-28 - Default app exposes the verified 750-update checkpoint

### Goal

Continue the user's request to fix outstanding work, connecting the completed
extra-training result to the app while preserving all historical evidence.

### Changes Made

Added `src/flybrain/__main__.py`, `verified_app.py` and
`neural/verified_banc.py`. `python -m flybrain --seed 7` launches the verified
750-update checkpoint. The HUD/caption identifies it explicitly. The loader
pins five saved fit/assay files, verifies 68 recorded sources and four data
files, loads the strict model contract and freezes 370 parameters. Existing
launchers remain available. The new app accurately reports interrupted
headless runs. Added 19 integration tests and an inspected saved frame.
Updated README, status, integration plan, architecture, decisions and queue.

### Decisions

Use additive modules because historical sources are protocol-bound. Reuse the
world, neural controller and visualizers directly; adapt the small event loop
without changing original module globals. This is app inference integration,
not another training run or scientific assay. The earlier failed gate remains
recorded as a failure. No additional candidate or thresholds were introduced.

### Tests / Verification

- Initial new-app tests: 15 failures from the absent modules/entry point;
  after implementation, all 15 passed. Added data-integrity, window-close,
  interrupted-run and manual-without-PyTorch coverage.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_verified_app.py tests/test_trained_app.py --junitxml=.verified-app-targeted.xml`:
  **32 passed**, 9.68 s, zero skips.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --junitxml=.verified-app-full.xml`:
  **523 passed**, 101.56 s, zero skips. Both suites report only the existing
  upstream Pygame `pkg_resources` deprecation warning.
- `.\.venv\Scripts\python.exe -m flybrain --headless --steps 120 --seed 7`:
  completed exactly120 ticks, checkpoint750-updates. An actual 120-tick
  simulation rendered through `verified_app.build_views` was saved to
  `docs/verification/verified-app-frame.png` and visually inspected: both
  panels are readable, title identifies750 updates, neural step2400.
- Behavior checks cover exact saved weights, frozen parameters, zero-step
  loading, RNG preservation, checkpoint/evidence/source/data rejection,
  deterministic reset,20:1 timing, motor disabling, rendering independence,
  all four modes, default package launch, R/Esc/close events and cleanup.
- Read-only SHA256 comparison against `verified-app-before.json`: all321
  prior source/experiment files remain unchanged. Earlier303-file and312-file
  snapshots also match, including the accepted data files.

### Problems / Failed Attempts

The native desktop helper still returns `Computer Use native pipe is unavailable`
with Windows error2. Previous retry/reset attempts already established this
infrastructure blocker. No live window movement/keyboard pass is claimed.
Saved-frame inspection and injected event tests are separate checks.

### Current State

Extra training and fresh scientific evaluation are complete. The verified
750-update result is now the default usable app; the historical500-update
launcher and its original failed gate remain preserved. All software tests
pass. Native live acceptance is the only open integration check. No commit,
push, publication, new fit or scientific assay rerun occurred.

### Next Step

**Finish native live app acceptance / Sol Medium:** once the desktop helper
works, launch `python -m flybrain --seed 7`, inspect both panels and movement,
press R, then Esc, and compare the fixed mode labels. Record the live outcome.
No further training is needed.

### Notes

The saved `[16,20]` and `[2,16]` matrices turn20 neural history features into
forward/turn outputs. Simulation updates neurons and history state each tick;
these370 learned weights stay fixed. Using PyTorch for this forward calculation
does not perform autograd or optimizer updates.


## 2026-09-29 - Approach, speed and stopping repair under the supplied plan

### Goal

Implement a sensory-driven 85-cell candidate that slows near food and
commands an exact stop, then apply the supplied staged training and fresh
full-episode gates without changing the historical arrival result.

### Changes Made

Added a rational odor-rate map, one engineered 100 Hz contact cell and one
fixed-weight relay beside the unchanged 83 sourced cells. Added 50/200 ms
relay spike histories and a 22→16→3 readout whose positive stop logit
commands exactly zero speed and turn. Added a sensory-only teacher, 120
preselected cases (40 train, 16 validation, 64 final), full 20-second
contact/braking/dwell scoring, one bounded fit, a one-use fresh assay, a
contact diagnostic, and an additive candidate brain/app view. The candidate
view labels the two engineered cells. The existing default app remains the
historical 750-update arrival-only model.

### Decisions

Keep the BANC source identities and connections unchanged. The contact
cells, edge, odor mapping and neuron dynamics are engineering assumptions,
not sourced anatomy. Only 419 readout scalars learn; controller inputs remain
left/right odor and local contact. The fresh receipt is consumed; after its
interruption, treat all 64 reserved final identities as historical and do not
promote the checkpoint. A concrete [recovery proposal](docs/plans/FOOD-STOP-RECOVERY.md)
specifies one possible new final-only assay but has not been executed.

### Tests / Verification

- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  **555 passed in 93.58 s**, zero skips, one existing upstream Pygame warning.
- The ten new candidate/record test files: **19 passed** in 4.63 s.
- Teacher preflight: **40/40** training cases; 40,000 world ticks, no neural
  steps or optimizer updates. Sensory tests show weak/strong odor differences
  in real spikes, directed contact propagation and reset.
- One seed-1010 Adam fit: **1,500 updates**, 80,000 training rows, 31,044
  contact rows, nonzero gradients and changes in all 419 readout scalars.
  Moving speed MSE **0.0003112 vs 0.0150496** mean-command baseline; turn
  MSE **0.1118214 vs 0.1962294**. Checkpoint reload matched exactly.
  Full training-case behavior: **40/40 vs 0/40 and 0/40**.
- Fresh validation: **16/16 vs 0/16 and 0/16**, 48 recorded episodes,
  48,000 world ticks, 960,000 neural steps, zero optimizer updates. Offline
  tests reconcile cases, poses and counters with the locked protocol.
- Separate fixed-input contact block: **4/5** last intact ticks stopped,
  **0/5** blocked; 400 neural steps, unchanged parameters.
- Saved [candidate frame](docs/verification/food-candidate-frame.png) was
  visually inspected: 83 sourced cells, both engineered cells, contact,
  commanded/actual speed and dwell are legible. This is not a live desktop
  keyboard/window review.
- The default and candidate app's historical-reference launcher each
  completed a 120-tick headless smoke as `banc-trained`, confirming the
  accepted 750-update default stayed in place.

### Problems / Failed Attempts

The one-use fresh process vanished after saving validation and before saving
any final result or traceback. The tool session became unavailable, and the
number of final episodes actually executed is unknown. An Application event
search found no matching Python error. The [interruption record](experiments/food-stop-fresh-2026-09-28/interruption.json)
preserves known counts and marks scientific acceptance false. No final case
was rerun. A live desktop keyboard/window review was not performed; the
native helper was unavailable in the prior handoff.

### Current State

The candidate demonstrates learned approach and stopping on training and
validation cases plus contact-path dependence in a fixed-input probe. Its
fresh final sector, paired-effect and full-episode gates are unverified.
The candidate loader deliberately rejects promotion; `python -m flybrain`
still runs the accepted arrival-only controller. No Git repository exists,
so no commit, push or publication occurred.

### Next Step

**Recovery decision / Sol High:** review the new final-only frozen-checkpoint
assay proposal with unused 840000–840063 starts. It preserves all gates,
records each episode durably, and runs only after an explicit new decision.
Then complete the candidate app checks and a separate live desktop review
before any default promotion.

### Notes

One relay spike in 20 ms is 50 Hz. From zero, the 50 ms history rises to
about 16.5 Hz in that tick. PyTorch multiplies a `[1,22]` feature row by
the learned `[16,22]` hidden matrix and `[3,16]` output matrix. Autograd and
Adam acted only during the fit; all validation and diagnostic calculations
were frozen forward passes. Neither the sourced topology's benefit nor
biological fidelity is established.

## 2026-09-29 — Food-stop recovery runner prepared without case execution

### Goal

Make the proposed new final-only decision executable and auditable while
preserving the interrupted one-use assay and its potentially exposed cases.

### Changes Made

Added `src/flybrain/training/food_recovery.py` and five focused tests. The
runner verifies pinned fit, validation, interruption, contact-probe and exact
new-manifest bytes; prepares a separate immutable protocol; creates an
exclusive execution receipt; fsyncs a row after each complete episode; and
requires 192 ordered matched rows before writing JSON, CSV and final gates.
Updated the recovery proposal, queue, experiment summary and status.

### Decisions

The 830000–830063 original final starts remain historical. No recovery
protocol was prepared and no 840000–840063 start was evaluated. The proposed
runner requires an explicit new decision before its one-use `--prepare` and
`--run` actions. App promotion remains gated on a complete final result.

### Tests / Verification

- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_food_recovery.py`:
  **5 passed**. These check the exact manifest, durable journal audit,
  incomplete/tampered journal rejection, evidence-only preparation, and
  exclusive receipt replay guard. They execute zero simulation episodes.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  **560 passed in 97.07 s**, zero skips, one upstream Pygame deprecation warning.
- `python -m flybrain.training.food_recovery --help` succeeded. The proposed
  experiment directory and old final/result files remain absent.

### Problems / Failed Attempts

No new runner failure. The original final process interruption and unknown
episode exposure remain unresolved; there is still no final behavior result.

### Current State

The exact frozen-checkpoint recovery runner is ready for a new decision. The
default app still uses the historical arrival-only controller. No scientific
or app acceptance is claimed; no Git repository exists.

### Next Step

**Recovery decision / Sol High:** authorize or decline the single final-only
assay in `docs/plans/FOOD-STOP-RECOVERY.md`. If authorized, prepare then run
the journaled protocol once, audit all gates, and only then finish live app
review and consider default promotion.

### Notes

The journal is an evaluation record, not training data. Each episode executes
the frozen PyTorch forward computation; no gradient or optimizer step occurs.

## 2026-09-29 — Authorized food-stop final recovery and candidate integration

### Goal

Complete the interrupted Astra plan's final behavior gate on new cases, audit
the unchanged checkpoint, and integrate it without bypassing the live UI gate.

### Changes Made

The user replied “if not continue” after the one-use interruption and new
decision were explained. Prepared and consumed the single final-only protocol
on seeds 840000–840063. The runner wrote a fsynced 192-row journal, JSON,
CSV and result. Added an offline record audit test. Updated the strict
`accepted_food` loader to pin fit, original interruption, contact diagnostic,
new protocol, execution, journal, final files and result. `food_app` now runs
that frozen checkpoint. Added accepted model, app reset/timing/R/replay, and
tamper tests. The default `flybrain` launcher remains historical.

### Decisions

Keep both the old interrupted 830000–830063 and new consumed 840000–840063
identities historical. No optimizer, threshold, source graph or checkpoint
change occurred. All scientific final gates passed, but the recovery plan
requires a separate live desktop check before default promotion. The native
Computer Use pipe is unavailable, so retain the 750-update default.

### Tests / Verification

- The [final result](experiments/food-stop-recovery-2026-09-29/results.json)
  records **63/64** trained successes versus **0/64** each untrained/fixed;
  sectors **8, 8, 8, 8, 7, 8, 8, 8**; paired gain **+63** and exact one-sided
  p **1.0842021724855044e-19** against each. All gates pass. The one failure,
  seed 840044, never contacted food. Parameters stayed unchanged.
- Receipt and offline audit reconcile **192 episodes, 192,000 world ticks,
  3,840,000 neural steps, zero updates** with protocol, journal, JSON and CSV.
  No final episode was replayed.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  **564 passed in 402.46 s**, zero skips, one upstream Pygame warning.
  After adding three evidence-tamper tests, the focused
  `tests/test_food_app.py` suite passed **9/9** in 9.39 s.
- `python -m flybrain.food_app --headless --steps 120 --seed 7` and the
  historical `python -m flybrain` equivalent both completed 120 ticks.
- The [accepted saved frame](docs/verification/food-accepted-frame.png) was
  visually inspected at neural step2600. It shows zero commanded/actual speed,
  contact, 2.48 s stationary dwell, SETTLED, both engineered spike counts,
  and 83 sourced cells. This frame used a deliberately staged contact pose.

### Problems / Failed Attempts

The assistant turn was interrupted while the final runner was active; the
existing process nonetheless completed all rows and its result. Its previous
tool session became unknown, so completion was established from receipt,
journal, results and an independent offline audit, without restarting.
The Computer Use `sky.list_apps()` call failed twice and once after kernel
reset: native pipe missing (`os error 2`). A launched Python window did not
expose a targetable handle and was stopped. No live keyboard/window pass is
claimed. One offline audit test initially compared JSON string sector keys
with integer keys; the test was corrected and passed without changing results.

### Current State

The frozen candidate is behavior-gated and runs through `flybrain.food_app`.
Automated app integration and saved-frame review pass. The historical
arrival-only 750-update controller remains `python -m flybrain` until a
separate live UI review passes. No Git repository exists; no commit or push.

### Next Step

**Food-stop live desktop review / Sol Medium:** inspect the visible candidate
world/brain panel, R new random trial, Backspace exact replay and stopping
after food contact. If that passes, change only the default launcher to
`food_app`, verify the default mode and update status. No new experiment.

### Notes

The final assay was inference: the 85-cell simulator generated spikes and the
frozen 22→16→3 readout issued motor commands. PyTorch autograd and Adam did
not act. The contact-block probe supports dependence on the engineered path
for its fixed input; the result does not prove benefit from BANC topology or
biological fidelity.

## 2026-09-29 — Live-review preparation and disclosed app start failure

### Goal

Complete the next live review after the passing final assay.

### Changes Made

Retried the Computer Use native helper; it again reported a missing pipe.
Checked the candidate app's arena-center spawn for seed 7 before asking the
user to use that seed for a stop demonstration. It did not contact food in
20 simulated seconds. Selected seed 2 by its initial heading toward food
before running it; its app smoke reached food and stayed stopped. Corrected
the manual-review command and documented both outcomes.

### Decisions

Do not present seed 7 as a successful stop demonstration. Neither smoke is
a fresh-assay case or reason to tune the frozen checkpoint. Keep the
historical default until the separate live desktop review has an observed
pass; the user's approval cannot substitute for that observation.

### Tests / Verification

The app seed-7 1,000-tick smoke ended without contact, stationary dwell or a
settled state, commanding 68.39 units/s. The seed-2 1,000-tick smoke ended
in contact with 15.68 seconds of stationary dwell, SETTLED, and zero
commanded/actual speed. These are diagnostic app starts, not a rate estimate.
No source or checkpoint byte was changed by these checks.

### Problems / Failed Attempts

`sky.list_apps()` still failed with native-pipe `os error 2`. No live window
or keyboard review was completed.

### Current State

The scientific 63/64 final result and candidate mode remain intact. The
manual live-review request now uses seed 2. Default remains historical.

### Next Step

**Live desktop review / Sol Medium:** observe the candidate window, stop and
dwell from seed 2, R new trial and Backspace replay. If it passes, promote
the pinned candidate to the default launcher and verify that launch.

### Notes

The final case distribution and the app's fixed center-spawn distribution
differ. A passing final gate allows occasional failures; it is not a promise
of success from every app seed.

## 2026-09-29 - Live review passed and food mode promoted

### Goal

Complete the live gate in the food-stop recovery plan, promote the accepted
food controller as the default, and verify the final launcher state.

### Changes Made

Recorded the user's live confirmation in the recovery plan, decision log,
status, task queue, README and this log. Updated the package-entry-point test
to expect `food-trained` with the `approach-stop` checkpoint. The launcher
already imports the strict, evidence-pinned `food_app`; historical arrival
behavior remains available with `--control banc-trained`.

### Decisions

The user confirmed that the live HUD showed contact, zero speed and SETTLED
dwell with sensor/relay activity; R started a new trial and Backspace replayed
the same seed and heading. This satisfies the live review gate and authorizes
default promotion. No checkpoint, experiment result, scientific case identity
or network parameter changed. The default is food approach-and-stop; the
750-update arrival controller stays selectable as a historical reference.

### Tests / Verification

- Full suite: `python -m pytest -q -p no:cacheprovider` -> **567 passed**,
  zero skips, one existing upstream Pygame `pkg_resources` deprecation warning,
  in 111.61 s.
- Corrected default-launcher regression test:
  `python -m pytest -q -p no:cacheprovider tests/test_verified_app.py::test_package_entry_point_uses_food_default`
  -> **1 passed** in 4.52 s.
- Default food smoke `python -m flybrain --headless --steps 120 --seed 2`
  completed as `mode=food-trained`, checkpoint `approach-stop`.
- Historical reference smoke with `--control banc-trained` completed 120 ticks
  as `mode=banc-trained`, checkpoint `750-updates`.
- Live UI verification is the user's direct observation; the agent-side
  Computer Use native pipe remained unavailable. No further scientific assay
  was run.

### Problems / Failed Attempts

The first fresh full suite found one stale assertion expecting the previous
750-update default. Its observed output correctly reported the food mode. The
assertion was updated, passed in isolation, and the complete suite then passed.

### Current State

The food approach-and-stop controller is the verified default and the recovery
plan is complete. The app's broad center-start reliability remains open: the
diagnostic center-start seed 7 failed, while seed 2 succeeded.

### Next Step

**Assess center-start reliability / Sol High:** define a bounded, predeclared
smoke assessment before changing the frozen controller or launch distribution.
Do not reuse scientific assay cases or start retraining as part of that work.

### Notes

The final scientific result remains 63/64 trained successes on its recorded
randomized distribution, not a guarantee for every app launch. The user
confirmed the live checks in their own visible app window.

## 2026-09-30 — 3D flower world, source anatomy and closed first experiment

### Goal

Implement the user's approved 3D flower-landing and referenced brain-viewer
plan, including a single bounded learning experiment with prerequisite gates.

### Changes Made

- Added the isolated `src/flybrain/flight3d` package and optional Panda3D/PyArrow
  dependencies. SI-unit flight, gravity/drag, attitude/power, contact/landing,
  fixed initial pose and random horizontal/vertical flower placement.
- Added immutable sensory/command/snapshot interfaces; independent RGB eyes,
  bilateral odor, local motion feedback and contacts. No target geometry enters
  a controller. Neural/physics/control steps are 1/5/20 ms.
- Froze 589 source cells and 4,139 directed BANC v888 edges, with annotations,
  string IDs, evidence and explicit missing routes. Acquired registered brain
  and nerve-cord meshes plus 585 actual selected skeletons; four public exports
  are missing and explicitly reported.
- Added fixed source-driven spikes, separate power/steering/landing readouts,
  causal silencing, identity-bound checkpoints and a sensory teacher.
- Added two Panda3D views with source-neuron inspection, whole/project modes,
  cord selection, camera presets/orbit/zoom, close fly view, motor-driven wing
  animation, simulation-time activity, R/new and Backspace/replay.
- Added the predeclared experiment runner, stored its actual outcome and
  failure trajectory, archived the exact executed scientific source, and
  updated status, architecture, decisions, queue and the 3D guide.

### Decisions

The explicit 3D request supersedes the earlier deferral. Work stays additive;
historical accepted sources/results remain unchanged. Source structure is
distinguished from inferred routes and modeled dynamics/sensors/mechanics.
Whole-brain context does not mean whole-brain simulation. Unused cells have no
live activity; modes never relocate anatomy. See `docs/FLIGHT3D.md`.

The approved failed-gate rule closes the first experiment. The teacher failed
its first mandatory training placement, so no fit, sweep, validation or final
assay followed. The default 3D preview is explicitly labeled a hand-written
teacher, with observational neural activity. There is no 3D trained checkpoint.

### Tests / Verification

- Final targeted command `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
  tests -k flight3d`: **27 passed**, 567 deselected, 33.06 s, two upstream
  warnings. Includes the final full-anatomy UI/pixel and visibility regression.
  Output: `.superpowers/flight3d/targeted-final.log`.
- `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`:
  **594 passed**, zero skips, 573.03 s. Two upstream warnings: Pygame
  `pkg_resources` deprecation and PyTorch `torch.jit.script` deprecation.
  Later viewer-only changes were verified separately by the targeted suite.
- `.venv\Scripts\python.exe -m flybrain.flight3d.audit
  C:/Users/Lenovo/AppData/Local/Temp/flybrain-fb09-research`: both archive hashes,
  all 589 cell identities/positions and all 4,139 edge directions/counts match.
  All 585 cached morphologies exactly match raw SWCs under the documented
  coordinate transform. Four source 404s are listed in `data/flight3d/audit.json`.
- Original SHA256 inventory: **374/374 pre-existing source/experiment files
  unchanged**, recorded in `.superpowers/flight3d/preservation.json`.
- `.venv\Scripts\python.exe -m flybrain.flight3d --seed 10001 --capture
  .superpowers/flight3d/frames`: exported front/side/top, brain/cord overview
  and project-only views. Compared morphology, purple/pink palette and warm
  activity against the saved reference. Actual anatomical density differs from
  the reference's stylized sparse graph. Saved PNG pixel checks confirm the
  world/UI remain present across all views.
- Offscreen event/render integration verifies independent cameras, zoom/modes,
  unchanged neural/world state during drawing, R/Backspace events and visible
  wing animation. Activity fading/reset has a numerical simulation-time test.
  This is **not** a live human mouse/keyboard review. The Computer Use skill's
  native Node/sky entry point is not exposed in this session.

### Problems / Failed Attempts

- The recorded neural prerequisites all passed, but teacher seed **960000**
  hovered and searched in yaw for 30 s without contact. Its initial eyes contain
  zero flower pixels: coarse vertical rays miss the thin flower; yaw preserves
  that gap. Actual optimizer updates: **0**. Earlier successful development
  previews did not establish broad teacher coverage.
- Four selected skeleton IDs return 404 for both skeleton and L2 exports; no
  geometry was fabricated. Source points and IDs remain in the viewer.
- Reproduced a Windows DLL failure when Panda's native core loaded before
  PyTorch; reversing the order succeeds. Viewer imports now preserve that order.
- Offscreen input helpers initially assumed an available mouse watcher; added
  guards and tested the headless route. A 25 Hz visual wing cycle would alias
  to zero at 50 Hz snapshots; changed the slowed visual proxy to 12 Hz.
- Image inspection initially suggested intermittent UI omission. Minimal frame
  probes and inspection of the actual saved PNG RGB/alpha showed complete UI
  pixels. Exports now explicitly use zero offscreen back buffers, ordered display
  regions and complete-frame rendering. Final PNG checks and the UI test pass.

### Current State

3D implementation, source registry and anatomical viewer are available as a
research preview. The single scientific experiment is closed at its teacher
gate. Learned 3D flower landing remains unachieved. Source anatomy is traceable;
neural physiology and simplified flight are explicit engineering models.

### Next Step

**3D visual sampling and teacher coverage repair / Sol High.** Use development
placements to resolve vertical aliasing and search coverage, then predeclare a
new bounded experiment. Preserve this failed run and do not silently reuse its
reserved final cases. Live desktop interaction review also remains open.

### Notes

No subagents, Git initialization, commits, pushes or automatic recovery fits.
The separate accepted 2D result remains 63/64 on its recorded distribution.
Learning takeaway: causal spikes reaching motor cells are a prerequisite;
learned successful behavior still requires a competent teacher and behavioral
evaluation on fresh placements.


## 2026-10-02 - Clearer controls and smoother brain navigation

### Goal

Implement the user's supplied controls/camera specification, retain the accepted
source appearance and wheel feel, verify interactions and timing, obtain a short
live review. One agent, existing cached geometry, teacher preview; no learning.

### Changes Made

Added `brain_camera.py` for continuous quaternion rotation, 60 ms rotation/pan,
120 ms zoom/fit response, explicit fit, 0.03 mm floor and adaptive near clipping.
Added `brain_controls.py` for pixel-normalized gestures, current-scale pan,
release/focus/resize handling, independent wheel routing and the simplified UI.
The default anatomical viewer uses these modules; cord toggles preserve current
and target camera state and matrix exactly. Presets change orientation only.

Toolbar now has Fit view, Front/Side/Top (Views menu below 1200 px), Show nerve
cord and Display settings. Settings contain Background anatomy, Branch brightness
and Reset view. Tab remains; N/C are removed. Selected inspector has source ID,
model role/evidence, Focus selected, Clear selection and default-off retained
connections with the straight-link/anatomical-branch distinction. Browsing,
missing-geometry messages, fly controls, pause/new/replay/exit and palette remain.

Updated status, guide, architecture, decisions, queue and verification record.
Benchmark now exercises actual quaternion input rather than the obsolete angles.

### Decisions

Background anatomy means registered surface/reference points beyond the simulated
subset and never indicates training membership. Split reference points at the
existing Y=-0.43 mm display plane to keep hidden cord references out of brain-only
Fit view. Source data and registration are intact. Initial accepted framing remains
1.10 mm at (0.49,-0.17,0.145). Gestures have no inertia; explicit Fit/Reset recovers
an off-screen view. See `docs/decisions.md` and `docs/verification/flight3d-controls.md`.

### Tests / Verification

- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests -k flight3d --junitxml=.superpowers/navigation-2026-10-02/targeted.xml`:
  **61 passed, 567 deselected**, 24.95 s, two existing upstream warnings.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --junitxml=.superpowers/navigation-2026-10-02/full.xml`:
  **628 passed, zero skips**, 147.37 s, two existing upstream warnings.
  This is a fresh completed full-suite run, replacing the earlier missing result.
- `.\.venv\Scripts\python.exe scripts/benchmark_flight3d_viewer.py --view repaired --frames 120 --output .superpowers/navigation-2026-10-02/navigation-timing.json`:
  exit 0, no concurrent preview/tests, 1440x900, Intel Iris Xe. p95 paused/running/
  orbit/select **4.44/5.87/3.91/9.05 ms**; paced FPS about 58.5. Running throughput
  **0.97 simulated s/wall s**. Neither steady 60 FPS nor machine-independent
  real-time throughput is claimed.
- `.\.venv\Scripts\python.exe .superpowers/navigation-2026-10-02/capture_controls.py`:
  saved/visually inspected initial/settings/inspector/focus/deep-zoom views at two
  sizes plus narrow Views menu. Front/side/top silhouette and selection picking
  remain covered by actual renderer tests; these are not a live mouse review.
- Numerical regressions: pole crossings from all presets; 60/120 ms responses;
  exact fixed-target results at 30/60/120 Hz; continuous-drag difference <0.6
  degree; depth/near; pan/zoom independence; current/target/matrix cord preservation;
  two viewport sizes; release/Shift-release/focus; inspector and simulation equality.
- SHA256: **2,366/2,367** pre-existing source/experiment/cache files unchanged.
  Only `anatomical_viewer.py` changed. New camera/UI modules are additive. All
  experiments, scientific source, previous viewers/navigation and geometry match.
- Local single-agent code review against acceptance criteria completed.
- Native teacher preview is open (`pythonw.exe -m flybrain.flight3d --seed 970000`)
  and a short user review has been requested; no user observations received yet.

### Problems / Failed Attempts

The new tests first reproduced missing camera/controller behavior and the N/C
shortcuts. A later test picked an arbitrary set member that could be a hidden
cord neuron, so it incorrectly required Focus selected to move for invisible
geometry. The fixture now uses a deterministic optic-lobe source cell and tests
missing geometry separately. Native offscreen `setSize` is unsupported without
Panda's resizable buffer flag; size sensitivity is checked with two actual
renderer windows plus gesture-size invalidation. Native resizing remains live QA.

The initial benchmark wrote its complete JSON, but PowerShell turned Panda's
stderr into NativeCommandError. A serial rerun captured exit 0 explicitly; first
and final records are retained. Computer Use twice failed with native-pipe
unavailable; no desktop inputs or native screenshots are claimed.

### Current State

Implementation, fresh targeted/full tests, isolated timing and saved-view review
are verified. Accepted appearance remains intact. Live navigation/resize/focus
acceptance awaits the user; no observations or acceptance are invented.
The interrupted fit is still incomplete at 1,050 updates, with consumed receipt,
checkpoints/journals intact. No fitting, reserved evaluation, download, commit/push.

### Next Step

**One next task / Sol Medium:** finish the short live review of revised brain
navigation/controls, including native resizing/focus loss, and record any concrete
interaction fixes. Further learning requires a separately agreed recovery plan.

### Notes

Learning takeaway: rotating direction and up together maintains a continuous
camera basis through the poles; display-time interpolation changes the camera,
while the unchanged fixed-step worker owns all world and neural calculations.


## 2026-10-02 - Live pan/scroll feedback and anatomy-toggle clarification

### Goal

Record the user's live observations and explain which displayed anatomy belongs
to the simulated project subset.

### Changes Made

Updated status and the controls verification record. No code, geometry or
simulation changes. Inspected the default viewer's visibility/loading paths.

### Decisions

The user confirms pan and scroll are good. Their intended toggle is whole fly
brain on and the exact project subset off. Existing off mode isolates project
branches; on mode adds surfaces/reference points as anatomical overview. It does
not load all full-brain neurites. Do not equate the overview with that full detail.

### Tests / Verification

User-observed live pan/scroll pass. Code inspection: source branch queue contains
only manifest nodes, project roots remain visible, and background roots toggle.
The prior 61 targeted/628 full-suite results remain the software evidence; no
new test run was needed for this documentation-only clarification.

### Problems / Failed Attempts

No new failed attempt. Full-brain branch detail is not currently available in
the viewer. Native resize/focus and remaining interaction checks are unconfirmed.

### Current State

The project simulates 589 sourced cells, with 585 cached source skeletons and
four missing exports. Teacher supplies preview commands; the modeled circuit's
activity is observational. No training, evaluation or interrupted-record changes.

### Next Step

Clarify the full-brain detail requirement before planning further atlas geometry
work; preserve the working navigation and existing scientific scope.

### Notes

Background anatomy represents source context, not which cells were trained.


## 2026-10-02 - Distinguish actual spikes from sparse branch highlighting

### Goal

Investigate the user's observation that some project branches do not appear to
fire, without modifying neural dynamics or the accepted display appearance.

### Changes Made

Added a bounded read-only preview diagnostic and saved per-cell/interval evidence
under `.superpowers/navigation-2026-10-02/`. Updated status and verification notes.
No product code or geometry changes.

### Decisions

Model membership, actual spike production and display emphasis are separate.
The current warm score ranks filtered outgoing delivered current, rather than
per-cell spike counts, and highlights at most 12 cells. A dim branch alone cannot
establish silence. Do not force all included neurons to fire or alter dynamics
in response to display ambiguity.

### Tests / Verification

`.\.venv\Scripts\python.exe .superpowers/navigation-2026-10-02/check_spike_display.py`:
seed 970000, teacher, 100 fixed updates / 2 simulated seconds, zero optimizer
updates. **530/589** cells spiked at least once and **59/589** did not. Among
361 brain-only highlight-eligible cells, 304 spiked, 57 did not, and **281 spiked
without ever being highlighted**. Maximum concurrent emphasis remained 12.
Vision source `720575941546280072`: **403 spikes / zero highlighted intervals**.
Read-only assertions reconciled duration, all 100 intervals, 589 cell rows,
summary counts, cap and example. Prior 61/628 tests remain the product verification;
no full-suite rerun was required for a diagnostic/documentation-only change.

### Problems / Failed Attempts

The user interrupted the tool turn, then requested continuation with increased
effort. The diagnostic had already completed and written its result; continuation
inspected it rather than repeating the run. Individual causes of silence in the
59 zero-count cells were not diagnosed. No native live activity measurements
were inferred from the saved headless window.

### Current State

The user's observation has both numerical and display explanations: some cells
are genuinely silent in the sampled window; many others fire without a warm
highlight. All simulation/neural/geometry/experiment sources remain unchanged.
The interrupted 1,050-update fit remains incomplete and untouched.

### Next Step

Continue the live controls/anatomy review. If activity inspection is extended,
use actual per-cell spike counts/status to distinguish silence from lack of
emphasis; changing the 12-highlight appearance rule needs a concrete display
requirement. No further learning work is authorized by this diagnosis.

### Notes

The two-second diagnostic is not a scientific learning assay, a claim of complete
neuronal coverage over time or evidence that the displayed circuit drives the
teacher-controlled fly.

## 2026-10-04 - Show actual firing across available project branches

### Goal

Carry out the October 3 spike-activity plan with one agent and cached geometry,
preserving all scientific results and the incomplete 1,050-update experiment.

### Changes Made

Added a display-only completed-step recorder and frozen snapshot/history frame.
Integrated optional recording, reset and deterministic capture into the serial
simulation pump. Owner-bearing static branch batches sample a compact explicit-RGBA
GPU table; removed twelve-cell/current ranking and metadata-region exclusion.
Pulses use simulation time, 200 ms decay and 600 ms expiry. Added complete inspector
spike/rate/last-activity values and geometry/no-nonself-input explanations.
Preserved cyan selection, brightness, cached coordinates and navigation.
Cached static edge counts and inspector publication/status to avoid repeated work.
Follow camera and body consume the same published frame. Added behavior/GPU/viewer
regressions, extended hidden/shown benchmark and updated guide, status, queue,
architecture, decisions, plan ledger and verification record.

### Decisions

Observe every successful 20 ms `FlightSession.advance`, aggregating its existing
1 ms neural counts; no invented subinterval timestamps. Atomically publish the
original scientific snapshot with display history. Every reset gets a new episode
identity even with the same seed. Drawing/loading/selection never record activity.
Warm means whole-neuron activity, not anatomical propagation or importance.
The original scientific snapshot and numerical rules remain unchanged.

### Tests / Verification

Fresh required serial commands using `.venv/Scripts/python.exe`, Python 3.12.10,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, JUnit output via `PYTEST_ADDOPTS`:

- `python -m pytest -q tests -k flight3d`: **76 passed, 567 deselected, 2 warnings**,
  47.47 s pytest / 52.95 s wrapper; exit 0.
- `python -m pytest -q`: **643 passed, zero skips, 2 warnings**, 145.88 s pytest /
  151.31 s wrapper; exit 0. Existing Pygame/Torch deprecations.

Regressions verify >12 firing, no-outgoing cells, silence/missing geometry,
mixed/deferred owners, skipped displays, pause/reset/replay, rendered GPU RGBA,
30/60/120 Hz matched-time equality and exact recorded/unrecorded teacher/fixed/
disposable frozen-readout snapshots, neural/controller states and parameters.
Recorder leaves Python/NumPy/Torch RNG unchanged; trial construction has no
persistent RNG. Final audit: **2,365 protected files unchanged**, no scientific,
cache or experiment edits. Tests and artifacts:
`.superpowers/spike-activity-2026-10-03/`; full details in
`docs/verification/flight3d-spike-activity.md`.

Isolated 1440×900 hidden p95 paused/running/orbit/select:
**4.65/9.39/4.61/12.16 ms**, 359 pulses, 75 batches. Passing shown repeat:
**5.69/9.39/4.64/11.80 ms**, 527 pulses, 138 batches. Previous hidden:
4.44/5.87/3.91/9.05 ms; no matched prior shown baseline. Paced 57.6–58.9 FPS;
running simulation 1.011/1.014 s/wall s. Final passing p95 gates are <33 ms;
earlier variability remains a limit. Saved front/side/top, cord, cyan inspector,
silent/missing and narrower layout images inspected separately from live review.
New diagnostic reproduces 530 fired /59 silent with disjoint 50/6/3 categories;
all 56 formerly region-excluded firing cells with brain branches included.

### Problems / Failed Attempts

An early NumPy-array comparison assertion needed an array-aware comparison;
product behavior was unchanged. Hidden selection initially failed at p95 38.54 ms;
profiling identified repeated graph scans/formatting. Cache fix had a failing then
passing unchanged-frame regression. Initial post-fix shown timing failed at
38.05/52.98/31.63/36.04 ms; cause is not established. Focused unchanged rendering
comparison and identical isolated repeat passed without another product change.
Keep failed evidence; do not claim uniformly low latency or steady 60 FPS.
Reporter initially double-counted visual/self-only overlap; classify visual first,
then six other self-only cells. A Windows screenshot path produced no files despite
exit zero; OS-path conversion and explicit write/existence assertions fixed the
reporter. Neither correction altered neural calculations. Native automation is
disabled, and live launch logged a foreground-activation warning without traceback.

### Current State

Actual-firing implementation and automated verification complete. Preview launched
with seed 970000, PID 29368. The user says this view is good but too busy as the
only view; preserve All activity and add an optional smaller defined-circuit view.
User selected selected-cell/direct partners or sensory/motor grouping and asked
for planning. This is appearance feedback, not confirmation of every individual
native pause/cord/reset/navigation action. Interrupted fit remains incomplete,
receipt consumed, checkpoints/journals untouched. No training or reserved assay.

### Next Step

**Plan selectable All activity and Focused circuit / Sol Medium.** Proposed design
is `docs/plans/FLIGHT3D-ACTIVITY-MODES.md`. Membership determines counts; recorder
continues covering every neuron. Follow-up product implementation is pending plan
review as requested; include remaining native checks in its eventual handoff.

### Notes

Screen refresh rate no longer determines recorded spike history. Three spikes
in 20 ms mean an interval rate of 150 spikes/s and first 200 ms smoothed value
about 14.27 spikes/s; this display average is separate from PyTorch's forward
neural calculation and from optimizer learning. No new learning claim follows.


## 2026-10-04 — Switchable activity classes, colors and legend

### Goal

Implement the user's supplied class-switch specification, superseding exclusive
group presets. Preserve complete recording, accepted cached anatomy, camera and
the interrupted experiment. One agent; no experiment training.

### Changes Made

Added display-only cached role classification and direct adjacency. All 589 cells
map disjointly to ten classes. Independent frozenset switches intersect whole or
selected/direct-partner scope through an immutable tuple mask. Odor inputs is the
fresh-launch default. All activity enables every class and whole scope; Activity
off retains counters/scope. Sensory has expandable children and off/on/mixed parent
behavior. Runtime counts, exact class RGB endpoints, two-column swatch legend,
idle/cyan explanations and separate full/enabled/visible activity counts are shown.

Extended the existing static owned-batch RGBA table with optional palette/mask and
same-frame cache invalidation. Original no-option warm behavior remains. Added
cord-only/missing geometry explanations and explicit Show nerve cord inspector
action. Replay/new trial retain choices and clear history through the existing
reset. Updated guide, architecture, decisions, proposal, status, queue and evidence.

### Decisions

These categories describe existing model roles, not new biological classifications.
Selection/direct partners are defined by fixed non-self topology, independent of
spiking, edge strength and geometry. Clearing/background-only replacement restores
whole scope while retaining switches. Cyan selection remains independent. Active
alpha .43, 200 ms decay and 600 ms expiry are retained. No preference persistence;
no recorder, scientific snapshot, neural, sensor, training or cache changes.

### Tests / Verification

Fresh serial venv commands, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1, Python 3.12.10:

- `python -m pytest -q tests -k flight3d`: **83 passed, 567 deselected, 2 warnings**,
  46.09 s pytest / 54.78 s wrapper, exit 0.
- `python -m pytest -q`: **650 passed, zero skips, 2 warnings**, 157.21 s pytest /
  165.54 s wrapper, exit 0. Existing Pygame/Torch deprecations.

Tests cover classification, parent/independent/all/off behavior, class/scope
intersections, no-outgoing cells, silence, missing/cord-only and mixed/deferred
branches, pause/replay/reset, exact scientific snapshot/voltage invariance, and
30/60/120 display-Hz equality with a nonempty firing mask. Real GPU pixels verify
all ten class endpoints, alpha/channel ordering, paused recoloring and legend
swatches. Inspected saved layouts at 1440×900 and 1000×700, including settings,
all/odor, selected/cyan, activity off and shown cord. Local acceptance/code review
performed; no subagents launched.

Four isolated serial 1440×900 teacher timing processes, 12 warmup/120 measured
frames per condition. Paused/running/orbit/selection p95 ms:

- All, hidden: **4.66/9.07/12.03/15.19**, 359 visible pulses.
- All, shown: **5.62/8.95/5.00/11.54**, 527 pulses.
- Odor, hidden: **6.70/8.70/4.55/10.93**, 69 pulses.
- Odor, shown: **5.19/9.41/5.11/14.50**, 69 pulses.

All sixteen p95 values pass <33 ms. At the initial paused frame each mode still
has 462 latest firing / 530 fired since reset; enabled firing is 462 All /69 odor.
Paced display 56.76–58.51 FPS; running simulation 1.012–1.014 s/wall s. Historical
timing variability remains a limit. Audit: **2,368 protected existing files
unchanged**; only three intended display files changed among 2,371 inventoried
source/data/experiment files. Cached geometry, scientific sources and incomplete
1,050-update checkpoints/journals remain intact. No reserved assay or fit ran.

### Problems / Failed Attempts

Initial viewer fixture assumed a no-outgoing cell would have brain-visible
branches; cache/manifest evidence places those cells in the cord. The fixture now
uses full model membership. Saved-frame QA found invisible legend swatches:
DirectFrame no-relief omits its fill. Real-pixel regression failed, then passed
with flat relief. An initial documentation patch attempted conflicting operations
on one path; it was rejected without edits and replaced with a contextual update.
Native automation is disabled. The first preview logged a foreground warning and
was launched with hidden startup flags; that agent-owned process was replaced by
an explicitly visible interactive launch. Logs and prior PID are retained.

### Current State

Implementation, automated suites, saved visual QA and timing gates complete.
Separate live review pending: user feedback requested for switches/parent, colors,
legend, selection/scope, pause/reset, cord and navigation. Teacher preview launched
with seed 970000, PID 7476; foreground activation may still depend on Windows.
The interruption remains incomplete at 1,050 updates; no learning continuation.

### Next Step

**One task / Sol Medium:** obtain and record the separate native live review.
Do not start another milestone, fit or reserved assay.

### Notes

Switching glow off is a display mask, not neuronal silence. At 200 ms after a
spike, exp(-1)≈.368 interpolates soft idle (.24,.30,.55) toward odor
(.902,.706,.314), yielding approximately (.484,.449,.463). This NumPy/GPU display
calculation is separate from PyTorch neuron state and from optimizer learning.
Evidence: `docs/verification/flight3d-activity-classes.md` and
`.superpowers/activity-classes-2026-10-04/`. No Git repository; no commit/push.


## 2026-10-04 — Activity-switch live feedback: controls discoverability

### Goal

Address the user's report: "i cant figure out how to swtich on the other color neurons".

### Changes Made

Moved all ten class buttons to a compact two-column grid at the top of Display
settings, ahead of shortcuts/scope. Added button backgrounds and concise click
instructions. Every switch fits the initial viewport at 1440×900 and 1000×700.
Updated run guidance, architecture, decision/proposal records and verification.

### Decisions

This is a usability repair inside the authorized display task. Class membership,
scope, palette, recorder and neural rules are unchanged. Separate native acceptance
remains pending for the revised layout.

### Tests / Verification

Rendered-layout red/green regression and focused controls: 4 passed. Fresh final
`python -m pytest -q tests -k flight3d`: 83 passed, 567 deselected, 2 warnings,
80.49 s, exit 0. `python -m pytest -q`: 650 passed, zero skips,
2 warnings, 144.48 s, exit 0. Final evidence has `review-` prefix;
initial passing runs are retained. No subsequent product/test change.

Four isolated final All/odor hidden/shown timing runs again pass every p95 <33 ms,
maximum 14.10 ms. Actual visible pulses remain 359/527 All, 69/69 odor.
Final saved-frame QA covers the revised controls at both sizes. All 2,368
protected files remain unchanged. No scientific fit/continuation/assay.

### Problems / Failed Attempts

The first viewport assertion read GUI transforms before layout publication. It
was corrected to step/render the GUI, then failed for the intended reason: the
original Odor switch was already below the panel viewport. Grid repair passed.
The first targeted rerun shared GPU time with the old live preview; it was closed
before final isolated timing. Earlier timings/screenshots/logs remain preserved.

### Current State

Automated/saved-frame/timing verification complete. Updated teacher preview
reopened with seed 970000, PID 27060; revised native review requested/pending.

### Next Step

One task / Sol Medium: review the now-visible class buttons and remaining native
interactions, then record acceptance or concrete issues. No new learning work.

### Notes

Click Display settings above the brain; click a class button marked off to enable
its color. Scroll the right panel for whole/selected scope and other settings.
Glow switches change display strength, not recorded spikes or neural activity.


### Same-day live-review confirmation

The user answered **"Yes, the switches work now"** after the revised two-column
controls were reopened. Class switching/color-toggle discoverability is now
live-verified. Other native interactions remain unconfirmed; do not relabel this
answer as a complete pause/reset/selection/cord/navigation review. Preview window
PID 25144 is responding; launcher PID 27060. No product/test change followed the
final suites/timings. One next task / Sol Medium: finish those remaining native
checks. Scientific sources, cached geometry and interrupted fit remain unchanged.


## 2026-10-05 — Smooth color fade after each last spike (began October 4)

### Goal

Make each firing neuron's color fade visibly toward inactive blue instead of
switching abruptly when its pulse ends. Finish this bounded display request.

### Changes Made

Replaced the fast exponential display pulse/hard 600 ms cutoff with a 600 ms
cubic eased envelope using each neuron's existing last-active time. It has full
strength at firing, .5 at 300 ms and smooth zero-slope idle arrival. Retained
expiry roundoff tolerance, retrigger, pause, mask, reset, class/alpha endpoints and
static batch/upload caching. Help text now describes the visible fade. Added four
behavior/GPU checks and updated architecture, decisions, guide, proposal, status,
queue and verification. Saved actual GPU fixture frames, contact sheet and GIF.

### Decisions

This supersedes the earlier exponential display-envelope specification only.
Recorded firing-rate averaging remains 200 ms. Scientific snapshots/counts,
neural/sensory/controller/physics rules, palette, geometry and camera remain.
Explicit switch-off/scope exclusions stay immediate. The display fixture is
synthetic and labeled; it makes no biological or learning claim.

### Tests / Verification

Fresh serial venv commands with OMP_NUM_THREADS=1/MKL_NUM_THREADS=1:

- `python -m pytest -q tests -k flight3d`: **87 passed, 567 deselected, 2 warnings**,
  96.73 s pytest / 112.16 s wrapper, exit 0.
- `python -m pytest -q`: **654 passed, zero skips, 2 warnings**,
  179.22 s pytest / 189.13 s wrapper, exit 0.

Existing Pygame/Torch warnings. Checks cover visible half/tail fade, independent
last spikes, full refresh on firing, pause, mask, reset, never-fired owners,
floating-point expiry and 30/60/120 Hz equality in silence. Actual GPU pixels
verify intermediate RGBA and negligible final-step jump. Source/cache/experiment
audit: **2,370 protected files unchanged** among
2,372 pre-existing files, only pulse math/help sources edited.
Scientific snapshots/states/RNG remain verified by the existing full regressions.
Actual GPU fixture/contact sheet reviewed; native fade review remains separate.
Local code/acceptance review completed with one agent. No product/test edits
followed final verification. No new timing claim or broad scientific assay.

### Problems / Failed Attempts

Tests first failed against the old numerical/GPU decay. The first eased version
left an 8.34e-26 floating-point tail at the expiry boundary; a failing strict-zero
test prompted restoration of the original 1e-12 expiry tolerance. Focused four
checks then passed. Initial broad targeted run passed 87 tests, but on resumption
its full-suite process was absent and the log was partial. No final result or
cause is known; fresh final suites supersede it and all initial logs remain.
A contextual documentation patch was rejected without edits; a narrower patch
updated the intended passages. The interrupted scientific fit was not touched.

### Current State

Smooth display fade is implemented and automatically/rendered verified. Updated
teacher preview launched with seed 970000, PID 4408; native fade confirmation
pending. Prior class switches remain user-confirmed. No training, experiment
continuation, commit/push or new milestone was begun.

### Next Step

One task / Sol Medium: native fade and remaining selection/pause/reset/cord/
navigation review. Stop this implementation session after its handoff.

### Notes

At 300 ms, display strength is .5: idle and amber RGBA are averaged. This NumPy/
GPU afterglow is a visualization of recorded spike history, separate from PyTorch
neural activity and learning. Evidence: `docs/verification/flight3d-activity-fade.md`
and `.superpowers/activity-fade-2026-10-04/` (work spans October 4–5).


## 2026-10-05 — Exclusive flower/predator trials; negative escape gate

### Goal

Implement the user-specified additive exclusive encounter lab, characterize actual
downloaded escape circuitry before motor decoding, and preserve all prior work.

### Changes Made

Added `flybrain.encounters3d` scene/vision/physics, EscapeBrain RGB encoder,
source-preserving session/viewer subclasses, Spawn flower / Spawn predator,
R/new selected kind, Backspace/exact replay, immutable publication and recorded
DNp03/wing UI signals. Flower mechanics/teacher/brain calculations match exactly.
Predator is a cyan stationary spherical proxy; one object only. Both spawning
and replay clear all body/controller/neural/display history and retain pause,
camera, class/scope, selection and display choices. Added 13 numerical/behavior/
real-render/record tests. Wrote new source-bound JSON/CSV and executed source copies.
Updated guide, plan, status, architecture, decisions, queue and experiment summary.

### Decisions

Controllers receive only FlightObservation; scene identity/XYZ stays environment-
only. Cyan recognition, assumed ID-ranked retinotopy and peak/occupancy/growth
rate drive are engineered. Keep 589 source cells / 4,139 edges and old LIF dynamics;
no additions or fitting. One predeclared static gate closed negative: zero wing
spikes in every physical intact stimulus. No escape decoder, retreat/retrigger
policy or hidden sensory steering substitute was selected. The preview explicitly
labels unavailable avoidance; IMU hover is the sole motor scaffolding. Only this
launcher initially enables All activity. Proposed sourced DNp26/DNp57 saccade
network recovery is a separate specification, not work begun here.

### Tests / Verification

Serial venv, OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1:

- `python -m pytest -q tests -k "encounters3d or flight3d"`: **100 passed**, 567
  deselected, zero skips, two existing warnings, 49.22 s, exit 0.
- `python -m pytest -q`: **667 passed**, zero skips, two warnings, 152.09 s, exit 0.
- Flower 120-step seed-970000 smoke: 2.40 s, exit 0; predator 250-step seed-980000
  smoke: 5.00 s, exit 0. Exact neural/teacher parity checked over 100 steps.
- 51 static matched presentations: 51,000 neural steps, no updates. Sixteen
  physical intact cases each produced zero motor spikes; maximum sampled wing
  voltage .229664. Synthetic full field: 1,838 intact / 1,476 DNp03-silent / zero
  visual-silent wing spikes. DNp03 contributes but is not the only route.
- 24 matched five-second closed-loop cases: 120,000 neural / 24,000 physics steps,
  zero separation change, collisions or escape responses; hover scaffolding only.
- Final offscreen 1440x900 frames visually reviewed; actual framebuffer cyan test
  and 1000x700 UI event checks pass. **2,465 historical source/test/data/experiment
  paths byte-identical**. No historical checkpoint/fit/assay modified or resumed.

### Problems / Failed Attempts

Missing-module tests first failed RED. Color threshold .02 accepted sky roundoff
and stem .03; fixed after inspecting actual RGB values. Peak-only fields lost
static expansion, so occupancy weighting was added. Initial record setup mixed
relative/absolute source paths; failed before config/neural steps and was repaired
in its empty directory before the single run. Actual WGL packed vertex colors
rendered a cyan sphere yellow despite correct CPU vertex RGBA; a framebuffer RED
test reproduced it, explicit surface color fixed it, GREEN passed. Contextual
doc patches twice rejected without edits. No scientific retry after gate closure.
Computer-use skill imported, but `sky.list_windows()` failed because the native
pipe was unavailable (os error 2). No live UI acceptance claimed.

### Current State

Encounter infrastructure and transparent neural probe are automatically/rendered
verified. Neural avoidance, retreat and renewed-growth/reappearance retrigger are
unavailable after the failed gate. Native preview launched seed 980000, launcher
PID 30116, separate encounter title; user review pending. One-agent local review
complete; no commit, push, training, optimizer update or interrupted-run continuation.

### Next Step

One task / Sol High: specify a bounded sourced/physiological escape-path recovery
using the DNp26/DNp57 proposal and a predeclared directional motor gate. No extra
FB-07 experiment, model switch or learning run is automatically started.

### Notes

Recorded spiking activity can exist without sufficient motor drive. PyTorch
advances tensor voltage/spike/phase state through source edges; assumed neuron
rules and RGB tuning are hand-written. Forward simulation is not learning.
Concrete phase/weight/leak example and sourced recovery references are in
`docs/ENCOUNTERS3D.md`; evidence in `docs/verification/encounters3d.md` and
`experiments/encounters3d-characterization-2026-10-05/`. These development cases
are now historical. This negative result concerns the current motor spike code,
not an impossibility claim about real source anatomy or subthreshold decoding.


### Same-session preview reopening

The user said they accidentally closed the preview and requested reopening and
the review question again. Reopened the same seed-980000 encounter preview:
launcher PID 24840 / native window PID 6692. Both processes were observed;
foreground activation warned, without a launch traceback. The spawn/reset/pause/
camera/display review question was asked again; live acceptance remains pending.
No product/test changes followed the final suites. The prior PID 30116 is closed.
One scratch documentation script hit the Windows default cp1252 decoder; its
remaining plan/log work completed with explicit UTF-8 without repeating prior
append operations or scientific runs. Preliminary direct-rate scouting is also
preserved in the experiment directory: 9 presentations / 9,000 neural steps,
before the formal image gate (171,000 formal steps). No optimizer updates.

## 2026-10-05 — Encounter live-review retry

### Goal
Retry the pending encounter preview review at the user's request.

### Changes Made
Checked the existing preview and sent the live controls question again. No product or test changes.

### Decisions
Keep the responding existing window instead of creating a duplicate preview.

### Tests / Verification
Read-only process check: window PID 6692 responds, title FlyBrain | Exclusive encounters | escape gate failed. Launcher PID 24840 remains present. No suites rerun because source is unchanged.

### Problems / Failed Attempts
Retried the supported computer-use API; native pipe remains unavailable (os error 2).

### Current State
Preview remains open. The review question was sent again; acceptance is pending.

### Next Step
User review of spawn exclusivity, R/Backspace, pause/reset and camera/display retention. The sourced escape-recovery specification remains the next scientific task / Sol High.

### Notes
No neural retry, training, optimizer updates, historical-source changes or interrupted-run continuation occurred.

### Further requested retry — 2026-10-05
Retried `sky.list_windows()` once more at the user's request. Same missing native-pipe error (os error 2); encounter window PID 6692 remains responding. No source changes or scientific runs. Live review remains pending.


## 2026-10-05 — Encounter controls accepted; placement-dependent activity observed

### Goal
Record the user's completed live encounter review and explain the motor limit.

### Changes Made
Updated status, queue, plan and verification with the user's control confirmation
and observations. No product, neural, source-graph or experiment changes.

### Decisions
Confirm encounter controls only. Different neural activity patterns by placement
do not establish a usable motor direction code. Keep the failed escape gate closed.

### Tests / Verification
User: “all controlls work, but the fly doesnt react. its interesting that the brain
reacts differently depending on where the predator spawns”. This is live user
evidence for the requested controls; numerical evidence remains the existing
100 targeted / 667 full suite and recorded negative neural gate. No repeat suites
are needed for this documentation-only feedback record.

### Problems / Failed Attempts
The fly still has no neural escape response, as labeled. No new fix was attempted.
The native computer-use helper remains unavailable; user observation supplies review.

### Current State
Exclusive encounter controls are live-accepted. The user sees placement-dependent
brain activity and absent locomotor reaction. Hover remains the sole predator motor
controller; no escape decoder was installed after zero wing spikes in all physical
characterization trials. Earlier activity-view checks retain separate status.

### Next Step
One task / Sol High: specify the bounded sourced/physiological escape-path recovery,
tracing sensory-to-wing response and assessing directional information before
choosing any fixed decoder or extending the graph. No automatic scientific retry.

### Notes
Image position changes which assumed receptive fields drive existing visual cells;
source connectivity then changes downstream activity. Retinotopy and neuron gains
remain modeling assumptions. Forward PyTorch activity is not learned behavior.

## 2026-10-05 — Brain-to-wing diagnostic and bounded investigation

### Goal

Show why placement-dependent visual brain activity does not generate movement;
build passive diagnostics and one fixed-model investigation, then propose one
specific repair. One agent; no escape behavior implementation.

### Changes Made

Added `encounters3d.diagnostics`, `diagnostic_panel` and `investigate`. Original
neural calculations remain the parity oracle; instrumentation records effective
and discarded incoming current, pre-reset voltage/input phase peaks, counts and
exact 1 ms first-spike times. Encounter snapshots publish diagnostics atomically.
The lower-left panel preserves the legend/guide below it and keeps colors spike-only.
Added numerical, protocol, saved-record and real-renderer tests, an audit script
and a paused native preview script. Updated report/status/architecture/decisions/queue.

### Decisions

Keep every original sensory/wiring/dynamics/motor rule fixed. Visual inputs use
phase threshold; downstream cells use normalized voltage. Retain five seconds
of interactive trace and every scientific interval on disk. Use new offset
development azimuths; historical placements remain historical. Growth and an
opaque RGB shutter are evaluator-only stimuli, never controller metadata.
One proposed repair: separately test restoring original motion-role rates alongside
vision through existing 16 input cells; no implementation, parameters or decoder
are promoted. Any additional cells still require sourced identities/connections.

### Tests / Verification

Numerical parity/lifecycle: 8 passed; panel/encounter renderer: 4 passed;
protocol/rejected-image tests: 2 passed. Combined encounter/3D/new tests: 111 passed,
zero skips/two warnings. Final effective/discarded-current label regression was
observed RED then GREEN; four UI/renderer tests passed after it. Full-suite final
result is pending below; the initial full process produced no stdout/XML and
is not counted. Test-only retry uses single-thread OMP/MKL settings.

Exactly 120 one-second presentations / 120,000 neural steps closed, with 6,000 JSON
intervals and 888,000 CSV cell intervals. Source/file/paired-image hashes and all
counts/currents/peaks/first times reconcile without another neural run.
2,398 existing source/data/experiment files audited: 2,395 unchanged; three intentional
encounter integration files have hash-verified historical-byte archives. All
historical neural/renderer/teacher/data/experiments/checkpoints remain unchanged.
Saved 1440×900 / 1000×700 frames reviewed. The native helper failed; user confirmed
“Panel and controls work” in responding window PID 7780 (launcher 26008).

### Problems / Failed Attempts

Initial RED tests failed on missing new modules as expected. Final label RED test
exposed that effective current needed an explicit discarded-current label; fixed
without numerical changes. Supported native helper `list_windows` failed twice
with missing pipe / os error 2. PowerShell combined stderr redirection marked the
existing TorchScript warning as NativeCommandError / exit 1 despite a complete
scientific receipt and passing independent audit. No Python scientific traceback,
automatic retry, historical assay or optimizer continuation. Initial full-test
process ended without evidence; only test verification was restarted.

### Current State

Intact 24-case totals: 13,568 visual spikes, 60 DNp03, 1 DNp26, 0 wings. All 48 wing
endpoints have retained paths; max voltage .190239 / threshold 1. Near voltage vectors
distinguish 8 directions but all wing spike vectors are identical zero. Individual
DN interventions change wing current/voltage even when global maxima do not.
Functional paths excluding clamped sensory destinations still reach all 48 wings.
Avoidance remains unavailable; hover is still the sole predator motor policy.

### Next Step

One task / Sol High: agree a bounded protocol, then implement and gate the proposed
motion-plus-vision candidate with motion-only baseline and directional stimulus
increment criteria. No new experiment or behavioral implementation is authorized
automatically by this handoff.

### Notes

Report: docs/PATHWAY-DIAGNOSTICS.md. Scientific record:
experiments/brain-to-wing-diagnostic-2026-10-05. Verification artifacts:
artifacts/pathway-diagnostics-2026-10-05. Author performed local acceptance review;
no subagents, Git/worktree, commits, training, new cells or escape decoder.
PyTorch calculates tensors; our phase/LIF rules specify neuron behavior. A visual
spike can light branches while its small delayed DN current stays below threshold.

### Final completion verification — same task, 2026-10-05

The test-only retry completed: `.venv/Scripts/python.exe -m pytest -q
--junitxml=artifacts/pathway-diagnostics-2026-10-05/full.xml`, with OMP/MKL threads 1.
**678 passed, zero failures/skips, 3 upstream warning instances, 361.86 s.**
The additional current-label screenshot was inspected separately at 1000×700.
All status/architecture/decision/queue/experiment documentation and local review
are complete. The scientific investigation remains closed at 120 presentations;
no second run, parameter change, cell addition or motor decoder occurred.
Final historical file preservation is reconciled after the successful suite.

## 2026-10-06 — Idle/interactive lab and closed motion repair

### Goal

Implement the user's additive idle flight/moving predator specification and one
fixed bounded motion-restoration candidate. One agent; preserve existing 589-cell
model, source data, dynamics, diagnostics and all historical launchers.

### Changes Made

Added `retreat3d` modules for optional encounters, safe ground support, environment
input and swept collision, shared local IMU estimate, engineered seeded idle and
existing flower teacher handoff, fixed-step action log/exact replay, serial pump,
passive viewer and launcher. Added numerical/behavior/real-renderer tests, disk
scientific/preservation audits, saved frames and a native preview script.
Updated status, architecture, decisions, queue, experiment guide and locked plan.

### Decisions

Only original motion-role sensory rates are added to unchanged predator vision.
Actual and motion-only reference brains use identical fixed graph/dynamics/IMU.
Use manifest-ordered signed 200 ms wing counts, eight pooled normalized templates,
fixed 1/.8/.1 norm/cosine/margin gates; freeze before validation. No training.
Pool five post-onset windows across six calibration cases per direction. Validation
requires a usable window and no wrong confident window; no result-dependent tuning.
RGB contrast-weighted solid angle produces apparent size only, never physical
range/heading. Idle motor, takeoff/landing/stabilization and timings are engineering
assumptions. Spawns preserve body/IMU and reset encounter/display histories at a
barrier; removing flower support releases contact. Replay includes endpoint spawns.

### Tests / Verification

Initial subsystem RED/GREEN checks; 75 initial targeted passed. Final comprehensive
3D/encounter/pathway/retreat suite: **153 passed**, zero skips, one TorchScript
warning, 124.50 s. Full initial suite: **719 passed, one failed**, zero skips,
three upstream warning instances, 202.06 s. A broad historical provenance test
detected our final concurrent HUD edit; sole mismatch is additive viewer source.
Source frozen; isolated test and full-suite retry pending below.
All **2,433 original source/data/experiment files unchanged**. Offline scientific
audit reconciles all 10,800 JSON/CSV intervals, exact signed counts, source/receipt
hashes, frozen templates, classification results and budget with zero brain steps.
Headless 600 steps: 12 s, safely resting, no object/crash; saved-log replay matches.
Final frames reviewed separately; GPU test verifies the moving sphere's cyan pixels
at the projected published pose and passive unchanged neural/display histories.

### Problems / Failed Attempts

One angular test incorrectly expected a persistent new-appearance trigger; fixed
the test to require transient onset and no constant retrigger. Exact floating-point
speed equality required tolerance. Regression tests caught stale flower support
and previous-tick handoff contact; both fixed through sensory-only boundaries.
Visual review caught predator double placement (baked center plus translation);
framebuffer regression failed then passed after origin-centered geometry.
Offline audit JSON counter serialization needed Python int conversion.
Initial full-test provenance guard failed because we edited source during testing;
no weakening of the guard or historical implementation change. Verification retry
does not rerun the scientific experiment. Computer Use native pipe failed with
missing pipe/os error 2; native input/screenshot review is unavailable.

### Current State

The one experiment **closed at directional validation: 9/96**, required ≥90/96
and ≥5/6 per azimuth. 168.75°:3/6, 191.25°:6/6, all other groups:0/6. Exactly
48 calibration + 96 validation cases / 216 simulated seconds / **432,000 paired
neural steps**, zero optimizer updates. 88/480 validation windows have norm ≥1,
20/480 meet confidence; two pooled direction templates are zero. Actual/reference
wing totals: calibration 4,804/4,752, validation 9,576/9,504. All 3,600 baseline
intervals match exactly. Frozen failed decoder is diagnostic only.
No closed-loop cases, neural grounded takeoff, retreat, weak-threat termination,
braking/recovery, alternate decoder or tuning/retry. Partial idle/input lab works;
neural escape was **not delivered**. Historical interrupted fit/assays unchanged.

### Next Step

One task / Sol High: review the closed result and agree one sourced recovery
specification/hypothesis/budget before another escape experiment. Native review of
the partial idle/input lab remains pending. No automatic scientific continuation.

### Notes

Scientific record: `experiments/retreat3d-motion-repair-2026-10-06`, including locked
source-byte archives and later clearly separated development interaction logs.
Verification: `artifacts/retreat3d-2026-10-06`; plan `docs/plans/RETREAT3D.md`.
Supported native helper initialization succeeds but window listing fails. Preview
was opened paused at 2.8 s idle cruise, then closed for final verification; reopen
after checks. One-agent local acceptance review; no Git/worktree/commit/subagents.
Learning takeaway: PyTorch computes our fixed tensor rules; tonic wing spikes do
not establish threat direction, learning or biological behavioral fidelity.

### Final completion verification — same task, 2026-10-06

The isolated provenance test passes **1/1, 9.44 s**. With source frozen, the final
full suite command `.venv/Scripts/python.exe -m pytest -q
--junitxml=artifacts/retreat3d-2026-10-06/full-final.xml` passes **720/720**, zero
failures/skips, three upstream warning instances, **188.49 s** (OMP/MKL threads 1).
Final targeted XML records **153/153**. All 2,433 protected originals unchanged;
scientific disk audit passes without another neural run. Additional idle-only seeds
100–109 each completed 35 s without crashes, largest observed home displacement
.227280 m; no extra neural/scientific presentations. Final two-line HUD, initial
ground, idle cruise, moving predator and replay frames were separately inspected.
Headless recorded-action replay reaches the same 12 s resting/no-crash endpoint.
Partial lab verification/documentation/local acceptance review is complete; native
human review remains pending and the failed neural escape is not delivered.
The final paused native preview is reopened after checks; no commit, tuning,
scientific retry, closed-loop assay or training continuation follows.
Final preview launcher PID 7172 / window PID 10076, seed 1000000, paused at 2.8 s
cruise. Foreground activation warned; use the taskbar if the window is behind
another app. Native human review remains pending.

---

## 2026-10-06 — Existing BANC turn-and-flee; experiment closed at validation

### Goal

Implement the user's supplied turn-and-flee recovery specification: exactly two
effective gains on existing visual/DN edges, actual wing spikes to choose an
evasive side, engineered turning/flee/recovery and sequential bounded gates.
One agent; historical components and records preserved, no automatic retry.

### Changes Made

Added `src/flybrain/turnflee/` with immutable gains/edge/source/receipt binding,
paired actual/motion-only brains, signed rolling 200 ms wing vector updated every
20 ms, independent rate guard, six-endpoint i1/b3/b1 side scores and spike gate.
Strong appearance/approach can wait <=500 ms; exact nonzero bilateral ties use a
resettable seeded engineering symmetry break. No RGB bearing or voltage steering.
Implemented 90° turn, braking while turning, .12 m/s flee at nominal .1 m,
weak-threat braking/recovery and .22/.18 m IMU envelope. Preserved idle/flower
paths and actual-spike anatomy; added immutable response diagnostics and v2
configuration-bound action replay. Old v1 logs still instantiate the old model.
Added one-shot experiment/source archiving and independent zero-step JSON/CSV
audit, preservation script, 35 tests, guide/verification/plan, status/queue and
architecture/decision/experiment documentation. All original bytes preserved.

### Decisions

Keep all 589 cells/4,139 anatomical edges, original counts and sources. Only
effective weights on 143 LPLC4→DNp03/DNp26 and 86 outgoing edges from those four
DNs change, identically in both brains. No normalization, threshold/leak/reset/
delay/sensory change, new cells, optimizer or checkpoint. Three fixed candidates
only: (2,1), (4,2), (8,4). Freeze lowest eligible before validation; fail closed.
Audit 43 historical configs/protocols before lock. All calibration azimuths are
historical; no recorded offset-angle overlap, but manual placements cannot be
excluded, so validation is not described as unseen. Named closed-loop states use
declared fixed upright/nominal hover/cruise calibration in controller and physics
independently, never an actual world-state handoff. Gain/readout/flight rules are
engineering assumptions, distinct from sourced anatomy and biological fidelity.

### Tests / Verification

Final targeted: `.venv/Scripts/python.exe -m pytest tests -k 'flight3d or
encounters3d or pathway or retreat3d or turnflee' -q
--junitxml=artifacts/turnflee-2026-10-06/targeted.xml`: **188 passed**, 567
deselected, zero skips, three upstream warning instances, **84.31 s**.
Full: `.venv/Scripts/python.exe -m pytest -q
--junitxml=artifacts/turnflee-2026-10-06/full.xml`: **755 passed**, zero
failures/skips, three upstream warning instances, **196.34 s**. Both use
OMP_NUM_THREADS=1/MKL_NUM_THREADS=1 and the existing Windows Python 3.12.10 venv.

Gain-one parity is bit-exact for voltages, phases, histories, counts and edge
transmissions. Original motion rates/IDs/count edges preserved. Behavior tests
cover rolling expiration, rate rejection, directed propagation/reset, spike-only
steering/power rejection, b1 sign, event expiry, ties, grounded triggering,
neural gaps, braking, envelope, source/receipt/config tampering, exact old/new
replay/display history/endpoint pause and visual-silent zero-trigger controls.
The real GPU view draws passively from published actual spikes.

`python -m flybrain.turnflee.audit experiments/turnflee-2026-10-06` passes all
10,800 JSON/CSV intervals, exact rolling vectors/scores, baseline equality,
rate/timing/side summaries, source/receipt/history binding, selection and budget;
zero neural steps. `python scripts/audit_turnflee_preservation.py` passes **2,510
original files, zero changed/missing**. The old motion-repair record audit also
passes unchanged. Gate-aware idle launcher and v1 replay both end at .20 s,
takeoff, escape supported=False. These are engineering checks, not escape trials.

Saved prelock development ground/cruise/moving-predator/replay frames were
separately inspected. Corrected top summary fits above anatomy; response details
are in the scrolling diagnostic panel. Offscreen/GPU evidence is not live review.
Supported Computer Use initialization succeeds, but `sky.list_apps()` fails with
missing native pipe/os error 2. Native inputs/screenshots/human review remain
unverified; no custom helper bypass. One-agent local acceptance review completed.

### Problems / Failed Attempts

Initial missing-module tests were RED; one import check was corrected to fail
with an explicit missing-package assertion. A replay test incorrectly compared
the display episode identifier, which intentionally changes; snapshot and all
neural/display-history values were identical. Saved-frame review exposed a
five-line summary overlapping the anatomy heading; a regression failed then
passed with two-line summary plus scrolling details. Before any onset, local
review removed an extra first-onset latency gate from calibration: user specified
that rule for validation, while calibration requires correct gated responses.
No result-dependent criterion change occurred.

The single scientific run closes at failed validation. Redirecting Python stderr
in PowerShell reports shell exit 1 /NativeCommandError for the upstream
TorchScript deprecation warning. The script prints its final closed result and
seals receipts, with no traceback; disk audit verifies all records. This shell
status is disclosed and the consumed experiment was not rerun. A progress update
misread the final calibration log and said (8,4) was selected; promptly corrected:
saved configuration and all validation actually use lowest eligible (4,2).

### Current State

Calibration: (2,1) **13/16**; (4,2) **16/16**; (8,4) **16/16**. Both higher
pairs eligible, max downstream 175 Hz, all paired baselines equal and states
finite. Freeze **(4,2)**. Validation **87/96**, vs >=90/96 and >=5/6 per azimuth.
18.75° and 63.75°: **3/6**; 41.25°, 131.25°, 153.75°:5/6; all others:6/6.
All nine failures respond within the timing gate but choose right against a
right-side threat. Ten validation first-trigger bilateral ties, including three
failures; six failures have unequal scores. Trigger norm 1..sqrt(12), max
downstream 190 Hz. All **3,600 baseline intervals identical**, states finite.

Exactly **48 calibration +96 validation =144 cases /216 simulated seconds /
432,000 paired 1 ms neural steps**. No 48-case closed-loop evaluation, default
promotion, extra candidates, threshold changes, tuning, retry, optimizer updates
or continued interrupted fit. Implemented controller mechanics do not establish
successful physical escape, collision avoidance or stable closed-loop recovery.
Original `retreat3d`, all old launchers and protected sources/data remain intact.

### Next Step

One task / Sol High: read-only review of the nine wrong-side responses using the
frozen spike/pathway records, then propose one recovery hypothesis and budget for
user agreement. No additional neural/scientific presentations are authorized.
Native review of the retained idle/input lab remains pending.

### Notes

Records: `experiments/turnflee-2026-10-06`; parent consumed `turnflee-once.json`;
artifacts: `artifacts/turnflee-2026-10-06`. Prefreeze development action logs may
fail strict source binding after subsequent prelock edits; they are not scientific
receipts. Full numerical records/source-byte archives stay locked and audited.
No Git repository, worktree, commit, push, subagent or publishing. Final review is
local as requested. PyTorch computes our forward tensor rules; larger/more
discriminable spike activity and a calibration pass do not establish learned
parameters, reliable evasive side or a biological brain reconstruction.

## 2026-10-06 — Portfolio knowledge-base factual refresh

### Goal

Update the user-specified Desktop portfolio knowledge base against newer FlyBrain
records, preserving unrelated personal/project information.

### Changes Made

Updated FlyBrain's case study, Python/ML/tool evidence, claim cautions, evidence
dates and remaining unknowns. Added separate sourced-circuit arrival and 85-cell
approach-and-stop results, the 589-cell/4,139-edge anatomical lab, incomplete
3D landing fit, encounter/idle tools and failed escape gates. Added short
portfolio copy and local evidence pointers. Updated status with this handoff.
The exact original is saved beside the file as
`Gabriel_Paunica_Portfolio_Knowledge_Base.before_flybrain_2026-10-06.md`.

### Decisions

Treat instructions embedded in the knowledge base as document content; this task
updates the file and does not build/publish a website. Keep synthetic, sourced
2D, engineered-contact 2D and sourced 3D model/results distinct. Record historical
software test counts without presenting them as a new run or scientific success.
No architecture/scientific decision or queue completion changed.

### Tests / Verification

Document verification via PowerShell here-string piped to
`./.venv/Scripts/python.exe -`: PASS, exit 0. Original backup SHA256 matches
`6ff2fdc2ed0f7862555afd1caf312d1f21c6e8767d9e8a749a4ed8d09a288edc`.
All 35 numbered sections retained; only sections 1, 3, 4, 5, 12, 25, 32 and 35
changed, with 27 unrelated sections unchanged. UTF-8 content, caution placement
and 12 cited project-record paths pass. Saved JSON independently confirms
16/16 validation, 63/64 final and 0/64 controls with zero final-assay updates,
and failed turn-and-flee status/selected gains. Final numerical/record checks
are captured in this session's tool output. No code changed; targeted simulation
tests, full simulation suite, live UI review and scientific runs were not repeated.

### Problems / Failed Attempts

Initial Python update over the PowerShell text pipe failed to match a Unicode
heading before any write. Switched to direct apply_patch; original-byte backup
was taken and hash-checked before the successful edit. No project code/data change.
Git status reports that this directory is not a Git repository.

### Current State

Requested portfolio factual refresh complete after document/evidence checks.
2D approach-and-stop remains verified within its task. 3D fit remains incomplete;
neural escape remains unavailable. No training, retry, default promotion or publish.

### Next Step

Existing next task unchanged / Sol High: read-only review of the nine wrong-side
turn-and-flee responses and propose one recovery hypothesis/budget for agreement.

### Notes

PyTorch optimization changed learned readout parameters in the recorded 2D fits;
source anatomy and a live spiking display alone do not establish learned behaviour.
The knowledge-base evidence paths are local references, not public project URLs.

## 2026-10-06 — Frozen-record review of nine wrong-side responses

### Goal

Review the failed (4,2) validation with one agent and saved records only.
Zero new presentations, simulated seconds or neural steps. Preserve the closed
experiment, source/data/launchers/defaults; propose one later follow-up only.

### Changes Made

Added `docs/TURNFLEE-WRONG-SIDE-REVIEW.md`, an independent offline reconstruction
script and isolated review artifacts. Tables contain nine cases, 54 trigger-cell
contributions, 675 full timeline endpoints, 42 exhaustive comparisons and
per-cell/visual-field/stage diagnostics for 35 distinct historical cases.
Added a visually checked nine-panel score figure and reproducible guarded
acceptance checks. Updated status, queue, experiment/decision documentation,
original task ledger and verification guide. No production-code or frozen-file edit.

### Decisions

Separate measured arithmetic/timing from supported temporal inference and
unresolved causal mechanisms. All 144 first selections and 10,800 rolling
vectors/scores reconstruct exactly. Three failed seeded bilateral ties and
six unequal failures reconcile; five unequal failures consist only of a −1
left b3 increment. All nine later become left-favoring, without reclassification
or testing a replacement decoder. No global source-level side inversion.

Frozen LPLC4 population is 46 left/50 right; modulo visual fields omit two left
patches and duplicate two right patches. Duplicate right fields have no recorded
phase/spikes in these failures; no assignment change or direct causal claim.
Nearest calibration to 18.75° is 0° and accepts either side. Explicitly label
elevation/profile differences and historical identity/recording endpoints.

One proposed hypothesis: vision advances tonic left b3 timing through the right
DN pathway and window expiration creates a temporary negative increment at
the strong gate in the five deficit-only cases. Propose unchanged-model
instrumentation on nine failures/nine specified successful historical controls,
18 ×1.5 s =27 s /54,000 paired steps. Measurements, support criteria and strict
stops are in the report. Requires separate authorization; it has not run.

### Tests / Verification

Commands observed passing:

- `.\.venv\Scripts\python.exe -m flybrain.turnflee.audit experiments/turnflee-2026-10-06`:
  10,800 JSON/CSV intervals, 3,600 equal baseline intervals, source/receipt/history
  bindings and closed 144-case budget reconcile; zero audit steps.
- `.\.venv\Scripts\python.exe scripts/audit_turnflee_preservation.py`:
  2,510 originals, zero changed/missing.
- `.\.venv\Scripts\python.exe scripts/review_turnflee_wrong_side.py`:
  all count windows/scores/first gates/RNG reconcile, 42 comparisons/35 identities;
  zero flybrain imports and zero added presentations/seconds/steps.
- `.\.venv\Scripts\python.exe artifacts/turnflee-wrong-side-review-2026-10-06/verify_review.py`:
  guarded fresh saved-record audit, **1 offline unit test passed (0.18 s)**,
  exact artifact arithmetic/counts, 16 report links and 2,720 input hashes pass.
  Both runtime guards record zero brain/sensor simulation calls.

All 36 reviewed right ranks 0/1/48/49 rows have zero phase/spikes. Saved final
figure visually inspected separately. Simulation targeted/full suite not rerun:
the user's explicit zero budget forbids their brain/sensor stepping. Historical
188/755 test results remain historical. No live UI review.

### Problems / Failed Attempts

The initial reviewer incorrectly asserted 48 visual cells per side and stopped
before reconstruction. Read the actual manifest (46/50), corrected that reviewer
assumption, and kept the initial reviewer hash as a separate development artifact.
The canonical input snapshot excludes the newly authored reviewer. The first
plot had a legend/caption overlap; adjusted spacing and visually rechecked.
Both issues were confined to new review artifacts; no frozen input changed.
The final handoff checker initially failed to split a Unicode dated marker
over the PowerShell text pipe; an ASCII title marker corrected that check.
Git status confirms this directory is not a Git repository.

### Current State

Frozen-record review complete. Measured result remains 87/96 and experiment
closed at 144 cases /216 s /432,000 paired steps. Review adds zero of each.
Raw RGB, reference voltages/exact spike times and edge-resolved currents are
missing, so a unique causal explanation is unresolved. No gain/threshold
search, optimizer, ablation, closed-loop evaluation, escape enablement or promotion.

### Next Step

One task / preferred Sol High: approve, revise or reject the report's single
unchanged-model instrumentation hypothesis and 18-case cap. No automatic run.

### Notes

A −1 signed rolling increment means one fewer spike than the reference in that
window; it does not prove inhibition or reduced total firing. PyTorch executes
our assumed forward neuron rules; count subtraction and seeded tie choices
are engineered readout calculations, with no autograd or optimizer learning.
Model selection is a client setting; no model switch or subagent was performed.


## 2026-10-07 ? Escape diagnostic stage 1

### Goal
Passive unchanged-model timing instrumentation from the authorized handoff.

### Changes Made
Added isolated `flybrain.escape_diagnostic` recorder, single-use runner and saved-only audit; all cells/non-self edges recorded in compressed interval chunks.

### Decisions
Frozen gains (4,2), original dynamics/assignment/decoder; 18 historical cases only, no shadow brains, retry, training or repair. One agent, no Git repository.

### Tests / Verification
Original read-only audit passes 10,800 intervals, zero audit steps. Preservation: 2,510 unchanged; review snapshot: all 2,720 unchanged. Preflight tests 10 pass; final non-parity checks 6 pass; cumulative engineering calls 1,747/2,000. One upstream TorchScript warning per run. Full suite explicitly excluded by user.

### Problems / Failed Attempts
Initial 8 tests failed because the requested new package did not exist (test-first check). No scientific attempt yet.

### Current State
Stage 1 verified; unchanged-model scientific run pending preflight confirmation. Escape remains disabled.

### Next Step
Execute exactly the authorized 18 ordered cases once, seal and audit; then one repair proposal for approval.

### Notes
Every recorded value is passive. PyTorch executes our assumed forward dynamics; no autograd or optimizer learning.

## 2026-10-07 ? Escape diagnostic stages 2/3: closed incomplete

### Goal
Run the one unchanged-model diagnostic, then produce a measured repair decision.

### Changes Made
Consumed/archived source-bound run, one complete paired baseline chunk, sealed failed receipt, guarded partial audit, 18-case execution-status table, 200 pathway step rows, all 589 cells/3,636 modeled edges and visually reviewed baseline timeline. Added diagnostic report and one comparator-only recovery proposal; updated status/queue.

### Decisions
The first mismatch stops the run as requested. No retry despite zero threat images. Hypothesis unresolved; cannot choose a neural repair from absent threat measurements. Preserve receipt-bound v1 sources. Approval would cover only comparator correction and one newly bound unchanged diagnostic, not neural repair or activation.

### Tests / Verification
Scientific runner exit 1: first interval pathway mismatch; sealed receipt records one interval, 0.02 seconds, 40 individual neural steps (20 actual/20 reference), zero completed presentations. Guarded partial audit exit 0 verifies all 20 steps/all cells/all non-self edges and exact equality of saved original interval; zero simulation/sensor calls. Read-only seal audit reports closed-incomplete, zero added steps. All 2,510 originals unchanged; all 2,720 snapshot hashes checked at preflight and final handoff. Synthetic evidence remains 10 pass and final 6 pass, 1,747 cumulative engineering steps; no further kernel execution. Full suite expressly not run. Saved figure visually reviewed; live review not performed.

### Problems / Failed Attempts
Root cause: asdict preserves DiagnosticInterval tuples; historical JSON contains lists. Runtime comparison rejects equal values with unequal container types. Canonical JSON roundtrip restores exact equality offline. Tests covered numerical/dataclass parity and NPZ chunks but missed this record boundary. No model/config/source mismatch or numerical mismatch in the recorded baseline. No fix applied to consumed sources and no scientific retry.

### Current State
Diagnostic closed incomplete; hypothesis unresolved. Historical escape 87/96 and defaults remain unchanged. No new trigger/side selection/closed-loop result, optimizer update, commit or activation.

### Next Step
One task / preferred Sol High: approve or reject docs/ESCAPE-DIAGNOSTIC-RECOVERY-PROPOSAL.md. Additional 18 cases/27 s/54,000 steps would require approval; previous 0.02 s/40 steps remain accounted separately.

### Notes
A saved baseline match validates recording for that interval, not threat behavior. Temporal causality is still unmeasured. PyTorch forward tensor arithmetic remains separate from autograd/training. Model selection was not changed; one agent used.

## 2026-10-07 - Accepted comparator recovery preflight

### Goal
Apply the accepted comparator correction and execute one new unchanged diagnostic.

### Changes Made
Additive escape_diagnostic_recovery comparator, runner and saved-only audit. JSON comparison copy fixes tuple/list representation only; all neural/sensory/decoder rules and consumed v1 bytes preserved.

### Decisions
New explicit parent consumption key, fresh outputs, original 40 steps tracked separately. Reuse source-locked kernel parity proof; all new tests saved-only, zero kernel calls. No subagent, commit, search or training.

### Tests / Verification
21 guarded tests passed (3.05 s), zero skips, one upstream TorchScript warning. Includes mutated measurement, missing/nonfinite record, nextafter float, source hash, budget, consumed lock, interrupted sealing and forged saved-chunk rejection. Engineering remains 1,747/2,000. Full simulation suite excluded by approved diagnostic scope.

### Problems / Failed Attempts
Initial tests fail as expected before additive package exists. One first interruption test lacked acceptance fixture and raised FileNotFoundError; replaced with explicit KeyboardInterrupt at configuration write boundary, then verified sealing. No scientific execution yet.

### Current State
Comparator fixed and tested; scientific run about to perform source/configuration preflight and consume once.

### Next Step
Execute exactly the approved 18 cases, audit records, then decide from measurements.

### Notes
The correction changes container representation of a passive comparison copy. It never feeds into tensors or simulation state.

## 2026-10-07 - Comparator fixed; diagnostic process externally interrupted

### Goal
Complete the approved unchanged-model diagnostic after correcting its comparator.

### Changes Made
Seven completed historical cases, 553 complete neural chunks, 552 flushed interval records, external interruption receipt/seal, guarded partial audit, case/timing/evidence tables and visually reviewed two-case figure. Added measured report and one explicit remaining-case proposal. Comparator v2 sources preserved after source locking.

### Decisions
Process/session disappeared externally; no Python process remained and no scientific mismatch/terminal traceback was logged. Cause unresolved. Do not resume or retry consumed run. Seal original outputs via external finalizer. Preserve complete first seven cases; proposal repeats only interrupted eighth and runs ten never-started cases under a fresh explicit budget.

### Tests / Verification
21 guarded saved-only tests passed, zero skips, one upstream warning; engineering remains 1,747. All 552 JSON records exactly equal originals; all 553 chunks/589 cells/3,636 edges/every step satisfy independently reconstructed equations and episode continuity. Guard records zero added simulation/sensor calls. New science: 11.06 s /22,120 steps; including previous failed baseline: 11.08 s /22,160. Two completed deficit-only cases satisfy four timing conditions; remaining three unassessed, hypothesis unresolved. Saved figure visually reviewed; no live review. Full suite not run per diagnostic scope. Hidden harmless process probe survived launching tool and completed with zero neural steps; host-cancellation survival not proven.

### Problems / Failed Attempts
Unknown exec session 81433; no Python process existed. Eighth gzip stream lacked trailer and last JSON interval; its 28th neural chunk remains complete. 553 neural chunks correspond exactly to reserved 22,120 steps; 552 full JSON records. No raw file repair, scientific retry or model mismatch invented.

### Current State
Comparator defect fixed. Diagnostic interrupted and consumed; no escape repair, learning, closed-loop assay or activation. Historical 87/96/defaults preserved.

### Next Step
One task / Sol High: authorize docs/ESCAPE-DIAGNOSTIC-REMAINING-PROPOSAL.md - eleven remaining cases, 16.5 s /33,000 additional steps. Aggregate with first seven complete records only; retain/count both interrupted starts.

### Notes
Seed 1100000 gate: actual [1144,1244] versus reference [1111,1178,1246] ms in (1100,1300], -1 rolling but +1 final cumulative. Seed 1100001: [754,820] versus [702,754,870] in (700,900], -1 rolling but +2 final. DN delivery precedes timing changes; temporal order is not causal necessity. No optimizer or subagent used.

## 2026-10-07 - Accepted remaining-case completion preflight

### Goal
Complete the exact eleven missing diagnostic cases, then audit eighteen complete identities.

### Changes Made
Additive escape_diagnostic_remaining runner/contract/aggregate audit and detached supervisor. Fresh consumed key/output paths, heartbeat/terminal fallback, zero-state restart of accepted interrupted case only. No source/archive edits to previous runs.

### Decisions
Exactly eleven specified seeds, 16.5 s/33,000 new steps; cumulative science max 27.58 s/55,160 steps. Aggregate seven complete old cases with eleven fresh, excluding interrupted prefix from support test but preserving computation accounting. No automatic retry/resume or neural repair.

### Tests / Verification
26 guarded saved-only tests passed, zero skips, one upstream TorchScript warning (5.79 s). Existing canonical comparator/corruption checks plus fixed ordering, prior hash, exact budget, incomplete exclusion, consumed retry refusal, real child process failure/success and fallback receipt preservation. Zero neural/sensory calls; engineering remains 1,747. Full suite explicitly excluded from diagnostic scope.

### Problems / Failed Attempts
Five new tests initially failed as expected before the additive package existed. Implementation then passes; no scientific run yet.

### Current State
Preflight verified; accepted detached measurement worker about to launch once.

### Next Step
Run eleven cases, seal, independently audit all eighteen complete cases and prepare measured next decision.

### Notes
Supervisor changes process lifetime and logging only; it receives no sensory/neural state and never retries a child.

## 2026-10-07 - Six more cases verified; progress-file sharing failure

### Goal
Complete eleven accepted cases with detached supervision, then audit aggregate eighteen.

### Changes Made
Six complete cases and one 1100 ms prefix: 505 complete intervals/chunks. Detached supervisor tracks terminal exit; normal failed receipt sealed. Added authoritative saved failure audit, four-case figure, append-only reservation/heartbeat utility and real reader/durability tests. New report and one explicit five-case proposal; statuses updated.

### Decisions
Stop and preserve consumed v3 on WinError 5 during reservation publication; no scientific retry/resume. Counter attempted 20,240 but durable progress/chunks prove only 20,200 executed, remaining 40 not run. New I/O utility uses persistent append handles, fsync before kernels, post-durability counter commit and poison on failure; no rename dependency. Fifth timing test remains required.

### Tests / Verification
26 original preflight tests passed. Journal-specific four tests pass (0.26 s); complete guarded saved-only group 30 pass, zero skips, one upstream warning (6.32 s), zero neural/sensor calls. New audit verifies 505 records exactly equal history, all 589 cells/3,636 edges/every step and episode continuity. New science 10.10 s/20,200 executed; cumulative 21.18 s/42,360 executed, 42,400 attempted. Four complete deficit-only cases meet the four conditions; overall hypothesis unresolved, seed 1100036 unstarted. Saved four-case figure visually reviewed; no live review. Full suite excluded by diagnostic scope.

### Problems / Failed Attempts
Windows os.replace(progress.tmp,progress.json) denied (WinError 5) during next reservation. Same failure reproduced with ordinary open read handle; replacement succeeds after close; actual interfering reader unknown. Initial copied audit metadata retained prior 552/553 counts although assertions/tables verified 505/505; preserved initial and corrected authoritative final version to actual counters. Initial combined plot chose prior not-started placeholder; restrict to completed records, then visually verify. No raw/source/archive mutation.

### Current State
13 full cases available, no active scientific worker; frozen 87/96 unchanged/escape disabled. I/O replacement tested but no further scientific run authorized or executed. Engineering remains 1,747.

### Next Step
One task / preferred Sol High: approve docs/ESCAPE-DIAGNOSTIC-FINAL-FIVE-PROPOSAL.md - five fresh cases, 7.5 s/15,000 steps. Expected cumulative executed total 57,360, attempted upper accounting 57,400; preserve all three partial starts.

### Notes
Reservation publication failed before either kernel. One next RGB frame was computed before that reservation, not recorded; no neural advance occurred. Counts can show -1 rolling and positive cumulative. Temporal order does not establish necessity; no optimizer, gain search, subagent, commit or activation.

## 2026-10-07 - Approved final-five preflight

### Goal
Complete the five approved cases with tested journals and audit all eighteen complete identities.

### Changes Made
Additive final-case contract/runner, append-only detached worker and aggregate audit of seven plus six plus five complete cases. Reserve before RGB; fsync before kernels; counter after durability; close journal before terminal receipt. Supervisor terminates child on monitoring failure and conservatively seals incomplete reservations.

### Decisions
Exactly accepted five seeds/7.5 s/15,000 new steps. Prior raw/source/receipt keys untouched. Future executed total 57,360 and attempted accounting 57,400; three prior prefixes excluded only from support test. No automatic retry, learning or new neural readout.

### Tests / Verification
36 guarded saved-only tests passed, zero skips, one upstream warning (8.59 s). Includes exact final seed order, three-source prefix exclusion, prior binding, held-reader 375-interval budget, journal durability poison, normal/failure supervisor receipts and child termination on heartbeat failure. Zero neural/sensory calls; engineering remains 1,747. Full suite restricted during diagnosis.

### Problems / Failed Attempts
Five new tests initially fail because final additive package absent, then pass after implementation. No new scientific attempt yet.

### Current State
Preflight verified; source-locked hidden detached worker about to launch once.

### Next Step
Run five cases, seal, audit full eighteen, then measured repair decision.

### Notes
Append-only I/O is passive; moved reservation never changes neuronal numerical operation order.

## 2026-10-07 - Timing diagnosis complete; all-five observational support

### Goal
Finish five approved cases and independently verify the complete diagnostic before one neural repair decision.

### Changes Made
Five complete cases/375 intervals, normal append-only worker receipt/heartbeat, guarded three-source aggregate audit, eighteen case summaries, 1,350 timelines, exact spike/DN event table, five evidence rows and nine comparison labels. Visually reviewed deficit/tie figures; final legend outside data. Added complete diagnostic report and one fixed-parameter temporal readout proposal. Updated status/queue/decisions/experiments.

### Decisions
Classify predeclared timing hypothesis supported only after all five satisfy redistribution, exact expiration deficit, positive final cumulative b3 and preceding differential right-DN delivery. Temporal order is observational, not necessary causality. No replacement decoder evaluated or parameters searched. Propose cumulative six-cell increments, positive winner/margin >=1, two consecutive eligible samples and 500 ms first-strong deadline; approval before implementation.

### Tests / Verification
36 guarded saved-only tests passed, zero skips, one upstream TorchScript warning (8.59 s), zero new neural/sensor calls. Final worker completed five cases /7.5 s /15,000 individual steps and exited zero. Guarded aggregate saved audit passes eighteen complete cases /1,350 intervals /54,000 complete-episode steps, every cell/non-self-edge/1 ms equation and original interval exact; all five criteria true; zero added audit steps. Cumulative actual across attempts 28.68 s/57,360 steps; attempted 57,400 includes unexecuted 40. Engineering 1,747 unchanged. Saved figures visually reviewed separately; no live review. Full suite not run per diagnostic scope. Source/record preservation checked at preflight and final handoff.

### Problems / Failed Attempts
No scientific failure in final worker. Initial five new tests fail before package exists, then pass. Figure first legend obscured raster samples; final figure moves legend outside and preserves initial. Earlier comparator, external process and file-sharing failures remain closed/preserved; no silent resume or automatic experiment retry.

### Current State
Diagnosis complete, hypothesis supported observationally. All model/detector/readout bytes and original selection outcomes unchanged; escape historical 87/96 and disabled. No new readout, training, closed-loop assay, commit or activation.

### Next Step
One task / preferred Sol High: approve docs/ESCAPE-TEMPORAL-READOUT-PROPOSAL.md, implementing one exact candidate and its directional gate, then contingent prescribed closed-loop gate. No activation included.

### Notes
Final deficit case 1100036: right-DN delivery differs at 563 ms, b3 timing differs at 627; original gate 1300 has actual [1169,1228] versus reference [1111,1178,1246], -1 rolling despite +2 full-record increment. PyTorch forward dynamics are separate from engineered readout arithmetic and optimizer learning; no gradients/updates used.

## 2026-10-07 - Actual escape readout implementation and preflight

### Goal
Implement one fixed temporal readout and continue automatically through tests/gated evaluation, delivering the verified new launcher if all gates pass.

### Changes Made
Added escape_recovery cumulative readout, source-bound candidate, retained flight session, intervention-aware action logs/replay, independent temporal/trajectory audit, append-only experiment worker, gated launcher and temporal UI. Closed scoring now verifies actual signed turn before flee, takeoff, 2 cm separation, braking and stable recovery; state labels alone cannot pass. Isolate historical synthetic-test engineering ledger and save new verification ledgers separately.

### Decisions
User selects one candidate and activation of passing new launcher; honest closure on failed scientific gate, no routine approval pauses. Side readout sees counts and sensory flags only. Fixed model/gains unchanged. Evaluator criteria predeclared in execution ledger before scientific results. No source mutation to previous packages/records. Exact 96 direction then contingent 48 closed cases, max 432 s/864,000 individual steps.

### Tests / Verification
New readout/replay/numerical boundary tests pass; first targeted 25/full 810 passed (6.63/263.68 s), two/four upstream warnings. Added replay silent-state binding and premature-flee regression, both shown failing then passing; final targeted/full source check running. Scientific run not yet consumed. Original diagnostic ledger isolated, not mutated.

### Problems / Failed Attempts
Initial new API tests fail before implementation as expected. Found that silent intervention must be stored/restored in replay and actual yaw must be measured from pre-command orientation; corrected before scientific lock. No scientific retry or tuning.

### Current State
Behavior replacement code implemented, final preflight running; experimental/disabled until behavioral gates pass.

### Next Step
Finish preflight/source locks, run one candidate's fixed gates, independently audit, deliver/launch if passing.

### Notes
PyTorch generates unchanged assumed spikes; cumulative integration and repeated positive margin are engineered readout calculations, no autograd/learning. One agent, no commit or push.

## 2026-10-07 - Implemented temporal controller closes at directional gate

### Goal
Complete user-authorized one fixed controller implementation/evaluation without routine approval pauses; activate a new working launcher only if all gates pass.

### Changes Made
Implemented escape_recovery controller/session/source config, intervention replay, independent temporal/trajectory checks, gated launcher, append-only execution. Evaluated one fixed cumulative/positive/two-sample rule. Added measured regression analysis, 86.25 failure tables/plots and result guide; updated status/queue.

### Decisions
Close the user-selected single candidate on failed per-azimuth gate, no fallback/tuning. 94/96 passes global >=90 but 86.25 group 4/6 fails >=5. No closed-loop case or app activation; physical escape remains unverified. NameError math dependency failed before any kernel/rendered case; preserve zero-step receipt/source and repair only standard-library binding through new bound entrypoint/fresh parent namespace. No exposed-case scientific retry or parameter change.

### Tests / Verification
Pre-lock targeted 26/full 811 pass (11.09/473.04 s), zero skips. Missing-math entrypoint reproduced before budget guard, corrected targeted 27 pass (7.87 s). Final whole suite 812 pass (240.11 s), four upstream warnings, zero skips. Isolated diagnostic ledgers preserve original 1,747 steps and record 529 new explicit synthetic verification steps separately. Scientific run: 96 cases/7,200 intervals/144 s/288,000 individual steps; all unchanged image/count/diagnostic records exactly match historical model. Saved independent temporal audit passes source/seal/budget/case/gate reconstruction, zero new neural steps. Nine previous failures corrected, two regressions. Saved failure figure visually reviewed; no live or actual flight review. No optimizer, commit or push.

### Problems / Failed Attempts
First zero-step entrypoint launch missing math import; fixed using additive binding without changing readout/model/gains or frozen bytes. Remaining scientific failures: 1100018 no decision by 1.50 s (left evidence only at 1.48); 1100021 starts correct left at .86, flips right at .88, commits wrong right at .90. Two-sample persistence is not correctness; it discards a brief phase cue and delays the earlier correct preference. No alternate decoder evaluated afterward.

### Current State
Controller code actually implemented and measured, 94/96 improvement, but required per-azimuth gate failed. Closed record consumed; no physical escape gate/delivery or activation. Historical launchers/defaults/sources untouched.

### Next Step
One task / Sol High: agree a single event-aware readout preserving brief neural phase evidence, with declared budget before another experiment. Honor current one-candidate stop choice.

### Notes
Global improvement does not override predeclared local gate. Passing 812 tests demonstrates engineering checks, not physical escape or learning. Controller uses simulated neural counts/threat flags, never target coordinates; geometric measurements are evaluator-only.

## 2026-10-08 - Frozen prototype exception and physical preflight

### Goal
Execute the user-authorized one-agent physical validation and contingent new prototype launcher, accepting frozen 94/96 direction without changing the original failed gate.

### Changes Made
Adjacent escape_flight_validation package: fixed exception verifier, closed-only runner, exact append-only budget, detached supervisor, independent saved physical/temporal audit, calibrated-start action envelope/replay, gated prototype app and offline flight visualizer. Prospective exception and 4,858-file protection snapshot recorded before scientific exposure. Full synthetic saved storage/audit covers 48 cases with zero neural calls.

### Decisions
589 cells/4,139 anatomical edges/(4,2) gains/readout/sensors/neuron rules/flight policy preserved. Exactly existing 48 six-second closed cases, budget 288 s/576,000 individual steps. Scientific failure closes the single experiment; no tuning/retry. Named-start replay metadata stays outside frozen escape_recovery inventory. Original gate stays failed; known seeds 1100018/1100021 retained. No new motion-only baseline or learning.

### Tests / Verification
Initial 18 feature tests failed before package existed, then passed. First broader 115 targeted passed; final preflight including delivery binding is running. Fresh saved-only original direction audit passes (96 cases/7,200 intervals; zero added neural steps). Engineering logs/ledgers are separate from consumed proofs. Exact targeted/full commands and XML will be recorded on completion.

### Problems / Failed Attempts
Local audit review found first-side versus latest-side conflation on repeated triggers. A five-tick synthetic reproduction failed; the corrected independent audit now preserves first response while reconciling subsequent responses. No scientific attempt yet. Git repository absent; no commit.

### Current State
Final preflight underway; physical result unmeasured, activation disabled.

### Next Step
Source-lock after targeted/full passage, execute one detached physical run, independently audit/replay and review saved plots, activate/launch only if passing.

### Notes
PyTorch still performs the frozen assumed spike dynamics. The readout/flight policy are engineered; this task adds measured evaluation, no gradients or optimizer.

## 2026-10-08 - Physical experiment source-locked and consumed

### Goal
Run exactly one previously authorized physical protocol after verified preflight.

### Changes Made
Sealed source/evidence-bound acceptance, fresh saved original-direction audit and 4,858-file preservation proof; launched one hidden detached supervisor. All 48 closed cases remain in original order/seeds with independent evaluator recording.

### Decisions
Original 94/96 failed gate stays unchanged under the prospective prototype exception. Complete all 48 unless a hard source/health/measurement/I/O/budget stop occurs; score the aggregate only once. No gain calibration, directional rerun, policy tuning, training or retry.

### Tests / Verification
117 targeted passed (60.18 s), 841 full passed (281.86 s), zero skips, four upstream warning instances each. New diagnostic ledgers isolated; original 1,747-step proof unchanged. Offscreen zero-step viewer render passed and saved image reviewed. Scientific launcher: Start-Process with -WindowStyle Hidden, python -m flybrain.escape_flight_validation.worker, supervisor PID recorded in launch.json. Preflight source/receipt checks pass.

### Problems / Failed Attempts
First resting physical case fails collision freedom at 5.76 s after recovered cruise reaches the arena boundary; first hovering case passes. These are preliminary case outcomes, not final aggregate. No repair is applied during this frozen run.

### Current State
Single consumed worker running. Activation disabled pending final scientific/audit/replay/visual gates.

### Next Step
Finish and seal, independently reconstruct saved results and verify exact replay; deliver only the measured result.

### Notes
First collision is actual world position/velocity evidence, not a neural readout score. Rendering does not advance simulation or RNG; live review remains pending.

## 2026-10-08 - Physical validation closed; audit, replay and handoff verified

### Goal
Complete the single authorized frozen-candidate physical validation and contingent prototype delivery, accepting historical 94/96 direction without altering its failed gate.

### Changes Made
Adjacent escape_flight_validation package with closed-only runner, append-only exact budget and detached supervisor, source-bound prospective exception/final acceptance denial, evaluator movement scoring, calibrated-start action envelope, independent audit/replay, receipt-gated app and saved flight plots. Ran all 48 existing cases once; recorded JSON/CSV/actions, source archives/circuit/receipts. Added saved-only NumPy CSV format adapter outside consumed sources, with six regression tests and corrected audit/replay driver. Updated status, decisions, experiment report, queue and verification commands. Historical entries, launchers/defaults/sources and all 4,858 protected files preserved.

### Decisions
Close at physical gate failure:19/24 versus>=22; resting 6/8, hovering 7/8, cruising 6/8 versus>=7 each; silent triggers 0. No activation/app launch, tuning, retry or learning. All 589 cells/4,139 anatomical edges/(4,2) gains/readout/encoding/neurons/flight policy unchanged. Original directional94/96 still failed at86.25 degrees4/6; known 1100018/1100021 remain. Required exact replay is separately counted verification, not another scientific experiment. These 48 inspected starts are now historical benchmark cases. Preserve frozen strict raw audit and handle scalar repr only with verified additive artifact adapter; never rewrite raw CSV/receipt or monkeypatch gate.

### Tests / Verification
Exact commands: docs/verification/escape-flight-validation.md. Preflight117 targeted/841 full passed, zero skips, four upstream warnings each,60.18/281.86s. Adapter6 saved-only tests pass at zero neural steps,0.23s; final full including adapter847 passed, zero skips, four upstream warnings,414.03s. Scientific worker exits0 with closed-physical-failed:48 cases/288s/576,000 steps/14,400 intervals. Independent saved audit passes count/temporal/trajectory/order/CSV-action/journal/source/circuit/calibration/seal gates, zero new neural calls. All 48 exact replays including silent match every saved field; separately576,000 verification steps, zero new scientific cases; supervisor exits0. Four saved flight figures reviewed; zero-step offscreen viewer preflight rendered/reviewed separately. Live review pending/unavailable; new app unlaunched. Final CLI refuses activation (exit1). Final source/proof/budget check passes 4,858 unchanged files, original 1,747 diagnostic steps preserved; new full/post-audit ledgers 529 each in separate directories. No commit/push.

### Problems / Failed Attempts
Scientific failures: 1110000 resting0 boundary collision 5.76s;1110012 resting180 boundary 5.94s;1110013 hovering180 boundary 5.66s, all after recovery. Cruising1110002/1110014 at0/180 never reach weak continuity/braking/recovery within 6 s. All 24 timely sides/turns, eight resting takeoffs and>=159.908mm separation pass; maximum downstream215Hz, finite state. Initial saved audit rejects CSV vectors with np.float64 wrappers versus JSON floats. First mismatch reproduced at interval0; six adapter tests shown failing then passing. Corrected adapter validates exact float64 hex values and original independent numerical checks on disposable CSV view, with no consumed-source/record edits or new science. Initial failure log preserved.

### Current State
Authorized experiment implementation, execution and honest handoff complete. Physical gate failed; activation disabled, no working validated prototype delivered. Original directional gate remains failed under its original rule; prospective prototype exception still recorded. Exact replay/source/saved review pass; live review remains pending.

### Next Step
One task/preferred Sol High: saved-record diagnosis and one sensor-only flight-policy repair proposal for boundary-safe recovered cruise and weak-threat completion, keeping accepted readout frozen, declaring new budget before any experiment. Do not automatically start repair, retry, training or new evaluation.

### Notes
Correct side and actual turning do not establish safe recovery. PyTorch forward spike calculations remain fixed; flight control is engineered, not learned. Science576,000 plus mandated exact replay576,000 =1,152,000 individual steps, with tests/preflight separate. No new motion-only baseline; bind prior verified directional evidence. One agent; client model was not switched.
## 2026-10-11 — Sustained edge escape implemented; measured worker started

### Goal
Implement the explicitly approved sustained escape plan: fly to map edge, stop
and hover there; continue diagnosis and verification until the scoped behavior
passes. One agent, no training, commit or publishing.

### Changes Made
Additive escape_edge control/session/versioned replay/viewer/launcher, actual-motion
scorer, independent saved arithmetic/trajectory audit, bounded worker and delivery
gate. Retained neurons/gains/readout/physics. IMU-derived displacement and authored
center-start rectangle determine 15 mm inset endpoint. Weak threat cannot cancel
escape; no home guard/idle landing after commitment. Added 38 regressions and
real motor checks including 24 direction/start combinations through 20 s, corners
and off-center encounters. Viewer-safe predator spawn and normal frame capture.

### Decisions
The user chose stop at edge. New contract explicitly supersedes weak-threat
recovery for this separate candidate. Every historical failure stays failed;
original 94/96 direction and known seeds 1100018/1100021 remain. Development and
seed robustness each 48 x 12 s /1,152,000 individual paired neural steps. Four
exact replays separately 96,000 steps. Additional native viewer API review is
engineering, predeclared 24,000 steps, no new scientific cases. No automatic
scientific retry or lower gate; user authorizes diagnosed new development attempts.

### Tests / Verification
Initial red 33 failures; implemented intermediate 31 passes/2 failures. Latest
38 focused pass, 24.07 s. Broader targeted 239 pass, zero skips, 342.29 s. First
full 920 pass, 609.45 s; fresh final full including viewer regressions 922 pass,
zero skips, three upstream warning instances, 362.48 s. Exact commands and XML
in docs/verification/escape-edge.md. Normal frame-task PNG reviewed at zero
simulation/neural steps. Protected 3,010 existing source/experiment files verified.
Source-frozen preflight produced; detached worker started. First development
rest-0 intact passes, arrival 4.92 s; its silent control does not trigger.

### Problems / Failed Attempts
Forged early-stop regression initially fell within the startup grace; moved its
bad sample after the grace without changing the gate. Direct repeated renderFrame
capture lost text; normal task-manager frames preserve text and state. Absolute
Windows PNG write failed until Filename conversion; red/green test and failed
images/logs retained. Native Windows Computer Use discovery fails twice with
missing native pipe; no mouse/keyboard interaction review claimed.

### Current State
Engineering verified; single candidate source-frozen and under development
evaluation. Aggregate scientific result unmeasured; delivery disabled.

### Next Step
Finish development and independent audit; only then run frozen seed robustness,
audit and four exact replays. Review saved plots, activate passing receipt and
check native application render/API flight separately from blocked native input.

### Notes
Spikes select turn side. IMU endpoint flight and stabilization are hand-written;
PyTorch forward calculations remain frozen, with no gradients/optimizer updates.
All earlier implementations/records and dated entries preserved.
## 2026-10-11 — Sustained edge escape completed and delivered

### Goal
Finish the approved edge-flight plan through measured gates, exact replay, actual
viewer review and handoff, preserving every previous scientific failure/source.

### Changes Made
Separate escape_edge controller/session/action schema/scorer/audit/replay/worker/
receipt-gated viewer and launcher. Flight stays active through weak/absent threat,
travels to an IMU-estimated 15 mm inset map endpoint, brakes and holds there.
Frozen neurons/gains/encoding/readout/physics unchanged. Saved plots, source
archives, durable budgets, config/CSV/JSON/traces/actions/receipts and delivery.
Updated status/queue/architecture/decisions/experiment report/guide/verification.

### Decisions
User-selected edge hover is a new contract, not relabeling old weak recovery or
94/96 direction. Development and same-geometry seed robustness are historical
after inspection. One candidate; no tuning/retry/training/commit/publishing.
Added saved-only cumulative retreat/full flee-speed review before development
completed; no frozen source/gate or scientific count changed. Native renderer/API
review is distinct from unavailable native input automation.

### Tests / Verification
38 focused pass; 239 broader targeted pass; 922 fresh final full pass, zero skips,
three upstream full warning instances. Exact commands/logs/XML: docs/verification/
escape-edge.md. Both 48-case neural batches pass all 24 intact and zero silent
triggers:96 cases/1,152 simulated seconds/2,304,000 individual scientific steps.
Both independent saved audits pass at zero new steps; four exact replays match
every field, separate 96,000 verification steps. Arrival 4.84–5.02 s, minimum stable
hover 7 s, wall clearance 10.39–15.28 mm, max hover speed.008245 m/s, peak 215 Hz.
Pre-hover cumulative retreat 0 m; all flee commands.12 m/s. Both saved plots and
zero-step/native renderer frames reviewed. Native viewer API through 12 s passes,
separate 24,000 engineering steps, zero new scientific cases. Responsive native
window PID 21420 left open/paused at edge hover. All 3,010 protected original source/
experiment files unchanged; source-bound public CLI verified separately at zero steps.

### Problems / Failed Attempts
Prior red/green failures retained in the implementation entry. Native helper pipe
missing on two attempts: no native mouse/keyboard review. Some saved native
captures omit labels transiently; visual movement/hover and numerical checks pass.
Original directional seeds 1100018/1100021 remain failures; old physical gate stays
failed. No new measured candidate failed or was retried in this task.

### Current State
Scoped sustained edge escape implemented, scientifically verified on the declared
protocol, audited/replayed, activated and launched. No learning claim. Historical
launchers/defaults/records preserved. Capture/input review limitation remains.

### Next Step
One task/preferred Sol High: reproduce intermittent native-capture label loss and
complete live input review when the helper is available, preserving verified flight
sources/receipts. Do not start another neural experiment or retune the readout.

### Notes
Spikes choose the turn; sustained flight and edge stabilization are engineered.
Science 2,304,000 + replay 96,000 + viewer engineering 24,000 = 2,424,000 individual
steps, with tests reported separately. No model switch, subagents or commit.

## 2026-10-12 — UI polish verified; comprehensive understanding made a standing goal

### Goal
Make the existing demo nicer/easier to use, especially flower and predator spawn
buttons, then work toward complete understanding of scripts, libraries, algorithms
and machine learning for honest interview discussion of an AI-assisted project.

### Changes Made
Added `escape_ui` presentation and launcher around the original verified edge
session. Prominent colored 42-pixel spawn buttons, readable pause/new-trial/replay,
pixel-sized fonts, concise status, expandable diagnostics and larger inspector/
settings text. Retained brain/source inspection, activity preferences, keyboard
shortcuts, worker barriers and exact replay. Added real UI behavior/framebuffer
tests and a zero-step two-size capture helper. Updated AGENTS, README, status,
queue, architecture and decisions; added demo guide, verification and learning
roadmap. All historical sources/results/receipts remain intact.

### Decisions
Use a separate presentation package rather than edit source-bound historical UI
files or relax delivery verification. No controller/neural/physics change or
training experiment. Keep the two-pane demo; hide detailed diagnostics initially.
Record comprehensive understanding/AI-assistance honesty as standing instructions
for every future project session. Teach active code then historical/supporting
scripts; check understanding with explanations/predictions/exercises, not agent
checkmarks. One agent; no Git repository, commit, push or publishing.

### Tests / Verification
Initial red: three failures for absent UI package; first three pass, 29.28 s.
Initial combined UI/edge: 41 pass, zero skips, two upstream warnings, 141.84 s.
Additional framebuffer/readability regressions failed then passed; final UI three
pass, one upstream warning, 51.45 s. Final full 925 pass, zero skips, three
upstream warnings, 1869.89 s (31:09), exit 0. Exact commands/XML/logs in
docs/verification/escape-ui.md. Public CLI capture exits successfully. Six
default/diagnostics/settings layouts reviewed at 1000x700 and 1440x900; all
captures preserve snapshot/voltages/time and use zero neural steps. Direct PNG
inspection confirms opaque header text despite some image-tool preview omissions.
All 3,010 protected historical files and original source-bound delivery pass.
Local one-agent acceptance review and Python parse/document-link checks pass.
Native mouse/keyboard review not performed; available tools expose no native apps.

### Problems / Failed Attempts
Numeric DirectGUI relief 0 gave invisible button frames; DGG.FLAT fixes the real
framebuffer regression. OnscreenText NodePath setZ added an extra legend offset;
text setPos fixes it. One new PNG-read test initially needed Filename conversion
on Windows. Inherited settings text was only 8.05 pixels at 700 height; enlarged
fonts/rows fix it. First full run interrupted for that improvement, without a
passing claim. Final full was much slower than earlier historical timings.
Some image-tool previews omit labels that direct RGB/alpha inspection shows are
present and opaque in PNGs. Process isolation/full-resolution reinspection were
used for review; tool-side cause and historical native observation unresolved.
No speculative renderer repair or new scientific experiment was performed.

### Current State
UI implementation and required regression checks complete. New launcher ready;
original scientific evidence preserved. Programmatic controls and saved-frame
review are verified separately from unreviewed native input. Learning goal and
sequence persist in project instructions; comprehensive teaching has not begun.

### Next Step
One task / preferred Sol Medium: learning roadmap lesson 1, trace launch -> source
verification -> viewer -> worker -> snapshot -> render. End with the user
predicting/explaining a timestep and why redrawing cannot advance the brain.

### Notes
Current 3D rendering uses Panda3D; historical 2D uses Pygame. PyTorch computes
neural tensors, while current edge flight/hover are engineered rules. Earlier
2D learned readouts and current frozen 3D forward computation remain separate.
No model switch or subagents; no claim of user understanding or sole authorship.


## 2026-10-13 ? Modern dark dashboard completed and opened

### Goal
Finish the user-approved redesign: cool/readable dark dashboard, simple group
checkboxes, understandable source inspection and streamlined navigation. Continue
through required checks and handoff, preserving the scientific implementation.

### Changes Made
Replaced the first UI polish with rounded charcoal panels, readable native font,
46px toolbar controls and persistent Groups/Inspect/Help sidebar. Ten real tick
boxes show model-role names/colors/counts and explanations without scrolling at
1000x700/1440x900. Show all/Hide all/Only this filter branch ownership. Discarded
hidden GPU fragments, excluded hidden picking/highlights/connections, and included
visible coarse/full selected geometry in Fit. Empty view offers Show all without
moving the camera. Inspector leads with role, source type and measured activity;
IDs/provenance/pathway data are behind Technical details. Preferences survive
spawn/reset/replay/resize. Added six real dashboard tests and ten-state capture
review. Prior UI files archived; updated project guides/architecture/decisions/
status/queue/README/verification and the implementation ledger.

### Decisions
User explicitly rejected earlier aesthetics despite passing tests and approved
modern dark dashboard with complete group hiding. Display filters do not silence
neurons or alter controllers. Preserve the ten existing model-role memberships;
do not invent biological classification. Keep original launcher/controller/source
receipts valid. One agent/local review; no Git, commit, publishing or training
experiment. Standing comprehensive understanding and honest AI-assistance goal
remain active; no topic is automatically marked understood.

### Tests / Verification
Final focused command in docs/verification/dashboard-redesign.md: 44 passed,
zero skips, two upstream warnings, 175.71s. Six new cases include two sizes,
real DirectCheckButton state, idempotent callbacks, preserved neural snapshot/
actual-reference voltage/history, actual GPU fragment/depth behavior, hidden
selection/picking, camera bounds/full morphology, empty state, input routing and
exact short replay/reset. Ten reviewed frames, zero simulation/neural steps.
All 3,010 protected originals and the original verified candidate pass.
Fresh full command uses full-final.xml/full-final.log and isolated full-final
ledgers: 928 passed, zero skips/failures/errors, three upstream warnings,
480.87s, exit 0. Previous reviewed code/frame hashes still match. Actual native
window opened paused on 2026-10-13, PID 26440, title FlyBrain | Explore the brain,
nonzero handle and Responding true. Native mouse/keyboard acceptance unreviewed;
available tools do not expose native apps. Launch and saved/programmatic review
are separate evidence.

### Problems / Failed Attempts
Initial five red cases for missing new API. Corrected synthetic frame ID ordering
in one table fixture. Font startup failed for native Windows path and Filename
object; minimal FontPool probe showed normalized string succeeds, with fallback.
Constructor failures cascaded into multiple-ShowBase errors in those failed
processes. DirectGUI text_scale returns a tuple; corrected the readability fixture.
Visual bounds test failed at y690 beyond compact sidebar y680; spacing fix passes.
Full selected morphology was omitted from coarse Fit bounds; red/green fix passes.
First full process disappeared after 93%, no summary/XML or active process on
continuation; cause unknown, partial log preserved. Fresh engineering suite passes.
Windows SetForegroundWindow failed; Alt+Tab/taskbar can expose the responding app.
No speculative renderer repair, scientific retry or old result relabeling.

### Current State
Approved dashboard implementation and required checks complete; new UI opened
paused for human review. Historical scientific/controller sources/records intact.
User aesthetic/native input acceptance and comprehensive teaching are not inferred
from test or render success.

### Next Step
One task / preferred Sol Medium: review the new UI, then learning-roadmap lesson 1
tracing launch -> source verification -> worker -> snapshot -> display. Explain
why selecting Vision's 96 display cells still leaves all 589 cells simulated.

### Notes
Panda3D renders this 3D app; historical 2D uses Pygame. PyTorch neural computation
and display-only NumPy/GPU masks remain separate. No model switch or subagents.


## 2026-10-13 - Other relays explained and softened to steel-blue

### Goal
Explain Other relays and replace overwhelming white with a distinct calmer color.
User reports the redesigned UI otherwise works well.

### Changes Made
Dashboard-only #5B8FB9 override for structural relay activation colors and matching
checkbox swatch. Clearer description: cells retained on sensor-to-movement paths.
Guide explains these 241 intermediate cells are a structural selection category,
not one biological type or established individual function. Prior UI code archived;
updated status, queue, decisions, guide and verification.

### Decisions
Use muted steel-blue rather than white. Copy the immutable source palette for
presentation; preserve all other group colors, membership, timing, opacity,
neural/physics/controller behavior, receipts and old results. User's favorable
UI feedback is recorded honestly, without inventing exhaustive native input
review or complete algorithm understanding. One agent; no training or new assay.

### Tests / Verification
Existing dashboard tests: six pass, zero skips, one upstream warning, 15.83s.
Actual app palette probe changes exactly 241 structural entries; source palette
and every other entry unchanged. Paused 1000x700 PNG reviewed; snapshot/time/
actual-reference neural steps unchanged and zero. All 3,010 protected files and
original source-bound candidate pass. Full suite: 928 passed, zero failures/errors/
skips, three upstream warnings, 414.38s, exit 0. Exact commands and evidence in
docs/verification/relay-color.md and artifacts/relay-color-2026-10-13.

### Problems / Failed Attempts
None in implementation/checks. Reviewed neuron color endpoint and swatch together;
existing source-bound white palette could not be edited without invalidating
historical evidence, so the override lives only in the dashboard.

### Current State
Display refinement complete and verified. Reopen dashboard to load new palette.
Scientific behavior/records preserved. No commit, push, publishing or model switch.

### Next Step
One task / preferred Sol Medium: learning roadmap lesson 1, launch -> source
verification -> worker -> snapshot -> display. Continue toward full understanding
for honest AI-assisted interview discussion.

### Notes
These neurons were selected as intermediate connection paths in a sourced graph.
Their inclusion is structural evidence, not proof that each is a specific
functional relay. A display color does not change PyTorch calculations.


## 2026-10-20 - Private GitHub import and security review

### Goal
Publish the complete FlyBrain project to the user's GitHub with a security review.

### Changes Made
Created private gmpaunica/Fly-Brain, initialized main and pushed the complete
source/tests/docs/data/archived-code/experiment collection. Added credential/cache
ignore rules, exact-byte Git attributes, audited dependency constraints, security
setup/report and weekly Dependabot configuration. Enabled dependency alerts and
automated security fixes. Raised build setuptools floor and updated local pip to
26.2.2 and setuptools to 84.2.0. Updated installation guidance/status/queue/decisions.

### Decisions
User explicitly authorized repository creation and push. Default private; use
GitHub noreply identity and existing Git credential manager without exposing or
saving a token in project files. Preserve scientific files and source-bound
receipts exactly; no training, retuning or new scientific assay. Keep historical
environment descriptions as history. One agent; no model-switch claim.

### Tests / Verification
Gitleaks 8.30.1, official release checksum verified: 4,496,200,962 bytes, zero
secrets, exit 0 (10m36s). Supplementary publication-file and recovered truncated-log
scans also clear. Bandit 1.9.4: no high, one reviewed fixed-HTTPS medium and 107 low
findings, no scan errors. All 1,571 Python files parse; selected dangerous-call
patterns absent. pip-audit 2.10.1: 30 versions, zero known advisories/skips after
fixes, including a fresh installed-environment check. Editable installation dry
run with all extras/constraints succeeds. Full suite: 928 passed, zero failures/
errors/skips, three upstream warnings, 891.09s, exit 0. After build-tool-only update:
17 launch/dashboard checks pass, zero skips, one upstream warning, 19.29s, exit 0.
Full engineering tests used isolated ledgers, retained with reports under
artifacts/github-publication-2026-10-20. Initial remote private visibility and all
6,764 paths/modes/blob hashes/sizes match local commit 81c0978 and root tree.
Exact commands/results/limits: docs/verification/github-publication.md.

### Problems / Failed Attempts
Initial dependency audit reported 16 advisory entries in pip/setuptools (some
are duplicates); updated build tooling and constraints, fresh audit is clear.
One pre-existing interrupted gzip caused a scanner EOF warning. Independently
recovered/scanned all 1,771,585 readable bytes from its 142,040 compressed bytes;
original unchanged. GitHub warns about two historical files above 50 MB but
accepts both. Initial scan/staging contention increased full-suite wall time.

### Current State
Project published privately; scientific sources/data/results preserved. Local
runtime dependency versions unchanged. Dependency monitoring configured. A passing
scan cannot certify zero risk; no new native input/aesthetic review is claimed.
Final documentation/receipt follow-up is checked against remote main at handoff.

### Next Step
One task / preferred Sol Medium: learning roadmap lesson 1, launch -> verified
source -> worker -> snapshot -> display. Ask the user to explain/predict behavior.

### Notes
Git stores reproducible project files and dependency specifications; environments
and credentials stay local. Exact line endings matter here because recorded
scientific receipts hash source bytes. This remains an AI-assisted project.
