# FlyBrain analysis and architecture

Status: M0-M5 and evaluation implemented; M6 navigation improvement verified
in the frozen fresh-case setup on 2026-09-23. These are engineering choices, not biological mechanisms. Read only
the relevant sections. Product intent is in [the preserved brief](reference/PROJECT_BRIEF.md).

## Assessment

The project has a useful progression: a deterministic world, a sensory interface, a small neural controller, a visible feedback loop, and only then learning and biological data. The main feasibility risk is the learning experiment. A working SNN and a moving fly alone do not establish learned behavior, or an advantage from connectivity. Treat each as a separately tested result.

The first demo should prove the eight-part loop in the original brief, even if navigation is poor. The first ML result should demonstrate measured improvement over the same fixed controller on unseen starts. A claim about biological topology requires sourced connectivity and fair controls; a hand-designed directional circuit cannot establish that claim.

Do not front-load a generic simulation framework. Build ordinary classes and functions as each milestone needs them. Keep the environment usable without Pygame and training usable without a display. Dense PyTorch tensors are a reasonable first choice at 50-200 neurons; decide on sparse storage after measuring or importing a larger graph.

## Boundaries

| Owner | Allowed inputs | Output / responsibility |
|---|---|---|
| `world/` | Body state, sources, motor command, fixed `dt` | Physics, bounds, antenna sampling; owns coordinates |
| `neural/` | Sensory values, neural state, neural `dt`, local generator | Tensor state, spikes, decoded motor activity; no world objects |
| `simulation.py` | World and controller | Ordered stepping, reset, snapshots; orchestration only |
| `visualization/` | Read-only world/neural snapshots | Draw arena, graph and HUD; never advance simulation |
| `training/` | Sensory episodes, objective, chosen parameters | Optimization, checkpoints, separate evaluation |
| `data/` (later) | Versioned external connectivity/annotations | Graph and provenance, with documented transformations |

Coordinates may be used by reward computation and evaluation. They must not enter the inference/controller input or be disguised as an action computed by the environment. Supervised targets, if used, must be documented separately from observations. Inspect the actual controller call boundary, not merely the name of an input type.

## M0-M1: geometry, motion, and timing

- World units are arbitrary simulation units. The initial viewport maps one unit to one pixel. Time is seconds; heading and angular speed are radians and radians/second.
- Origin is the upper-left; +x is right and +y is down. Heading 0 points right. Positive rotation is clockwise/right. Normalize heading to `[0, 2*pi)`.
- Default arena: 800 by 600. Fly collision radius: 6. Maximum forward speed: 100 units/second; maximum angular speed: `pi` radians/second. These are scaffolding defaults, not biological measurements.
- `MotorCommand(forward, turn)`: forward in `[0, 1]`; turn in `[-1, 1]`. Turn -1 means left, +1 right. Reject nonfinite/out-of-range values. The same command type will later be used by the neural decoder.
- `Fly.step(command, dt)` mutates body state. For positive dt, set commanded speeds, update heading, then move using the updated heading. This semi-implicit ordering is part of the contract. `dt == 0` changes nothing; negative/nonfinite dt is an error.
- Clamp the fly center to `[radius, width-radius]` and `[radius, height-radius]`. No bounce or wrap. Preserve heading and commanded speeds at a boundary; zero actual displacement at the wall is possible. Report whether the attempted step crossed a boundary.
- Reject nonpositive/nonfinite dimensions/radius and an arena too small for the fly diameter. Start/reset centered, with heading drawn using a local `random.Random(seed)`; same seed yields the same initial state. No global RNG use.
- Physics ticks are exactly **0.02 s (50 Hz)**. Render at approximately 60 FPS using an accumulator. Rendering speed cannot change the number or order of simulated ticks for the same accumulated duration and command schedule.
- Keep wall-clock accumulation separate from a single fixed step. Reject negative/nonfinite elapsed time. Allow a small numerical roundoff guard at tick boundaries. Unit tests compare frame partitions; they must not demand identical paths for different physics step sizes.
- The interactive adapter may clamp a single wall-clock frame delay to 0.25 s to avoid catch-up after a dragged/paused window. Document that the interactive simulation then lags real time. Headless runs use exact ticks and never drop them.
- Keyboard mapping: W/up forward; A/left turn -1; D/right turn +1; opposing turn keys cancel. Key release yields zero command. R resets seed/body/accumulator; Escape/window close quits. Controls are explicitly marked temporary manual debugging.

## M2: antenna geometry and odor

For heading `h`, forward is `(cos(h), sin(h))`; the left unit vector is `(sin(h), -cos(h))`. Antennae sit ahead of the body with opposite lateral offsets. Test left/right geometry at headings 0 and `pi/2` to catch a mirrored steering convention.

Start with one static scalar odor: `strength / (distance_squared + softening_squared)`, with positive softening. Keep raw concentrations finite/nonnegative; define any encoder normalization explicitly instead of silently tuning world physics to make neurons spike. Start with deterministic current encoding in M3, so circuit failures are reproducible without Poisson noise. Add seeded stochastic encoding only as a separate tested change.

Implemented M2 sensory contract: immutable `SensoryObservation(odor_left: float,
odor_right: float)` in `world/sensing.py`. It carries only finite nonnegative raw
concentrations. `sample_odor` owns coordinate-based sampling, separate from the
observation boundary. Temporary manual movement continues until M4.

M2 uses forward/lateral antenna offsets 10/5, a static source at 75% of arena
width and 50% height, strength 1000 and softening 20. These are arbitrary
scaffolding. The field extends outside walls; antenna points are not clamped.
Simulation publishes current-pose readings at initialization, after movement and
on reset; renderers only read them. This HUD observation is post-movement, so M4
inference explicitly samples the pre-movement pose separately. No encoder
normalization is selected yet. Source state is immutable and unchanged by reset.

## M3-M5: neural loop and visualization

M3 implementation: standalone CPU PyTorch under `neural/`, independent of
world, app and renderer. Full assumptions and fixtures are in
[M3 plan](plans/M3.md) and the M3 decision in `decisions.md`.

Time is seconds. Voltage, external current and synaptic weights are arbitrary
normalized model units, not biological measurements. Defaults: dt=0.001,
tau=0.020, rest=reset=0, threshold=1. Each step uses
`candidate = rest + (v-rest)*exp(-dt/tau) + I*(1-exp(-dt/tau)) + W @ prior_spikes`.
The passive term is the exact solution with constant I (a steady-state
voltage displacement from rest); W is a delayed instantaneous end-of-step
voltage jump. Equality fires, overshoot is discarded by hard reset, and
there is no refractory counter (at most one spike per neuron per substep).
All recurrence uses prior spikes; no same-step cascade. Generic LIF populations
permit signed currents and weights. Invalid/nonfinite input is rejected before
state advances; nonfinite transitions are rejected without modifying that step.

The fixed synthetic graph has groups of 10/10 sensory left/right, 25/25
interneurons left/right, and 10/10/10 motor left/forward/right. Only same-side
sensory→interneuron→motor edges exist, with locally seeded positive jumps in
[0.096,0.144). Every neuron has a unique label and group. Forward motor current
1.3 is tonic scaffolding. Independent sensory currents use
`2*min(concentration/0.025,1)`; zero odor produces zero sensory current, while
tonic forward activity remains. This hand-designed circuit has no navigation,
learning or biological-topology claim.

`SyntheticNetwork.advance(observation, substeps=20)` holds encoded/tonic
currents for 20 one-millisecond steps and returns int64 per-neuron counts,
start/end completed-step IDs, elapsed simulation seconds and a resulting
post-substep snapshot. Last-step bool spikes alone omit earlier activity;
M4 uses full counts and labels the sensory observation pre-movement.
Float32 is default; float64 is supported. CPU allocations are explicit,
including under ambient device contexts. Weights/state/snapshot access returns
copies, drawing/reading cannot advance the model, and reset clears voltages,
pending spikes and step count while retaining topology. Inference uses no
autograd route. The optional `neural` extra leaves manual world imports usable
without PyTorch.

M4 tick order:

1. Sample left/right antennae at the current body pose.
2. Encode sensory current and hold it for this world tick.
3. Advance 20 neural substeps; accumulate all population spike counts.
4. Decode normalized motor activity to `MotorCommand`; move the world once by 0.02 s.
5. Publish a snapshot with tick ID, observation, neural activity summary and resulting body state.

Implemented M4 keeps these boundaries explicit. `NeuralController.decide` accepts
only `SensoryObservation`; `closed_loop.py` owns coordinate-based sampling and
world movement. Each ten-neuron motor group is reduced from the int64 count
tensor by `min(group_sum / 10, 1)`. Forward uses the forward value and turn is
`right - left`, so positive remains clockwise/right. A fixed EMA with alpha 0.5
smooths both values before `MotorCommand` clamping. One spike per neuron per
20 ms interval (50 Hz mean activity) is therefore the arbitrary full-scale
reference. The scaling, sign mapping and smoothing are hand-written actuator
scaffolding, not PyTorch learning or biological measurements.

The observation stored in `ClosedLoopSnapshot.pre_movement_observation` belongs
to the pre-movement pose; its immutable body state is post-movement. Reset clears
the network, motor smoothing, body/world timing, closed-loop frame accumulator
and last snapshot while retaining the source and fixed topology. A tested output
gate can silence `MotorCommand` without stopping neural/decoder state, proving
that body movement is caused by decoded output. Manual and neural app modes are
explicit; the neural path is imported lazily so the manual app remains usable
without PyTorch.

M5's viewer consumes every `ClosedLoopSnapshot` completed before a display
frame, because a 60 FPS renderer can otherwise miss activity accumulated over
1 ms neural steps. `BrainActivityModel.observe` decays traces using each
activity interval's simulation seconds and flashes neurons from the full int64
count tensor. `BrainView.draw` only reads those traces, so repeated frames do
not change activity, world state, neural state, or randomness.

Neural mode uses a 1200 x 600 window: the original 800 x 600 world plus a
400 x 600 brain panel. Node positions are a fixed functional layout by the
seven synthetic groups; they are not anatomical coordinates. Group colours
remain visible during flashes. A deterministic, group-balanced subset of 80
nonzero `W[post,pre]` edges is drawn in pre-to-post direction. The panel shows
tick/substep/simulation timing, recent motor-population traces, and the decoded
forward/turn command. Reset clears visual traces independently while the normal
closed-loop reset clears simulation state. Manual mode remains 800 x 600 and
does not import Torch.

## M6-M8: learning and scientific gates

FB-06 fixes the first evaluation protocol before any optimization. The source
is at (600,300), success means fly-center distance at most 20 world units, and
an episode stops immediately on success or after 600 world ticks (12 seconds).
Boundary contact counts but does not terminate. Locally seeded starts occupy a
60–180 unit annulus; train, validation, and final seeds are disjoint and all
generated poses are recorded. The primary objective is arrival rate; secondary
all-case progress and success-only time/path efficiency keep their denominators
explicit. The evaluator owns scoring coordinates and builds a fresh closed loop
per case. It never sends those coordinates to the controller. See the
[FB-06 plan](plans/M6-BASELINE.md) and [recorded baseline](experiments.md).

Boolean thresholding alone does not supply a useful autograd path for synaptic weights. Before building a trainer, state what is trainable and how the objective reaches those parameters. PyTorch documents limits around [non-differentiable operations](https://docs.pytorch.org/docs/2.14/notes/autograd.html); choose and test a deliberate learning route.

Recommended first investigation: freeze the spiking core and optimize a small PyTorch motor readout from filtered spike features. A supervised sensory-only teacher is an optional calibration experiment; label it imitation and do not claim it proves reward learning or biologically constrained synaptic learning. The first ML milestone still needs improved held-out embodied behavior. If features are insufficient, measure that limitation before increasing the model.

Training internal pathway gains is a later option requiring a documented surrogate gradient or a valid reward-based estimator. Neither the Python world nor a hard threshold becomes differentiable by setting `requires_grad=True`. Verify a finite, nonzero gradient and an actual update to every intended parameter on a controlled example. Verify frozen parameters and forbidden edges remain unchanged.

Before training, lock distinct training, validation and final evaluation cases. Tune on training/validation only. Compare fixed and trained controllers on identical held-out cases, episode horizons, sensor scales and budgets. Add a capacity-matched randomized graph control, with its degree/weight matching choices documented, before claiming a topology advantage. Report failed runs and uncertainty across seeds; do not select only a favorable seed.

Save configuration, seeds, initial conditions, software/device, checkpoint, and per-episode metrics. Include failures when reporting success/time metrics; do not make successful-only times look like all-episode averages. Check reload reproduces evaluation outputs on the same environment. PyTorch cautions that [reproducibility is not guaranteed across releases/platforms](https://docs.pytorch.org/docs/2.14/notes/randomness.html), so record that environment.

Only mark M6 complete when recorded results show the required improvement. A trainer whose loss changes but whose behavior does not improve is an honest partial result. M8 must preserve data source/version/license, neuron IDs, filtering, edge direction, signs if known, and mapping assumptions. Replacing a toy graph with a dataset is not automatically a complete sensory-to-motor biological pathway.

The [2026-09-22 recovery review](plans/M6-RECOVERY.md) supersedes the proposed
FB-07K escalation. Before another imitation intervention, measure the sensory
teacher in the world and compare turn fitting with a constant train-target
mean. Before a learning claim, compare trained and untrained versions with
identical encoder/features/decoder settings, alongside the original fixed
baseline. Retire final seeds 200-207 from confirmatory use: FB-07 and FB-07J
have both evaluated them. A new validation gate cannot restore their novelty.

### Optional neural charge readout (2026-09-22)

`record_charge=True` adds observation of actual LIF transitions without
changing their voltages or spikes. Each substep records leak loss
`v_before - v_relaxed` and reset loss `v_candidate - v_after` in float64.
The interval's voltage change plus these losses, divided by its substep
count and `1-exp(-dt/tau)`, is a drive-equivalent charge measurement.
`NeuralActivity.integrated_charge` is optional; default activities preserve
their previous contract. It includes synaptic jumps outside the sensory
groups. The charge filter requires sensory groups with no incoming edges.

The new readout keeps seven spike EMAs and replaces the two endpoint voltages
with independently clamped `logit(Q/2.5)`. With the rational encoder these
approximate `log(concentration/0.005)`. This engineering transform avoids
spike-reset phase discontinuities and compensates encoder compression; it
does not compute bilateral contrast or teacher steering. Actual simulated
membrane changes are the decoder's input source. Because the transform
reconstructs sensory drive, successful decoding does not establish benefit
from the downstream graph, biological fidelity, or learned spiking dynamics.

For example, odor `[0.008, 0.010]` gives charge about `[1.53846, 1.66667]`
even when endpoint voltages are `[0.97249, 0.08128]`. The last two features
are about `[0.47000, 0.69315]`. Together with seven spike values they form
a `[9]` tensor; the learned weight matrix is `[2,9]` (output,input). PyTorch
computes two weighted sums, then sigmoid/tanh bounds forward/turn. Autograd
and Adam update those 20 readout scalars during offline fitting; the neural
simulation has no autograd path. These are separate operations.

Charge checkpoints use schema 3 and explicit transform constants, feature
names, encoder and graph hash. Old schema-2 voltage checkpoints remain
separate. The candidate passed training calibration and the fresh matched
evaluation under [the locked protocol](../experiments/m6-charge-repair/heldout_protocol.json):
32/32 final arrivals versus 1/32 for each control. The scope and limits are in
[the experiment record](experiments.md#m6-fresh-matched-evaluation--2026-09-23).
The live app still uses the fixed decoder.

### M7 / FB-08 experimental interventions (2026-09-23)

`training/fb08_experiments.py` builds independent experimental network copies;
the original neural classes and checkpoints remain unchanged. Current ablation
is applied before LIF integration. Synaptic interventions replace both the
experimental topology's weights and the population's owned weight matrix,
preserving group labels and neuron equations. No controller gains access to
geometry, evaluation cases or target answers.

The [recorded assay](../experiments/fb08-interventions/results.json) completed
ten matched conditions on sixteen diagnostic starts, zero training updates.
Intact arrivals were 15/16. Sensory ablation gave 0/16; removing synapses,
changing sensory-pathway gain and three constrained random wirings each
retained 15/16. Thus this assay does not establish a navigation benefit from
downstream connectivity. M7's saved-config/result/plot requirements pass;
M8 needs a separate biological source/pathway feasibility decision.

### M8 / FB-09 BANC source contract (2026-09-24)

The [feasibility decision](plans/FB-09.md) selects BANC v888 Dataverse release
3.0 under CC BY 4.0. Static metadata and v2 edge files are pinned by Dataverse
file ID, byte size and MD5. Preserve `root_888` as opaque string metadata and
derive contiguous tensor indices deterministically. Source edges are
`pre -> post`; the project tensor remains `W[post,pre]`.

The proposed 83-neuron subset contains 68 proofread VA2 ORNs with qualifying
ipsilateral projection-neuron edges, bilateral `VA2_adPN`, five reproducibly
selected corridor neurons, bilateral DNa02 and six named leg motor neurons.
Keeping every internal edge with at least five synapses yields 176 directed
edges. Raw synapse count and postsynaptic-input fraction remain provenance.
The first simulation transform is unsigned and non-negative; its global gain,
LIF dynamics, equal-current ORN encoding and 2D motor mapping are explicitly
synthetic, not supplied by BANC.

The bilateral sensory observation may drive only the matching ORN pools. A
future topology-dependent output must exclude charge features that reconstruct
sensory drive. Right-minus-left DNa02 activity is the steering observable;
mean activity of selected coxa-anterior motor cells is only a forward proxy
because the simulated body has no legs or gait. No import, dynamics claim or
topology-benefit claim is complete at this decision stage.

### M8 / FB-09 imported subset and dynamics gate (2026-09-24)

The 90,011-byte BANC artifact is packaged in `flybrain.data`. The optional
PyArrow extractor reads only pinned full-source files after size/MD5 checks.
The runtime loader validates release/license, string IDs, unique endpoints,
edge counts, normalization and corridor directions, then creates a copied
83 x 83 PyTorch tensor with `W[post,pre] = connectome_gain * norm`.
`BancNetwork` feeds the existing `LIFPopulation` through the existing bilateral
odor encoder. It has no tonic drive and does not alter the synthetic app mode.

The first bounded dynamics check used gains 1 and 600. The low gain stopped
before projection neurons; the high gain caused right-side activity from
left-only odor and sustained firing after odor ceased. Both remained finite.
The imported topology is available for simulation and its tiny-edge dynamics
are correct, but no real-graph gain passed selective propagation/washout.
No biological motor decoder or training run is authorized by this result.

### FB-09 normalized recovery interface (2026-09-25; not accepted for behavior)

`NeuralTopology` and `SpikingNetwork` structural protocols let the existing
encoder/controller/view consume either graph without changing the simulation
clock. Manual mode still imports no PyTorch. Explicit `BancDynamicsConfig`
selects the recovery transform; omission retains the historical raw transform.
Source and effective topology are separate objects. The latter masks self
edges and normalizes incoming retained counts; no raw artifact is rewritten.

`BancMotorDecoder` receives only full-interval endpoint spike counts. Its
float64 CPU feature vector has shape `[8]`: two DNa02, three left motor and
three right motor cells, sorted by source ID within roles. Counts/dt gives Hz;
a 100 ms EMA is updated once per 20 ms tick and reset per episode. Fixed
forward/turn reductions are diagnostic scaffolding, not learned parameters.
`BrainView(kind='banc')` reads snapshots and depicts functional groups, not
anatomical positions. Neither drawing nor frame rate advances neural state.

The declared dynamics gate failed, so `--control banc` is unavailable in
practice until a verified config is supplied. A future receipt must travel
with original `banc_dynamics_results.json` and `banc_dynamics_protocol.json`
sidecars; hashes, success flags, selected config and installed-source hashes
must agree. This is a provenance consistency check, not tamper-proof signing.
No old failed result can become accepted by changing receipt labels alone.

### BANC dynamics variants and accepted measurement contract (2026-09-26)

`sensory_mode='phase_rate'` independently maps antenna concentration to
`min(96,176*sqrt(c))` Hz, with sorted-ID uniform initial phases. Each 1 ms
step schedules external ORN current pulses; previewing `encode` is pure and
reset restores all phases. A retained ORN-to-ORN source edge can add activity
at its recipient. External waveforms and actual neuronal spikes are distinct.

The approved `projection_dynamics='charge_preserving'` variant changes only
the two PNs to tau25ms, threshold1.1 and subtractive reset, with one spike per
step. Remaining cells retain tau20ms, threshold1 and hard reset0. These are
engineering assumptions; source counts, edges and endpoint features stay fixed.
The alternative 5ms PN synaptic filter is a separate closed failed experiment,
never combined with the phase-rate/charge candidate. Historical defaults remain.

Causal population reconstruction retains the selected dynamics. Every ORN is
compared individually; only the actual sensory-edge receiver may differ after
an all-synapse cut. Downstream-only cuts retain that sensory edge. All four
controls passed in the approved latency-aligned protocol. Rendering tests
verify the candidate's future trajectory remains identical after extra draws;
they do not establish scientific acceptance or live visual usability.

The accepted package now bundles its configuration, original successful
protocol and result. Core/source hashes must still match on load. The separate
offline measurement windows derive from sourced PN-to-DNa02 path lengths:
left delay4ms, right3ms. Their duration remains750ms and thresholds1Hz/2%.
Original simultaneous-window counts/gates remain in the record; their mid-right
failure is not erased. No live input, neural clock, decoder feature or output
is delay-aligned, and no future data enters the controller.

Teacher-readiness execution records accepted bundle/raw subset/all source and
prior-file hashes before stepping, verifies actual effective weights afterward,
and limits eight teacher plus eight fixed episodes to9,600ticks. It ran5,808
ticks, passed teacher8/8 and feature gates, and did not optimize. Teacher
commands are labels/data-collection scaffolding, never live BANC feature input.
The fixed BANC baseline is0/8; learning belongs to FB-10.

### FB-10 endpoint-readout protocol (2026-09-26)

`neural/banc_readout.py` adds a separate 8->16->2 CPU float32 model, with tanh
hidden units and sigmoid-forward/tanh-turn outputs. Its 178 trainable weights
and biases receive offline imitation gradients. The source graph, phase-rate
encoder and neuron dynamics remain inference-only. The decoder composes the
unchanged `BancMotorDecoder` to obtain exactly its eight 100 ms endpoint EMA
rates; the fixed proxy command is discarded. No odor values, coordinates,
membrane charge or teacher commands enter the readout.

Training mean and population standard deviation (floor1 Hz) are fixed buffers,
computed from the preserved 1,008 training samples before fitting. Checkpoints
bind feature IDs/order/units, EMA timing, raw source hash, transform, accepted
bundle/dynamics, actual effective weights, architecture, protocol hash and
normalization. Loading rejects mismatches and nonfinite state. Reset clears
neural and EMA history without changing model parameters.

`training/banc_protocol.py` prepares an exclusively written protocol and hash;
it performs no scientific stepping or fitting. The protocol binds its output
directory, original readiness bytes, training cases, software and source/history
hashes. `training/banc_training.py` separately consumes it via an exclusive
receipt before optimization. One 500-update Adam run (seed1010, lr0.01) has
separate turn-error, integrity and matched behavior gates. It runs at most25
training episodes /15,000 world ticks /300,000 neural steps, including replay.
Actual neural counters must agree with world ticks. A nonfinite fit stops
immediately; a finite fit-gate miss still records training behavior controls.
All attempts remain consumed. No automatic retry or held-out execution exists.

Preparation was verified on 2026-09-26. The separately authorized fit ran on
2026-09-27 and is closed negative: turn MSE 0.06678675, trained arrivals 2/8
versus untrained 1/8 and fixed 0/8. The circuit and contracts are unchanged;
all integrity/replay gates passed, but turn-fit and arrival gates failed.
See [FB-10](plans/FB-10.md) and its [result](../experiments/fb10-banc-readout/results.json).
Fresh held-out starts still require a later locked protocol after all training
gates pass. Recovery begins with a train-only diagnostic protocol, not a retry.

### FB-10 measured readout recovery (2026-09-27)

Separate modules preserve all old protocol-bound source. Fixed pair conditioning
and 100/500 ms causal endpoint histories precede learned heads. The latest
`neural/banc_projection.py` observes 16 endpoint histories and four PN incoming
event histories. At each actual 1 ms step it reads
`W[PN,pre] @ previous_spikes`, accumulates the real internal events, and divides
by row budget 3 and interval seconds. It never reads external sensory current,
voltage, odor columns, teacher labels or world geometry. Every spike/voltage
and original endpoint history remains identical to the original circuit.

Twenty raw neural features are re-expressed as endpoint pair means/differences
and fast/slow PN pair means/differences, then standardized by fixed training
buffers. The 20->16->2 head has 370 learned weights/biases. For a batch
`[15864,20]`, PyTorch's first matrix is `[16,20]`, then `[2,16]`; forward is
`X @ W.T + b`. Filters and graph have no gradient path. Reading PN inputs moves
the readout boundary upstream: downstream mediation/necessity is not claimed.

Training collection uses teacher labels on predetermined teacher and frozen
student trajectories. Labels cannot replace a student body command. The PN
collection replay matched all 64 outcomes and every target. Original slow
history reconstruction differs from live `1-expm1` filtering by <=1.27898e-13Hz;
the failed bit-exact raw gate stays recorded. `training/banc_projection_fit.py`
validates a separate <=1e-12Hz file-only canonicalization, preserves all original
16 endpoint values, and appends the four measured PN histories unchanged.

The latest original-encoder head passes original fit/navigation/replay gates,
but fails a separately added covered-data absolute gate. Its fit is closed.
Only an explicit acceptance decision can admit it to a different frozen assay
under the original gates; the old result/receipt cannot be edited or reopened.
The [proposal](plans/FB-10-FROZEN-CANDIDATE-ASSAY.md) was subsequently accepted;
the recorded frozen assay passed while retaining the original failure.

`neural/banc_range.py` is an isolated fixed sqrt/log encoder prototype. Its
entry probe failed endpoint contrast and is closed; `training/banc_range.py`
blocks scientific collection/fitting. It does not replace the accepted package,
app encoder or any completed fit. No scientific source guard was weakened.

### Frozen trained app adapter (2026-09-28)

`trained_app.py` is a separate entry point because the original app and views
are hash-bound to historical protocols. It reuses `ClosedLoopSimulation`,
the real projection controller and original drawing, with additive HUD
subclasses that identify the frozen learned readout. Existing modes delegate
to the original app. The original manual dependency path stays lazy.

`neural/frozen_banc.py` pins five checkpoint/training/assay artifacts and verifies
the assay's 64 source hashes and four accepted data files. It uses the strict
projection checkpoint contract and saved normalization; all 370 parameters
have gradients disabled. It reads no training samples, runs no assay and
consumes no receipt. All historical sources and artifacts are preserved.
This is a checkout-based demo requiring its saved experiment evidence, not
a standalone redistributed model package. The extra covered-fit failure
remains pinned. See [the task](plans/TRAINED-BANC-APP.md) for its verification
limits and a concrete feature-to-matrix calculation.


### Default verified app (2026-09-28)

`python -m flybrain` invokes additive `verified_app.py` with the separately
verified 750-update readout. Its loader `neural/verified_banc.py` pins the new
five fit/assay files and verifies all 68 recorded sources and four accepted data
files. All learned parameters have gradients disabled. The app preserves the
sensory-only controller boundary and read-only renderer. The main loop is
adapted locally because the earlier entry point is protocol-bound; views and
neural calculations are reused directly. `trained_app.py` retains its historical
500-update model; `app.py` retains its manual default and fixed BANC mode.
This remains a checkout-based demo requiring the saved evidence files.

### Approach-and-stop candidate (2026-09-28)

`world/food.py` samples local body/patch overlap and provides
`FoodObservation(odor_left, odor_right, food_contact)`. The candidate network
in `neural/food_contact.py` retains the 83 sourced BANC IDs and their
effective connections, maps each antenna concentration to the declared
`96*c/(c+0.1)` Hz engineering rate, then appends an engineered 100 Hz contact
cell and a fixed-weight relay. The first 83 cells remain sourced; the last
two cells and their edge are not part of the BANC subset.

`neural/food_readout.py` filters actual relay spikes over 50 and 200 ms and
appends those histories to the existing 20 endpoint/PN histories. The fixed
neural graph and filters produce a 22-value row. Only the 22→16→3 PyTorch
head learns: the first two outputs encode forward speed and turn; a positive
third stop logit produces `MotorCommand(0, 0)`. Rendering reads snapshots
and cannot advance neural state, the RNG, or simulation time.

`training/food_teacher.py` is hand-written demonstration scaffolding that
reads only the same three sensory values. The preflight, fit, and fresh assay
live in separate additive modules. Fit data and normalization use training
cases only; validation gates final evaluation; all scientific result files
are preserved after a failed gate.

## Isolated 3D subsystem (2026-09-30)

The user-approved [3D plan](plans/FLIGHT3D.md) supersedes the earlier deferral
for this subsystem. Existing 2D modules and frozen experiment sources remain
unchanged. `python -m flybrain.flight3d` launches its own Panda3D application.

- `contracts.py`: immutable observations, commands, snapshots and trial setup.
  SI metres/seconds, +Z up; initial position `(0.3, 0.3, 0.1)`, yaw zero (+Y).
- `world.py` / `sensors.py`: 5 ms physics, 20 ms observations. Geometry is
  restricted to setup, physics, sensing and evaluation. The controller receives
  RGB eyes, bilateral odor, body-local angular velocity/specific force and
  physical contacts; it receives no coordinates or ground-truth bearing.
- `anatomy.py` / `assets.py` / `audit.py`: pinned BANC v888 registry and directed
  source edges, native SWC/registered mesh assets, cache provenance and exact
  source reconciliation. One transform `(X,-Y,Z)*1e-6` maps nm to viewer mm.
- `neural.py`: 1 ms assumed LIF spikes through fixed sourced topology. Motor
  endpoint histories feed independent power, steering and landing readout heads.
  Silence gates each head after normalization so learned biases cannot drive a
  silent endpoint. Checkpoints bind to the circuit digest and freeze inference.
- `teacher.py`: hand-written visual servo and integrated local motion estimates.
  Known flower size and actuator calibration are declared assumptions.
- `experiment.py`: independent headless evaluator, frozen cases/protocol,
  prerequisites, teacher collection, one optimizer budget and strict gates.
  A failed stage closes the run. Exact executed source is archived with this
  first run; teacher failure prevented any fit or final-case access.
- `session.py`: headless orchestration and immutable snapshots. `viewer.py`
  advances the session only in its fixed-step scheduler; passive drawing,
  camera controls and anatomy loading never advance physics or neurons.
  Activity decays against snapshot time, including pause and replay behavior.

At a distance the viewer shows source root/reference positions and registered
region meshes, plus simplified project skeleton chains. Selection loads full
morphology in the same coordinates. Visibility modes affect drawing only.
Unused atlas records are not simulated and never get activity colors.
Missing positions, untyped records and morphology coverage remain explicit.
See [implementation assumptions and controls](FLIGHT3D.md).

### October 1 repair contracts

The sensory observation shape remains immutable `[2,24,48,3]`. Eight deterministic
subpixel rays undergo independent nearest-surface intersection before RGB averaging.
The shared `sensors.flower_contrast` function operates solely on RGB; both teacher
and neural visual encoder use it. Teacher altitude/search estimates integrate local
motion signals and reset with each episode.

Neural dynamics/readouts remain unchanged. `_advance` additionally records delivered
edge current, excluding discarded input currents and silenced destinations. Snapshots
carry immutable manifest-ordered interval totals and 300 ms filtered current rates.
`TransmissionTrace` updates inside the session on each simulation interval. The
renderer consumes those snapshots; display time affects camera interpolation only.

`network.py` retains group membership, anatomical centroids and exact bundle edge
indices. Unpositioned cells remain explicit, without fabricated geometry. The historical
`network_viewer.py` extends the retained detailed `viewer.py`; `navigation.py` owns
full panel rectangles and bounded camera interpolation. Rebuilding scrollable UI
preserves selection. Destroying a viewer removes its scheduler task.

`protocol.py` freezes source snapshots, coordinates, settings, software metadata,
circuit digest and protocol hash after 49/49 development checks. `experiment.py`
consumes those exact cases/configurations. A project-level exclusive authorization
receipt survives run-directory renaming. Durable journals distinguish stage starts,
episode completion, update starts/completions, interruption and closed gates. Every
optimizer update publishes a candidate checkpoint and cumulative parameter changes.


### Anatomical performance repair (2026-10-01)

`anatomical_viewer.py` is the default launcher. It reuses navigation/scroll UI
without constructing centroid bundles. `anatomical_display.py` caches immutable
source morphology and originally ranked at most twelve neurons from snapshot edge
activity (superseded by the October 3 display-only recorder below).
Registered surfaces/reference points support the silhouette; source branches use
batched coarse buffers, full selection is separate, hidden cord buffers are
deferred. Camera and display choices do not enter sensory/neural computation.

`simulation_pump.py` gives one serial worker ownership of pending fixed steps.
Only completed immutable snapshots cross into drawing. Pause/reset/exit barriers
prevent session mutation after a reset or shutdown. Wall-time credit is bounded
at 100 ms; the consequence on a slow machine is slower simulated time, never an
altered timestep. The original neural/sensor/world/session modules are unchanged.
The prior viewer modules remain intact for comparable performance measurements.


### Clearer controls and continuous camera (2026-10-02)

`brain_camera.py` owns the default brain camera's scalar-first quaternion, pivot,
pan, distance and input targets. Quaternion rotation carries right/up/back
vectors together; no polar up-vector switch exists. `brain_controls.py` routes
window-normalized input through viewport pixels and builds the toolbar/settings.
Rotation and pan interpolate with `1-exp(-dt/0.060)`; zoom and fit pivot use
`1-exp(-dt/0.120)`. Gesture release/focus loss discards pending rotation/pan.
Pan uses perspective scale without altering the distance floor. Cord/background
visibility never resets or fits the camera. Fit is explicit; presets change only
orientation. Camera and UI modules import no world/sensor/neural/session logic.

Cached geometry supplies fit bounds. Source-reference points use the same
Y=-0.43 mm brain/cord visibility plane as branches, rather than region labels
that can leave a few cord-positioned references visible with cord hidden.
Original arrays/annotations remain intact. Initial framing retains 1.10 mm at
(0.49,-0.17,0.145); background anatomy is on and is unrelated to trained cells.
The historical viewers/navigation remain intact for comparative benchmarking.

### Complete spike publication and owned branch colors (2026-10-03)

`activity.py` observes interval counts after every successful `FlightSession.advance`
through the optional recorder in `SimulationPump`. The scientific `FlightSnapshot`
contract and neural/session calculations are unchanged. A frozen `DisplayFrame`
contains the original snapshot plus episode/step identity and tuple-valued cumulative
counts, last-active interval end times and 200 ms exponentially averaged firing rates.
Publication uses a lock; the renderer acquires one complete frame for body, follow
camera, branch coloring, coverage and inspector. Reset is a worker barrier that
increments episode identity even for same-seed replay. `advance_fixed` pauses the
worker, drops pending wall credit and uses the same observed path for capture/checks.

Display pulses refresh for any positive interval count and ease smoothly to zero
over `.600` seconds without activity. The October 4 fade follow-up replaced the
rapid exponential display decay and hard cutoff with cubic smoothstep; the 200 ms
firing-rate average remains unchanged. No rendering event writes spike
history. Times describe completed control intervals, not neural substep timestamps.
There is no outgoing-edge, top-count or metadata-region activity filter.

`branch_activity.py` builds static XYZ+owner vertex buffers. The existing Y=-0.43 mm
midpoint branch split and 50,000-segment batching carry owner arrays alongside
coordinates, including deferred cord buffers. A nearest-filtered texture with
explicit RGBA upload stores one color per manifest-ordered neuron. Existing idle
brightness presets interpolate toward the existing warm range. Full cyan selection
is independent; background context is not assigned simulated activity. Resident
owner sets determine unique visible pulse counts, not camera screen coverage.
This is whole-neuron display activity, not anatomical spike propagation.

### Independent activity switches (2026-10-04)

`activity_classes.py` derives ten display categories from unchanged manifest roles,
caching disjoint membership and direct non-self adjacency. `BrainControls` owns a
frozenset of enabled classes and a whole/selected scope. Their intersection is a
tuple-valued neuron mask; all recorder and scientific snapshot values remain
complete. Selection clearing/background replacement falls back to whole scope;
replay/new trial retain choices and clear history in the existing pump reset.

`BranchActivityTable` optionally accepts one immutable RGB activation endpoint per
neuron and a display mask in `update`. Existing callers without options keep the
original warm interpolation. Class endpoints use alpha .43 and the existing
600 ms eased fade (superseding the initial exponential decay). Both renderer and upload tokens include the mask,
allowing immediate same-frame paused recoloring without static geometry changes.
Owner sets cached during branch splitting explain cord-only anatomy without
scanning segment arrays during inspector refresh. GPU pixels verify palette RGBA
and UI legend swatches. The legend is display-only; it adds no biological taxonomy.
Class switches occupy a compact two-column grid at the top of Display settings;
rendered-layout regressions require every class button inside the initial panel
viewport at 1440×900 and 1000×700. Shortcuts/scope follow below that grid.

### Continuous activity-color fade (2026-10-04 follow-up)

`pulse_strengths` uses only each frozen frame's simulation time and per-neuron
last-active timestamp. With `u=clip(age/.600,0,1)`, display strength is
`(1-u)^2*(1+2u)` (one minus cubic smoothstep). It starts at one, is .5 after
300 ms and reaches exactly zero with zero endpoint slope. New spikes refresh it;
repeated paused frames are unchanged. Never-fired cells and reset episodes stay
idle. Existing class RGB/alpha interpolation and tuple-mask upload keys remain.

This display envelope is independent of the existing 200 ms rate-average
calculation. Recorder counts, timestamps, scientific snapshots, neural rules and
simulation timing do not change. Off switches/scope exclusions still take effect
immediately as explicit display choices. GPU tests verify intermediate RGBA fade
and negligible final-step color change, alongside 30/60/120 Hz equality in silence.


### Exclusive encounters extension (2026-10-05)

`python -m flybrain.encounters3d` is additive; every historical Python source,
checkpoint, cached source asset, experiment and original launcher is preserved.
`scene.SceneSpec(seed, kind, position)` is an immutable environment-only exclusive
object definition. Flower physics/sensing delegate to the old code exactly.
Predator physics retain 5 ms flight steps with swept sphere contact and no flower
or odor path. Its cyan spherical surface uses independently occluded rays and the
same eight-ray RGB averaging. Geometry exists only in setup, physics, sensing,
presentation and evaluation; the controller signature remains FlightObservation.

`neural.EscapeBrain` subclasses the old FlightBrain, changing only sensory input
to engineered cyan detection on the existing 96 LPLC4 cells. The unchanged
589-cell/4,139-edge registry retains source IDs, directions, counts, geometry and
roles. ID-ranked 8x6 fields, peak/occupancy/positive-growth weighting and vision-only
drive are declared assumptions. The old 100 ms leak, threshold 1, subtractive reset,
1 ms propagation and unsigned incoming-count scaling are reused without learning.

The bounded image characterization failed the existing wing spike-history gate:
all physical one-object stimuli yielded zero wing spikes. No motor decoder or
retreat/retrigger policy was installed. `controller.PredatorController` advances
real circuit activity but actuates only `LocalHover`, a clearly labeled IMU
stabilizer with fixed initial rest/calibration assumptions and no RGB/geometry
steering. The limitation is specific to this modeled drive and motor representation,
not a proof that the source circuit cannot support alternative physiological models.

`session.EncounterSession` reuses old advance/replay orchestration, with an immutable
EncounterSnapshot (object kind/position for rendering/evaluation only).
`viewer.EncounterViewer` reuses the anatomical viewer, closing its initial idle
teacher worker before installing the new session; no monkeypatch or historical
factory edits. Buttons and R reset at the existing worker/recorder barrier.
Backspace replays exactly. Camera, display classes/scope, selected neuron and pause
are retained. Only this launcher enables All activity initially. DNp03 rates and
wing interval counts come from one atomic recorded DisplayFrame, never the RGB
detector. Predator color uses explicit ColorAttrib to match eyes on this WGL path.

`characterize.py` and `evaluate.py` keep image intervention and geometry scoring
outside controller inference, store new source-bound JSON/CSV, and refuse record
overwrites. No optimizer, checkpoint creation, interrupted-run continuation or
reserved assay occurs. See [encounter guide and recovery proposal](ENCOUNTERS3D.md).

### Passive brain-to-wing diagnostic (2026-10-05)

`encounters3d.diagnostics` adds mixin subclasses of the original flower and
predator brains. `_advance_measured` preserves each original TorchScript operation,
ordering and float64 precision, observing incoming increments and threshold values
before reset. The historical `flight3d.neural._advance` remains unchanged as the
parity oracle. Instrumentation does not feed any quantity back into stepping.

Each frozen `DiagnosticInterval` contains start/end neural time, effective incoming
current sum/peak, discarded current sum, peak pre-reset voltage, input phase peak,
counts, interval/episode first spike times and intervention IDs, in manifest order.
Currents/voltages are normalized model quantities, not amperes/millivolts. Incoming
edges to phase inputs or silenced cells are recorded as discarded, matching their
original voltage clamp. First spike timestamps are the exact 1 ms grid, unlike the
original display recorder's last-active completed interval endpoint.

`EncounterSession._snapshot` includes `pathway` in the same immutable body/spike
publication. The original pump lock/barriers own all mutations. Replay/reset clear
the diagnostic record, five-second/250-interval trace ring and first-spike history.
`diagnostic_panel.pathway_text` reads only a published `DisplayFrame` and manifest;
it cannot step the brain. The subclass viewer inserts diagnostics above the
retained lower-left legend/guide. Existing GPU class colors still use spike history,
never voltage/current or panel data. Cameras, pause, scene controls, teacher and
preferences retain their original contracts.

`encounters3d.investigate` locks one fresh 120-presentation protocol before onset,
streams all 6,000 intervals to compressed JSONL and pathway-cell CSV, stores paired
RGB arrays and records software/device/source hashes. Scene geometry and shutter
masking belong solely to the evaluator; the brain receives the unchanged sensory
observation boundary. No decoded movement, parameter updates or source extensions.
Scientific records close at their declared budget; saved-record postprocessing
does not rerun neural state. See [diagnostic report](PATHWAY-DIAGNOSTICS.md).

### Additive idle and moving encounters (2026-10-06, partial delivery)

`retreat3d.world` preserves historical world files and implements optional one-object
scenes, safe ground support and fixed 5 ms predator movement. `PredatorInput` is
environment-only and never enters `FlightObservation` or the fly command. Diagonal
input is normalized, speed capped at .08 m/s and the full sphere clamped to the
arena. Relative swept fly/predator segments include supported resting flies.
Removing a supporting flower releases contact while preserving pose/momentum.
Original above-ground flower physics and eyes have numerical parity tests.

`controller.KinematicEstimate` integrates only body-local IMU, assuming initial
upright grounded rest at the calibrated leg height. Orientation/velocity/relative
displacement and altitude stay shared across encounters. `IdleFlight` is explicitly
engineered: smooth seeded heading, inward bias from estimated home displacement,
.06 m/s desired cruise, .1 m altitude, seeded 3–6 s cruise and 2–5 s rest. The old
flower teacher retains its visual motor calculation; the shared estimate supplies
its state and suppresses duplicate integration, including engineered takeoff from
ground. Spawn mode/current foot observation carry no pose or steering answer.

`neural.MotionVisionBrain` combines the untouched predator visual formula with
the exact old 16-motion-input formula. `MotionBrain` is an independent motion-only
reference with identical observations/dynamics. Manifest-ordered actual-minus-
reference counts form nonoverlapping 200 ms signed 48-wing vectors. `FrozenDecoder`
holds eight normalized directional templates and accepts only that vector; norm,
cosine and margin gates are 1/.8/.1. Its failed-run diagnostic loader verifies
receipt, circuit and template identity. The diagnostic decoder never actuates the
fly. Anatomy always displays actual-brain spikes and its passive pathway data.

`threat` estimates contrast-weighted apparent visible solid angle and equivalent
circular angular diameter using existing eye geometry. No physical distance or
image-derived heading is returned. Five-sample 100 ms size/positive-growth averages,
appearance/growth thresholds and fifteen-step weak continuity are engineering
calculations only; no closed-loop stopping behavior was promoted.

`session` owns physics, paired brains, sensory control and immutable snapshots.
Spawn resets paired neural/transmission/decoder/threat state at unchanged world
time; the snapshot explicitly carries the neural time origin. `interaction` records
every applied 20 ms predator command plus ordered same-step spawns. `InteractionPump`
extends the original serial worker with a pause barrier that reconstructs each
recorded interval and resets display histories on replayed spawns, including an
endpoint spawn. Replay pauses at the exact endpoint; R resets to no object.
Pause/focus loss drops keys and pending worker input/credit. Object positions and
brain/behavior/angle diagnostics are drawn from published snapshots, not inferred
inside rendering. Moving sphere vertices are origin-centered and translated once.
Historical viewer/session/physics/neural modules are unchanged.

`experiment` locked one hypothesis before onset, calibrated 48 cases, froze
templates and validated 96. The gate failed 9/96; execution closed at 216 s /
432,000 paired neural steps. `audit_retreat_records.py` reconciles stored windows,
spikes, timestamps, source hashes and budgets with zero neural stepping. No
closed-loop assay or grounded neural takeoff/retreat/braking/recovery is delivered.
See [partial lab and outcome](RETREAT3D.md).

### Effective-gain turn-and-flee candidate (2026-10-06, closed negative)

Additive `turnflee` reuses `retreat3d` and leaves every historical implementation
byte unchanged. A frozen `ResponseConfig` contains gains, 143 retained
LPLC4→DNp03/DNp26 edges, 86 outgoing edges from those four cells, circuit/source
hashes and calibration receipt. Only the effective weights are multiplied after
historical count normalization; anatomical counts and LIF/encoding rules remain
unchanged. Both actual and independent motion-only reference receive the gains.

`ResponsePair` maintains rolling ten-interval/200 ms signed 48-wing counts every
20 ms and both brains' downstream rate audit. `WingScores` uses the six existing
i1/b3/b1 cells: i1/b3 increments favor the same side, b1 the opposite. Six-cell
norm >=1 plus a <=500 ms strong-event latch gates `EscapePolicy`; RGB cannot
choose turn side. Exact bilateral ties use a seeded engineering symmetry break.
The policy targets a 90° IMU-relative turn, brakes while turning, then approaches
.12 m/s at .1 m altitude. Weak-threat braking waits for horizontal and vertical
speed <.02 m/s. An IMU displacement guard uses .22/.18 m hysteresis and a
chosen-side tangent with inward component. These are explicit engineering rules.

`TurnFleeSession` inherits spawn/worker/replay barriers, preserves the body and
IMU across encounters, resets all new state and publishes immutable scores,
side, strength, escape state and configuration alongside actual spikes. New v2
logs bind the configuration; v1 logs delegate to the original model. Anatomy
continues to use actual spikes. Defaults require every final scientific gate;
development construction remains labeled experimental/default disabled.

One experiment selected lowest eligible (4,2), then closed at **87/96** signed
validation, below >=90 and >=5/6 each azimuth. No closed-loop cases or default
promotion. An independent saved-record audit reconstructs all 10,800 intervals,
windows, gates and budgets without advancing neurons. See [turn-and-flee guide](TURNFLEE.md).
### Sustained edge escape (2026-10-11, verified declared protocol)

Additive `escape_edge` inherits the unchanged paired neurons/temporal readout
and physics. Spikes commit a signed 90-degree turn. An IMU-derived estimate
then drives a ray endpoint within relative [-.285,.285] m horizontal bounds,
the fixed center-start calibration for a .6 m square with 15 mm wall inset.
Actual world/predator coordinates never enter control; evaluator/viewer geometry
is separate. Cruise .12 m/s, altitude .10 m; final 60 mm uses capped 2/s endpoint
feedback. States idle/turning/fleeing/edge_braking/edge_hover are explicit;
weak threat cannot cancel a committed escape. Hover stays latched until reset
or encounter change, without idle landing/home bias or neuron/readout resets.
Action schema `escape-edge-actions-v1` binds sources, candidate, calibrated start,
intervention and environmental actions. Source-bound delivery requires both
physical batches, audits, exact replay and engineering/visual evidence. Results
apply to the declared static-predator cases; historical directional failures stay
failed. This is engineered motor control, not learning or a reconstructed brain.

### Demo presentation (2026-10-12)

`escape_ui` is an additive Panda3D presentation subclass of the original
`escape_edge.viewer.EdgeViewer`. Its CLI calls the original verified-candidate
loader. No source-bound controller/viewer, archived experiment or receipt is
edited or reissued. Both launchers continue to require original evidence.
Primary buttons reuse existing spawn/pause/reset/replay methods and worker
barriers. The new UI provides pixel-sized text/targets, collapsible diagnostics
and the retained neuron inspector/display preferences. Diagnostics are formatted
from immutable published snapshots; UI layout and camera operations do not
advance time, simulate spikes or consume simulation RNG. Rendering uses normal
frame tasks while paused for zero-step captures. These engineering/visual checks
are separate from scientific evaluation and native mouse/keyboard acceptance.

### Modern dashboard and branch visibility (2026-10-12)

The approved redesign replaces lower research panels with a persistent sidebar.
Pixel layout is shared by widget placement, viewport rectangles and input routing.
Existing ten model-role memberships remain authoritative; friendly descriptions
are explanatory presentation, not a new biological taxonomy.

`GroupActivityTable` extends the original owned-branch RGBA table. Unchecked
owners receive zero alpha and their fragments are discarded, preventing invisible
depth occlusion. No vertex ownership, neural state/history or controller input is
changed. The same display mask excludes picking, selection/connection highlights
and camera bounds. Bounds include enabled coarse branches, visible full selected
morphology and optional source outline/cord; empty group selection leaves the
camera stable. This is display filtering, not a neural ablation.

The new inspector leads with role/type and recorded spike values; identifiers,
provenance and pathway measurements are opt-in. Filters and navigation preferences
survive reset/replay/resize. Native Segoe UI is loaded from the user's Windows
installation, never bundled; the standard font remains the portable fallback.
