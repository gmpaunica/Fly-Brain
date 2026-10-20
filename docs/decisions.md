# FlyBrain decisions

## Decision: M2 raw sensory boundary and sampling phase (2026-09-18)

### Context
The user accepted FB-01's manual checks and asked to move on, deferring visual
polish. M2 needs distinct spatial samples without providing an oracle controller.

### Decision
Use an immutable static food odor source and raw two-value sensory observation.
Offsets 10 forward/5 lateral and field strength 1000/softening 20 are synthetic
scaffolding. Sampling extends outside arena walls. Source defaults to (600,300)
in the app; pure simulations may omit it for zero input. Source parameters reject
nonfinite/invalid values and unrepresentable peak concentration. Hypot-based
evaluation avoids overflow from directly squaring large distances.

Publish current-pose readings after movement and on reset for the HUD. M4
inference samples pre-movement sensory values separately. No source identity
encoding, normalization, consumption, reward or automatic steering is added.

### Reason
Keep coordinates within world sampling while making movement-to-sensation and
left/right conventions independently testable. Avoid choosing neural encoding
scales before the neuron dynamics are specified.

### Consequences
There is one scalar food odor. Reproducibility includes sensory readings on the
recorded environment; no biological accuracy or navigation claim is justified.
Visual style remains functional and is deferred for later polish.

## Decision: FB-01 runtime and frame guard (2026-09-18)

### Context
Python 3.14.2 was installed, but binary-only installation found no compatible
Pygame wheel. Interactive long frame delays need a bounded catch-up policy.

### Decision
Use the approved user-scope Python 3.12.10 installation and a local virtual
environment with Pygame 2.6.1. Cap interactive frame elapsed time at 0.25 seconds;
headless mode advances exact physics ticks. Accumulator comparisons allow a
1e-12 tick roundoff guard, without rounding body state.

### Reason
Avoid compiling SDL and keep rendering independent of deterministic physics.

### Consequences
Interactive simulation can lag wall time after stalls. Replay guarantees require
identical per-tick commands. Visible controls acceptance is still pending;
numerical and dummy-driver tests cannot establish it.

## Decision: Sol tasks with a small startup context (2026-09-18)

### Context

The user wants to build with Sol Medium/High and reduce token use. The original agent brief is 24,835 bytes and includes long-term stages unrelated to the first build.

### Decision

Keep a compact `AGENTS.md`, status, append-only log, a task queue and a detailed active plan. Preserve the complete original brief byte-for-byte in `docs/reference/PROJECT_BRIEF.md`. Default to one Sol agent, Medium for ordinary tasks and High for mathematical/scientific decisions. Use task gates and focused local review.

### Reason

Precise interfaces and observable completion checks reduce repeated interpretation and broad context loading. A project-local launcher is convenient without modifying user-wide model settings.

### Consequences

The client still controls the running model. This is not a measured cost/performance claim. New tasks need updated status and evidence; future roadmap entries are not implementation instructions for the current session.

## Decision: Pure fixed-step world with a Pygame adapter (2026-09-18)

### Context

Rendering and future neural computation operate at different rates; both experiments and interactive use need the same physics.

### Decision

Keep geometry/physics independent of Pygame. Use a 0.02-second world tick, an accumulator in the interactive adapter, radians/seconds, screen-coordinate orientation, and radius-aware clamping. M3 will use 0.001-second neural substeps, 20 per world tick. Add dependencies only when the active task needs them.

### Reason

Headless tests can verify actual motion. Integer neural/world timing avoids a fractional substep scheduler. Explicit turn conventions prevent mirrored antenna/motor behavior.

### Consequences

This is simplified scaffolding. The interactive view may lag wall time after long stalls, while headless experiments retain every tick. A changed integration rule or step size requires tests and a decision update.

## Decision: Establish evaluability before committing to a learning method (2026-09-18)

### Context

A hard-spiking PyTorch simulation does not automatically provide gradients or useful navigation. A synthetic hand-wired controller cannot establish a biological-topology advantage.

### Decision

Build fixed baselines and held-out evaluation first. Investigate a small trainable readout as the first learning route; require a gradient/update check before a trainer. Internal-gain learning needs its own surrogate/reward-estimator decision. No final training algorithm is selected yet.

### Reason

This isolates representation, optimization and evaluation failures while keeping PyTorch central. It leaves room to learn from a negative result without disguising it as success.

### Consequences

Readout-only training must be labeled as such. M6 requires observed behavioral improvement; imitation loss alone does not meet that gate. Biological claims wait for sourced data and matched controls.

## Decision: M3 normalized LIF and explicitly synthetic pathways (2026-09-18)

### Context
M3 needs a precise equation and small directed test before a larger network.
The odor field has arbitrary units, and no sourced neural parameters exist.

### Decision
Use exact passive integration with constant external current and an end-of-step
voltage jump from the previous spike vector: `rest + (v-rest)*exp(-dt/tau) +
I*(1-exp(-dt/tau)) + W @ previous_spikes`. Time is seconds; other quantities
are normalized arbitrary model units. Defaults: dt 0.001, tau 0.020, rest/reset
0, threshold 1. Equality fires, overshoot is discarded, no refractory counter,
and at most one spike per neuron per substep. Signed jumps/currents are allowed.
The graph fixes `W[post,pre]`; recurrence is synchronous and delayed one step.

Build 100 neurons in seven labeled groups: 10 left and 10 right sensory,
25 left and 25 right interneurons, 10 each left/forward/right motor.
Only same-side sensory→interneuron→motor edges exist; local seeded positive
weights are in [0.096,0.144). Independently encode each antenna using
`2*min(concentration/0.025,1)`, and add tonic current 1.3 to forward motors.
All these choices are hand-written scaffolding. No inhibition, adaptation,
synaptic current kernel, biological data, learning or navigation claim.

### Reason
The passive solution avoids an implicit integration assumption. Discrete jumps
make a hand-checkable A→B fixture and preserve the documented spike-matrix
orientation. Explicit normalization makes sensory firing reproducible without
changing world physics. CPU float32/float64 and local RNG simplify verification.

### Consequences
Large jumps can cause consecutive spikes; rates are capped by the timestep.
Hard thresholds and inference-only state updates supply no training gradient.
Tonic forward activity means zero odor is not an entirely silent network.
The directional pathways predispose future steering; they do not demonstrate
learning or topology benefit. Neural inference and motor decoding wait for M4.
PyTorch is an optional `neural` extra; world/manual app imports remain independent.
Full equations and fixtures: [M3 plan](plans/M3.md).

## Decision: M4 count-based causal motor decoding (2026-09-20)

### Context

The fixed graph emits sparse spikes across twenty one-millisecond substeps.
Using only the last spike vector loses earlier activity, while theoretical
one-kilohertz normalization would make the current synthetic motor output tiny.
Embodied integration must also keep coordinates out of the controller boundary.

### Decision

Pass only `SensoryObservation(odor_left, odor_right)` into
`NeuralController.decide`. The separate closed-loop orchestrator samples the
current pose, advances exactly 20 neural steps, decodes, moves once for 0.02 s,
and records pre-movement sensation plus post-movement body state.

Decode the full int64 count tensor. For each ten-neuron motor group use
`min(group_sum / 10, 1)`, so one spike per neuron in 20 ms (50 Hz mean) is the
arbitrary full-scale reference. Forward uses `motor_forward`; turn is
`motor_right - motor_left`, matching positive-clockwise screen coordinates.
Apply an EMA with alpha 0.5, clamp through `MotorCommand`, and clear the EMA on
reset. Provide a boolean output gate after decoding so tests can silence physical
movement while real neural and decoder state continue to advance.

Keep the default app manual. `--control neural` explicitly selects the closed
loop and lazily imports PyTorch-dependent modules. Movement keys are disabled in
that mode and the window/HUD use a distinct `NEURAL CLOSED LOOP` label.

### Reason

Complete counts preserve all events the 60 Hz world/view can otherwise miss.
The 50 Hz reference makes the current graph's verified population bursts produce
bounded observable commands without changing neural dynamics. Separating the
orchestrator from the controller makes the sensory-only information boundary
testable, and the output gate establishes physical causality without a fake
network.

### Consequences

The normalization, sign convention and smoothing are hand-written engineering
choices, not learned parameters or biological estimates. PyTorch performs the
count tensor indexing and sums; ordinary Python applies normalization/EMA and
constructs `MotorCommand`. No autograd graph, gradient, optimizer, reward or
navigation-success claim exists. The gate is a diagnostic seam and retains its
enabled/disabled setting across episode reset. Full contract and verification:
[M4 plan](plans/M4.md).

## Decision: M5 functional diagram driven by simulation time (2026-09-20)

### Context

The project has a real synthetic spike graph but has not imported anatomical
coordinates or a sourced connectome. A display frame can cover zero, one, or
several 20 ms world ticks, while each tick summarizes twenty 1 ms neural steps.

### Decision

Display the current 100-neuron graph as a deterministic functional diagram,
grouped into sensory, interneuron, and motor columns. Label it `LIVE SYNTHETIC
BRAIN` and state on-panel that the layout is diagrammatic. Do not imply that its
shape represents fly neuroanatomy.

Pass every completed closed-loop snapshot to a separate visual activity model.
Set a neuron's trace from its accumulated interval spike count and linearly fade
the trace over 0.2 seconds of neural simulation time. Drawing does not decay or
clear traces. Show a group-balanced deterministic subset of at most 80 nonzero
edges, recent motor-group traces, and the already decoded motor command.

Use a 1200 x 600 neural window with the original world on the left and a 400 x
600 panel on the right. Keep manual mode at 800 x 600 and retain lazy neural
imports. Reset the visual model alongside the closed-loop episode.

### Reason

This makes the verified simulated information flow visible without inventing
an anatomical brain. Consuming every snapshot prevents render rate from hiding
spikes; simulation-time fading makes replay independent of display frame count.
Fixed layout and edge selection avoid adding another random stream.

### Consequences

The view is useful for debugging and demonstration, but it cannot support
anatomical claims. A later sourced connectome milestone must provide provenance
and coordinates or a documented layout mapping. Motor bars show recent neural
activity; the numeric output remains the hand-written M4 decoder result. Full
contract and verification: [M5 plan](plans/M5.md).

## Decision: FB-06 locked embodied evaluation (2026-09-21)

### Context

The live fixed controller has been observed circling, but no numerical food
arrival baseline existed. Training requires a frozen comparison protocol.

### Decision

Use 600 world ticks (12 seconds), a 20-unit fly-center success radius, and
locally seeded initial poses 60–180 units from the fixed source. Lock disjoint
train 0–7, validation 100–103, and final 200–207 seeds before optimization.
Run each case with fresh world, network and decoder state. Success at the
starting pose or final allowed tick counts; wall crossing is counted but does
not terminate. Use held-out success rate as primary objective, with all-case
normalized progress as secondary and successful-only time/path efficiency
reported with their denominator. FB-06 runs the unchanged fixed controller.

### Reason

Explicit starts, timing and termination make a later comparison reproducible
and prevent hidden selection of favorable cases. The evaluator can score using
coordinates without giving them to `NeuralController.decide`.

### Consequences

The first validation/final result is a small diagnostic baseline, not proof of
population performance or learning. Final cases must be reserved from FB-07
tuning. See [FB-06 plan](plans/M6-BASELINE.md) and [recorded results](experiments.md).

## Decision: FB-07 trainable readout and negative navigation result (2026-09-21)

### Context

The fixed spiking controller has a locked evaluation baseline. Hard spike
thresholds have no chosen surrogate gradient, so direct graph-weight training
would introduce another scientific assumption before the first learning test.

### Decision

Freeze the synthetic 100-neuron graph and sensory encoder. Train a 7-to-2
linear readout with bias from EMA-filtered mean group spike counts. Its 16
PyTorch parameters receive gradients through sigmoid/tanh motor outputs. Use a
sensory-only hand-written forward/turn teacher and mean squared error on 1,600
ticks from the eight locked train trajectories. Use one predeclared 80-step
full-batch Adam run at learning rate 0.05. Save the checkpoint, reload it before
evaluation, and compare the locked validation/final cases with FB-06.

### Reason

This gives a small, testable autograd path while preserving the graph and the
controller's sensory-only input boundary. A teacher based on antenna readings
can be implemented without feeding world pose or target geometry to inference.

### Consequences

All 16 readout scalars received nonzero gradients and changed, but held-out
arrival did not improve (validation 0/4 both; final 0/8 learned versus 1/8
fixed). This is a measured supervised imitation attempt, not reward learning
or a completed M6 navigation milestone. All eight train starts map to the same
first-tick spike feature vector despite eight distinct teacher turns. Diagnose
the sensory saturation and observability boundary before enlarging the model.
See [FB-07 plan](plans/M6-LEARNING.md) and [recorded results](experiments.md).

## Decision: FB-07D encoder-scale diagnosis (2026-09-21)

### Context

FB-07's first-tick spike features hid distinct train sensory observations.
The encoder saturated at 0.025, but a scale change could also suppress spikes
when odor is weak. A wider readout alone cannot recover information absent
from its seven feature inputs.

### Decision

Replay locked train and validation fixed-controller observations through the
original 0.025 encoder and one predeclared 0.05 encoder. Count exact feature
collisions and conflicting teacher turns. Train one otherwise matched readout
at scale 0.05 and compare validation cases only. Keep final starts reserved.
Treat the scale as synthetic hand-written scaffolding; default app and FB-07
behavior retain 0.025.

### Reason

The paired replay isolates encoding from movement, while the embodied
validation run measures what the altered encoder actually does in control.
The scale was chosen from train/validation concentration ranges before the
embodied outcome. Topology, network size, teacher, training budget and locked
cases remain fixed.

### Consequences

Train aliasing fell from 1,030/1,600 to 452/1,600 ticks, but validation
aliasing rose from 1,446/2,400 to 1,589/2,400. Wider validation arrival tied
the original learned and fixed controllers at 0/4. Mean progress improved but
remained negative. Do not run another readout optimization under the present
encoding. First find a sensory representation that preserves weak-signal
differences and spike activity on both train and validation. See
[FB-07D plan](plans/FB-07D.md) and [recorded result](experiments.md).

## Decision: FB-07E weak-signal gate (2026-09-21)

### Context

Widening the linear encoder reduced train saturation but silenced every weak
antenna reading in the FB-07D paired probe. Validation aliasing worsened.
The next question was whether one zero-preserving nonlinear current could
retain weak spikes and improve seven-feature discriminability on both splits.

### Decision

Probe one fixed rational rule, `I(c)=2.5*c/(c+0.005)`, on the same locked
train/validation fixed-controller observations. Compare exact alias ticks,
turn-only MSE floor and weak-reading spike activity with both prior encoders.
Require all three predeclared criteria on both splits before another bounded
readout run. Keep the default app encoder linear at 0.025, the synthetic
topology unchanged and final cases sealed.

### Reason

The rational rule is continuous and zero-preserving, while its low-end gain
can drive the assumed LIF neurons with weak positive odor. Same-observation
replay isolates this information effect from movement and readout fitting.

### Consequences

Weak spike activity rose sharply and the teacher-turn MSE floor fell on both
splits. Validation aliasing fell to 992/2,400, but train aliasing was 627/1,600
versus 452 at the better prior scale. The gate failed; no new training or
embodied result was claimed. The next diagnosis should locate remaining
collisions across current, spikes, group reduction and EMA. See
[FB-07E plan](plans/FB-07E.md) and [recorded result](experiments.md).

## Decision: FB-07F collision-stage diagnosis (2026-09-22)

### Context

The rational FB-07E encoder retained weak spikes and reduced validation
aliasing, but train aliasing exceeded the wider linear encoder. More current
detail alone did not establish a better seven-feature readout input.

### Decision

Trace exact current, sensory counts, all-neuron counts, seven group means and
EMA features on the same locked train/validation observations. For final
feature collisions with different teacher turns, label the latest earlier
representation that differed. Use this as a diagnostic snapshot comparison;
do not interpret it as causal proof or change any network parameter.

### Reason

The all-100 stage separates spike-count loss from group averaging. Pairwise
labels locate remaining final collisions more specifically than independent
stage alias totals, which can change nonmonotonically because of neural and
EMA state.

### Consequences

The rational current pairs remained distinct for all distinct observed raw
pairs, while every final conflicting candidate pair had identical all-100
spike counts and group means. The measured target is the 20 ms spike-count
representation, not group averaging or EMA merging. Next test two sensory
post-interval voltages alongside seven EMA spike features as a controlled
information probe before training. Topology, app behavior and final cases
remain unchanged. See [FB-07F plan](plans/FB-07F.md) and
[recorded result](experiments.md).

## Decision: FB-07G post-interval sensory voltage probe (2026-09-22)

### Context

FB-07F found that all final conflicting rational-encoder EMA pairs retained
distinct current but identical 20 ms spike counts. The network snapshot also
holds post-interval membrane voltage, which may preserve subthreshold state.

### Decision

Append the mean voltage of each ten-neuron sensory group to seven spike EMA
features in a read-only candidate replay. Require lower exact alias counts,
lower bucket-optimal sensory-teacher turn MSE floor, and at least one resolved
conflicting pair on both locked train and validation splits. Do not change the
default controller, graph, optimizer, or final-case evaluation in this probe.

### Reason and consequences

The two voltage means are a compact, explicit representation of the assumed
LIF state. On recorded ticks, they reduced train aliases from 627 to 18 and
validation aliases from 992 to 46. The turn floor fell on both splits, so the
information gate passed. Exact float32 distinctions can be fragile, and these
metrics do not show that a trainable readout can use them or that a fly reaches
food. FB-08's M6 improvement prerequisite is still open. See
[FB-07G plan](plans/FB-07G.md) and [recorded result](experiments.md).

## Decision: FB-07H matched voltage readout test (2026-09-22)

### Context

Two sensory voltage means sharply reduced exact train/validation feature
collisions in FB-07G, but no trainable readout or embodied behavior had been
tested with them. FB-08 requires an M6 navigation improvement.

### Decision

Train seven- and nine-feature linear readouts on identical rational-encoder
train observations using the same sensory-only teacher, zero initialization,
80 Adam steps and LR 0.05. Reload both and compare locked validation cases
with the recorded fixed baseline. Open final cases only if nine has more
validation arrivals than both controls and higher mean progress than fixed.

### Reason and consequences

The paired seven-feature model separates the effect of appending voltage from
the encoder and data change, though the models differ in parameter count
(16 versus 20). Both models learned the imitation target numerically, but all
controllers had 0/4 validation arrivals. Nine's mean progress was -1.770
versus fixed -1.688, so the gate failed and final cases remained sealed.
Exact voltage distinctions and lower training loss did not translate into
measured navigation benefit under this teacher and linear readout. The next
step is a focused failure diagnosis before changing the objective or model.
See [FB-07H plan](plans/FB-07H.md) and [recorded result](experiments.md).

## Decision: FB-07I scoring-only embodied trace (2026-09-22)

### Context

The FB-07H nine-feature readout had lower train imitation loss but no
validation arrival. Validation progress was worse than the unchanged fixed
controller, leaving the source of the behavioral failure unclear.

### Decision

Replay saved fixed, seven- and nine-feature controllers on locked train and
validation starts. Record pre-movement sensory teacher turn, actual command,
source bearing, distance change and wall contact per tick. Use source position
only in this scoring diagnostic. Compare turn signs only beyond declared
0.1-radian bearing and 0.05-command thresholds. Do not retrain or open final.

### Reason and consequences

The trace separates teacher sign from learned-command sign and realized
movement without treating any as an optimal policy. On validation, all 12
controller runs retreated before first wall contact. For the nine-feature
model, teacher sign matched bearing on 2,091/2,091 eligible ticks, while
command sign matched on 1,564/1,915; 1,112/2,400 ticks contacted a wall.
The observations point to early retreat and actuator behavior, but do not
establish whether turn magnitude, forward drive, policy capacity or sensory
history is the causal bottleneck. Probe that distinction before another
training change. See [FB-07I plan](plans/FB-07I.md) and
[recorded result](experiments.md).

## Decision: FB-07J one matched motor gain (2026-09-22)

### Context

FB-07I showed early retreat before wall contact on all validation controllers.
The learned turn sign sometimes differed from the sensory teacher, and sign
agreement alone could not establish sufficient turning relative to forward
motion.

### Decision

Multiply decoded turn by one predeclared gain of 2.0, clamp to motor bounds,
and preserve forward output. Apply it to both saved nine-feature and fixed
controllers, keeping checkpoints, graph, sensory input and world fixed. Open
final starts once only if nine gain-2 beats nine gain-1 and fixed gain-2 in
validation arrivals and mean progress.

### Reason and consequences

The matched fixed gain-2 control distinguishes an actuator gain effect from
the learned readout's effect. Validation met the gate (1/4 versus 0/4 and
better progress), so one final run was made. Final arrival tied 1/8 across
nine gain-2, fixed gain-2 and original fixed, although nine's mean progress
was less negative. The primary M6 arrival improvement remains unverified.
No further gain selection should use these final starts. A new learning
objective and fresh held-out protocol require an explicit next decision;
FB-08 stays gated. See [FB-07J plan](plans/FB-07J.md) and
[recorded result](experiments.md).

## Decision: close FB-07 expansion and recover the existing fit (2026-09-22)

### Context

The user challenged the expanding letter sequence and requested a review and
correction. Completed experiments had not improved the primary arrival
objective. The proposed reward-learning pivot preceded an embodied teacher
control and a constant-command fitting baseline.

### Decision

Supersede FB-07K, retain all historical results and keep M6 open. The only next
task is [M6 readout recovery](plans/M6-RECOVERY.md#next-task-m6-readout-recovery):
one evidence-based fitting correction, at most 500 readout optimizer steps,
with declared turn-fit/training-arrival gates and a stop for user input if
they fail. Do not automatically create further letters, change objectives or
expand the network. A successful fit yields a saved candidate and a concrete
fresh held-out protocol; it does not automatically authorize final evaluation.

### Reason and consequences

The unchanged sensory teacher reaches 8/8 training starts. The existing nine
features/readout reduce turn error only 5.72% versus a constant mean-turn
predictor; about 79% of the combined loss reduction is forward-speed fitting.
There is an imitation gap to resolve before claiming the teacher or reward
objective must change. This review does not isolate optimizer from feature
capacity or closed-loop data shift, and it does not establish generalization.

Require a same-candidate frozen-untrained control as well as the original
fixed baseline to separate parameter learning from encoder/decoder changes.
Retire seeds 200-207 from new final claims: both FB-07 and FB-07J used them.
Earlier "sealed" descriptions meant no extra run in that particular task.
No physics, neural dynamics, topology or deployed controller changes here.

## Decision: close the bounded recovery fit after its failed gate (2026-09-22)

### Context

The saved nine-feature readout retained nonzero turn gradients, and the raw
training matrix was full rank but had a condition ratio near 73. A linear
train-only diagnostic reached turn MSE about 0.15539, well above the declared
0.016752 threshold. That diagnostic did not prove a lower bound for the tanh
readout.

### Decision

Use exactly one conditioning correction: standardize each feature from the
1,600 training rows while fitting, then fold the transform into the original
nine-input readout. Keep Adam learning rate 0.05 and stop at 500 full-batch
updates. The correction changes no online feature, topology, sensory boundary,
motor gain or live app path. The run reached turn MSE 0.155287 and 1/8 train
arrivals; both recovery gates failed. Close this fit attempt and await a user
decision before any different representation, capacity or learning approach.

### Reason and consequences

Scaling lowered the training matrix's condition ratio to about 11.5, but
turn fitting improved only modestly over FB-07H. Finite gradients, updates,
topology and reload checks passed, so numerical failure to update is not the
observed explanation. This result does not isolate information loss from
readout capacity or closed-loop data shift. No validation/final starts were
evaluated, no checkpoint entered the app, and M6 navigation learning remains
unverified. See the [experiment record](experiments.md#m6-readout-recovery-fit--2026-09-22).

## Decision: neural charge features under renewed repair authorization (2026-09-23)

### Context

The user explicitly requested another repair with Astra and fixes for other
broken project behavior. The earlier 500-update fitting correction remains
a closed negative result. Astra reproduced a representation problem: post-reset
voltage can fall sharply when odor increases, and retains spike phase and
old state when the teacher depends on current bilateral concentrations.

### Decision

Measure charge from actual membrane change plus leak/reset losses, with
float64 accumulation. Keep the existing seven spike EMAs and replace the
two endpoint voltages with separate log-charge features. Compensate the
rational encoder independently on each side, before the learned readout.
Keep old behavior available by default and require schema-3 checkpoints
for this different feature meaning. No raw odor or teacher/contrast value
enters the decoder. Run one newly declared candidate with at most 500 Adam
updates, then apply the prior training gates and matched training controls.

### Reason and consequences

The one candidate passed on 2026-09-22: turn MSE 0.001762; trained 8/8 training
arrivals versus untrained 0/8 and fixed 0/8. This fixes the training calibration
gap. It does not show generalization, learning inside spiking neurons, or
benefit from downstream connectivity: the engineered charge identity
reconstructs sensory drive. Fresh evaluation is prepared for review with
frozen weights, new cases, matched controls and a failure stop. M6 remains
open pending that evaluation. No checkpoint was promoted to the app.

Also protect existing experiment directories before computation and use
exclusive checkpoint file creation. Regression tests reproduced previous
overwrites in temporary directories; historical recorded data remain intact.


## Decision: accept the frozen M6 navigation gate within its declared scope (2026-09-23)

### Context

The charge candidate passed its bounded training calibration. The user then
approved the concrete frozen fresh-case protocol, including matched controls,
validation-before-final gating, a zero-update budget and failure stopping rules.

### Decision

Execute that protocol once. Bind the execution receipt to its content hash in
the fixed checkpoint directory so a copied filename cannot permit reuse.
Keep source/checkpoint/software integrity checks and disable readout gradients.
Accept M6 navigation improvement only within the recorded evaluation scope.

### Reason and consequences

Validation passed 16/16 versus 1/16 for each control. Final passed 32/32 versus
1/32 for each control; both one-sided paired p values were 4.656612873e-10.
All 144 episodes completed with zero updates and unchanged frozen artifacts.
This supports an effect of learned motor weights in the engineered charge
representation. It establishes no downstream topology or biological benefit.
The final cases are now historical. The live app remains on the fixed decoder;
optional saved-checkpoint integration is the next task, with separate visible
and reset/timing checks. No reward-learning pivot or new FB-07 letter follows.

## Decision: proceed directly to FB-08 with frozen intervention controls (2026-09-23)

### Context

The user requested the work necessary to reach the next FB section. FB-08's
M6 improvement prerequisite has passed. Optional app checkpoint integration
does not block the renderer-free experiments required by M7.

### Decision

Run the bounded [FB-08 protocol](plans/FB-08.md): ten conditions, sixteen new
matched diagnostic starts, at most 160 episodes and zero optimizer updates.
Compare sensory-current ablation, all-synapse removal, two sensory-pathway
gains and three constrained random wirings with the intact saved readout.
Require intact >=13/16 arrivals before proceeding to interventions.

### Reason and consequences

The charge features reconstruct sensory drive, so downstream wiring may be
dispensable for this trained controller. Disabling sensory current and deleting
downstream synapses test different dependencies. A `[100,100]` weight matrix
uses `W[destination, source]`: multiplying rows 20-69, columns 0-19 by 0.5
halves sensory-to-interneuron voltage jumps without changing sensory charge.
Random controls permute weights within each allowed source layer, preserving
each destination's incoming weight multiset. They do not preserve source
out-degree or left/right blocks; those limits must accompany any conclusion.

Use the same frozen readout for every condition. Loading checks its original
contract before explicitly constructing each intervention; do not relabel
perturbed graphs as original compatible checkpoints. Record per-condition
hashes and preserve all previous artifacts. New seeds 51000-51015 are an
exploratory assay, not a new M6 final test. The next biological-subset stage
requires a separate source/pathway feasibility decision.

FB-08 outcome: intact 15/16, each sensory ablation 0/16, and all seven intact/
synaptic-control conditions 15/16. The declared hypothesis was supported;
no downstream topology benefit was established. The 160-episode run preserved
all 47 prior artifacts. M7 acceptance is complete with results and inspected
plots. FB-09 source selection must account for the fact that simply replacing
downstream wiring may leave this readout's navigation substantially unchanged.

## Decision: select a pinned BANC v888 olfactory-to-motor subset (2026-09-24)

### Context

FB-09 must establish source access, licensing, stable IDs, edge semantics and
sensory/motor coverage before importing biological data. FB-08 also showed that
the current charge readout can navigate while ignoring downstream wiring.
Since the queue was written, the final BANC brain-and-nerve-cord connectome was
published with a static archive, making a single-specimen motor path possible.

### Decision

Use Harvard Dataverse BANC v888 release 3.0 under CC BY 4.0, pinned by the file
IDs, sizes and MD5 values in [FB-09](plans/FB-09.md). Preserve `root_888` as a
string and convert directed `pre -> post` edges to `W[post,pre]`. Start from a
deterministic 83-neuron subset: 68 qualifying bilateral `ORN_VA2` cells,
bilateral `VA2_adPN`, five structural corridor neurons, bilateral DNa02 and six
named coxa-anterior leg motor neurons. Retain all internal edges with at least
five synapses.

Treat the first transform as unsigned structural connectivity:
`W[post,pre] = connectome_gain * norm`. Preserve raw counts, normalized input
fractions, transmitter annotations and quality flags, but do not infer
receptor-specific signs. Drive only left/right ORN pools from the corresponding
simulated antenna. Exclude the current charge features from a future
topology-dependent decoder; use DNa02 asymmetry for steering and label any
motor-neuron forward reduction as synthetic scaffolding.

### Reason and consequences

BANC uniquely provides ORNs, central brain, descending neurons and named motor
neurons in one female specimen. The audited source contains the complete
declared corridor, so no cross-animal ID join is required. It does not provide
membrane parameters, receptor signs, a calibrated VA2 odor input or a 2D gait
decoder. The specimen was unusually right-turn biased and the selected
corridors are asymmetric. Therefore the next task is only a reproducible
loader/transform with propagation and stability tests. Any navigation or
topology claim requires later matched ablation/randomization controls and an
output path that cannot bypass the imported graph.

## Decision: preserve the audited corridor and close the first BANC gain probe (2026-09-24)

The extractor replay revealed that the prior feasibility note incorrectly
called the left pinned corridor a strongest-first shortest path. The archived
edges also contain a stronger four-hop left route through `v2LN30`,
`M_l2PNl20` and `CB0356`. The 83-cell artifact keeps the originally audited,
explicitly listed corridor IDs; its selection rule is now a pinned curated
route with verified directed edges, not a shortest-path claim.

The small packaged artifact is connected to the existing PyTorch LIF core
through a separate `BancNetwork`, preserving the synthetic controller. Source
`norm` remains the fraction of full postsynaptic input, not renormalized over
the selected subset. The unsigned transform retains all 31 self-edges and 62
cross-side edges as provenance. In the fixed gain-1/600 diagnostic, gain 1 did
not propagate beyond VA2 ORNs, while gain 600 propagated but crossed sides and
did not wash out. Neither candidate is adopted for behavior. The next task
must predeclare and test one dynamics mapping before a motor decoder or
learning experiment. See [FB-09](plans/FB-09.md) and the
[recorded probe](../experiments/fb09-gain-probe/results.json).

## Decision: normalized BANC recovery remains gated (2026-09-25)

The user approved masking self-edges in simulation only and normalizing each
retained nonself input row by its selected synapse-count sum. Original counts,
IDs and source normalization remain untouched. All 62 cross-side edges remain;
contralateral activity is expected and is not itself a failure. The simulated
graph has 145 edges and no directed cycles. Current gain, ORN sensitivity,
unsigned weights, LIF rules and motor reduction are engineering assumptions.

One fixed experiment tested row budgets 1.25, 2 and 3, with per-side ORN scales
log-spaced from 0.005 to 0.05. Exactly 72,000 steps ran with no optimizer.
Budgets 2 and 3 resolved persistent firing and passed individual pulse gates,
but neither passed all contrast gates. The high-band failure is retained, not
removed by narrowing the accepted concentration range after the fact.

No candidate is accepted, no causal controls or teacher-readiness episodes ran,
and no BANC app configuration is bundled. The eight-endpoint spike-rate decoder
and app/view are gated preparation. The future loader requires matching
successful result/protocol sidecars and current-source hashes; receipt labels
alone cannot enable behavior. That hardening followed the closed experiment
and did not change its evidence or neuron calculations.

FB-10 remains blocked. A revised encoder is a proposed next hypothesis, not a
proven fix or an automatically authorized new experiment. Its exact budget and
unchanged scientific gates must be approved before another run.

## Decision: diagnose PN loss and test one temporal assumption (2026-09-25/26)

The user renewed authorization to continue to FB-10 readiness. A 4,500-step
unchanged-parameter diagnostic exactly replayed all counts and showed identical
PN spike times for equal versus right-strong high-band input. Different input
was absorbed by leak/reset losses. A single wider sensory range (.005-.375)
then passed all propagation/stability gates but failed middle/high contrast;
that 24,000-step experiment is closed without trying more scale values.

One mechanism test applied area-preserving 5 ms exponential filtering to
incoming jumps at the two PNs only. This targeted the measured loss boundary;
other neurons retained instantaneous jumps and hard reset so downstream gain-3
paths were not converted into multi-spike amplifiers. Wiring, counts, neuron
equations, tau20ms, decoder and concentration/acceptance gates stayed unchanged.
The BANC-specific subclass resets its two filter states, and causal interventions
construct the same configured population type rather than silently dropping it.

This is an engineered temporal assumption, not source biology. The test closed
after24,000 steps, with low-left propagation and mid/high contrast failures.
There is no accepted filter configuration. No tau/gain/range search follows.
These two failed repairs require an architectural debugging reset before any
new candidate. Neither software tests nor reduced overshoot establish readiness.

## Decision: phase-rate input and user-approved PN-only reset (2026-09-26)

One phase-distributed square-root rate encoder used the frozen budget-3 graph
and original neuron dynamics. It failed only the high-left contrast after
24,000 steps. The real within-ORN edge was preserved; causal criteria were
corrected before measurement to separate external waveform from actual ORN
counts, with a fourth downstream-only cut and per-cell comparison.

The user explicitly approved changing only the two PNs' assumed dynamics.
The separately predeclared candidate used tau25ms, threshold1.1, subtractive
reset, unchanged encoder/graph/decoder, and all original gates. Its 24,000-step
assay corrected high-left contrast but narrowly failed mid-right: four
bias-corrected spikes against required 4.02. All pulse gates passed. It remains
a failed candidate: no rounding allowance, extended window, post-hoc gate
change, automatic parameter search or verified default. No optimizer ran.

Astra reviewed implementation and the saved failure boundary. Counts suggest
temporal relay sensitivity but do not establish particular lost spike events.
The next decision is a bounded observation-only investigation, not an assumed
parameter fix. BANC import is complete; FB-10 readiness is not.

## Decision: exact relay diagnosis before further changes (2026-09-26)

The user approved exactly3,000 observation-only steps on the two recorded
mid-band cases. All83 pulse/steady counts replayed exactly; every PN spike
matched DNa02 after the actual corridor delay (left4ms, right3ms), and every
intermediate edge transmitted one-for-one with1ms delay. There is no measured
relay spike loss in these cases. Counting-window crossings explain the entire
six-to-four bias-corrected contrast change. Preserve the original failed gate.

No further neuron or encoder tuning is justified by this finding. Astra
recommended a latency-aligned sustained-rate measurement only if that is the
intended scientific requirement. The [proposal](plans/FB-09-LATENCY-PROPOSAL.md)
retains750ms duration and1Hz/2% thresholds but changes timing; it explicitly
requires user approval. It is historically informed, not independent evidence
or proof of simultaneous-window robustness. No new protocol was executed.

## Decision: accept delay-aligned sustained-rate contract; finish FB-09 (2026-09-26)

After receiving the explicit measurement choice, the user directed continuation
through FB-09 and confirmed continuation after the approval interpretation was
stated. The new protocol mechanically delays the common PN counting interval
by the source corridor length: left4ms/right3ms. It keeps750ms duration,1Hz/2%
thresholds, every other gate, and the exact PN-charge model. Original raw
simultaneous metrics and all prior failed evidence remain recorded unchanged.
This is a historically informed measurement-definition change, not independent
confirmation or a claim that the old simultaneous gate now passes.

The one capped assay passes dynamics and all four causal controls at30,000steps,
0updates. A verified receipt with byte-identical evidence sidecars is packaged.
The readiness assay passes teacher8/8, finite varying endpoint features and
provenance/budget checks at5,808worldticks,0updates; fixed untrained BANC0/8 is
descriptive baseline evidence. Do not confuse teacher scaffolding with learned
neural navigation. The live decoder remains eight endpoint-rate features only.

The current FB-09 contract is complete with358passing tests, three app smokes,
static visual inspection and independent Astra review. Static inspection is
not live manual UI acceptance. Feature saturation and wider observed odor range
remain explicit FB-10 risks; no training or topology-benefit claim follows.

## Decision: prepare one frozen BANC endpoint-readout attempt (2026-09-26)

The user requested starting FB-10. The active handoff's first task is protocol
preparation. Implement and verify that task now; keep the actual fit and a
subsequent fresh evaluation separate. Artificial-data optimizer tests establish
software behavior, not a BANC experiment or learning result.

Choose one small nonlinear 8->16->2 readout (178 weights/biases), because the
accepted endpoint rates vary and their exact-alias error floor0.00238492 is
below the constant-turn baseline0.08901093. This motivates one bounded test,
not an assurance that the representation suffices. Do not add sensory features
or tune the accepted source circuit. Reuse its identical endpoint EMA by
composition, preserving the receipt-bound core files.

Freeze train-only mean/std with a 1 Hz floor. Seed1010, Adam lr0.01 and500 updates
are one predeclared candidate, without a sweep. Require turn MSE<=25% of the
constant baseline, >=7/8 training arrivals, strict arrival and all-case progress
improvement over matched untrained and fixed BANC, gradients/parameter changes,
exact checkpoint replay, frozen core/normalization and provenance. No gate is
claimed achieved until its actual run; held-out improvement remains necessary
for learned-navigation claims.

Bind the protocol to one output directory and consume it before updates so
copying it cannot silently reopen an attempt. Partial execution remains a
record, not permission to retry. Maximum behavior budget is25 episodes,
15,000 worldticks and300,000 neural steps. New fresh splits are not prepared or
inspected here. Source biology, assumed dynamics, engineered normalization and
future learned readout parameters remain explicitly distinct.

## Decision: close the failed FB-10 capped fit (2026-09-27)

The user explicitly authorized continuing with the single prepared 500-update
fit and matched controls. Execute that existing protocol without resealing,
changing its candidate or interpreting preparation as a learning result.

The finite fit completed 500 updates; all 178 readout scalars changed. Turn MSE
0.06678674714883089 missed the frozen ceiling 0.02225273146561865. Trained
arrivals 2/8 missed the required 7/8, despite beating matched untrained 1/8 and
fixed BANC 0/8 in arrivals and all-case progress. All integrity/replay/budget
gates passed at 25 episodes /14,131 world ticks /282,620 neural steps.

Close the attempt under its original stop rule. Preserve all outcomes, the
consumed receipt and both checkpoints; do not add updates, lower gates, retry,
change learning method or start held-out evaluation. FB-10 overall remains
unfinished and learned BANC navigation is unestablished. The next concrete
task is a train-only recovery review and predeclared bounded diagnostic
protocol, before an explicit new-candidate decision. This result does not
isolate the cause or justify a scientific architecture change.

## Decision: measured FB-10 recovery under renewed authorization (2026-09-27)

The user explicitly asked to fix whatever is needed and finish FB-10, twice.
This renews the scope beyond preparation/one task, including necessary bounded
repairs and a separate fresh evaluation only after all training gates pass.
It does not reopen consumed experiments or permit favorable result selection.

The first recorded repair separated common activity and bilateral contrast
without changing source spikes or readout capacity. Its .06408428 turn MSE and
2/8 arrivals failed. Preserve/close it; do not extend its fit. The measured
conditioning problem was insufficient to explain/repair the whole failure.

After two failed fits, a minimal 9,000-step equal/mirrored pulse reproduction
found 3.88796 Hz equal-input contrast fluctuation, exceeding the mean directional
change for a 10% stimulus difference. A 1 ms causal delay did little; reject it.
A 500 ms EMA reduced equal-input noise to .815565 Hz. Exact offline reconstruction
of original interval counts (max error 1.78e-15) raised train-only contrast/target
correlation from .34580 to .67369. No optimizer updates or world ticks were used
for either diagnosis; the spike probe alone advanced 9,000 neural steps.

Choose one fixed dual-timescale feature interface: actual endpoint spikes feed
100 ms/500 ms causal EMAs, conditioned separately. A 16->16->2 readout learns
306 scalars; retain seed 1010, Adam .01, 500 updates and the original fit,
arrival, matched-control and integrity gates. This is an engineered temporal
readout repair, not sourced physiology or trainable connectome dynamics.
Source graph/encoder/neuron rules stay byte-identical. Preserve every previous
implementation and record the new candidate in its own consumed protocol.
Close on failure; no implicit next candidate. On a training pass, freeze a
separate fresh-case matched evaluation before execution, without model changes.

## Decision: freeze the passing temporal BANC candidate for fresh evaluation (2026-09-27)

The measured temporal repair passed every original training gate at 500 updates:
turn MSE .00837231, trained 8/8 versus matched untrained/fixed 0/8, exact replay,
306 changed parameters, fixed graph/normalization/provenance and bounded actual
25 episodes /10,775 world ticks /215,500 neural steps. Full suite 426 passes.
This is training calibration only until independent starts are measured.

Under the user's renewed instruction to finish FB-10, freeze the candidate and
lock a separate fresh assay before stepping. Seeds 620000-620015 validation
and 621000-621031 final use the same 60-180 unit area-uniform annulus and random
headings; reject overlap with earlier recorded seed identities. Require >=13/16
validation arrivals and strict arrival/all-case-progress improvements over both
controls. Only then inspect final behavior: >=26/32, paired arrival gain >=8,
one-sided exact p<=.025 against each of the two controls, and strict all-case
progress improvement. Retain all failures and exact source/parameter integrity.

Budget 144 matched episodes /86,400 ticks /1,728,000 neural steps maximum,
zero updates. One destination-bound protocol and a central consumed-candidate
receipt prevent renamed retries. No tuning, checkpoint changes, outcome-based
case selection or topology-benefit claim. The chosen thresholds follow the
previous project's fresh navigation gate and are frozen before FB-10 outcomes.

## Decision: repair measured FB-10 training coverage without using failed validation as training (2026-09-27)

The first temporal candidate passed training but missed fresh validation:
11/16 versus 0/16 and 1/16, below required 13/16. Close its consumed assay;
final was not stepped. A bounded five-case diagnosis exactly reproduced all
failed rollouts and measured teacher 5/5 arrivals. Student tail turns about
-.28 are too weak versus teacher about -.98, producing stable orbits.

The original 1,008 labels have no turn below -.8, with minimum -.46347 and
maximum +.89150. Failed tails are 7.20 normalized feature units on average
from nearest original data. Choose a documented coverage repair: 32 separately
seeded training starts, 700000-700031, with both teacher trajectories and
frozen-student trajectories labelled by the sensory teacher. Keep every case
and the original samples; do not train on the diagnosed validation cases.

Freeze collection budget 64 episodes /38,400 ticks /768,000 steps /zero updates
and teacher 32/32 readiness. Then one fixed 306-parameter, seed-1010, Adam-.01,
500-update candidate with original matched behavior/replay budget. Preserve
the absolute original turn ceiling as well as <=25% of the aggregate constant
predictor and the original-data turn gate. Source graph, filters and neuron
rules remain fixed. This extends supervised state coverage, not the learning
method or source biology. Record independent collection/fit receipts.

A passing candidate may run a new separately frozen validation/final assay on
720000-720015 /721000-721031, after checking all prior seed identities. Retain
13/16 validation and 26/32 final, strict matched progress, paired gain>=8 and
one-sided p<=.025 against each control. No favorable case selection, lowered
gates or reuse of the closed assay. The broad user request authorizes this
measured recovery decision; any failed new gate again closes that experiment.

## Decision: one bounded readout-capacity test after covered-state underfit (2026-09-27)

The 32-case collection passed teacher 32/32 and recorded every frozen-student
rollout (22/32 arrivals), adding 14,856 rows with zero updates. Combined data
has 15,864 rows, 4,954 strong-left labels and 231 strong-right labels, and
constant turn MSE .31523325. No historical validation row was used for training.

The 306-scalar fit reached aggregate .03579290 and original-data .02694830,
missing both frozen absolute ceilings despite original 8/8 arrivals and all
integrity checks. Close and preserve it. Later student states account for
substantial residual error (.03974), so initial unobservable ticks alone do
not explain the miss. Do not lower gates or append updates to this attempt.

Choose one explicit capacity test, 16->64->2 /1,218 motor-readout scalars, on
exactly the same data and preprocessing. Keep seed 1010, Adam .01, 500 updates,
world/replay budgets and all scientific gates. The spiking graph, encoder and
neurons remain unchanged. Separate modules/contracts preserve all earlier
implementations. A failure closes this test; it is not a sweep or permission
for another automatic architecture. A pass can unlock the already specified
separate fresh assay under the renewed user finish request.
<!-- FB-10 projection precision decision -->

### 2026-09-27: Preserve closed recovery failures; request an explicit acceptance decision

The fixed PN-input readout passes every original FB-10 training gate: original
turn MSE 0.01088492, original arrivals 8/8 versus 0/8 and 0/8, strict progress,
finite gradients/370 changed scalars, exact replay and unchanged core/history.
It misses an additional recovery aggregate absolute ceiling: 0.02354470 versus
0.02225273. That experiment stays closed negative. Its 500 updates, result,
gate flags and source must not be changed to make it eligible silently.

A measured fixed encoder revision preserved all 26 transient pulse checks but
failed the declared relative endpoint contrast threshold. It closed after
39,000 neural steps and no fitting. The original circuit/encoder/app remain
unchanged. A separate fixed PN unmixing calculation does not justify a fit:
covered PN contrast stds already exceed the normalization floor, and unmixed
standardization is effectively equivalent. No encoder/capacity sweep follows.

The pending user decision in `plans/FB-10-FROZEN-CANDIDATE-ASSAY.md` is whether
to admit the existing frozen candidate using the original training gates,
while carrying its extra failed gate into a separately locked fresh assay,
or retain the extra gate and stop. This is a material acceptance assumption,
not an implementation permission request. Neither exception nor fresh assay
has been executed. All fresh thresholds, paired controls and zero-update
budgets remain unchanged in the proposal.

### 2026-09-27: Observe actual PN incoming spikes in the FB-10 motor readout

The covered 16-unit and 64-unit endpoint heads both failed the unchanged error
ceiling. A zero-update training replay found PN incoming weighted spike
contrast correlates .87787 with teacher turn, while preserving endpoint output
exactly. The renewed fix-and-finish scope authorizes one readout-only repair:
four causal PN input-event histories alongside the sixteen endpoint histories,
with a 370-scalar head, unchanged graph/encoder/LIF and one 500-update budget.
The full experiment design and stop are in `plans/FB-10-RECOVERY.md`.
This moves the learned readout upstream as well as downstream. A passing result
would establish learning from this fixed circuit's internal spike signals;
it would not establish that its downstream endpoints or BANC topology are
necessary, nor that this is a biological motor decoder. No direct sensory
value, external current, voltage, geometric answer or teacher label enters
inference. All old attempts and protocol-bound source remain unchanged.


### 2026-09-28: Accept one frozen original-gate FB-10 assay; preserve covered failure

After the explicit choice between the frozen original-gate assay and stopping
recovery, the user replied "conitnue". The assistant acknowledged proceeding
under the proposed original gates while retaining the additional recovery
failure. The exact response and interpretation are in
`plans/FB-10-FROZEN-ACCEPTANCE.json`, embedded in the locked assay protocol.
This decision supersedes the pending status of the 2026-09-27 proposal.

The 370-scalar candidate, six prior fits, source core, encoder, dynamics and
all old artifacts remain unchanged. Original eligibility uses turn MSE
0.0108849243 against the original 0.0222527315 ceiling, original 8/8 arrivals
and strict matched/integrity checks. Covered MSE 0.0235447047 still fails that
fit's additional absolute ceiling; `training_gate_passed` stays false in its
historical result. The exception is explicit, not a relabeled successful fit.

One new protocol locked unused 720000/721000 starts, zero updates, at most
144 episodes / 86,400 world ticks / 1,728,000 neural steps, and unchanged
validation/final thresholds. The additive runner independently verifies
checkpoint errors, exact seed-1010 initialization, all nine integrity flags,
training summaries and complete provenance. JSON/JSONL/CSV history checks,
a central consumed receipt and partial counters prevent unnoticed reuse.

The assay passed: validation 15/16 versus 1/16 and 1/16; final 31/32 versus
4/32 and 3/32. Paired final gains +27/+28 and p=7.45e-9/3.73e-9 pass; all-case
progress and frozen integrity pass. The budget consumed 144 episodes,
63,599 ticks and 1,271,980 neural steps, zero optimizer updates. FB-10 is
complete for this frozen internal-spike readout and recorded case distribution.
All assay identities are now historical. No retry or new fit is authorized.

Astra reviewed the implementation and saved evidence. This result does not
establish downstream mediation, a benefit from BANC topology, or biological
fidelity. The fixed app mode is unchanged; optional trained BANC app integration
is a separate next task, with no new scientific training/evaluation needed.

### 2026-09-28 — Additive frozen trained app, preserved historical source

The user continued the next app task while explicitly retaining the extra
training-threshold failure. Add `flybrain.trained_app --control banc-trained`
and a verified runtime loader instead of editing protocol-bound app, view,
neural or training files. Existing modes delegate to the historical app.
HUD subclasses identify the frozen PN-input/endpoint readout and assumed
dynamics. This small duplication of the app loop preserves source provenance.

Pin the exact trained checkpoint, training protocol, failed fit result,
successful frozen assay protocol and result. Verify recorded source/data bytes
and strict checkpoint contracts before display or stepping. The demo requires
the checkout's saved evidence. Parameters are frozen; no optimizer or new
scientific evaluation is introduced. This does not establish downstream
mediation, topology benefit or biological fidelity.

Automated integration/event checks and the saved-frame visual review are
separate from live desktop verification. The latter remains pending because
the native Computer Use pipe is unavailable after retry/reset. Do not mark
that acceptance item complete from a headless smoke.

### 2026-09-28 — One same-method extension for the extra covered fit gate

The user's continued request reopened the *question* of extra training, not
the closed 500-update attempt or its historical result. The predeclared
[follow-up](plans/FB-10-EXTRA-TRAINING.md) tested whether more optimization
time on the same data, seed, 370-scalar PN-input model, objective and Adam
learning rate would cross the unchanged 0.0222527315 covered ceiling. A new
750-update fit started from the seed-1010 initialization because the original
Adam optimizer state was not saved. Its update-500 curve matched the old run
exactly. No width, encoder, data, seed, learning-rate or method sweep ran.

This distinct fit passed its covered and original-data error gates and its
matched original training behavior. Its separate one-use fresh assay passed
the original 16/32-case arrival, paired-effect, progress and integrity gates
on new 730000/731000 identities. The earlier covered fit remains a failure;
the previous frozen acceptance and successful 720000/721000 assay stay valid
as separate history. No source graph, neural rule, feature boundary or
topology-benefit claim changed. The new model is not silently substituted in
the app's earlier pinned 500-update mode.


## 2026-09-28 - Expose the verified extra-training model in the default app

The user's request to fix outstanding work authorizes routine app integration.
Use additive `__main__.py`, `verified_app.py` and `neural/verified_banc.py` so the
new default command loads the passing 750-update checkpoint while every existing
protocol-bound source and experiment remains byte-identical. Pin its fit and
fresh-assay evidence, validate original sources/data, and strictly load frozen
parameters. Reuse views, controller and neural math without replacing module
globals. Adapt the event loop in the new module because editing the historical
entry point would invalidate recorded provenance. No training or scientific
assay rerun is part of integration. The older gate failure remains a failure.
Native keyboard/window review stays explicitly pending until the helper works.

## 2026-09-28 - Sensory approach-and-stop candidate

The new candidate appends one engineered food-contact sensory cell and one
engineered relay to the unchanged 83 sourced cells. Its fixed relay jump is
2 normalized voltage units, and local contact schedules the sensory cell at
100 Hz. These cells, their edge, the odor-to-rate map
`96*c/(c+0.1)` Hz, and the assumed LIF dynamics are engineering choices.
No source identity or BANC edge is reclassified as a contact pathway.

The 22 input histories consist of the earlier 20 endpoint/PN-event histories
and 50 ms and 200 ms filters of actual contact-relay spikes. A 22-to-16-to-3
head is the only trainable component (419 weights and biases). Its positive
stop logit yields exactly zero speed and turn. Geometry produces only the
local contact observation and never overrides the motor output. The
hand-written sensory-only demonstration controller supplies training labels;
it is not a biological mechanism or a learned policy.

The user's Astra plan authorizes one declared fit and a gated fresh assay.
Training rows, normalization, Adam settings, stop condition and paired gates
are locked before fitting. A failed numerical, training, or validation gate
closes the attempt without a new candidate or final-split run.

### 2026-09-29 — Interrupted final split

The one-use fresh assay wrote a complete 16/16 validation record, then its
process disappeared without a final result or traceback. Its 64 reserved
final identities are now historical because partial exposure cannot be
excluded. The frozen checkpoint remains unpromoted despite passing training,
validation and a separate contact-block diagnostic. A new final-only assay
would require an explicit new decision and unused cases; the concrete
[recovery proposal](plans/FOOD-STOP-RECOVERY.md) leaves the model and gates
unchanged and has not been run.

### 2026-09-29 — Run one final-only recovery; keep default gated on live review

After being told the original final attempt was interrupted, its cases were
historical, and a new decision was required, the user replied “if not
continue.” This authorized the prepared single final-only assay on new
840000–840063 cases, with no retraining or change to the checkpoint or gates.
The earlier 830000–830063 identities remain historical. The one new run
completed all 192 matched episodes and passed: 63/64 trained final successes,
seven or eight successes in every sector, 0/64 for each control, +63 paired
gain and one-sided exact p ≈ 1.08e-19. Seed 840044 is the one trained failure.
The fitted readout remained frozen.

Expose this behavior-gated checkpoint through the additive `food_app` mode
with pinned fit, interruption, intervention, protocol, journal and result
bytes. Keep `python -m flybrain` on the historical 750-update arrival model
until the separate live desktop review required by the recovery proposal
passes. The saved-frame and automated checks do not substitute for that
review. No extra fit, case split, threshold or topology claim is authorized.

### 2026-09-29 — User-observed live review passed; promote food mode

The user watched the running food app and confirmed the requested live
behavior: food contact, zero-speed settled dwell and sensor/relay activity
were visible, R started a new trial, and Backspace replayed the same seed and
heading. This satisfies the live-review gate in the recovery plan. The default
`python -m flybrain` launcher now loads the pinned food approach-and-stop
checkpoint; the accepted historical arrival controller remains available via
`--control banc-trained`.

This changes only the launcher default and handoff documentation. The frozen
checkpoint, consumed recovery receipt, experiment records and historical case
identities remain unchanged. The candidate's 63/64 rate is limited to the
recorded randomized distribution; diagnostic app start seed 7 failed from the
center while seed 2 succeeded. No further training or scientific assay is
authorized by this decision.

### 2026-09-30 — User-authorized 3D flight and anatomical viewer

The explicit implementation request authorizes the approved 3D plan and its
single bounded experiment, superseding the old 3D/full-context deferral.
Implementation is additive in `src/flybrain/flight3d`; historical accepted
sources and results are preserved. Panda3D is an optional dependency.

Freeze a source registry before fitting: 589 proofread BANC v888 cells, 4,139
retained directed count>=5 edges, string IDs, annotations, positions and evidence.
Intermediate corridors are structural inferences. Use actual BANC-space
skeletons and registered brain/nerve-cord meshes; the reference supplies style
only. Use one nm-to-mm transform, with no layout changes between visibility modes.

Physiology is explicitly modeled: unsigned normalized weights, 100 ms LIF leak,
threshold/reset rules, visual-field assignments, ideal local motion feedback
and actuator mappings. Independent readouts access corresponding power,
steering and landing endpoints. Zero endpoint activity removes that drive.
Whole context is not whole-brain simulation; missing geometry is reported.

The first frozen experiment passed neural prerequisites but failed the teacher's
first required case. The stop rule closes it with zero optimizer updates and
untouched validation/final assays. Do not append an automatic recovery sweep.
Next work is development visual-sampling/teacher repair, followed by a new
explicitly predeclared experiment. Preserve this run's exact source and failure.

### 2026-10-01 — Visual repair, grouped network and one separate experiment

The user's verbatim request is preserved in the development evidence directory.
It authorizes one agent, the specified repair/viewer work, and exactly one new
bounded experiment after 49/49 development successes. The previous failed run,
its source snapshot, and accepted 2D sources/results/checkpoints remain protected.

Use deterministic 2x4 independently occluded subpixel rays; shared RGB contrast
above 0.01; and a local-IMU-estimated eight-second, +/-1.5 cm search cycle.
These are engineered sensory/control assumptions. No geometry crosses the controller
boundary. The single development implementation passed all 49 full episodes.

Default to 96 source-member groups inside hemisphere/region boundaries. Show
centroid connections only for retained directed edges, with a registered silhouette;
retain explicit Anatomy mode. Missing source positions never receive fabricated
coordinates. Current telemetry is passive: actual delayed delivery totals, fixed
manifest edge order, 300 ms simulation-time smoothing and fixed normalization.

Freeze seed 940404, fresh training 980000–980047, validation 981000–981023 and
final 982000–982095 after overlap checks. Keep all original gates and the one
2,000-update Adam fit. Consume a single project-level receipt before execution;
renaming a directory cannot grant another run. Failed prerequisites/gates close
the experiment, and interruption is incomplete. No automatic retraining or sweep.


## 2026-10-01: source anatomy and independent display scheduling

The user rejected the grouped default as visually unreadable and laggy. Restore
registered surfaces and cached skeletons on black, with subdued blue/violet
context and up to twelve warm neuron-level highlights. Centroid connections are
restricted to explicit selected-neuron inspection. Illumination does not imply
measured conduction position/speed. Overview simplification preserves branching
and coordinates; selected morphology uses exact cached source segments.

Batch static geometry, reuse immutable arrays, bound progressive uploads, and
cache activity/text updates. One serial simulation worker preserves the existing
fixed numerical steps while decoupling frame latency from simulation computation.
Pause/replay/new/exit use barriers. At most 100 ms of pending wall-time credit
prevents runaway catch-up; simulation throughput is reported separately from FPS.
No neural dynamics, sensor calculations or teacher logic change in this repair.

Reconcile the consumed October 1 fit as incomplete after 1,050 updates. Preserve
all run/source/checkpoint/journal bytes, including the stale 1,049 parameter-change
summary. Record the discrepancy and computed deltas outside the frozen directory.
No restart, extra fitting, reload/behavioral gate or reserved evaluation is
performed. Local review follows the explicit single-agent instruction; no commit.


## 2026-10-02 - Display-only quaternion camera and explicit framing

The user's controls specification authorizes a dedicated camera for the existing
anatomical viewer. Preserve the accepted black/blue-violet/warm/cyan appearance,
0.88/1.14 wheel multipliers and 120 ms zoom response; use continuous quaternion
rotation at 0.006 rad/pixel with 60 ms response. Release cancels pending gesture
rotation/pan, so there is no inertial spin. Presets preserve pan/zoom/pivot;
cord toggles preserve all rendered state and targets exactly. Fit/Reset view
are explicit. Pan is unbounded at current perspective scale and never raises
the 0.03 mm zoom floor; Reset view recovers any off-screen anatomy.

Background anatomy denotes registered surface/source reference points beyond
the simulated subset, not training membership. Source points use the existing
Y=-0.43 mm visibility split, matching branches. Source records/cache bytes and
registration are unchanged. N/C shortcuts are removed; brightness belongs in
settings and retained straight connections belong in the selected inspector.
The teacher remains the preview controller. No dynamics, sensing, fitting,
checkpoint, interrupted experiment or reserved evaluation changes are authorized.

## Decision: Display actual interval spikes across owned source branches (2026-10-03)

The user's actual-firing specification supersedes the twelve-cell outgoing-current
emphasis rule in the default anatomical viewer. Record every completed fixed control
interval outside scientific state; publish the original immutable snapshot and
display history atomically. A monotonic display episode identity clears same-seed
replay history. Use a 200 ms pulse decay with a 600 ms idle cutoff and a separate
200 ms exponential firing-rate average in spikes/s.

Preserve neuron owner indices alongside static branch geometry through spatial
brain/cord splitting, batching and deferred loading. Color through an explicitly
RGBA-ordered nearest-filtered GPU table, with no active-cell scene objects. Branch
visibility follows geometry, never metadata region or outgoing topology. Selection
remains cyan. Whole-neuron illumination does not measure propagation along anatomy.
No neural rule, sensory input, scientific snapshot, teacher control, source cache,
learning protocol or interrupted 1,050-update experiment is changed.

## Independent activity classes — 2026-10-04

The user replaced exclusive role presets with independent display switches and
Whole model/Selected circuit scope. Fresh launches enable only Odor inputs;
All activity enables every class and whole scope. Activity off retains history.
Switches are a frozenset; complete/disjoint role-derived membership and fixed
one-hop neighbors produce an immutable display mask. Scope/selection never filter
recording, neural computation or scientific snapshots. No preferences persist.

The ten specified RGB endpoints interpolate from existing idle anatomy with active
alpha .43 and the existing decay/expiry. Cyan selection is independent. UI legend
categories are model assignments, not newly established biological classes.
Clearing or replacing a modeled selection with a background cell restores whole
scope while retaining switches. Cord-only/missing geometry is explained; Show
nerve cord is an explicit action that preserves all camera state. The interrupted
1,050-update experiment remains untouched; no learning or scientific assay runs.

Live-review feedback found other class switches difficult to locate beneath the
settings introduction and shortcuts. The class switches now precede shortcuts and
scope in a compact two-column grid with button backgrounds. All ten are initially
visible at both requested sizes; a rendered-layout regression enforces that.
This changes discoverability only, retaining class/scope/recorder semantics.

## Smooth after-spike color fade — 2026-10-04 follow-up

The user requested that firing neurons fade back to cold blue rather than switch
abruptly after firing. The previous 200 ms exponential display decay still had
about 5% strength immediately before its hard 600 ms cutoff. Replace that display
envelope with a visible eased fade over the same 600 ms total window, using
`(1-u)^2*(1+2u)` for `u=clip(age/.600,0,1)`. It is continuous with zero slope at
both ends and .5 strength at 300 ms. New spikes refresh full class color; pause
freezes the fade. Class/scope exclusions remain immediate user display choices.

This supersedes the prior exponential pulse specification only. The 200 ms
firing-rate average, spike recorder, last-active timestamps, scientific snapshots,
neural computation, geometry, palette/alpha endpoints and reset semantics remain
unchanged. The source/cache/experiment audit and numerical/GPU/refresh-rate tests
protect that boundary. No experiment training or assay is authorized by this work.



## 2026-10-05 — Exclusive encounters and closed negative escape gate

- The user explicitly authorizes a stationary predator in an additive current-3D
  lab extension. The previous general predator deferral is superseded only here.
  No simultaneous flower/predator scene or historical source edits are needed.
- Environment SceneSpec carries seed/kind/XYZ; controllers retain FlightObservation
  only. A cyan sphere is a visual predator proxy; recognition, existing ID-ranked
  retinotopy and image peak/occupancy/growth drive are engineered assumptions.
- Preserve all 589 source cells and 4,139 directed edges. No cells were added.
  Existing unsigned LIF dynamics are assumptions, not sourced physiology.
- One predeclared static characterization closes negative: zero wing spikes in
  16 intact physical presentations. DNp03 contributes under full-field stimulation
  (1,838 intact / 1,476 DNp03-silent / zero visual-silent wing spikes), but the
  model has other visual routes. These are numerical contributions, not evidence
  of biological necessity or successful escape. No fixed escape decoder, sensory
  steering substitute, retreat/retrigger policy or training follows this gate.
- Predator UI explicitly reports unavailable avoidance and the sole local hover
  stabilizer. Its direction-free lift/motion damping reads only local IMU signals.
  Threat display is recorded DNp03 activity; no subjective fear claim is made.
- Spawn advances seed and clears body/controller/neural/transmission/display
  state; R uses selected kind, Backspace replays it. Pause and camera/display
  preferences persist. All activity is this launcher's initial display choice.
- The sourced recovery proposal is LPLC4 -> DNp26/DNp57 -> saccade DN network ->
  contralateral i1/b3, motivated by Figure 5G of
  [Croke et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977095/).
  DNp26 is already present; DNp57 is absent. Source missing paths and review
  physiology/spatial calibration under a separate bounded specification first.
  The current negative result does not rule out a subthreshold decoder, which
  would require its own calibration and verification. No extension started.
- Development stimuli/evaluation seeds in the saved record are now historical.
  No trained/final benchmark was accessed or optimizer/checkpoint written.
  [Guide](ENCOUNTERS3D.md), [plan](plans/ENCOUNTERS3D.md),
  [results](../experiments/encounters3d-characterization-2026-10-05/results.json).

## 2026-10-05 — Brain-to-wing diagnostic and closed investigation

- The user authorizes numerical instrumentation and one fixed-model investigation,
  not retreat, training, a sweep, a parameter change or an added source cell.
- Preserve the historical stepping kernel as the exact parity oracle. Add a
  separate instrumented kernel with unchanged arithmetic and sensory rules;
  record voltage before subtractive reset and spike times on the 1 ms grid.
- Separate phase-driven visual inputs from voltage-driven downstream cells.
  Record effective incoming and discarded electrical increments explicitly.
  Keep current/voltage units normalized and all branch colors spike-only.
- Attach immutable interval diagnostics to the existing atomic encounter snapshot.
  Retain five seconds interactively; stream every interval in the investigation.
  Rendering and panel refresh never advance simulation or consume its RNG.
- Use 22.5°-offset development directions at .09/.17 m. Temporal sequences combine
  growth, a declared opaque binocular image shutter and reappearance. Do not
  label these an unseen behavioral assay or rename historical inspected cases.
- Close exactly 120 one-second presentations / 120,000 neural steps. All wing
  spike outputs remain zero. Retained paths reach every wing endpoint; weak
  electrical drive and spike-threshold information loss are demonstrated.
  Near wing voltages distinguish eight directions; this does not establish a
  decoder, biological accuracy or a retreat behavior.
- The one proposed repair is to restore the original body-local motion-role
  inputs alongside predator vision, using existing source cells/edges and fixed
  dynamics. It is unimplemented and untested. Its next task needs a separate
  declared budget, motion-only baseline and directional stimulus increment gate.
  [Croke et al., Figure 2F](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977095/)
  motivates flight-state dependence, not our engineering IMU rate formula.
- No new cell identities or connections are invented. LC4/LPLC1/DNp57 remain
  absent; any addition requires sourced identities and connections and a
  separately agreed implementation task.
- Native automation failed; the user independently confirmed “Panel and controls
  work” in the open paused preview. Separate numerical/saved-frame evidence and
  human live acceptance are retained. No neural avoidance claim follows.

[Report](PATHWAY-DIAGNOSTICS.md), [plan](plans/PATHWAY-DIAGNOSTICS.md),
[verification](verification/pathway-diagnostics.md).

## 2026-10-06 — Idle encounter lab and failed existing-motion repair

- The user explicitly authorizes additive 3D idle cycles, continuous one-object
  encounters and moving predator controls, plus one fixed bounded repair. Preserve
  589 cells, all retained source wiring/dynamics/diagnostics and historical launchers.
  Use one agent, no commit or optimizer and no scientific tuning/retry on failure.
- Initial state is calibrated upright grounded rest with no object. Timings,
  cruise/altitude, home envelope, safe support and stabilization remain declared
  engineering assumptions. Continuous spawn preserves body and IMU estimates;
  supporting flower removal changes contact only. The teacher inherits sensory
  estimates and never gets world-pose initialization.
- Keep predator commands environment-only, diagonal normalized and clamped,
  applied in 5 ms steps with relative swept collision. Replay includes every
  applied control interval and same-step/endpoint spawns; replay pauses at endpoint.
  Explicit worker barriers reset histories while keeping cameras/preferences/pause.
- Restore only exact original motion-role rates alongside unchanged predator
  vision. Run a separate identical-IMU motion-only reference. Only actual spikes
  illuminate anatomy; signed subtraction/cosine are labeled control diagnostics.
- Pool five 200 ms post-onset signed wing vectors across six calibration cases per
  direction; normalize/freeze eight templates before validation. Constant upright
  rest IMU for sensory calibration/validation; original state-dependent closed-loop
  IMU would be used only after passing. A validation case needs at least one usable
  window and no wrong confident window. Fixed norm/cosine/margin gates 1/.8/.1.
- RGB contrast-weighted pixel solid angle supplies equivalent circular angular
  size, not physical range or a steering heading. Use five-sample 100 ms averages
  and a fifteen-step 300 ms weak timer. Partial occlusion reduces visible size.
  Neural-dependent triggering/stopping is not installed after the earlier gate fails.
- The single run closes at **48 calibration + 96 validation**, **216 s /
  432,000 actual/reference steps**. Validation **9/96**, per-direction **0/6 except
  168.75° 3/6 and 191.25° 6/6**. No 48-case closed-loop evaluation, grounded neural
  takeoff, escape, stable recovery, alternate decoder, weight change or retry.
  Frozen failed templates may be inspected as diagnostics only; the partial lab
  always labels avoidance unavailable. Tonic wing spikes are not threat direction.
- [Croke et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977095/) motivates
  looming/flight-state questions but does not validate our motor or threshold
  defaults. [BANC](https://www.nature.com/articles/s41586-026-10735-w) supplies
  anatomical structure, not a complete physiological/behavioral reconstruction.
- Store source-byte archives, hashes, config, frozen templates and all numerical
  traces in the fresh experiment directory. Recorded placements are now historical;
  old assays and the interrupted 1,050-update fit remain untouched. Native live
  review is pending because the supported helper pipe fails.

[Guide](RETREAT3D.md), [locked plan](plans/RETREAT3D.md),
[closed result](../experiments/retreat3d-motion-repair-2026-10-06/results.json).

## 2026-10-06 — Two effective gains and spike-only turn-and-flee (closed)

- User supplied a separate recovery specification, replacing eight-direction
  classification with a signed evasive side. Keep 589 cells/4,139 edges and all
  original bytes. Gains are engineering assumptions, not anatomical edits or
  physiological measurements. Multiply only 143 visual→DN and 86 DN-output
  effective edges, identically in actual/reference; no renormalization.
- Exactly (2,1), (4,2), (8,4), 16 historical zero-elevation cases each. Original
  threshold/leak/reset/delay/input formulas remain unchanged. Eligibility requires
  correct gated output, exact baseline spikes, finite state, <=240 Hz downstream
  in 200 ms. Lowest eligible (4,2) freezes before offset validation.
- Equal per-spike wing scores use i1/b3 toward their side and b1 opposite; only
  those six endpoints enter the norm. A seeded exact bilateral tie is explicitly
  engineering symmetry breaking. Strong appearance/growth waits <=500 ms; no
  voltage or image-derived bearing chooses movement.
- 90° turn, horizontal braking, .12 m/s flee, .1 m altitude, weak continuity and
  <.02 m/s horizontal/vertical recovery, .22/.18 m IMU envelope are engineered.
  Closed-loop named-state setup assumes upright rest/nominal hover/cruise,
  initialized independently in control and physics. No world-state handoff.
- Immutable configuration carries identities, source hashes and checked
  calibration receipt. Default enablement requires final gates; action logs bind
  the configuration. Old logs use original dynamics and exact endpoint replay.
- Audit 43 old configs/protocols before lock. Calibration angles are historical;
  offset validation is not claimed unseen because old manual placements cannot
  be excluded. No calibration-only promotion, extra candidate or automatic retry.
- Result closes at **87/96**, with 18.75°/63.75° groups **3/6** and nine wrong-side
  responses. All states finite, 3,600 paired baselines identical, max 190 Hz.
  **144 cases / 216 s / 432,000 paired steps**; no closed-loop evaluation, updates
  or default promotion. Historical interrupted fit is untouched.

[Guide](TURNFLEE.md), [locked plan](plans/TURNFLEE.md),
[result](../experiments/turnflee-2026-10-06/results.json).

## 2026-10-06 — Frozen-record wrong-side review and one pending hypothesis

- Review only; zero brain/sensor simulation, decoder replacement or performance
  reclassification. Preserve closed (4,2) failure, source/data hashes, launchers
  and defaults. All nine later score reversals remain descriptive observations.
- Exact reconstruction separates three seeded tie failures from six unequal
  scores; five unequal failures are a local −1 left b3 rolling deficit despite
  a positive full-record increment. Negative signed counts do not establish
  inhibitory current. All original effective weights remain positive shares.
- Correct reviewer assumption to the frozen 46-left/50-right LPLC4 manifest.
  The existing modulo field assignment is engineered; no field or model edit.
  Side conventions and direct cross-side DN-to-i1/b3 edges agree with source.
- Missing raw RGB/reference substep states/edge currents limit attribution.
  Propose only one unchanged-model instrumentation hypothesis concerning b3
  timing and window expiration, capped at 18 historical presentations /27 s /
  54,000 paired steps. It requires separate authorization and has not run.
- Simulation targeted/full-suite checks are explicitly excluded by the user's
  zero review budget. Offline arithmetic, receipts, guarded audit and preservation
  checks supply task-specific verification; historical test counts stay historical.

[Review and exact proposal](TURNFLEE-WRONG-SIDE-REVIEW.md).

## 2026-10-07 ? Passive escape timing diagnostic

User authorizes the diagnosis-first handoff: frozen 589 cells/4,139 edges, (4,2), unchanged field/motion/LIF/readout, two independent brains only. Record all cells and non-self edges rather than choosing a pathway after results. Integer 1 ms IDs and float64 neural measurements; raw original float32 RGB preserved. Stream compressed 20 ms chunks. Exactly 18 historical cases/27 seconds/54,000 individual kernel steps; at most 2,000 cumulative synthetic engineering steps. Fresh parent consumption and output seal prohibit retry/resume. The full simulation suite is deferred by explicit user instruction until an approved repair. A concrete repair decision requires approval after measured evidence; no repair or activation authorized here.

## 2026-10-07 ? Close diagnostic on comparison fault; preserve consumed run

The scientific gate closed at first baseline interval due to tuple/list representation inequality in pathway diagnostics. Saved serialized records equal historical records exactly; guarded numerical audit confirms every recorded step. Preserve consumed instrumentation sources and failed receipt; do not resume or retry even though no threat image rendered. Classify the timing hypothesis unresolved. Only propose comparator canonicalization plus separately approved fresh unchanged-model measurements. Do not select a neural repair without those measurements. Explicitly carry 0.02 seconds/40 steps into any cumulative diagnostic accounting. See [report](ESCAPE-DIAGNOSTIC.md) and [single approval proposal](ESCAPE-DIAGNOSTIC-RECOVERY-PROPOSAL.md).

## 2026-10-07 - Accepted comparator-only recovery

User approved the existing proposal and continuation. Apply canonical finite JSON equality in a new runner, preserve v1, and authorize one fresh 18-case measurement budget. Reuse unchanged kernel proof; 21 saved-only tests add zero steps. Prior 40 scientific steps remain explicit in cumulative accounting.

## 2026-10-07 - Seal external diagnostic interruption without resume

Comparator-only repair is verified over seven complete cases and all flushed records. Execution process vanished during eighth; cause unresolved, no model mismatch logged. Externally seal partial source-bound outputs after checking complete chunks match every reserved step. The all-five hypothesis remains unresolved; two observed successes cannot be generalized. Prepare one remaining-eleven-case proposal with additional 16.5 s/33,000 steps and aggregate only complete cases. Previous partial computations remain counted; no automatic retry. A detached worker probe confirms survival of a normal launching invocation, not all host termination. [Report](ESCAPE-DIAGNOSTIC-RECOVERY-RESULTS.md), [proposal](ESCAPE-DIAGNOSTIC-REMAINING-PROPOSAL.md).

## 2026-10-07 - Accepted remaining-case completion

User accepted exact eleven-case proposal. New source-locked runner, hidden detached supervisor and aggregate audit preserve interrupted receipts. First seven complete prior cases plus eleven fresh form the support test; prefixes excluded and counted separately. 26 guarded tests add zero neural steps.

## 2026-10-07 - Append-only journal after reproduced Windows sharing failure

Atomic replacement can be denied by a reader without delete sharing; reproduced WinError 5 exactly, actual experiment handle unknown. Preserve consumed v3. Implement separate append-only persistent reservation and heartbeat handles, flush/fsync before kernels, commit counter after durability and poison on failure. Four direct tests and 30 guarded saved-only checks pass, zero neural calls. Distinguish attempted reservation 20,240 from executed 20,200. Do not retry scientific case automatically. One five-case proposal retains thirteen complete records and requires explicit additional budget. [Report](ESCAPE-DIAGNOSTIC-JOURNAL-RESULTS.md), [proposal](ESCAPE-DIAGNOSTIC-FINAL-FIVE-PROPOSAL.md).

## 2026-10-07 - Approved final-five journals and aggregation

User approved exact final-five proposal. Fresh append-only reservation/heartbeat worker, no replacement, conservative terminal fallback, no child retry. Aggregate seven plus six plus five complete cases, excluding prefixes but counting their execution. 36 saved-only guarded tests pass, zero neural calls.

## 2026-10-07 - Supported timing diagnosis; one temporal readout proposal

All eighteen complete historical presentations reproduce original records; all five predeclared deficit-only timing criteria pass. Classify supported observationally, not necessary causality. Preserve earlier failed/partial receipts and count actual versus attempted computation. The supported mechanism motivates one minimal cumulative paired six-endpoint readout with positive winner and unit margin, two consecutive eligible 20 ms samples, fixed 500 ms first-strong deadline and deferred ties. Fixed values chosen from integer resolution/minimal temporal agreement, not validation search. Proposal only, no evaluation/implementation; retain all later directional/closed-loop gates and separate activation approval. [Report](ESCAPE-DIAGNOSTIC-COMPLETE.md), [exact proposal](ESCAPE-TEMPORAL-READOUT-PROPOSAL.md).

## 2026-10-07 - Close implemented readout on 86.25-degree regressions

User chose one candidate/honest scientific stop and new launcher only after passage. Implemented cumulative positive/two-sample gate improves 87 to94/96 and fixes all old failures, but new 86.25 group4/6 fails >=5/6. Preserve result, no threshold/streak tuning, fallback or closed-loop evaluation/activation. Valid information can last one sample; two-sample agreement can wait into wrong preference. Next decision requires an event-aware design and explicit new scientific budget. Zero-step missing stdlib dependency was corrected via additive bound entrypoint; no exposed case rerun or neuron/readout parameter change. [Results](ESCAPE-RECOVERY-RESULTS.md).

## 2026-10-08 - Prospective frozen-candidate prototype exception

The user accepts 94/96 for this prototype and explicitly supersedes the next-readout task. Candidate ea70e22d8879874f14df4299ba797170d7624e0b26c6e0baeddf720d3494dcab retains its original failed directional gate (86.25 degrees 4/6) and failures 1100018/1100021. No historical result is relabeled and no directional case is rerun. A separate source-bound exception links that failed record, calibration and already verified motion-only baseline to exactly one prospective 48-case physical validation (288 seconds /576,000 individual neural steps). All 589 cells, 4,139 edges, (4,2) gains, sensory encoding, readout and flight policy stay frozen. Geometry is recorded/scored only by the evaluator. Named start metadata is added outside the frozen package for exact replay. Physical failure closes the experiment without tuning/retry; all physical/audit/replay/saved-review gates must pass before the separately receipt-gated prototype is activated and launched. Learning remains a later task. [Execution plan](plans/ESCAPE-FLIGHT-VALIDATION.md).

## 2026-10-08 - Close physical validation; retain frozen source and CSV representation

The one authorized physical gate fails at 19/24, with resting 6/8, hovering 7/8 and cruising 6/8; silent escape triggers zero. All 24 timely sides/actual turns, eight resting takeoffs and >=159.908 mm separation pass, but three post-recovery boundary collisions and two six-second weak/braking/recovery omissions fail. Preserve complete48/288 s/576,000-step result; no activation, policy/readout tuning, retry or learning. Original 94/96 failed directional gate and known seeds1100018/1100021 remain unchanged under the prospective exception.

The frozen CSV stores NumPy scalar repr in vectors while JSON stores plain numbers. First raw audit rejects textual CSV equality; raw records/seals/sources remain unchanged. Add a saved-only artifact adapter, checking finite literal triples by exact float64 hexadecimal values before the original independent numerical audit runs on a temporary canonical CSV view. Six regression tests pass; no gate or scientific source is monkeypatched. Source-bound corrected audit passes14,400 intervals at zero added neural steps. Exact action replay uses the unchanged frozen replay function with a separate verification journal/driver; no scientific rerun. The strict raw audit CLI retains its representation limitation; the corrected command is documented. [Measured result](ESCAPE-FLIGHT-VALIDATION-RESULTS.md).


## 2026-10-08 - Authorized escape completion repair

The user explicitly requests diagnosis and implementation until escape works.
This authorizes the previously proposed flight-policy repair and verification,
superseding the proposal-only next task. Historical sources/results remain
unchanged; use adjacent escape_repair code and a separate activation receipt.
The accepted neural readout, gains, graph, neuron rules and neural visual encoding
remain fixed. One agent; no learning, automatic commit or publishing.

Two reproduced causes determine the change: positive-only per-frame apparent-size
differences bias growth upward under silhouette sampling jitter, preventing weak
clearance; idle cruise resumes outward velocity at the recovery handoff while its
heading correction turns too slowly to prevent collision. Strong onset remains
exactly unchanged. Weak clearance requires 15 continuously small size samples and
net smoothed-size growth <=5 deg/s over their 0.28 s first-to-last interval.
The idle velocity servo caps outward radial speed from IMU-integrated displacement
to 2*(0.22-radius) m/s, clipped to +/-0.06 m/s, outside radius 0.18 m. This is a
calibrated home envelope, not direct access to arena walls or predator position.

Predeclared one candidate, 96 six-second cases (48 historical, 48 new seeds at
the same historical geometry), 1,152,000 individual neural steps. Require all
48 intact successes and no silent triggers with the existing physical criteria.
Four fixed action replays add at most 48,000 verification steps. A failed gate
closes that experiment; no relabeling, automatic retry or parameter search.
See docs/plans/ESCAPE-REPAIR.md. Evaluation is pending at this decision entry.
# 2026-10-11 — Sustained edge escape replaces weak-threat recovery

The user explicitly chose continued flight to a map edge, then stop/hover there.
Add a separate escape_edge policy rather than alter any source-bound historical
controller. Commit escape through weak/absent threat, remove its center-seeking
guard, and hold at an endpoint inset 15 mm. The IMU estimate and existing fixed
center-start/arena calibration determine that endpoint; actual body/predator
coordinates and an environment-computed steering answer remain forbidden inputs.
This is engineered navigation/stabilization, not learning or a reconstructed
biological escape circuit. Original 94/96 directional failures remain documented.
New acceptance replaces weak-threat braking/recovery with sustained travel and
two-second edge hover. All failed records/gates remain failed. The user authorizes
diagnosed development iterations until verified, each with declared cases/budget,
source freeze and immutable terminal record; no retry/threshold relabeling.

## 2026-10-12 — UI presentation and comprehensive project understanding

The user considers the behavior sufficient and requests UI polish, prominent
flower/predator spawning, then comprehensive understanding for honest interview
discussion of an AI-assisted project. This supersedes the capture-only next
task; it authorizes routine reversible presentation fixes, not neural tuning,
training, new scientific experiments or relabeling earlier results.

Use a separate `escape_ui` presentation package. Existing `escape_edge` sources,
inherited controller/neural/physics modules, experiment archives and delivery
receipts remain byte-identical. The new launcher uses the original verified
candidate check. This avoids invalidating source-bound evidence by editing its
viewer or relaxing verification. Presentation controls reuse existing spawn,
pause/reset and replay barriers; renderer reads immutable display frames.

Keep world and brain side by side, give primary actions visible 42-pixel
buttons, and collapse detailed pathway diagnostics by default. Preserve neuron
inspection and activity settings. Saved-frame review and programmatic callbacks
are distinct from native mouse/keyboard review; report any unavailable review.

Record the standing educational requirement in AGENTS.md, status, README and
docs/LEARNING_ROADMAP.md. Start the walkthrough after UI handoff. Cover each
distinct script and its algorithms, libraries, training/evaluation and limits.
Do not infer understanding from an explanation or imply sole human authorship.

## 2026-10-12 — Approved readable dashboard and geometry filters

The user rejected the earlier presentation as visually insufficient and difficult
to understand/navigate, particularly its textual neuron toggles. Passing behavior
tests did not establish the desired usability. They selected modern dark dashboard
and full branch hiding, then explicitly approved the supplied redesign plan.

Keep the launch command/source-bound controller. Implement only in escape_ui;
archive the earlier presentation before changing it. Preserve all original
scientific sources/receipts/results, neural computation, physics and action schema.
Expose the ten existing model roles with real checked-state controls, friendly
descriptions and counts; source biological type remains in Inspect. All neurons
stay simulated/recorded even if their group is hidden. Do not label these controls
as experimental neuronal silencing or inferred biological classification.

Apply the same owner visibility to RGBA/discard, picking, selection highlights,
connection overlays and camera bounds, including full selected source morphology.
Use source soft outline rather than the dense reference-point cloud as default
context. Empty group selection offers Show all and leaves the camera stable.
Pixel layout drives both scene/input rectangles and GUI placement. Defaults:
Groups tab, all roles checked, faint outline on, nerve cord off. Detailed source
IDs/provenance/pathway data are opt-in; checked state and navigation survive
spawn/reset/replay/resize. Use Windows' native readable font without redistributing
it; portable fallback remains. Verify actual pixels and zero-step previews before
the full regression run. The teaching goal remains next after UI review.

## 2026-10-13 — Softer relay color and clearer structural meaning

User reports the redesigned UI works well and requests a less overwhelming,
distinct color for Other relays. Override only the dashboard structural-group
palette with steel-blue #5B8FB9, using the same color for branches and swatch.
Preserve the source-bound original white palette, group membership, pulse timing,
opacity, neuron rules and controller. Archive prior presentation code separately.
Explain the 241 cells as structurally selected intermediate paths, not a single
biological type or demonstrated function for every retained cell. The original
path selector's own caveat remains: structural reachability is not evidence of
functional mediation. User review is reported as feedback, not exhaustive native
input validation or demonstrated understanding of all algorithms.

## 2026-10-20 - Private GitHub import and byte preservation

The user explicitly requested creating and pushing a GitHub repository with a
security review. Default to private visibility; include source, data, archived
code and measured experiment records, while excluding environments and secrets.
Use a GitHub noreply author address. Disable automatic line-ending conversion
because existing source-bound receipts verify exact file bytes across machines.
Keep the scientific sources and recorded results unchanged. Fix known vulnerable
build-tool versions through the build floor and audited installation constraints;
retain historical environment/results as historical evidence. Enable dependency
alerts and updates. A passing scan is limited to known advisories and detectable
patterns; it is not a guarantee of zero security risk.

