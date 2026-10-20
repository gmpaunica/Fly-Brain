# 3D flight lab

The isolated `flybrain.flight3d` application implements the approved
[3D plan](plans/FLIGHT3D.md). The 2D app remains the independently accepted
approach-and-stop demo. **3D learned landing is not accepted.** The September 30
experiment remains closed at the teacher prerequisite. The October 1 sensory repair and one-use experiment are recorded in
[the repair plan](plans/FLIGHT3D-REPAIR.md). That fit is **incomplete after 1,050
updates**, with checkpoints and journals preserved. The current
[anatomical repair](plans/FLIGHT3D-ANATOMICAL-REPAIR.md) adds no fitting or assay.

## Run

```powershell
.\.venv\Scripts\python.exe -m flybrain.flight3d
.\.venv\Scripts\python.exe -m flybrain.flight3d --control manual --seed 10001
.\.venv\Scripts\python.exe -m flybrain.flight3d --control fixed --seed 10001
```

The default is a clearly labeled **hand-written sensory teacher**. In this
mode the BANC circuit receives the observations and its displayed activity is
observational; the teacher supplies the commands. `fixed` actually drives the
body through the sourced circuit's motor endpoints. `trained` and `untrained`
require an explicit identity-bound `--checkpoint`; neither silently substitutes
a teacher. The October 1 partial candidate is not an accepted or fully verified trained
controller. The teacher remains the preview default.

| Control | Action |
| --- | --- |
| R / Backspace | New random flower / exact trial replay |
| Space | Pause simulation and activity fading |
| Drag / Shift-drag | Continuous rotation / pan at the current viewing scale |
| Wheel | Smooth zoom over a viewport; scroll over the full details/help rectangle |
| Display settings | Independent activity classes, Whole model/Selected circuit scope, Background anatomy, Branch brightness, Reset view |
| Front / Side / Top, or 1 / 2 / 3 | Change orientation while keeping zoom, pivot and pan; Views menu below 1200 px |
| Tab | Toggle Background anatomy: registered surface and reference points beyond the simulated subset |
| Show nerve cord / V | Change visible geometry only; preserve the entire camera state |
| Fit view | Smoothly frame all currently visible brain and cord geometry |
| Reset view / B | Restore front orientation and fit currently visible anatomy |
| F | Close follow view of the fly / arena overview |
| Click branches / Neurons button | Select a source neuron, full morphology and evidence |
| Focus selected / Clear selection | Explicitly frame selected visible source morphology / clear selection |
| Show connections (inspector) | Off by default; straight retained connections, not anatomical branches |
| Up / Down | Manual power above/below hover |
| W S / A D / Q E | Manual pitch / roll / yaw |
| L | Hold to deploy legs in manual mode |
| Escape | Exit |

The default starts at the accepted front framing with a margin around both side
lobes and the cord hidden. The fly camera retains its independent controls.
Branch selection changes the inspector and cyan geometry without moving the camera.
N and C have been removed as global shortcuts. Background anatomy says nothing
about which cells were trained; the teacher remains the preview controller.

The dedicated display-only quaternion camera rotates its viewing direction and up
vector together. Drag sensitivity is 0.006 radians per viewport pixel, independent
of window dimensions. Rotation/pan use a 60 ms exponential response and gestures
stop on release, pointer loss, focus loss or resize. Wheel multipliers remain
0.88/1.14 with the existing 120 ms response. Brain distance can reach **0.03 mm**,
with cord visible or hidden; near clipping follows distance down to 0.0003 mm at
that floor. Shift-pan uses perspective scale and never forces zoom outward.
The Neurons button browses the 589 simulated IDs in bounded pages, including IDs
with missing geometry. Details/help scroll over their whole rectangles; essential
trial, pause and exit controls stay outside those panels.

### Source anatomy and actual spike activity

The default is actual locally cached skeletons in their registered coordinates,
with a faint blue/violet source-reference cloud and translucent registered surface
on black. There are no group-centroid links in this view. The prior Anatomy and
Network implementations remain available to the benchmark for comparison.

The overview uses the existing audited degree-two-chain simplification; branch
junctions and anatomical coordinates are retained. Full source segments load on
selection and appear cyan. Full detail is reserved for selected neurons, rather
than every cell at distant zoom. All modes reuse one immutable morphology cache.
Static geometry is batched in at most 50,000 segments per upload. Two local I/O
threads prepare arrays, with a 2 ms main-thread loading budget between uploads.
A buffer upload is indivisible; the budget is a scheduling target, not a hard
latency guarantee. Cord render buffers are deferred until the cord is shown.

Every firing modeled neuron in enabled classes/scope colors its available visible branches with its class color,
including cells without outgoing edges and cells whose region annotation differs
from their branch location. No count cap or current-ranking filter applies.
Each completed 20 ms simulation interval containing spikes refreshes a pulse.
The class color eases smoothly back to blue/violet idle over **600 ms after the
last spike**, with no abrupt cutoff. A new spike refreshes full color and pause
freezes the fade. Colored branches show whole-neuron activity; they
do not indicate measured anatomical spike propagation or conduction speed.
Cyan still marks selection. Brightness does not establish behavioral importance;
teacher/manual activity is observational.

Static branch endpoints carry their neuron owner through brain/cord splitting
and deferred loading. A compact per-neuron RGBA GPU texture colors the existing
batches, rather than adding an object for every active neuron. Branch brightness
settings preserve the idle presets. Source geometry and camera controls are unchanged.

The worker records every completed interval, including steps between screen
refreshes. One immutable display frame pairs the original scientific snapshot
with a distinct episode identity, cumulative spikes, last-active interval times
and a 200 ms exponential average in spikes/s. The renderer reads that pair once.
Repeated drawing, selection, loading or visibility changes never record spikes.
Pause freezes activity; replay/new trial clear history, even with the same seed.
Latest-interval firing, recent visible pulses and fired-since-reset counts are
shown separately. “Visible” counts unique owners with branches uploaded in enabled
layers, rather than testing whether pixels are currently on screen.

The selected-neuron inspector shows latest-interval spikes, spikes since reset,
recent rate and time since the last active interval. “Not since reset” means no
recorded spikes. Geometry can be missing, still loading, empty or hidden; those
conditions do not establish silence. No non-self incoming connections is also
explained where applicable: source self-connections do not participate in dynamics,
while sensory inputs can receive external stimulation.

In the historical seed-970000 two-second teacher preview, **530/589 cells fired**
and **59 did not**: 50 unstimulated visual inputs, six other cells with only self-connections,
and three whose upstream cells stayed silent. Those are limits of that sampled
window and modeled inputs, not a claim of permanent silence or learning.

Text changes only when its
content changes. Connection diagrams are available only through Show connections in the inspector;
they join source reference points and are explicitly not reconstructed axons.

One serial simulation worker executes the unchanged 20 ms control intervals,
5 ms physics and 1 ms neural steps. The UI reads published immutable display frames.
Pause waits for an in-flight step; replay/new trial reset at a barrier. Pending
wall-time credit is limited to 100 ms so a slow simulation cannot create an
unbounded catch-up loop. Slow hardware runs fewer simulation seconds per wall
second; no numerical timestep is enlarged or neural calculation omitted.

Saved front/side/top anatomy, activity, cord and full-selection previews and the
three-view historical benchmarks are under `.superpowers/anatomical-repair-2026-10-01/`.
Sequential benchmarks show repaired running p95 **14.64 ms** versus **134.91 ms**
for Network and **65.47 ms** for prior Anatomy. Paused/orbit/selection p95 is
5.10/6.35/13.70 ms. Paced display measured 55.8-58.8 FPS; steady 60 is not verified.
Simulation throughput is 0.57 simulated seconds per wall second. See
[verification details](verification/flight3d-anatomical-repair.md). A native live
review is separate from offscreen checks and remains pending.

### October 4 actual firing verification and feedback

Fresh **76 targeted / 643 full-suite tests pass**, zero full-suite skips. Per-neuron
rendered GPU colors and matched simulation-time activity at 30/60/120 display Hz
are verified. Passing 1440×900 hidden/shown timings have p95 ≤12.16 ms with
359/527 visible pulses, but an earlier shown run failed and timing variability
remains a limitation. Scientific sources, all cached geometry and the interrupted
1,050-update experiment are unchanged. See
[exact evidence and limits](verification/flight3d-spike-activity.md).

The user approved the full-activity view as an option and subsequently specified
independent class switches and Selected circuit scope, implemented below. Individual
native interaction checks are unconfirmed. Space freezes pulses; R/new and
Backspace/replay clear display history. Pause before resetting to inspect zero
counters before fresh activity. Colored pulses describe activity, not causal importance.

### Independent activity switches — October 4

Fresh launches enable **Odor inputs** (amber) only. In Display settings, each of
the ten class switches operates independently. Click **Display settings** above
the brain, then click a class button to switch its color on/off. All ten buttons
are visible at the top in two columns; scroll the right panel for scope and other
display choices. Sensory groups odor, vision, optic
flow, motion, taste and touch. Its parent reports mixed when some children are
enabled. Clicking off/mixed enables all six; clicking on disables them. Expanding
or collapsing the child list changes no switch state. Runtime membership counts
are shown on every class. Categories are model assignments, not newly established
biological classifications. The complete color/role table is in the
[updated proposal](plans/FLIGHT3D-ACTIVITY-MODES.md).

**All activity** enables every class and Whole model. **Activity off** hides glow
and retains history and scope. Selected circuit includes a selected modeled cell
and its direct non-self incoming/outgoing partners, intersected with enabled
classes. It is disabled without a modeled selection. Clearing selection or
choosing a background-only cell restores Whole model without changing switches.
Cyan inspection remains independent. Cord-only/missing geometry is explained;
Show nerve cord does not automatically fit or move the camera.

The left panel starts with a two-column color legend and explicit on/off states;
navigation help follows. Blue/violet is idle anatomy and cyan is selection.
Full-model latest firing, firing within enabled/scoped membership and resident
visible colored pulses are counted separately. Disabling a class does not silence
its neurons. Paused switches immediately recolor the existing frame. Replay/new
trial retain switches/scope and reset display history. New launches start odor-only;
preferences are not persisted. See [verification and live-review status](verification/flight3d-activity-classes.md).

### October 2 controls verification (historical)

Fresh 61 targeted and 628 full-suite tests pass; zero full-suite skips. Isolated
navigation p95 is 3.91-9.05 ms, paced display about 58.5 FPS, with 0.97 simulated
seconds per wall second in that run. Steady 60 FPS is not established. All prior
experiments and cached geometry are byte-identical. The live teacher preview is
open for the requested short review; native drag/resize/focus acceptance remains
pending. See [checks, exact commands and limits](verification/flight3d-controls.md).

### October 1 sensory repair

Each unchanged `[2,24,48,3]` RGB image averages 2 horizontal by 4 vertical
subpixel rays, resolving nearest-surface occlusion separately for every ray.
Teacher and visual encoder share `R - (G+B)/2`, retaining positive contrast above
0.01. Sky, checker floor and stem fixtures are rejected. The teacher weights
image directions by contrast and uses apparent angular extent for range.

During visual loss the teacher adds an eight-second vertical cycle of amplitude
1.5 cm around its estimated search-entry altitude. It integrates body-local motion
feedback; no world pose enters control. Reacquisition clears the search immediately.
The development record contains **48/48** balanced seeds 970000–970047 plus the
explicitly historical **960000** regression, all successful over complete 30-second
episodes. Historical 960000 now contacts at 2.765 s and finishes with 27.24 s dwell.
This is teacher development evidence, not learned landing.

### A concrete PyTorch calculation

On development seed 970000, RGB pixel `[eye=1,row=14,column=2]` is approximately
`[0.46625, 0.20500, 0.29625]`. Its contrast is
`0.46625 - (0.20500 + 0.29625)/2 = 0.215625`, above 0.01.
The visual encoder's gain of three gives 0.646875; the selected visual entry cell
`720575941429767343` receives about `240 * 0.646875 = 155.25 Hz` on the first
observation. The complete input-rate vector has shape `[589]`.

PyTorch calculates phase increments of `rate * 0.001` each neural millisecond.
An input spike is delivered on the next step: an illustrative weight 0.9 times
previous spike 1 adds 0.9 to its destination. The sparse edge lists are equivalent
to a matrix indexed **[post, pre]**; they do not reverse source edge direction.
These phase, leak, threshold and reset rules are hand-written engineering choices,
not physiology supplied by PyTorch or by the BANC graph.

Two endpoint rate histories produce a `[2,589]` state. Selecting motor groups
yields feature sizes `[48,48,56]`, concatenated to `[152]`: power, steering and
leg heads can each read only their assigned group. Holding this initial observation
for two seconds gives a zero fixed motor command in the recorded illustration,
despite visual input activity. The hand-written teacher's initial command is
approximately `[0.4911,0.1037,0.0646,0.0577,1.0]`; it does not demonstrate learned
motor control. See [the numerical record](../experiments/flight3d-development-2026-10-01/teaching-example.json).

Forward tensor computation produces activity and commands. Only the separate fit
uses `loss.backward()` to calculate readout gradients and Adam to change readout
parameters. Running the neural simulation, or drawing illuminated links, does not
itself establish learning or successful behavior.

Use a close follow view to see the millimetre-scale body and animated wings
inside the 60 cm arena. The fly starts at the same position and orientation
every time; the flower changes horizontal position and elevation.

## Anatomy and coverage

`data/flight3d/circuit.json` freezes **589 string neuron IDs and 4,139 directed
connections** from the pinned BANC v888 Dataverse 3.0 archives. It retains the
source annotations, synapse counts, coordinates, evidence categories and
selected directed paths. Its SHA256 receipt is checked on every load.

The selected cells include 35 proofread DNg02 cells, a DNa04 pair, four
DNp07/DNp10 landing cells, 24 annotated wing-power and 24 wing-steering cells,
12 wing-tension cells and 28 front-leg extension motor cells. Sensory inputs
include olfactory, visual, haltere mechanosensory, taste and touch cells.
241 intermediate cells were selected for structural reachability. One
olfactory input lacks a retained route to each DN group; this is recorded in
`missing_paths`, without inserting a synthetic connection.

The overview contains 175,180 source records after excluding annotated glia,
tracheal, circulatory and non-neuronal records. **18,707 have no reference
position; 28,632 are untyped.** The viewer reports these limits. Reference
points are source root positions, not claims about soma centres. Detailed
selected skeletons and the brain/nerve-cord region surfaces come from the
official BANC storage bucket. Per-object generations, source hashes and raw
SWCs are cached. **585/589 selected skeletons are available.** Four exports are
missing (one LPLC4 and three corridor cells); both public SWC names return 404.
Their source positions are retained and the UI reports the omissions. See
`data/flight3d/coverage.json` for IDs and `data/flight3d/audit.json` for the exact
reconciliation of all available geometry, identities and directed edges.

The whole view initially uses coarse reference points and registered surfaces.
Project skeletons load progressively at a reduced geometric resolution. Clicking
a cached simulated branch or browsing its ID loads full source morphology;
unavailable source geometry stays explicit. Unused atlas records never
receive live activity colors. This does **not** load every full-brain skeleton
or simulate every atlas record. Full morphology would require the much larger
source archive and a separate performance budget.

All geometry uses one transform: **(X,Y,Z) nm → (X,-Y,Z) × 10⁻⁶ mm**.
No per-cell layout or separate centering occurs when modes change. Front is
viewed along source +Z; side along transformed +X; top along transformed -Y.
These are documented presentation presets, not an additional atlas registration.
The viewer's nerve-cord clipping plane is transformed Y = -0.43 mm. Source
geometry is retained intact on disk. A selected source-edge diagram joins
reference points; those lines are explicitly not anatomical axon geometry.

## What is biological and what is modeled

Identities, retained edges, synapse counts, annotations and morphology are
sourced. Structural paths establish reachability, not established physiological
mediation. The saved visual reference supplies the palette and navigation
direction, not invented cell positions or edges.

LIF dynamics, unsigned synaptic gain, sensory tuning, visual receptive-field
assignment, motor readouts and simplified flight mechanics are engineering
assumptions. Source transmitter annotations are preserved; postsynaptic receptor
signs were not supplied, so this model does not claim realistic excitation and
inhibition. Visual fields are assigned by a documented cell-order rule, not
measured BANC retinotopy. Motion feedback is an ideal body-local gyro and
specific-force proxy assigned to mechanosensory entries.

Physics uses metres/seconds, +Z upward, 5 ms steps, 20 ms controls and 1 ms neural
steps. Power supplies thrust along body up; a simplified attitude response
converts steering into pitch, roll and yaw. Gravity, linear drag, ground/arena,
stem and landing-surface contacts are modeled. This is not fluid dynamics or
reconstructed wing biomechanics. Wingbeat animation is a slowed visual proxy
whose amplitude follows power. Petal visuals are decorative lobes around the
sensor model's oblate flower surface and circular landing footprint.

## Neural calculation in plain language

An eye produces a `[2, 24, 48, 3]` RGB tensor. The encoder drives only selected
sensory cells. Source edges use `pre → post`; a dense equivalent would put the
weight at `[post, pre]`. The implementation scatters presynaptic contributions
into postsynaptic voltages. A cell leaks, receives delayed spikes, crosses a
threshold and resets by subtracting that threshold.

For example, one motor spike in 20 ms is 50 Hz. Starting from zero, the 100 ms
rate filter gives `50 × (1 - exp(-0.02/0.1)) ≈ 9.06 Hz`. Two filtered histories
per endpoint form separate wing-power, wing-steering and landing feature groups.
Only the corresponding group's readout can drive that actuator. Silencing the
motor cells clears their histories and removes their drive, including bias.

PyTorch computes these tensor operations. The neuron rules and sourced graph
are fixed. Adam would adjust only the readout parameters using gradients from
imitation error. The September 30 run reached zero updates. The October 1 run completed
1,050 updates before an execution interruption; that fact alone does not
establish learned landing.

## Recorded experiment and next work

The [protocol](../experiments/flight3d-landing-2026-09-30/protocol.json) declares
48/24/96 placements, 30-second episodes, 2,000 Adam updates and the approved
behavioral/statistical gates. Signal-distinction, propagation, motor silence
and DN-cut prerequisites passed. The teacher failed the first required training
case, seed 960000: it hovered and searched in yaw for 30 seconds, without food
contact. No other training cases, validation or final cases were evaluated.

The [diagnosis](../experiments/flight3d-landing-2026-09-30/diagnosis.json) records
zero flower pixels in the initial low-resolution eye samples. Yaw search
preserves elevation, so the thin flower can remain between vertical ray rows.
The experiment is **closed**, with zero updates and no learned checkpoint.
The trajectory, initial eye images, frozen protocol and exact executed source
snapshot are retained. There was no automatic retry or parameter sweep.

The separately authorized October 1 protocol passed **48/48 training teacher
cases** and completed **1,050 Adam updates**. A Windows access-denied error during
atomic progress-file replacement interrupted execution. Candidate and interrupted
checkpoints contain identical parameter tensors at update 1,050, with positive
recorded gradients and 12 changed parameter tensors. The parameter-change summary
on disk stops at 1,049; a separate read-only reconciliation computes the actual
1,050 deltas without modifying the run. No reload gate, trained behavioral gate,
validation or final evaluation completed. The one-use authorization remains
consumed. **Incomplete, not accepted; no restart or further fitting.**
See [reconciliation](../.superpowers/anatomical-repair-2026-10-01/interruption-reconciliation.json).

One next task: finish the short live review of the revised camera and controls.
Recommended effort: Sol Medium. Further learning
work needs a separately agreed recovery plan and belongs outside visual repair.

## Preparation on a fresh checkout

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[flight3d]'
.\.venv\Scripts\python.exe -m flybrain.flight3d.anatomy PATH_TO_PINNED_ARCHIVES
.\.venv\Scripts\python.exe -m flybrain.flight3d.assets
.\.venv\Scripts\python.exe -m flybrain.flight3d --capture .superpowers/flight3d/frames
```

The registry builder requires the two pinned archives and verifies their MD5s.
Asset preparation may resume downloads; it does not rerun a scientific
experiment. Camera controls and saved-frame export never fit a model.

Sources: [BANC dataset](https://doi.org/10.7910/DVN/7WTH1N),
[BANC paper](https://doi.org/10.1038/s41586-026-10735-w),
[DNg02 flight-amplitude work](https://doi.org/10.1016/j.cub.2022.01.008),
[landing pathways](https://doi.org/10.1038/s41593-019-0413-4),
[Panda3D independent display regions](https://docs.panda3d.org/1.10/python/programming/rendering-process/display-regions).
