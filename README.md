# FlyBrain

Before installing, read [SECURITY.md](SECURITY.md) for current build-tool minimums
and the audited dependency constraints. Source, data and saved experiment records
are included; local virtual environments and credentials are excluded.

## Interactive demo and learning goal — 2026-10-12

Run `.\.venv\Scripts\python.exe -m flybrain.escape_ui` for the redesigned dark
dashboard: world and brain side by side, real neuron-group checkboxes, readable
descriptions, and Groups/Inspect/Help navigation. Unticking a group hides its
branches; all neurons still run. The toolbar offers flower/predator spawning,
pause, new trial and replay. It uses the original source-bound verified controller.
[Controls](docs/ESCAPE-UI.md).
Redesign verification completed 2026-10-13: 44 focused/928 full passed, zero skips. Ten
zero-step layouts reviewed at two sizes; native input acceptance remains
unreviewed. [Current evidence](docs/verification/dashboard-redesign.md).
The earlier UI's 925-test record is preserved in [prior verification](docs/verification/escape-ui.md).

This is an **AI-assisted educational/portfolio project**. After UI polish, the
standing priority is a comprehensive walkthrough of each script, algorithms,
Python, Pygame, Panda3D, PyTorch, training and evaluation so the user can explain
and defend the project in interviews. [Learning roadmap](docs/LEARNING_ROADMAP.md).
The roadmap records a goal, not a claim of understanding or sole authorship.

## Sustained edge escape — verified 2026-10-11

Run `.\.venv\Scripts\python.exe -m flybrain.escape_edge` for the new 3D escape
viewer. It turns, keeps fleeing to a map edge, then brakes and hovers there.
Both declared neural batches passed: 48/48 intact escapes, zero triggers in
48 silent controls; four exact replays and 922 final full tests pass. The
brain/readout are unchanged; edge flight is engineered IMU control, not learned.
Historical 94/96 directional limitations remain. Native input review/capture
label cleanup remain a separate UI task. [Results and limits](docs/ESCAPE-EDGE-RESULTS.md).
The default 2D app and historical 3D launchers are unchanged.

## 3D flight lab — 2026-09-30

Run `.\.venv\Scripts\python.exe -m flybrain.flight3d` for the separate Panda3D
flower world and BANC anatomical viewer. It has fixed starting pose, flowers at
random positions/heights, physical flight/landing, source-neuron inspection and
whole/project visibility modes. See [controls and scientific limits](docs/FLIGHT3D.md).
**The 3D teacher prerequisite failed; no 3D learned-landing claim or trained
checkpoint is available.** The default 3D preview is clearly labeled as a
hand-written teacher. The accepted 2D result below remains separate.

**Behavior repair (2026-09-29): the 85-cell approach-and-stop candidate passed
its new final behavior assay.** It passed 40/40 training, 16/16 validation,
and **63/64** new full 20-second final cases versus **0/64** for both matched
controls. Every heading sector met its gate (seven or eight successes of
eight). The original interrupted final attempt remains historical. After the
user's live review confirmed stopping, the sensor/relay panel, R and Backspace,
the candidate is now the default. See [the measured result](experiments/food-stop-recovery-2026-09-29/results.json)
and [the limits](docs/experiments.md#approach-and-stop-single-readout-fit--2026-09-28).
The result applies to its recorded randomized case distribution; an app smoke
from center-start seed 7 failed to contact food, while seed 2 succeeded.

The current BANC app now offers **R: new random trial**, **Backspace: replay this
trial**, and **Esc: exit**. It shows the trial seed, speed, food contact, and
stationary-contact duration. Omit `--seed` for a fresh launch orientation; provide
one to reproduce the trial sequence. The accepted candidate has an engineered
contact-to-relay pathway and a frozen 22→16→3 readout. Run it with:

```powershell
.\.venv\Scripts\python.exe -m flybrain --seed 2
```

Historical BANC arrival model (2026-09-28): **A separate 750-update readout passed the
extra covered-training threshold and its fresh assay: 16/16 validation and
31/32 final arrivals, against matched 0/16 and 2/16 validation and 2/32 for
each final control.** The earlier 500-update fit's extra-threshold failure and
its explicitly accepted frozen assay remain preserved as separate records.
The verified 750-update controller remains available as a reference mode:

```powershell
.\.venv\Scripts\python.exe -m flybrain --control banc-trained --seed 7
```

This loads the saved 370-parameter PN-input/endpoint head without online training.
The historical `flybrain.trained_app` command retains the earlier 500-update
checkpoint. `flybrain.food_app` is also available explicitly. The default app
uses the 85-cell food approach-and-stop head in [the fit](experiments/food-stop-fit-2026-09-28/results.json)
and [fresh final assay](experiments/food-stop-recovery-2026-09-29/results.json).
Current fixed comparison: `python -m flybrain --control banc --seed 7`.
The original `flybrain.app --control banc` retains its historical controls.
The repository's saved experiment files are required for verified loading.
See [current status](PROJECT_STATUS.md), [integration scope and verification](docs/plans/TRAINED-BANC-APP.md),
and the [inspected food-app frame](docs/verification/food-accepted-frame.png).
The user completed the live review for the current food app.

The approved follow-up diagnostic found no lost relay spikes in the two tested
cases: unequal path delays move spikes across the fixed counting-window edges.
The approved [latency-aligned protocol](docs/plans/FB-09-LATENCY-PROPOSAL.md)
passed all response and causal checks. It changes the offline measurement
definition; old failed experiments remain failed, and simultaneous-window
robustness or biological fidelity is not claimed.

Educational Python/Pygame 2D fly simulation. M0-M2 provide a deterministic bounded
world, temporary **MANUAL DEBUG** controls, one food odor source and two antennae.
Body geometry and odor parameters are synthetic scaffolding, not biological
measurements. FB-03 adds a synthetic PyTorch spiking network; FB-04 connects its
sensory-only activity to bounded motor commands. FB-07 adds an offline trainable
motor readout experiment; the original app uses the fixed decoder.

**Historical milestone (2026-09-24):** M6 navigation improvement passed its fresh-case
gate: the trained charge readout reached food on 16/16 validation and 32/32
final starts, versus 1/16 and 1/32 for each matched untrained and fixed control.
This result applies to the frozen arena, start distribution and network seed;
it does not establish biological fidelity or a benefit from downstream topology.
See the [recorded comparison](docs/experiments.md#m6-fresh-matched-evaluation--2026-09-23)
and [current status](PROJECT_STATUS.md). At that point the default app exposed
the 750-update arrival-trained BANC head; the later food-stop repair is
summarized at the top of this README.

**FB-08 / M7 is also complete:** 160 matched intervention episodes and
[recorded plots](experiments/fb08-interventions/interventions.png). Intact
navigation reached 15/16 targets; disabling either sensory side gave 0/16.
Deleting or randomizing downstream wiring retained 15/16. This demonstrates
sensory dependence in the tested setup without evidence of a downstream
wiring benefit. FB-09's [source and import record](docs/plans/FB-09.md) now
packages a pinned CC BY 4.0 BANC v888 83-neuron olfactory-to-motor subset.
Its first unsigned dynamics probes failed and remain preserved. The accepted
engineered dynamics now run in the BANC app with eight endpoint spike-rate
features. [View the inspected frame](experiments/fb09-app-verification/banc-frame.png).
See the [FB-08 experiment plan](docs/plans/FB-08.md)
for the prior assay's scope and limits.

The checked-in [BANC subset](src/flybrain/data/banc_v888_va2_dna02.json) can
already be stepped with the neural extra installed:

```python
from flybrain.data.banc import load_banc_subset
from flybrain.neural.banc_network import BancNetwork, load_verified_banc_config
from flybrain.world.sensing import SensoryObservation

graph = load_banc_subset(connectome_gain=1.0)
brain = BancNetwork(graph, dynamics_config=load_verified_banc_config())
activity = brain.advance(SensoryObservation(0.025, 0.0))
```

The raw gain is a source-loading parameter; the verified configuration applies
the recorded normalized transform and assumed dynamics. Run the app with:

```powershell
.\.venv\Scripts\python.exe -m flybrain.app --seed 7 --control banc
```

## Install (PowerShell)

Tested: Windows AMD64, Python 3.12.10, Pygame 2.6.1 (SDL 2.28.4), pytest 9.1.1.
Metadata allows Python 3.11+, but 3.14.2 had no compatible Pygame wheel.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip>=26.2.2" "setuptools>=84.2.0"
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
```

`requirements.txt` installs the dev extra with [audited constraints](constraints-audited.txt).
The historical environment is recorded in [docs/environment.md](docs/environment.md).

FB-03 neural tests require the optional PyTorch dependency. For the checked CPU
setup, install it from the [official CPU wheel index](https://download.pytorch.org/whl/cpu)
before installing the extra (no CUDA, torchvision or torchaudio needed):

```powershell
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -c constraints-audited.txt "torch>=2.6,<3" "numpy>=1.26,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -c constraints-audited.txt -e ".[dev,neural]"
```

The world and manual app remain usable with the original installation alone.
If the neural extra is absent, run the world/manual cases with `python -m pytest
tests/test_arena.py tests/test_fly.py tests/test_odor.py tests/test_sensing.py
tests/test_simulation.py tests/test_app.py tests/test_world_view.py -k "not neural"
-q`; the neural CLI case itself requires PyTorch.

## Run and test

```powershell
.\.venv\Scripts\python.exe -m flybrain.app --seed 7
.\.venv\Scripts\python.exe -m flybrain.app --seed 7 --control neural
.\.venv\Scripts\python.exe -m flybrain.app --headless --steps 120 --seed 7
.\.venv\Scripts\python.exe -m flybrain.app --headless --steps 120 --seed 7 --control neural
.\.venv\Scripts\python.exe -m pytest -q
```

W/Up: forward. A/Left: left. D/Right: right. Opposing turns cancel; release stops
commanded motion. R resets the same seed/body/timing. Escape/window close quits.
`--steps N` limits positive physics ticks; headless requires it and steps with
a zero manual command.

The default and `--control manual` modes are the temporary keyboard scaffold.
`--control neural` explicitly enables the sensory-neural-motor loop, changes the
caption/HUD to **NEURAL CLOSED LOOP**, and disables movement keys. Reset clears
body/timing, neural state, motor smoothing and the closed-loop accumulator.

The 800 × 600 world uses seconds and radians, +x right and +y down. Heading zero
points right, positive turns clockwise. Each 0.02-second tick turns before moving
and clamps the center by body radius at walls, preserving heading/commanded speeds.
Pure physics never imports Pygame; drawing never steps or consumes randomness.
Interactive rendering targets 60 FPS and caps frame delays to 0.25 seconds, so
simulation lags real time after longer stalls. Replay equivalence requires the
same per-tick commands, not different keyboard polling schedules.

The green marker is a static food odor source, not a consumable reward. Blue is
the left antenna, pink the right. The HUD displays raw concentrations at the
current pose, updated after each tick and reset. The field is
`strength / (distance² + softening²)`, with strength 1000 and softening 20.
Antennae are 10 units forward and 5 units sideways; they are not clamped at walls.
Sampling is separate from the sensory-only observation, which carries no pose
or target coordinates. The manual app does not call the neural encoder or steer
automatically.

## Spiking network, causal control, and live view (FB-03–FB-05)

```python
from flybrain.neural.network import SyntheticNetwork
from flybrain.world.sensing import SensoryObservation

brain = SyntheticNetwork(seed=7)  # CPU float32; topology RNG is local
activity = brain.advance(SensoryObservation(0.025, 0), substeps=20)
print(activity.spike_counts)  # every spike in the 20 ms interval
brain.reset()  # clears voltage, pending spikes and substep count
```

The graph has 100 uniquely labeled neurons in left/right sensory and
interneuron groups and left/forward/right motor groups. It is synthetic,
with same-side excitatory pathways and tonic forward current, designed by
hand. Separate deterministic antenna currents saturate at 2 normalized units
at concentration 0.025. These gains are assumptions, not biological estimates.
No coordinates, world objects or precomputed steering enter inference.

Each 1 ms LIF step integrates passive leak exactly with tau 20 ms; recurrence
uses the previous spike vector and `W[post,pre]` voltage jumps. Threshold
equality fires, hard reset discards overshoot, and there is no refractory
counter. Snapshot tensors are detached copies. Counts retain spikes missed by
the last-step vector. Equations, group sizes and limitations are in the
[M3 plan](docs/plans/M3.md) and [decisions](docs/decisions.md).
In neural mode, each 20 ms world tick samples both antennae before movement,
holds those two values for 20 one-millisecond neural steps, and collects a
`(100,)` CPU `int64` spike-count tensor. The ten indices for each motor group are
summed. A group sum of 10 maps to activity 1.0; larger bursts clamp to 1.0.
Forward uses the forward group, while `right - left` sets clockwise-positive
turn. A fixed 0.5 exponential smoother precedes the bounded command.

For a concrete verified calculation, left-only input `SensoryObservation(0.025,
0)` produces first-tick motor sums `(left=10, forward=0, right=0)`. PyTorch's
tensor indexing and sum produce those integers. Our hand-written decoder gives
raw turn `0 - 1 = -1`, then smoothing from zero gives command turn `-0.5`.
This fixed controller performs forward inference only: there is no autograd
graph or parameter update in this path.

Neural mode opens a 1200 x 600 two-panel view. The world remains 800 x 600;
the right panel shows all 100 synthetic neurons, a limited deterministic edge
subset, recent spike traces, tick/substep time, motor-population activity, and
the decoded command. Its layout groups neurons by their hand-written function.
It is explicitly a diagram, not the anatomical shape of a downloaded fly brain.
Every completed neural interval is observed even when several world ticks occur
between display frames. Traces fade with simulation time, so pausing or drawing
extra frames cannot change them. Details and limits are in the
[M5 plan](docs/plans/M5.md).

## Fixed navigation baseline (FB-06)

Run the renderer-free fixed controller on locked validation and final starts:

```powershell
.\.venv\Scripts\python.exe -m flybrain.training.evaluation --output-dir experiments/fb06-fixed-baseline
```

The command records exact starts, configuration, software/device versions,
per-episode metrics, and split summaries as JSON and CSV. The source-arrival
criterion is fly-center distance at most 20 world units within 12 seconds.
Train starts are reserved for optimization. The [recorded baseline and
its limitations](docs/experiments.md) show poor navigation by the unchanged
synthetic controller.

## First trainable readout experiment (FB-07)

```powershell
.\.venv\Scripts\python.exe -m flybrain.training.readout_experiment --output-dir experiments/fb07-readout --baseline experiments/fb06-fixed-baseline/results.json
.\.venv\Scripts\python.exe -m pytest tests/test_readout.py tests/test_readout_experiment.py -q -p no:cacheprovider
```

The trainer freezes the 100-neuron graph and learns only a 7-to-2 linear
readout with bias: 16 scalar parameters. The seven inputs are filtered mean
spike counts for the existing neuron groups. A hand-written teacher uses only
the two antenna concentrations to specify a desired forward and turn command.
PyTorch computes mean squared error, differentiates that error through the
linear readout, and Adam updates those 16 scalars. The spiking network's hard
threshold and graph do not receive gradients. This is supervised imitation,
not reinforcement learning or a biologically learned connectome.

For one small calculation, let the seven input features be
`[1,1,1,1,1,0,1]`. The readout weight matrix is `(2,7)`: each row combines all
seven features to produce one raw output. Starting from zero weights and bias,
sigmoid maps the first raw output to forward `0.5`, and tanh maps the second to
turn `0`. With teacher target `[0.7,0.3]`, mean squared error is `0.065`.
Autograd gives raw-output gradients `-0.05` and `-0.3`; an active input feature
multiplies these into its two weight gradients. The optimizer then changes the
weights. This update is separate from advancing simulated neural time.

The [recorded attempt](docs/experiments.md) lowered imitation loss but did
not improve held-out food arrival. The readout cannot infer different turns
from identical spike features at all eight train starts. The checkpoint, curve,
and all episode results are under `experiments/fb07-readout/`. At that stage, M6 navigation
learning was still unverified.

The 2026-09-22 charge repair passed all 261 tests and both manual and neural
120-tick headless smokes. Existing experiment destinations and checkpoint files
are now protected against accidental overwrite. Use
`-p no:cacheprovider` when sandbox permissions block pytest cache writes. The full
suite emits one upstream Pygame `pkg_resources` deprecation warning.
FB-01's manual visible checks were accepted by the user; FB-02's source,
antennae and sensory HUD were visually inspected. The FB-05 two-panel output
was inspected from an exact-size rendered frame; live animation remains to be
reviewed interactively. Visual polish is deferred. See
[project status](PROJECT_STATUS.md).

## Original BANC endpoint-readout experiment (FB-10)

The [FB-10 protocol](docs/plans/FB-10.md) adds one small 8->16->2 readout above
the frozen83-cell BANC circuit. Inputs are only its eight endpoint spike-rate
EMAs. Train-only normalization and178 readout parameters are separate from the
sourced wiring and our assumed neuron dynamics. The prepared artifact is
`experiments/fb10-banc-readout/protocol.json`; preparation performs zero
scientific world ticks and zero experiment optimizer updates.

Preparation command (already consumed; do not repeat at the same destination):

```powershell
.\.venv\Scripts\python.exe -m flybrain.training.banc_protocol
```

Training command (executed 2026-09-27; consumed, do not rerun):

```powershell
.\.venv\Scripts\python.exe -m flybrain.training.banc_training --protocol experiments/fb10-banc-readout/protocol.json
```

The single fit completed 500 updates and 25 episodes including replay,
14,131 world ticks and 282,620 neural steps. Turn MSE 0.06679 missed the required
0.02225; trained arrivals 2/8 missed 7/8, versus untrained 1/8 and fixed 0/8.
Integrity and exact replay checks passed; the suite then passed 396 tests. This
original attempt is closed negative and had no held-out evaluation. Subsequent
recovery and its failed first fresh assay are recorded in the current status.

A receipt prevents rerunning a partial or completed attempt. The app still uses
the fixed BANC decoder. See the [results](experiments/fb10-banc-readout/results.json),
[plot](experiments/fb10-banc-readout/training_summary.png) and
[current acceptance proposal](docs/plans/FB-10-FROZEN-CANDIDATE-ASSAY.md).
