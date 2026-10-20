# Exclusive flower / predator 3D trials

```powershell
.\.venv\Scripts\python.exe -m flybrain.encounters3d
.\.venv\Scripts\python.exe -m flybrain.encounters3d --kind predator --seed 980000
.\.venv\Scripts\python.exe -m flybrain.encounters3d --headless --kind predator --steps 250 --seed 980000
```

**Spawn flower** and **Spawn predator** each start a fresh trial containing just
that object. R makes a new trial of the selected kind; Backspace replays the
current seed/placement exactly. Both reset the fly, controller, brain, interval
transmissions and display history. Camera, selection, activity switches, scope,
cord and display preferences persist. Paused trials stay paused after spawning.
The encounter launcher starts with All activity enabled; the old flight launcher
still starts odor-only. Spawn buttons remain visible at the bottom of the window.

Flowers retain the existing hand-written teacher and landing mechanics exactly.
Their brain activity is observational. The cyan sphere is a stationary predator
proxy, with the same appearance in ray-sampled RGB and the world renderer.
Recognizing cyan is an engineered color-selective sensory assumption. Controllers
receive only the existing `FlightObservation`, including `[2,24,48,3]` RGB images.
The environment-only scene seed, kind and XYZ never enter controller inference.

## Escape characterization failed; hover preview only

The current 589-cell, 4,139-edge source registry remains unchanged, including its
96 LPLC4 cells, bilateral DNp03 and wing endpoints. Eight directions at two image
sizes produced **zero wing-endpoint spikes in all 16 intact one-second trials**;
the maximum sampled endpoint voltage was **0.229664**, below threshold 1.
DNp03 fired in some placements but did not consistently reach threshold itself.
Front and rear are physically symmetric binocular presentations. Matched visual
silencing and DNp03 silencing were measured using the exact same eye images.

The symmetric, synthetic full-field cyan diagnostic produced 1,838 wing spikes
intact, 1,476 with DNp03 silenced, and zero with visual inputs silenced. This
supports a DNp03 contribution under these assumed dynamics and this diagnostic;
it also demonstrates remaining visual-to-wing routes. It establishes no
DNp03-specific behavioral necessity, biological fidelity or physical retreat.

No fixed escape decoder was selected. The predator preview labels **Avoidance
unavailable** and uses an explicitly engineered IMU hover stabilizer. Its
actuators do not read RGB or neural threat activity. Twenty-four matched five-second
closed-loop trials produced zero separation change, zero collisions and no escape
response. These are stable-hover measurements, not an avoidance success rate.
Retreat, hover-after-retreat and reappearance/growth retrigger behavior remain
unimplemented after the gate failure. No subjective fear or learned behavior is
claimed. This result concerns the current spike-history motor representation;
it does not prove that source anatomy or subthreshold activity can never support
a differently calibrated model.

The top-right signal shows **recorded DNp03 firing rates** and actual interval
wing spikes. It is independent of RGB contrast and actuator commands. Every
neuron's spiking and edge transmission remain recorded, even when a display
class is switched off. Vision and structural activity can therefore appear
while the fly stays still. DNp03 belongs to Structural relays in the existing
role-derived display index, rather than the Descending display class.

## Modeling and a small PyTorch calculation

The front end computes `min(G,B)-R`, rejects contrast <= .05, and drives only
existing LPLC4 input cells. Each side's cells retain the previous ID-ranked 8x6
receptive-field assignment. ID order is an assumed retinotopy, not a biological
spatial map. Rate is `240*clip(.75*patch_peak + .25*patch_mean +
.5*positive_growth_peak, 0, 1)` Hz. Patch occupancy and positive image changes
strengthen stimulation as the image expands. No other sensory drive enters the
escape probe, keeping pathway characterization independent of motion baselines.

PyTorch stores voltage, previous spikes and sensory phase as 589-element tensors.
`pre` and `post` index 3,636 non-self source edges; 503 self edges stay in the manifest.
There is no dense matrix here: `voltage.index_add_(0, post, weights*previous[pre])`
sums directed inputs at their destinations. Its dense equivalent has **rows =
destinations, columns = sources**, multiplying a source-spike vector.

For a concrete phase example, a 240 Hz input with phase .99 adds .24 during 1 ms,
emits one spike and leaves phase .23. The destination sees that spike on the next
millisecond. One actual edge is LPLC4 `720575941393550596` to left DNp03
`720575941539809997`: count 13 out of 618 non-self incoming synapses gives
assumed weight `.9*13/618 = .018932`. A DNp03 cell at voltage .5 first leaks to
`.5*exp(-.001/.1) = .495025`; that spike raises it to `.513957`, still below
threshold. Voltage >= 1 spikes and subtracts 1.
These leak/threshold/reset rules and `.9*edge_count/total_incoming_count` weights
are hand-written assumptions implemented by tensor calculations. Input cells are
phase encoders rather than LIF cells. Forward stepping runs under `no_grad`;
there is no loss, autograd, gradient, optimizer or learned motor mapping here.

The local hover stabilizer integrates the unchanged body angular-rate/specific-
force senses, assuming the declared initial rest state, gravity and actuator
calibration. It estimates relative height and velocity and damps motion. It never
reads environment coordinates or supplies an escape direction.

## Specific recovery proposal; not executed

First review the sensory/physiology assumptions: sparse-field drive is diluted by
normalization across all retained incoming edges, and source counts do not supply
measured gain, signs, receptive fields or motor calibration. Do not simply lower
thresholds until a behavior happens. A sourced extension to assess is the
**LPLC4 → DNp26 / DNp57 → saccade DN network → contralateral i1/b3 wing motors**
described in Figure 5G of [Croke et al., Current Biology](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977095/).
The same paper reports other visual inputs to DNp03 and incomplete loss of
avoidance after DNp03 silencing. Its Figure 5G network motivates a specific
candidate extension; it does not guarantee this simulator will retreat.

DNp26 is already present bilaterally in the downloaded subset; DNp57, LC4 and
LPLC1 are absent. A future bounded task would source the missing DNp57 identities
and actual retained paths from a pinned connectome, assess existing DNp26 routes,
and declare spatial tuning and physiology calibration before motor decoding.
Any added cells must serve those traceable paths. No extension, fitting or
training started in this task. Recommended effort: **Sol High**.

Evidence: [characterization JSON](../experiments/encounters3d-characterization-2026-10-05/results.json),
[responses CSV](../experiments/encounters3d-characterization-2026-10-05/responses.csv),
[closed-loop measurements](../experiments/encounters3d-characterization-2026-10-05/closed_loop_results.json)
and [implementation plan](plans/ENCOUNTERS3D.md). These development cases are now
historical; renaming a future task does not make them unseen evaluation data.

Run encounter/3D tests: `python -m pytest -q tests -k "encounters3d or flight3d"`.
Full suite: `python -m pytest -q`.
Save offscreen frames: `python -m flybrain.encounters3d --seed 980000 --capture PATH`.
The scientific record commands refuse overwriting an existing run; they are
consumed records, not instructions to rerun this experiment.
