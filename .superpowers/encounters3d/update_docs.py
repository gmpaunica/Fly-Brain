"""Append encounter records without rewriting historical documentation entries."""
from pathlib import Path

def append(path, text):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write('\n\n'+text.strip()+'\n')

append('docs/architecture.md', '''### Exclusive encounters extension (2026-10-05)

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
reserved assay occurs. See [encounter guide and recovery proposal](ENCOUNTERS3D.md).''')

append('docs/decisions.md', '''## 2026-10-05 — Exclusive encounters and closed negative escape gate

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
  [results](../experiments/encounters3d-characterization-2026-10-05/results.json).''')

append('docs/experiments.md', '''## Exclusive 3D predator pathway characterization — 2026-10-05

One predeclared, training-free characterization tested the unchanged 589-cell,
4,139-edge BANC slice. Sixteen physical RGB presentations (eight azimuths at two
sizes, including symmetric front/rear) plus a synthetic bilateral full-field
diagnostic were each held for one second under intact, visual-silent and
DNp03-silent conditions: 51 presentations, 51,000 neural steps, zero optimizer
updates. All 16 intact physical cases had zero wing endpoint spikes; maximum
sampled wing voltage was 0.229664, below threshold 1. Source binding and complete
per-cell counts are recorded. The fixed spike-history escape decoder gate closed.

The synthetic full-field diagnostic produced 1,838 intact wing spikes, 1,476 with
DNp03 silenced and zero with visual inputs silenced. DNp03 contributes to those
modeled responses but is not their sole route. Twenty-four matched five-second
closed-loop trials (eight directions x three conditions, 120,000 neural steps)
measured zero separation gain, zero collisions and no escape response. They use
the explicitly labeled hover-only failed-gate preview; they are not avoidance
successes. Recorded total across both phases: 171,000 neural steps, zero updates,
no learning rate or checkpoint. No historical fit/assay was resumed.

These are development cases and now historical. The result concerns the current
engineered visual drive and assumed spike-history motor model, not the anatomical
circuit's ultimate capacity. No hidden visual steering rule, gain sweep, decoder
fit or sourced extension followed the failed gate. A specific DNp26/DNp57 saccade
network extension and physiological-calibration review is proposed, not executed.
See [guide](ENCOUNTERS3D.md),
[config](../experiments/encounters3d-characterization-2026-10-05/config.json),
[static JSON/CSV](../experiments/encounters3d-characterization-2026-10-05/results.json),
[closed-loop measurements](../experiments/encounters3d-characterization-2026-10-05/closed_loop_results.json).''')

path=Path('PROJECT_STATUS.md')
text=path.read_text(encoding='utf-8-sig')
text=text.replace('# FlyBrain Current Status\n','# FlyBrain Current Status\n\n'+'''## Exclusive 3D encounters — 2026-10-05

Additive `python -m flybrain.encounters3d` provides Spawn flower / Spawn predator,
one stationary object per trial, unchanged teacher-controlled flowers, actual
recorded visual-circuit activity, R/new selected object and Backspace/exact replay.
Reset clears body/controller/neural/display history and retains pause/camera/
display preferences. The predator is a high-contrast cyan sphere in world and eyes.

**Avoidance gate failed:** all 16 intact physical image cases produced zero wing
endpoint spikes. No escape decoder or retreat/retrigger policy was selected.
Predator mode explicitly labels unavailable avoidance and IMU hover scaffolding.
Synthetic full-field drive: 1,838 intact / 1,476 DNp03-silent / zero visual-silent
wing spikes; DNp03 contributes but is not the sole route. All 24 five-second
closed-loop cases hovered with zero separation gain/collisions and absent retreat.
No training, optimizer updates, historical assay or interrupted fit continuation.

Fresh verification and live-review limits are recorded in
[encounter verification](docs/verification/encounters3d.md).
[Guide / sourced recovery proposal](docs/ENCOUNTERS3D.md),
[active task](docs/plans/ENCOUNTERS3D.md).

''',1)
text=text.replace('## Current Milestone\n','## Current Milestone\n\n- Exclusive encounters: infrastructure and actual visual-circuit probe delivered;\n  directional neural avoidance failed the characterization gate and remains\n  unavailable. No retreat, retrigger or learning claim.\n',1)
text=text.replace('## What Works\n','## What Works\n\n- Exclusive flower/predator scene, RGB visibility/occlusion/expansion, source-ID\n  preservation, teacher parity, deterministic reset/replay, pause barrier and\n  passive rendering. Recorded DNp03 and wing signals are independent of motor\n  commands; explicit predator color matches eye images in the framebuffer.\n',1)
text=text.replace('## In Progress\n','## In Progress\n\n- Encounter native UI review is pending: the computer-use helper failed to\n  connect. Offscreen event/render review does not substitute for live acceptance.\n  No scientific recovery run is in progress.\n',1)
text=text.replace('## Known Issues / Limits\n','## Known Issues / Limits\n\n- Current predator circuitry does not support a verified spike-history escape\n  decoder under declared inputs/dynamics; UI is a hover-only circuit probe.\n  Retreat and reappearance/growth retrigger remain unavailable. Physiological\n  signs/gains/retinotopy and a subthreshold decoder are not established.\n',1)
text=text.replace('## Important Decisions\n','## Important Decisions\n\n- User-authorized predator extension is exclusive/additive. Keep controllers\n  sensory-only and close the failed neural gate without hidden steering or\n  training. Source proposal: review LPLC4 -> DNp26/DNp57 -> saccade DN -> i1/b3\n  routes and physiological calibration as a separate bounded task.\n',1)
start=text.index('## Next Task\n'); end=text.index('## How To Run\n',start)
text=text[:start]+'''## Next Task

**One next task / Sol High: specify a bounded sourced escape-path recovery.**
Review the closed negative evidence and the concrete DNp26/DNp57 pathway proposal
in [the encounter guide](docs/ENCOUNTERS3D.md), including spatial/physiological
calibration and a predeclared directional motor gate. No extension or training is
automatically authorized. Native encounter controls and earlier activity checks
also remain pending user review; they are not scientific recovery experiments.

'''+text[end:]
text=text.replace('## How To Run\n','## How To Run\n\nExclusive encounters: `.\\.venv\\Scripts\\python.exe -m flybrain.encounters3d`.\nUse `--kind predator --seed 980000` for a reproducible hover-only circuit probe.\nSpawn buttons select one object; R/new selected kind, Backspace/exact replay.\nThe existing launchers below are unchanged.\n',1)
text=text.replace('Full connectome, predators, fluid dynamics, biomechanics and large RL systems remain deferred.',
                  'Full connectome, fluid dynamics, biomechanics and large RL systems remain deferred. Exclusive stationary 3D predator trials are now explicitly authorized.')
path.write_text(text,encoding='utf-8')
