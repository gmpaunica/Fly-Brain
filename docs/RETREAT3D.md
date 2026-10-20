# Idle encounter lab — partial delivery, escape gate failed

The additive `python -m flybrain.retreat3d` launcher starts with no object and an
engineered idle cycle: takeoff toward 10 cm, cruise at a desired 6 cm/s for a
seeded 3–6 seconds, safe landing and a seeded 2–5 second rest. These are modeling
defaults. Existing `flybrain`, `flight3d` and `encounters3d` launchers are unchanged.

Spawn flower / Spawn predator replaces the sole object while preserving fly
pose, velocity and local IMU estimates. Flowers use the existing visual teacher;
grounded flowers receive engineered takeoff before teacher handoff. Removing a
supporting flower removes contact without teleporting the fly. Predator movement
uses W/S for arena ±Y, A/D for ±X, Q/E to lower/raise, normalized diagonals,
8 cm/s maximum and arena clamping. Movement and relative swept contact occur in
fixed 5 ms physics steps. Pause and focus loss clear keys and pending input credit.
R starts a fresh no-object trial. Backspace replays all recorded inputs/spawns to
their recorded endpoint and pauses. Camera/display preferences persist.

```powershell
.\.venv\Scripts\python.exe -m flybrain.retreat3d --seed 1000000
.\.venv\Scripts\python.exe -m flybrain.retreat3d --headless --seed 1000000 --steps 600
.\.venv\Scripts\python.exe -m flybrain.retreat3d --headless --replay-log artifacts/retreat3d-2026-10-06/headless-actions.json
```

`--record <new.json>` saves the current trial on exit; `--capture <directory>`
saves offscreen review frames and an interaction log. Neither command runs an
experiment or optimizer. Recorded actions bind seed, fixed timestep and circuit
hash; reproducibility applies to the recorded software/device environment.

## Closed scientific result

The locked hypothesis was that restoring the original 16 motion inputs alongside
unchanged predator vision would supply sufficiently directional wing spikes.
All 589 cells, 4,139 retained source edges, weights, leak, thresholds, subtractive
reset and 1 ms propagation remain fixed. The actual brain has restored motion plus
vision; an independent motion-only reference receives identical IMU observations.
Only actual spikes color anatomy. Subtraction and cosine matching are explicitly
diagnostic/control calculations, not additional cells or learned neural activity.

Exactly **48 calibration + 96 validation cases** ran once: **216 simulated
seconds, 432,000 combined actual/reference neural steps**, zero optimizer updates.
Five 200 ms signed post-onset windows were pooled across six calibration cases
per direction; eight normalized templates were frozen before validation.
Only **9/96 validation cases passed**, versus ≥90/96 and ≥5/6 per azimuth.
The 168.75° group passed 3/6, 191.25° passed 6/6, all other groups passed 0/6.
Only 88/480 validation windows had norm ≥1; 20/480 met decoder confidence gates.
Two calibration directions produced zero pooled templates. Actual/reference
calibration wing totals were 4,804/4,752; validation totals 9,576/9,504. All 3,600
motion-baseline intervals had identical paired spikes.

The experiment **closed at the failed directional gate**. No tuning, retry,
alternate decoder, new cells, weight changes or closed-loop assay followed.
The 48 closed-loop cases, grounded neural takeoff, opposite-heading escape,
weak-threat termination/braking, stable recovery and visual-silent escape controls
were **not run or delivered**. Idle movement is engineered, not neural escape.
The frozen failed decoder is available only as a labeled passive diagnostic.
It cannot actuate the fly; the viewer always labels neural escape unavailable.

The RGB-only monitor implements a contrast-weighted visible solid angle and an
equivalent circular diameter using the existing eye geometry. It supplies only
apparent size/positive growth, never predator range or heading. Tests cover rear
views, seams, ±20° elevation, disappearance, reappearance and shrinking/repeated
approaches. A five-sample 100 ms average and fifteen-step 300 ms weak timer retain
the specified engineering thresholds; their closed-loop behavioral usefulness
remains unverified because the earlier gate failed.

The [flight circuit study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977095/)
supports looming responses and flight-state modulation. It does not validate our
motion formula, idle schedule or stopping thresholds. The
[BANC paper](https://www.nature.com/articles/s41586-026-10735-w) supplies anatomical
wiring/annotations, not these neuron dynamics or a complete behavioral simulator.

## A small PyTorch calculation

The brain stores a `[589]` voltage tensor and previous-spike tensor. A retained
edge identifies source `pre` and destination `post`; its weight multiplies the
source's previous spike, then `index_add_` adds this increment at the destination.
For example, destination voltage `.5` leaks to `.4950249` in 1 ms; a delayed
source spike with weight `.1` raises it to `.5950249`. Threshold is 1, so it
does not spike. This arithmetic is PyTorch; the leak/threshold/reset rules are
our assumed model. There is no autograd or optimizer update in this experiment.

For a 48-wing 200 ms window, actual counts `[3,1,...]` and motion-only counts
`[2,1,...]` yield the signed vector `[1,0,...]`. Each `[8,48]` template row is one
direction. Matrix multiplication with the normalized `[48]` vector produces
eight cosine scores; norm ≥1, best score ≥.8 and margin ≥.1 define usable output.
Spiking wing cells by itself does not establish a reliable threat direction.

## Evidence and next task

[Locked plan](plans/RETREAT3D.md),
[results](../experiments/retreat3d-motion-repair-2026-10-06/results.json),
[offline audit](../experiments/retreat3d-motion-repair-2026-10-06/audit.json),
[verification](verification/retreat3d.md). Configuration, frozen templates,
source-byte archives, hashes, 10,800 JSONL/CSV intervals and software/device are
preserved in the fresh experiment directory. Interactive action logs and frames
are separate development artifacts. These placements are now historical.

One next task / Sol High: review this closed result and agree one sourced
recovery specification with a new hypothesis/budget before any further escape
experiment. Native review of the partial idle/input lab remains pending; no
automatic scientific continuation is authorized.
