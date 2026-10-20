# Actual firing display verification — 2026-10-04

Implementation began October 3 and finished October 4, using one agent and cached
geometry. The uncapped actual-firing view is implemented. The user subsequently
said this view is good and should remain an option, while requesting a planned
smaller view based on a defined circuit. Individual native pause/reset/navigation
checks have not been confirmed by that feedback.

## Behavior and scientific preservation

`SpikeRecorder` observes every successful 20 ms `FlightSession.advance`, including
steps between display refreshes. It publishes the original immutable scientific
snapshot together with episode identity, cumulative counts, interval-end last
activity and a 200 ms exponential average in spikes/s. Both background stepping
and deterministic capture use the same observation path. Same-seed replay starts
a new display episode. Drawing, selection, loading and visibility never add spikes.

Every firing owner with resident branches in enabled layers gets warm color.
Static endpoints retain ownership through splitting, chunking and deferred cord
loading, sampling one RGBA texel per neuron. Explicit RGBA upload and real rendered
sentinel pixels verify color and alpha ordering. No activity overlays scale with
the number of firing cells. Cyan selection, idle presets and camera behavior remain.
Pulses decay with a 200 ms time constant, end at 600 ms, and freeze when paused.
They indicate whole-neuron activity, not measured anatomical propagation.

Regressions cover more than twelve firing cells, no outgoing edges, genuinely
silent cells, missing geometry, mixed/deferred branches, skipped display intervals,
pause, replay, new trial and repeated passive draws. Matched completed-step streams
at 30/60/120 display Hz have identical frames, pulse strengths, GPU owner colors
and visible counts at shared simulation times, including subsequent silence.
These checks establish simulated-time equality, not a wall-time throughput guarantee.

Teacher, fixed and a disposable frozen trained readout produce exactly the same
snapshots, neural state, controller state and parameters with/without recording.
Recorder operations preserve Python, NumPy and Torch global RNG state. Trial RNG
is local to trial construction and discarded; there is no persistent session RNG
to compare. The final hash audit found **2,365 protected existing files unchanged**
and zero protected changes. Of 2,369 inventoried existing source/data/experiment
files, only the four authorized display/pump source files changed. Neural, sensing,
teacher, scientific contracts, cached geometry and the incomplete **1,050-update**
experiment remain byte-identical. No fitting or reserved scientific evaluation ran.

## Fresh required suites

Executed sequentially using `.venv/Scripts/python.exe` (Python 3.12.10), with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and JUnit paths supplied through
`PYTEST_ADDOPTS`. Product/test code did not change after these final suites.

| Command | Observed result | Pytest duration | Exit |
|---|---|---:|---:|
| `python -m pytest -q tests -k flight3d` | 76 passed, 567 deselected, 2 warnings | 47.47 s | 0 |
| `python -m pytest -q` | 643 passed, zero skips, 2 warnings | 145.88 s | 0 |

Warnings are the existing Pygame `pkg_resources` and Torch `jit.script`
deprecations. Wrapper wall durations were 52.95 s and 151.31 s. Logs, XML,
commands, environment and exits are in
[final-targeted.log](../../.superpowers/spike-activity-2026-10-03/final-targeted.log),
[final-full.log](../../.superpowers/spike-activity-2026-10-03/final-full.log) and
[final-test-exits.json](../../.superpowers/spike-activity-2026-10-03/final-test-exits.json).
Earlier passing runs and red/green regression evidence are retained separately.

## Isolated navigation timing

1440×900, seed 970000 teacher, Intel Iris Xe, Torch 2.14.0 CPU, 12 warmup and
120 measured frames per condition, 60 Hz pacing. Cached geometry was drained;
warmup used 100 observed simulation intervals. No preview or test suite ran
concurrently. Frame work includes drawing/synchronization and selection where
applicable; pacing sleep is excluded. JSON contains means, p95, maxima, pause
barriers, simulation timings and scene counts.

| Condition | Previous hidden p95 | Actual firing hidden p95 | Actual firing shown p95 |
|---|---:|---:|---:|
| Paused | 4.44 ms | 4.65 ms | 5.69 ms |
| Running | 5.87 ms | 9.39 ms | 9.39 ms |
| Orbiting | 3.91 ms | 4.61 ms | 4.64 ms |
| Selecting | 9.05 ms | 12.16 ms | 11.80 ms |

The passing hidden recording has **359 visible pulses / 75 source batches**;
shown has **527 / 138**. Geometry nodes are 79/142 without selection and 81/144
with selection. Both recordings meet the hundreds-active setup and p95 <33 ms
gates. Paced display is 57.6–58.9 FPS; running simulation throughput is
1.011/1.014 simulated seconds per wall second. There is no matched historical
shown-cord baseline.

Failures remain part of the evidence. Initial hidden selection p95 was 38.54 ms;
profiling found repeated graph scans/inspector formatting on unchanged frames.
Static edge counts and inspector publication/status caching fixed this, with a
red/green regression. The first post-fix shown run failed at p95
38.05/52.98/31.63/36.04 ms. Its cause is **not established**. A focused unchanged
6,865,928-segment/138-batch rendering comparison and an identical isolated repeat
passed without a subsequent product change. The passing shown paused maximum
was still 33.78 ms. This establishes passing sampled gates with observed timing
variability, not uniformly sub-33 ms work or steady 60 FPS.

Records:
[hidden](../../.superpowers/spike-activity-2026-10-03/final-timing-hidden.json),
[passing shown repeat](../../.superpowers/spike-activity-2026-10-03/repeat-timing-shown.json),
[failed shown](../../.superpowers/spike-activity-2026-10-03/final-timing-shown.json),
[focused comparison](../../.superpowers/spike-activity-2026-10-03/cord-profile.json).
Run the benchmark with `python scripts/benchmark_flight3d_viewer.py --view repaired
--cord hidden --output <path>`; substitute `shown` for the other condition.

## Recorded activity and visual review

The new two-second diagnostic reproduces **530/589 fired, 59 silent**: 50
unstimulated visual cells, six other self-only cells and three with silent
upstream cells. These disjoint categories concern this window, not permanent
silence. Thirty-two of the visual cells also have only self-connections; the
first reporter assertion double-counted this overlap and was corrected without
changing simulation. All **56 firing metadata-cord cells with resident brain
branches** are included when cord is hidden. Four missing exports remain explicit.

Fifteen PNGs were saved at 1440×900 and 1000×700. Agent inspection included idle,
warm front/side/top, brain-and-cord, cyan selected inspector, silent and missing
geometry views, plus narrower inspector/missing layouts. The first Windows image
path attempt wrote no files despite a successful reporter exit; final capture
uses Panda's OS-path conversion and asserts image write/file existence. Final
reporter exit is zero. The numerical inspector stays readable in the scroll panel.
[Frames](../../.superpowers/spike-activity-2026-10-03/frames/1440-activity-front.png)
and [per-cell diagnostic](../../.superpowers/spike-activity-2026-10-03/activity-diagnostic.json)
are automated/saved-image evidence.

Launched `.venv/Scripts/pythonw.exe -m flybrain.flight3d --seed 970000` for live
review after timing, recorded PID 29368. Logs have no traceback but include
`SetForegroundWindow() failed!`; launch alone did not establish foreground display.
The user's subsequent feedback approves the full-activity appearance and requests
it remain selectable alongside a defined-circuit mode. It does not confirm every
individual Space/V/R/Backspace or resize/focus check. Native automation is disabled.
Pause before reset/replay to inspect cleared counters before fresh spikes arrive.

## Handoff

Full firing is implemented and automatically verified; native interaction items
remain individually unconfirmed. The next task is **review the selectable All
activity and Focused circuit plan / Sol Medium**, retaining all-neuron recording
and making subset membership explicit. No further learning run is authorized.
Screen refresh rate no longer decides which spikes the display records.
