# Brain controls verification - 2026-10-02

One agent, existing cached geometry, teacher preview. No fitting or reserved evaluation.
The accepted black/blue-violet/warm/cyan appearance is retained. The default brain
framing remains 1.10 mm at (0.49,-0.17,0.145), cord hidden, background anatomy on.

## Checks and commands

- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests -k flight3d --junitxml=.superpowers/navigation-2026-10-02/targeted.xml`
  **61 passed, 567 deselected**, 24.95 s, two existing upstream warnings.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --junitxml=.superpowers/navigation-2026-10-02/full.xml`
  Fresh completed full suite: **628 passed, zero skips**, 147.37 s, two existing
  Pygame/PyTorch deprecation warnings. XML and complete logs are saved.
- `.\.venv\Scripts\python.exe scripts/benchmark_flight3d_viewer.py --view repaired --frames 120 --output .superpowers/navigation-2026-10-02/navigation-timing.json`
  Exit 0 recorded in `timing-exit.json`. Sequential, 1440x900, 120 synchronized
  frames/condition, 60 Hz requested pacing, Intel Iris Xe, Python 3.12.10,
  CPU PyTorch 2.14.0. No concurrent previews or tests.
- `.\.venv\Scripts\python.exe .superpowers/navigation-2026-10-02/capture_controls.py`
  Saved and inspected initial, settings, inspector, explicit focus and deep zoom
  at 1440x900 and 1000x700, plus narrow Views menu. Original front/side/top/cord
  source morphology remains covered by the targeted renderer test.

## Navigation evidence

Numerical tests cover continuous pole crossings from front/side/top, orthogonal
up/direction vectors, unit quaternions, 60 ms rotation and 120 ms zoom constants,
0.03 mm distance, adaptive near clipping, scale-aware pan without changing zoom,
explicit smooth fit and preset pan/pivot/zoom preservation. Fixed-target
responses at 30/60/120 Hz agree to 1e-12; a continuous 120x80 px/s drag differs
by less than 0.01 rad (0.6 degree) between 30 and 120 Hz because of mouse sampling.

Actual renderer/event tests cover exact camera matrix and current/target state
preservation across cord toggles, both camera wheel routes, scrolling/gutters,
menu input blocking, pixel-equivalent drags in two actual window sizes, discarded
resize gestures, release/Shift-release/focus-loss handling, source selection and
missing geometry, inspector controls, pause, new trial, replay and exit. Existing
serial-vs-rendered sessions compare snapshots and voltage tensors exactly.

The offscreen buffer does not support native `setSize` without Panda's resizable
buffer flag. Size equivalence is verified in two independently created renderer
windows, with gesture-size invalidation tested separately; live window resizing
and desktop focus remain a separate human check. No mock numerical checks.

## Isolated timing

| Condition | Mean ms | p95 ms | Max ms | Paced FPS | Simulated s / wall s |
|---|---:|---:|---:|---:|---:|
| paused | 2.86 | 4.44 | 5.64 | 58.5 | 0.000 |
| running | 4.23 | 5.87 | 13.85 | 58.4 | 0.973 |
| orbiting | 2.72 | 3.91 | 6.09 | 58.6 | 0.000 |
| selecting | 3.48 | 9.05 | 12.11 | 58.6 | 0.000 |

Navigation p95 is below 33 ms. Paced FPS is about 58.5, so steady 60 FPS is not
established. Running throughput was about 0.97 simulated seconds per wall second
in this isolated run; this is machine/load-specific display scheduling evidence,
not a new scientific result. Numerical timestep sizes remain unchanged.
The first benchmark also wrote a complete record, but PowerShell interpreted
Panda stderr as NativeCommandError. A serial subprocess rerun captured native
exit 0 and the final timings above. Both first/final records are preserved.

## Preservation and local review

SHA256 inventory: 2366/2367 existing files unchanged. The only modified
existing source is `anatomical_viewer.py`; camera/UI modules are additive. All
experiments, geometry caches, previous viewers/navigation, scientific sources
and October 1 checkpoints/journals are byte-identical. That consumed fit remains
incomplete at 1,050 updates. No fit, evaluation, download, commit or push occurred.

Locally reviewed against the user's complete specification: no simulation
inputs/dependencies in the camera, independent world/brain state, geometry-only
cord toggles, explicit framing, default-off retained connections, removed N/C,
responsive Views menu and meaning of background anatomy. Loading newly created
source batches respects the currently selected branch brightness.

## Live review

Teacher preview was launched with `pythonw.exe -m flybrain.flight3d --seed 970000`.
A native FlyBrain window/process is present. The Windows inspection helper twice
reported that its native pipe was unavailable; no native UI observations are
claimed. A short review was requested from the user. **Awaiting their observations**,
including native drag/resize/focus behavior. Saved frames and injected events
are distinct from live acceptance.


### User feedback - 2026-10-02

The user confirms panning and scrolling are good in the live preview. Other
native checks are not yet confirmed. They clarify the intended comparison:
whole fly brain on, exact project nerves off. Current off mode isolates the
project's cached source branches; current on mode adds the registered surface
and reference points, not all full-brain branch skeletons. That distinction
must remain explicit; a full-neurite atlas view is not implemented or authorized
for acquisition by this clarification alone.


### Spike/display diagnostic - 2026-10-02

The user observed dim/silent-looking project branches. A bounded headless preview
diagnostic compared actual per-cell spike counts with the unchanged default
`NeuronIllumination` rule. Hypothesis: some project cells are genuinely silent in
this window, while some firing cells lack highlights. Budget/stop: seed 970000,
teacher mode, 100 control updates = 2 simulated seconds, no fitting or reserved
cases. It completed before the user's interruption; the saved result was read
and reconciled afterward, without rerunning it.

Command: `.\.venv\Scripts\python.exe .superpowers/navigation-2026-10-02/check_spike_display.py`.
Result: `.superpowers/navigation-2026-10-02/spike-display-diagnostic.json`.
Actual spikes: **530/589** cells fired at least once; **59/589** produced none.
Of the 361 cells eligible for warm emphasis with cord hidden and caches resident,
304 fired, 57 were silent, and **281 fired without ever receiving a highlight**.
Only 23 distinct cells were highlighted over the entire window; never more than
12 in one frame. One source vision cell, `720575941546280072`, produced **403
spikes** but received **zero highlights**. Four spiking cells had no retained
outgoing edge, and this display score cannot show their firing at all.

Warm score is filtered *delivered outgoing current averaged per retained outgoing
edge*, not the cell's own spike count. A 1.5% display-score threshold, top-12 cap,
geometry availability and cord visibility also affect emphasis. A dim branch
therefore cannot establish neural silence. The 59 zero-count cells are silent
only in this particular two-second window; their individual causes and lifetime
activity were not assessed. Model membership is fixed independently of whether a
cell fires in a particular trial. No visual or neural behavior was changed.

Read-only reconciliation confirmed 100 frames, exact 2.0 s, 589 rows, all summary
counts, highlight cap and the example's counts. This is display-debug evidence,
not a teacher gate, learned behavior assessment or scientific acceptance result.
