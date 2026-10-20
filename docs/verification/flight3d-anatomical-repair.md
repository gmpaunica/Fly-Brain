# Anatomical repair verification - 2026-10-02

One agent; CPU teacher preview. No new experiment, fit or reserved evaluation.

## Performance

Sequential offscreen measurements on Intel Iris Xe, Python 3.12.10, PyTorch 2.14.0+cpu, 1440x900, seed 970000, 120 synchronized frames per condition. Same loaded cache, presets, 50-step warm-up and 60 Hz pacing request. The two historical views share 0.96 mm brain-focus distance; the repaired default uses 1.10 mm for silhouette margin. Timings include selection, UI, activity and synchronized GPU frames; sleep pacing is excluded from frame latency and included in wall throughput. Prior views step inside the frame; repaired view uses a serial worker. Simulation has unchanged 20 ms controls/5 ms physics/1 ms neural steps. These are short benchmark samples, not a native live FPS guarantee.

| View | Condition | Mean ms | p95 ms | Max ms | Paced FPS | Sim seconds / wall second |
|---|---|---:|---:|---:|---:|---:|
| previous-anatomy | paused | 21.52 | 25.72 | 29.41 | 46.4 | 0.000 |
| previous-anatomy | running | 52.33 | 65.47 | 92.48 | 19.1 | 0.318 |
| previous-anatomy | orbiting | 25.48 | 30.25 | 43.85 | 39.2 | 0.000 |
| previous-anatomy | selecting | 26.71 | 32.54 | 93.43 | 37.4 | 0.000 |
| network | paused | 76.72 | 90.51 | 106.62 | 13.0 | 0.000 |
| network | running | 115.07 | 134.91 | 196.51 | 8.7 | 0.145 |
| network | orbiting | 94.32 | 129.26 | 142.63 | 10.6 | 0.000 |
| network | selecting | 89.58 | 118.49 | 148.27 | 11.2 | 0.000 |
| repaired | paused | 4.01 | 5.10 | 5.61 | 58.8 | 0.000 |
| repaired | running | 9.60 | 14.64 | 41.19 | 55.8 | 0.567 |
| repaired | orbiting | 4.68 | 6.35 | 8.15 | 58.8 | 0.000 |
| repaired | selecting | 5.78 | 13.70 | 34.68 | 57.8 | 0.000 |

The repaired p95 values meet the <33 ms target. Measured paced FPS is 55.8-58.8, so steady 60 FPS is not established. Running frame service capacity is about 104 FPS; the unchanged computation sustains only 0.57 simulated seconds/wall second here. Running simulation-step p95 is 43.84 ms and pause waited 24.44 ms for its in-flight step. Selection max 34.68 ms and running max 41.19 ms remain disclosed; a p95 gate does not rule out occasional slower frames. Expensive glow/postprocessing is omitted.

## Source and scientific preservation

SHA256 inventory: 594 existing source/experiment files; 593 unchanged. Only the 3D entry-point import changed. Both previous viewer modules, sensor/teacher/neural/world/session source, all 2D sources, prior experiments and October 1 checkpoints/journals remain byte-identical. New viewer/display/worker modules are additive.

Interrupted run reconciliation: 48/48 teacher cases, 1,050 completed updates; candidate/interrupted tensors identical, 12 changed tensors and positive gradients. Original parameter-change summary stops at 1,049. Reconciled deltas live outside the run; no reload/behavioral gate or validation/final executed. Receipt remains consumed.

## Visual QA and live acceptance

Front/side/top source previews were inspected before activity polish. Real registered source geometry shows both optic lobes and central brain; faint surface/reference cloud fills context beyond the selected 589 skeletons. The cached subset cannot reproduce every neurite in the full FlyWire reference. Full source selection is cyan; warm whole-neuron activity is sparse and does not establish conduction location/speed. Cameras were backed off to retain silhouette margin.

Saved anatomy/activity front/side/top, cord and full-selection PNGs are in `previews/`. Header pixels were checked in every PNG: apparent omission of static UI in the image display tool was contradicted by identical complete PNG pixels. Tests separately check UI pixels, source picking, panels, toggles, pause/replay/new/exit and render independence. Native mouse/resize/Shift-drag interaction still requires user review.

## Local code review

Reviewed against the approved plan without delegation, per user instruction. Fixed the hidden-cord pick mask, changed browse label to Neurons, deferred hidden cord GPU buffers, added full-morphology completion text and ensured exit joins the worker. Pause/reset barriers and exact serial/background numerical equality are covered by tests. Morphology arrays are immutable and exact source coordinates remain unchanged. Threading changes wall-time scheduling, not neural equations.

Test results and live observations are recorded in the project log and XML reports.
