# Fly-Brain: visual repair, readable brain network, and one bounded experiment

## Summary

Repair flower detection and teacher coverage, replace the default microscopic anatomy display with a grouped neural network, improve navigation, then conduct exactly one newly frozen experiment.

Read-only reproduction confirmed seed **960000** still produces zero flower detections and no contact after 30 seconds. An in-memory eight-sample averaging probe recovered visible color contrast, but landing remains unverified.

Use one agent. Preserve the failed experiment, its source snapshot, and accepted 2D source/checkpoints/results byte-for-byte. No commits, automatic sweeps, or additional experiments.

## 1. Repair sensing and teacher coverage

- Record a preservation hash inventory before edits. Keep all new development evidence separate from historical experiments.
- Preserve the immutable `[2,24,48,3]` RGB observation interface. Replace single rays with deterministic **2 horizontal × 4 vertical subpixel samples**, resolving nearest-surface occlusion separately before averaging RGB.
- Introduce one shared RGB contrast calculation for the teacher and neural visual encoder: positive flower-color contrast based on `R − (G+B)/2`. Test the initial threshold of `0.01` against flower, sky, floor, and stem samples. Keep all detection inside the sensory-processing path; expose no geometry or hit labels to controllers.
- Use contrast-weighted image directions and apparent extent for the teacher’s existing visual estimate. Verify acquisition, approach, close-range descent, and contact separately.
- Add bounded vertical search alongside yaw when visual detection is lost: an eight-second cycle within approximately ±1.5 cm of the internally estimated search-entry altitude. Derive movement estimates from local motion sensors; reset search history on replay and stop searching immediately on reacquisition.
- Use development seeds **970000–970047**, balanced across eight sectors and three heights, plus historical seed 960000 as an explicitly historical regression. Add fixed sensor fixtures for thin silhouettes, image boundaries, above/below/behind targets, occlusion, and temporary loss.
- Require all 48 development episodes and the historical regression to achieve the existing 30-second landing/dwell criterion before freezing the experiment. Development fixes require recorded causes; after two unsuccessful fixes to the same failure, produce a minimal reproduction and reassess. Do not consume the experiment while development prerequisites remain unresolved.

## 2. Build the grouped brain network and navigation

**Chosen presentation:** grouped network, with smooth camera zoom and scrollable information panels. No generated reference image is required.

- Add a default **Network** view with approximately 96 spatial groups of simulated neurons. Group deterministically within hemisphere and brain/cord boundaries, retaining source membership and anatomical coordinates.
- Draw subdued purple connection bundles between group centroids only where retained directed source edges exist. Keep a faint registered brain silhouette for recognizable shape. Label bundle geometry as a connection diagram; it is not reconstructed axon geometry.
- Use prominent connections and small junctions, with orange/yellow activity distributed along links. Preserve the detailed **Anatomy** view as an explicit alternative. Start focused on the brain; keep the nerve cord available through its existing toggle.
- Clicking a group opens its roles, member count, source IDs, and incoming/outgoing connections. Selecting a member reveals existing source morphology and evidence.
- Extend neural telemetry with immutable, interval-based edge transmission totals aligned to a fixed edge order. Record the contributions actually delivered by the existing delayed-spike calculation, excluding ineffective inputs and silenced destinations. Do not change neuron dynamics or readouts.
- Aggregate those contributions into bundles and smooth their display over **300 ms of simulation time**, using fixed normalization rather than rescaling each frame. Pause freezes activity; replay clears it; repeated rendering cannot accumulate activity.
- Keep teacher/manual activity labeled **observational**. In neural modes, describe illumination as modeled signal transmission, without implying that each bright link has proven behavioral importance.
- Route wheel events by the full panel rectangle: wheel over a viewport zooms that camera; wheel over details scrolls text. Add smooth zoom, Shift-drag panning, visible reset controls, and bounds that prevent clipping through the brain. Camera animation may use display time but must not affect simulation or activity time.
- Make details/help scrollable and resize-aware, keeping essential controls visible.

## 3. Freeze and run exactly one experiment

- Make the runner consume its frozen protocol instead of hard-coded historical placements.
- Reserve **980000–980047 training**, **981000–981023 validation**, and **982000–982095 final**, with initialization seed **940404**. Check historical records and development manifests for overlap before freezing; collisions block execution rather than silently changing cases.
- Freeze source snapshots, sensor/search settings, circuit digest, software/device metadata, case coordinates, hypothesis, and protocol hash in a new experiment directory. Add a one-use authorization receipt so renaming the directory cannot authorize another run.
- Retain the approved hypothesis: motor-endpoint readouts can learn safe flower landing from local sensory activity, with fixed topology and dynamics.

| Stage | Unchanged requirement |
|---|---|
| Neural prerequisites | Signal distinction, propagation, DN interventions, motor silence, mechanics |
| Teacher | 48/48 successful training placements |
| Fit | One 2,000-update Adam fit; learning rate .001; batch 512; training-only normalization; two-hour collection/fitting ceiling |
| Training gate | ≥44/48 successes and strictly lower MSE than the train-only mean predictor in every continuous command channel |
| Validation | ≥20/24 successes |
| Final | ≥82/96 overall, ≥26/32 in each height group, paired one-sided p≤.025 against each matched untrained/fixed control |

- Score complete 30-second episodes with safe landing and at least two seconds of stationary food contact maintained through the end.
- Stop immediately at a failed prerequisite or gate. Record actual optimizer updates, gradients, parameter changes, reload checks, trajectories, and stage outcomes durably. Execution interruption or budget exhaustion is **incomplete**, never a pass or permission to restart.
- Never evaluate reserved final behavior before earlier gates pass. No retraining after validation/final feedback. Preserve any failed candidate and keep the teacher as the default preview.

## 4. Verification and handoff

- Add numerical regressions for anti-aliasing, contrast detection, occlusion, sensory-only control, bounded search, reacquisition, landing, and deterministic reset.
- Verify grouping/source-edge accounting, transmitted activity, silence, pause/reset behavior, and independence from camera/render frequency.
- Test wheel routing, zoom/pan limits, scroll overflow, resizing, selection, view toggles, presets, R, Backspace, pause, and exit. Inspect saved front/side/top views for readable connections and unclipped UI.
- Run:
  - `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests -k flight3d`
  - `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`
- Complete live interaction review through native automation if available; otherwise obtain user-observed results using a short checklist. The current native helper fails to connect, so automated event tests and saved frames must remain distinct from live verification.
- Reconcile experiment journals, counters, checkpoints, and preservation hashes. Update status, append the dated project log, and revise the 3D guide/plan, architecture decisions, experiment summary, and task queue.
- Report actual learning results and unexecuted stages. Claim successful learned landing only if every behavioral gate passes. Include one concrete sensory-to-tensor-to-motor example and leave one evidence-based next task with recommended effort.
