# Actual modeled firing implementation plan

> For agentic workers: use `superpowers:executing-plans` to implement these checkboxes sequentially. The user's one-agent requirement overrides delegation and automatic commit suggestions.

**Status:** Implemented and automatically verified 2026-10-04. User approves the full-activity appearance as an option; individual native interactions remain unconfirmed. Follow-up selectable circuit view is planning only. No training or scientific assay ran.

**Goal:** Show actual firing across available project branches while keeping genuine silence identifiable and simulation results unchanged.

**Architecture:** Observe each completed fixed control interval in the simulation pump and publish one immutable display frame containing its original scientific snapshot and display history. Static branch vertices retain their neuron ownership and sample a compact per-neuron GPU color table. Drawing, selection and loading only consume published data.

**Tech stack:** Existing Python, NumPy, PyTorch and Panda3D; cached geometry only; no new dependency.

**Spec:** User's 2026-10-03 “Show actual firing across the modeled brain” request, captured in the requirements below. Existing scientific contracts remain in `docs/architecture.md` and `docs/reference/PROJECT_BRIEF.md`.

## Requirements and constraints

- Every firing neuron with available, visible branches receives a warm pulse: no count cap, outgoing-edge requirement or metadata-region exclusion.
- Pulse on each simulation interval containing spikes, fading with a 200 ms time constant and returning to idle after 600 ms without spikes. Pause freezes activity.
- Preserve blue/violet idle branches, restrained orange/yellow activity, cyan selection, branch brightness settings, geometry, camera and navigation.
- Observe every completed fixed simulation step, including intervals between display refreshes. Here a completed step means `FlightSession.advance`: one 20 ms control interval whose snapshot aggregates its 1 ms neural steps. Do not invent subinterval spike timestamps.
- Publish the original `FlightSnapshot`, episode identity, cumulative counts, last-active times and a 200 ms smoothed firing rate together, immutably and atomically.
- Replay and new trial clear history, including same-seed replay. Renderer actions never accumulate spikes.
- Preserve scientific snapshot contracts, neural calculations, sensing, teacher control and the interrupted 1,050-update experiment. Use one agent and cached geometry. No fitting, reserved evaluation, downloads, commits or publishing.
- Historical two-second preview: 530/589 fired; 59 silent comprise 50 unstimulated visual inputs, six self-only cells excluded from dynamics and three cells with silent upstream inputs. Explain this window and those limitations without claiming permanent silence.
- Include firing cells with no outgoing edges and the 56 firing neurons with visible brain branches previously excluded by region metadata.
- Inspect latest-interval spikes, spikes since reset, recent firing rate, time since last activity, missing geometry and lack of non-self incoming connections.
- Replace “up to twelve” copy with modeled/visible activity counts. Warm branches mean whole-neuron activity, without claiming anatomical spike propagation.

## Review focus

1. Same seed/time after replay must not reuse an old display frame or color upload.
2. Geometry completed or cord shown while paused must use current history without creating activity.
3. A neuron with both brain and cord branches must color both from the same owner; region labels never determine branch visibility.
4. Missing, empty, loading and hidden geometry must be distinguishable from zero spikes.
5. Concurrent publication/reset or a worker failure must never expose a new snapshot paired with old history or allow capture to bypass observation.

## Design decisions

The user's compact GPU table approach fits the existing static batching. Per-neuron overlays would scale scene objects with firing count; per-frame vertex recoloring would scale work with geometry size. Neither is needed.

New display-only `activity.py` defines a frozen `DisplayFrame` with `snapshot: FlightSnapshot`, monotonic `episode_id: int`, `step_index: int`, `cumulative_counts: tuple[int, ...]`, `last_active_times: tuple[float | None, ...]`, and `firing_rates_hz: tuple[float, ...]`. The original snapshot carries neuron IDs and latest-interval counts. No mutable array crosses publication.

`SpikeRecorder(snapshot: FlightSnapshot)` starts with zero history. `observe(snapshot: FlightSnapshot) -> DisplayFrame` accepts a strictly later completed interval with the same neuron order and seed; duplicate observation is rejected, not counted twice. `reset(snapshot: FlightSnapshot) -> DisplayFrame` increments episode identity, clears history and starts at step zero. Never infer replay from seed alone. Validate finite increasing time and nonnegative integer counts; reject changed IDs or seed without reset.

For interval duration `dt` and `tau = 0.200 s`, update the recent rate as `rate = exp(-dt/tau) * previous_rate + (1-exp(-dt/tau)) * count/dt`. This is an exponential average of interval firing rate, in spikes/s, not a biological firing-rate estimator. Cumulative counts add each interval once; a positive count sets last-active time to the interval end. Pulse strength is `exp(-age/0.200)` for `0 <= age < 0.600 s` and exactly zero at/after 0.600 s, or if the cell has never fired. Every positive interval refreshes strength to one, independently of its count or outgoing edges.

For example, three spikes in 20 ms give an interval rate of 150 spikes/s and a first smoothed rate of about 14.27 spikes/s. A further silent 20 ms reduces that rate to about 12.92 spikes/s and pulse strength to about 0.905. Selection never changes these values. This is display arithmetic; PyTorch's forward neural calculation and any optimizer remain separate.

GPU `branch_activity.py` owns static branch construction and color upload, leaving the general historical `viewer.geometry` helper intact. Each endpoint carries the same manifest index as its segment's owner. Keep `(segments, owner_indices)` paired through the existing midpoint Y=-0.43 mm split, 50,000-segment chunking, pending partial batches and deferred cord queue. Keep coordinate bytes unchanged.

Use a nearest-filtered, uncompressed RGBA texture with one texel per neuron and center-of-texel sampling; upload with explicit RGBA channel ordering, not Panda's implicit native channel order. Table colors interpolate the existing idle brightness presets `(0.24,0.30,0.55,0.22)` / `(0.32,0.40,0.67,0.42)` toward restrained warm colors using pulse strength. Reuse the existing warm range `(1, 0.36+0.48*strength, 0.07, 0.18+0.25*strength)` as the active endpoint. Keep cyan full-selection geometry separate. No per-active-cell scene objects; cache uploads by episode/step/brightness, not seed/time alone.

“Visible activity” counts unique pulsing owners with branches actually uploaded into enabled project layers, rather than metadata membership, queued geometry or camera pixel coverage. “Latest interval firing” and “fired since reset” are separate counts; a 600 ms pulse is not proof of firing in the latest interval. Geometry counts include available/loading/missing distinctions. Background context stays idle because it is not simulated.

## Task 1: Completed-step recording and atomic publication

**Files:** Create `src/flybrain/flight3d/activity.py`, `tests/test_flight3d_activity.py`; modify `src/flybrain/flight3d/simulation_pump.py`, `tests/test_flight3d_simulation_pump.py`.

**Interfaces:** Recorder/frame as above. Extend `SimulationPump(session, recorder: SpikeRecorder | None = None)`. Expose `display_frame: DisplayFrame | None` via a publication lock. Add `advance_fixed(steps: int, command=None) -> FlightSnapshot` for deterministic synchronous stepping at a worker pause barrier; share the same private advance-and-observe path with the worker. This returns the original final snapshot. With no recorder, existing behavior remains unchanged.

- [x] Add failing behavior tests for cumulative sum across skipped renders, latest counts, rate formula, pulse at 0/200/599/600 ms, never-fired cells, immutable earlier frames and observation validation. Pin same-seed reset identity/history and changed neuron order rejection.
- [x] Run `python -m pytest -q tests/test_flight3d_activity.py tests/test_flight3d_simulation_pump.py`; confirm failures concern missing behavior.
- [x] Implement recorder and optional pump observation after successful `session.advance`. Publish frame only after full recording. Reset recorder and publish zero history inside the existing reset barrier after the operation succeeds. Readers acquire one frame and use it throughout drawing.
- [x] Add pump tests for observing all five steps in a 100 ms request, pause retaining the same frame, reset discarding credit, deterministic stepping equivalence, recorder-free use, and worker failure surfacing with the last complete frame intact.
- [x] Compare recorded/unrecorded teacher sessions and a neural-controller session after each identical command: exact snapshot, voltage, previous spikes, phase, history, previous eye state, neural steps, transmission trace, controller state and parameter tensors. Verify applicable local RNG states plus Python/NumPy/Torch global RNG states are unchanged by the recorder itself; do not assume a generator exists. Include replay and new trial.
- [x] Run the focused task checks and inspect observed results before proceeding.

## Task 2: Ownership-preserving GPU branch coloring

**Files:** Create `src/flybrain/flight3d/branch_activity.py`, `tests/test_flight3d_branch_activity.py`; modify `src/flybrain/flight3d/anatomical_display.py`, `src/flybrain/flight3d/anatomical_viewer.py`, `tests/test_flight3d_anatomical_display.py`, `tests/test_flight3d_anatomical_viewer.py`.

**Interfaces:** `pulse_strengths(frame: DisplayFrame) -> np.ndarray` is pure. `BranchActivityTable(ids: tuple[str, ...])` supplies `make_batch(segments: np.ndarray, owner_indices: np.ndarray, name: str) -> NodePath` and `update(frame: DisplayFrame, bright: bool) -> None`. Store resident unique owner sets per layer for visible counts. `draw_frame(frame: DisplayFrame) -> None` replaces default viewer calls to `draw_snapshot`; do not fabricate history in a compatibility drawing method.

- [x] Replace cap tests with >12 firing owners, a firing owner without outgoing edges and a silent owner remaining idle. Test mixed brain/cord ownership, batches spanning owners/chunks, empty geometry, deferred cord and geometry loaded at unchanged simulation time.
- [x] Run the new checks and confirm expected missing-behavior failures.
- [x] Implement compact ownership-bearing static geometry, shader/table and explicit RGBA upload. Test distinct sentinel RGBA values through the real texture and shader, with several adjacent owners to detect interpolation/index errors. Read rendered diagnostic pixels to verify owner-specific colors, including alpha; raw byte assertions alone are insufficient.
- [x] Attach the recorder to the default viewer's pump; capture obtains a pause barrier and calls `advance_fixed` rather than `session.advance`. Read exactly one display frame in each render path. Remove the ranked activity overlay cache and its region filter; preserve full cyan selection and settings.
- [x] Update existing default-viewer tests that step the session directly to step through the observed pump path. Preserve scientific/reference viewer contracts. Test repeated drawing, presets, selection, background/brightness/cord toggles and progressive geometry while paused leave history and neural state unchanged.
- [x] Verify >12 real pulsing visible owners in cached teacher preview, no per-firing-neuron overlay objects, four missing exports retained in counts and a metadata cord neuron with real brain branches visible when cord is hidden.
- [x] Feed identical completed-step streams while drawing at 30/60/120 Hz; compare history, pulses, owner colors and visible counts at shared simulation times (including one second), then after 600 ms of silence. Test intervals skipped by drawing explicitly. Report simulated-time equality separately from wall-time worker throughput.
- [x] Run task checks, existing camera/controls/navigation checks and inspect saved front/side/top, cord and selected frames.

## Task 3: Inspector and honest activity explanation

**Files:** Modify `src/flybrain/flight3d/anatomical_viewer.py`, `tests/test_flight3d_anatomical_viewer.py`, `tests/test_flight3d_controls.py`.

- [x] Add failing assertions for latest-interval count, cumulative count, smoothed spikes/s and time since last activity. Never-fired cells show “not since reset”; source-only cells show “not simulated”. Count non-self incoming edges separately from source self-connections.
- [x] Implement frame-driven inspector value refresh without rebuilding buttons or losing scroll/selection. Preserve async morphology loading status independently of live numbers. Explain missing geometry, loading/hidden branches and no non-self incoming connections where applicable; sensory inputs may receive external stimulation.
- [x] Replace top-twelve coverage copy with distinct modeled latest-interval, recent visible pulse and since-reset counts. Explain whole-neuron pulses, the 200/600 ms rule, and observational teacher/manual modes without anatomical propagation claims.
- [x] Preserve the historical 530/589 diagnosis and 50/6/3 silence explanation in the guide/verification record, explicitly scoped to that window. Derive live statuses from current evidence; do not hardcode those counts as permanent app facts.
- [x] Run inspector, selection/missing geometry and resize/navigation regressions; review readability in saved 1440x900 and narrower layouts.

## Task 4: Verification, performance and live handoff

**Files:** Modify `scripts/benchmark_flight3d_viewer.py`, `PROJECT_STATUS.md`, `PROJECT_LOG.md`, `docs/FLIGHT3D.md`, `docs/BUILD_QUEUE.md`, `docs/architecture.md`, `docs/decisions.md`; create `docs/verification/flight3d-spike-activity.md`. Evidence goes to a fresh `.superpowers/spike-activity-2026-10-03/` directory (choose a fresh suffix if it exists).

- [x] Inventory protected scientific, cache and experiment hashes before implementation; compare afterward. Expected scientific snapshot/session/neural/world/sensor/teacher/experiment bytes stay unchanged. No scientific milestone or training continuation.
- [x] Run exactly `python -m pytest -q tests -k flight3d`, then a fresh `python -m pytest -q`; save command, stdout/stderr, exit status, duration and JUnit reports. Use this project's activated venv and record executable/version. Report failures/skips/warnings as observed, never reuse prior full-suite evidence.
- [x] Extend benchmark setup to observe deterministic warmup through the pump. Run isolated subprocesses for cord hidden and shown at 1440x900, cached geometry drained, hundreds of pulsing neurons verified from the display frame/resident owners. Use real recorded activity, not fabricated neural results. If a snapshot fails the hundreds-active setup condition, report setup failure rather than a passing load test.
- [x] Measure paused/running/orbiting/selecting conditions with the existing 120 measured frames per condition and warmup. Save mean/p95/max frame work, paced FPS, simulation throughput, active counts, batch/scene-object counts, environment and pause barrier cost. Require p95 frame work below 33 ms in both cord states. Keep frame work separate from pacing sleep and simulation throughput. Compare previous hidden-cord p95 **4.44/5.87/3.91/9.05 ms** respectively; the prior report has no matched shown-cord baseline, so do not invent one. Stop and narrow any failed performance gate; after two unsuccessful fixes gather a minimal reproduction.
- [x] Launch `pythonw.exe -m flybrain.flight3d --seed 970000` with captured logs for a short live user review. Record actual launch evidence. Ask for warm activity across branches, cyan selection, paused fading freeze, cord toggle/navigation, R clearing history and Backspace same-seed clearing history. Saved images/injected events are automated evidence; only record user review as passed after user feedback. Native automation being unavailable does not become a claimed visual pass.
- [x] Locally review the changed code against every requirement, atomicity, ownership/channel ordering and protected-state equivalence; use one agent. Update status sections, dated log fields, guide, decisions, architecture, queue and this ledger with actual evidence. Keep historical “up to twelve” entries identifiable as historical rather than rewriting old results.
- [x] Handoff concise changes, exact verification commands/results/limits, one learning takeaway and one concrete next task with recommended Sol effort. If live review remains pending, that is the next task (Sol Medium); never start another learning experiment automatically.

## Planning review

Coverage: completed-step publication, uncapped actual spikes, pulse timing, geometry ownership, GPU RGBA ordering, inspector/silence explanations, replay/pause, rate independence, scientific/RNG equality, required suites, both-cord performance, live review and records each have a task above. Preserve the existing scientific snapshot instead of modifying it. User supplied execution method was one agent; implementation is delivered. See the execution record below for verification limits and subsequent user direction.

## Execution record — 2026-10-04

Tasks 1–3 and Task 4 automated checks, launch, local review and records complete.
Fresh required targeted/full suites passed 76/643; logs/XML/exits retained under
`.superpowers/spike-activity-2026-10-03/`. Both passing hidden/shown recordings
meet p95 <33 ms with hundreds active. Earlier hidden selection failure prompted
inspector caching; an earlier failed shown run and unresolved timing variability
remain recorded. 2,365 protected files unchanged. Fifteen final screenshots saved;
front/side/top, cord, selection, silence, missing and narrower inspector reviewed.
See [complete evidence](../verification/flight3d-spike-activity.md).

Live launch is recorded separately from user acceptance. The user said the full
view is good and must remain an option, and chose a defined circuit for a smaller
view. This does not confirm individual native Space/V/R/Backspace or resize/focus
checks. The next task is the requested [activity mode proposal](FLIGHT3D-ACTIVITY-MODES.md)
/ Sol Medium; product implementation pending its review. No learning continuation.

- [ ] Confirm remaining native interaction items through user observations.