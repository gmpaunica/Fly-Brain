# October 1 development and verification evidence

This directory is development evidence, separate from both frozen experiments.
`REQUEST.md` preserves the user's scope verbatim. No commits or subagents were used.

## Preservation

Before edits, `preservation-before.json` inventoried 405 source/experiment files
and `source-before/` copied the 13 existing 3D Python modules. Only live 3D sources
were mutable. `preservation-check.json` verifies 392/392 protected historical files
unchanged, including accepted 2D source/checkpoints/results and the failed 3D run.

## Reproduction and causes

- The historical 960000 acquisition regression failed before implementation:
  `visual_measurement` returned None. Single vertical rays missed the thin target;
  old teacher and encoder thresholds also rejected averaged faint silhouettes.
- The blank-image search regression showed no vertical excursion. Search previously
  only changed yaw, preserving the vertical sampling gap.
- Replaced single rays with the requested 2x4 separately occluded samples; shared
  positive RGB contrast above .01; contrast-weighted image direction/apparent-size
  range; local-motion-estimated bounded vertical search. The three initial failing
  regressions then passed. No development repair sweep was needed.

## Full development episodes

`check_coverage.py sampling-search-v1` recorded all 49 complete 30-second episodes:
970000–970047, balanced over 8 sectors and 3 heights, plus historical 960000.
**49/49 passed.** Manifests, full trajectories and fsynced episode rows are in
`sampling-search-v1/`. `phases.json` analyzes those same saved trajectories:
all acquired on the initial image, approached, descended at close range, contacted
and maintained stationary food dwell through the end. It did not run new episodes.
Historical 960000 contacted at 2.765 s and finished with 27.24 s dwell.

## Viewer and telemetry

96 groups retain all 589 source IDs. There are 1,537 directed bundles, accounting
for all 4,139 source edges. Five explicit groups contain the 11 unpositioned cells;
34 bundles lack positioned endpoints and are accounted but not drawn. All 503
source self edges remain in accounting with zero modeled transmission.

Telemetry tests check delayed delivered current numerically, input-clamp/silenced
destination exclusion, reset and immutable tuple ordering. Filter tests check the
exact 300 ms exponential response, duplicate timestamps, pause and backwards-time
reset. Session snapshots are unchanged by repeated rendering.

UI integration exercises group/member selection, list overflow scrolling, full
rectangle wheel routing, separate cameras, Network/Anatomy, context/cord, presets,
focus/follow, pause, R and Backspace. Layout reflow/selection persistence is checked
at a real 1000x700 offscreen output. A Panda offscreen buffer cannot itself resize
without BF_resizeable: the attempted setSize check failed for that explicit reason,
so it is not claimed as a live-window resize test.

### UI failures found and fixed

- A destroyed viewer left its task in Panda's global scheduler. The next viewer
  invoked the destroyed instance. Removing the task on destroy resolved the
  sequential-viewer test; targeted suites subsequently passed.
- RGBA exports had unreliable presentation in the image viewer despite complete
  RGB UI pixels. Capture now writes opaque complete frames; numerical UI pixel
  checks accompany the saved image review. The image display can still omit text
  intermittently even though the same PNG's pixel audit finds it; live review is
  kept separate.
- Local review found Panda's Shift-modified mouse event was not registered.
  A failing event-registration regression reproduces it. Explicit modified-mouse
  bindings repair the path. A separate numerical failure showed zoom plus pan could
  lose volume clearance; the camera now adds pan magnitude to its minimum distance.
  These last two fixes occurred after experimental freezing and are isolated in
  `post-freeze-ui/` and `post-freeze-ui.json`. Every scientific source file remains
  identical to the executed frozen snapshot; the two UI files are never imported by
  the experiment runner.

## Verification boundaries

`frames-reviewed/` contains actual rendered front/side/top, cord, Anatomy and
group details. Source coordinates are not moved for presentation. Brain connections
and registered outline are visible; world and fixed controls occupy separate panels.
The native `@oai/sky` helper returned: "Computer Use native pipe is unavailable:
failed to connect native pipe: The system cannot find the file specified. (os error 2)".
A visible preview was launched and the user was asked for the live checklist.
Automated events, saved frames and live human observation are distinct evidence.

Exact commands and logs:

- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests -k flight3d`
  — `targeted-final.log`: 46 passed, 567 deselected; two upstream warnings.
- `.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`
  — `full.log`: 613 passed, two upstream warnings, 159.53 s before the final UI-only
  fixes. `full-final.log` records the final full-suite run.
- `.\.venv\Scripts\python.exe -m flybrain.flight3d --seed 10001 --capture
  experiments/flight3d-development-2026-10-01/frames-reviewed`
  — exited 0 and produced six opaque PNGs. This is offscreen, not live interaction.

`teaching-example.json` contains a concrete RGB-to-rate-to-endpoint calculation.
It is an explicitly held-observation development illustration, not a fresh
behavioral experiment. Only `../flight3d-landing-2026-10-01/` consumes the new
scientific authorization; its global receipt prevents renamed-directory retries.
