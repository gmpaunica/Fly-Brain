# Demo UI verification — 2026-10-12

All commands run from the project root with the existing venv,
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. No training experiments, new scientific
evaluation, receipt changes, commits or publishing were performed.

## Behavioral checks

Initial red: `python -m pytest tests/test_escape_ui.py -q --tb=short` failed all
three cases because the new presentation package was absent. First implementation
passed three cases (29.28 s). Initial combined check:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_escape_ui.py tests/test_escape_edge.py -q --tb=short --junitxml=artifacts/escape-ui-2026-10-12/targeted.xml -p flybrain.escape_recovery.pytest_plugin
```

Observed **41 passed**, zero skips, two TorchScript warning instances, 141.84 s.
The 38 original edge tests remained passing. Visual review then added checks for
visible green button fill, legend placement and readable inspector/settings text.

The new framebuffer test first failed on a Windows path fixture: PNMImage needs
`Filename.fromOsSpecific`, as the historical capture code does. After correcting
the fixture, both sizes failed on the intended invisible-fill assertion. Local
Panda3D source confirmed numeric relief 0 suppresses the frame; `DGG.FLAT` draws
it. OnscreenText uses its text position; NodePath `setZ` added a second offset to
the legend. Corrected both. Settings readability then reproduced at 8.05 pixels
in the 700-pixel window; pixel-based text and larger row spacing fix it.

Final presentation check:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_escape_ui.py -q --tb=short --junitxml=artifacts/escape-ui-2026-10-12/ui-final.xml
```

Observed **3 passed**, zero skips, one upstream TorchScript deprecation warning,
51.45 s. The tests use real Panda3D widgets and the actual frozen session:
42-pixel targets outside viewports, visible framebuffer fill, pause via button
and Space, spawn exclusivity/input release, exact three-interval replay, new
trial, inspection/settings retention, and original delivery still valid.
Capture/diagnostics preserve world time, snapshot, voltages and camera state.

## Full regression suite

The first full run was deliberately interrupted when visual review exposed
tiny inherited settings controls. No passing full-suite claim was made for it.
The final run uses an isolated diagnostic ledger:

```powershell
$env:FLYBRAIN_TEST_LEDGERS='artifacts/escape-ui-2026-10-12/final-full-ledgers'
.\.venv\Scripts\python.exe -m pytest -q tests artifacts/escape-flight-validation-2026-10-08/test_serialized_audit.py --junitxml=artifacts/escape-ui-2026-10-12/full-final.xml -p flybrain.escape_recovery.pytest_plugin
```

Observed **925 passed**, zero skips, three upstream warning instances (Pygame
pkg_resources and TorchScript deprecation), 1869.89 s /31:09. Exit code 0.
Logs/XML: `artifacts/escape-ui-2026-10-12/full-final.log` and `full-final.xml`.
This run was substantially slower than the prior recorded suite; no duration
or performance improvement is claimed. Historical unit tests include small
synthetic gradient/optimizer checks; no project training experiment was run.

## Visual review and limits

Public CLI capture smoke: `python -m flybrain.escape_ui --capture
artifacts/escape-ui-2026-10-12/demo-1440.png` exits successfully. Review script:

```powershell
.\.venv\Scripts\python.exe scripts/review_escape_ui.py artifacts/escape-ui-2026-10-12/isolated-frames
```

Initial and improved default/diagnostics/settings layouts were examined at
1000x700 and 1440x900. Some image-tool previews omitted header text; direct PNG
pixel inspection found 1,152 opaque bright header pixels in every 1000 capture
and 1,021 in every 1440 capture. Full-resolution reinspection of the 1440
diagnostics frame also showed the headers. Thus those preview omissions do not
establish missing text in the saved PNGs or a renderer fault. Their tool-side
cause remains unresolved; no speculative renderer repair was made.

A separate fresh 1440 renderer retained 1,021 bright header pixels in each of
four repeated captures, including diagnostics and settings. Final review captures
each size in its own process, following the actual launcher lifecycle, and checks
the header pixels. All six captures pass with zero simulation/neural steps.
The historical native label-loss observation is not relabeled as universally fixed.

The review record is `artifacts/escape-ui-2026-10-12/visual-review.json` and the
raw capture checks are in `isolated-frames/capture.json`. The review script checks
all **3,010 protected historical source/experiment files**
and the original delivery before/after capture. All captures are paused and
assert zero simulation seconds and zero actual/reference neural steps.
Native mouse/keyboard review is not performed; the provided computer-use tools
do not expose native app control. Real widget callbacks and saved-frame review
are reported separately and do not establish live input acceptance.

Local one-agent acceptance review covers preserved controller/sensory boundaries,
worker barriers, consistent reset/replay behavior, display-only controls, source
verification and honest learning/biological labels. The future learning goal is
recorded in AGENTS.md and docs/LEARNING_ROADMAP.md; no understanding is inferred
from implementing or explaining this task.
