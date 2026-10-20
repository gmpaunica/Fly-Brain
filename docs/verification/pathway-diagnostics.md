# Pathway diagnostic verification — 2026-10-05

Scope: user's brain-to-wing diagnostic; one agent, fixed scientific model.
No Git repository or commit. Existing source integration changes are limited to
the encounter scene/session/viewer. Their exact original bytes are archived and
hash-checked under `artifacts/pathway-diagnostics-2026-10-05/original/`.
All historical `flight3d` neural/renderer/teacher sources remain unchanged.

## Numerical and lifecycle verification

- `.venv/Scripts/python.exe -m pytest tests/test_pathway_diagnostics.py tests/test_encounters3d_session.py -q`
  → **8 passed**, one existing TorchScript deprecation warning.
  Bit-identical voltages, previous spikes, phases, counts, two rate histories,
  edge transmissions and visual image state against original stepping for
  flower/predator, varied intervals and all intervention sets.
- `.venv/Scripts/python.exe -m pytest tests/test_pathway_panel.py tests/test_encounters3d_viewer.py -q`
  → **4 passed**, one existing warning. Actual Panda3D rendering at 1000×700 and
  1440×900; passive panel/frame rendering, pause, reset, exact replay, preferences
  and unchanged predator RGB. These are automated UI checks, not human review.
- `.venv/Scripts/python.exe -m pytest tests/test_pathway_protocol.py -q`
  → **2 passed**, one existing warning. Exactly 24 paired stimuli/five interventions;
  RGB growth, shutter and reappearance; rejected flower/background with zero
  diagnostic currents/spikes. Does not execute the scientific protocol.
- Exact leak, threshold, subtractive reset, next-step propagation, pre-reset
  voltage peaks, input phase and first-spike timestamps are numerically checked.
  Silenced destinations have zero delivered current/voltage/spikes and expose
  discarded incoming increments; all episode diagnostic history clears on reset.

## Single bounded scientific record

Command: `.venv/Scripts/python.exe -m flybrain.encounters3d.investigate experiments/brain-to-wing-diagnostic-2026-10-05`

Exactly 120 presentations / 120,000 neural steps completed once; 6,000 completed
intervals, 888,000 CSV pathway cell intervals, all 589 cells in each JSON interval.
No parameters, optimizer, checkpoint, topology, sensory rules or commands changed.
Fresh directory creation refuses any existing directory, even an empty one.
The negative wing spike result closed the investigation with no retry.
The intact path check excluding edges entering clamped sensory cells is also
saved as `functional_paths.json`; it reaches all 48 wings in every intact case.

`.venv/Scripts/python.exe scripts/audit_pathway_records.py experiments/brain-to-wing-diagnostic-2026-10-05`
→ JSON audit confirms source/file/image hashes, paired images, interval count,
spike/current/voltage aggregates and exact first-spike grid. No neural rerun.
Output: `artifacts/pathway-diagnostics-2026-10-05/record-audit.json`.
Postprocessing of saved traces compares interventions per cell and ranks delivered
currents; its record is separate from the locked scientific receipt. A structural
check excluding paths through clamped sensory destinations still reaches all
48 wing endpoints from each intact case's driven visual cells.

The original output/capture commands used PowerShell combined stderr redirection.
PowerShell reported exit 1 for the existing TorchScript warning as a
`NativeCommandError`, without a Python traceback. Scientific receipt, output
1–120, file reconciliation and separately passing tests confirm completion;
the original exit indicator is preserved honestly in the session log.

## Saved-frame review and separate live review

Saved images under `artifacts/pathway-diagnostics-2026-10-05/frames/` include
initial/reset, predator/flower at 1 s and a separate 1000×700 predator capture.
Actual screenshots were inspected: three stages and both DN subgroups are visible,
numerical measurements are legible, the legend/guide is retained below by scrolling,
existing world/brain appearance and exclusive cyan predator remain intact.

Computer Use initialized through the supported `@oai/sky` API, then two native
`list_windows()` calls failed with missing pipe / os error 2. No unsupported
automation fallback was used. A native preview was launched with
`scripts/preview_pathway_diagnostics.py`; launcher PID 26008 / window PID 7780,
responding, titled `FlyBrain | Pathway diagnostics | avoidance unavailable`.
The user answered **“Panel and controls work”** to the explicit panel readability,
pause/update/reset/replay/spawn live-review prompt. This supplies the separate
live acceptance; it does not establish retreat or learning.

## Completion checks

Targeted command (PowerShell):

```powershell
$diagnosticTests = @(Get-ChildItem tests/test_encounters3d_*.py,tests/test_flight3d_*.py,tests/test_pathway_*.py | ForEach-Object { $_.FullName })
& .venv/Scripts/python.exe -m pytest @diagnosticTests -q --junitxml=artifacts/pathway-diagnostics-2026-10-05/targeted.xml
```

Observed: **111 passed**, zero skips, two existing TorchScript warnings, 290.03 s.
After the final display-label clarification, the four panel/encounter viewer
tests passed again (31.52 s, one warning; `panel-final.xml`). The clarification
exposes discarded incoming current explicitly; it changes no measured values.
Its regression assertion was observed failing before the label was added.

Preservation audit: **2,398 preexisting files** under source/data/experiments
(excluding bytecode); **2,395 unchanged**, three intentional encounter integration
edits with exact original-byte archives verified against the initial SHA-256
manifest. No historical neural/teacher/renderer, data, experiment or checkpoint
bytes changed. Evidence: `artifacts/pathway-diagnostics-2026-10-05/preservation.json`.

Local one-agent acceptance review checked the original/current integration diff,
instrumented/original operation order, sensory-only boundary, immutable publication,
input clamp/first-time semantics, replay/reset, source preservation and row-level
scientific failure categories against the user's specification. No unresolved
correctness blocker. This is author self-review as requested, not independent
subagent review. No Git/worktree/commit workflow applies in this non-Git workspace.

An earlier
full-suite process ended without stdout/XML; it supplies no passing evidence.
The verification retry uses `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1`; it reruns
tests only, not the closed scientific investigation.

Final full-suite command:

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
.venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/pathway-diagnostics-2026-10-05/full.xml
```

Observed: **678 passed**, zero failures/skips, **3 upstream warning instances**
(one Pygame pkg_resources deprecation, two TorchScript deprecations), 361.86 s.
Final source/data/experiment hash reconciliation is repeated after the suite.
Final current-label screenshot was separately inspected at 1000×700; numerical
semantics and all controls match the live-accepted panel. No further code changes
followed the successful full suite. No commit, publication or repair run followed.
