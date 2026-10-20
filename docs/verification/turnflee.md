# Turn-and-flee verification — 2026-10-06

Scientific result: closed at validation, **87/96** vs required >=90/96 and >=5/6
each azimuth. 18.75°:3/6, 41.25°:5/6, 63.75°:3/6, 86.25°:6/6, 108.75°:6/6,
131.25°:5/6, 153.75°:5/6; all other groups:6/6. All nine failures are timely
right turns against right-side threats. No closed-loop physical avoidance or
stable recovery assay ran; default `retreat3d` remains unchanged.

Calibration pairs: (2,1) 13/16; (4,2) 16/16; (8,4) 16/16. Both higher pairs
eligible with exact baseline parity, finite state and max downstream 175 Hz.
Select lowest (4,2), freeze before offset validation. Validation maximum 190 Hz.
Budget: 48 calibration +96 validation =144 cases, 216 s, 432,000 paired 1 ms
steps. Exactly 10,800 intervals, all 3,600 baseline intervals match. No learning,
new cells, threshold/encoding changes, additional candidates or retry.

Stored first-trigger score audit: ten exactly tied bilateral validation responses,
three of them among the nine failures; six failures have unequal scores. First
trigger six-cell norms range 1..sqrt(12). This is zero-step postprocessing.

The prescribed protocol was executed once using:

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m flybrain.turnflee.experiment experiments/turnflee-2026-10-06 > artifacts/turnflee-2026-10-06/experiment.log 2>&1
```

Do not rerun it. PowerShell reported exit 1 with a `NativeCommandError` for the
upstream TorchScript deprecation warning on redirected stderr. The experiment
printed its closed result and finalized all receipts; no traceback is present.
The independent saved-record audit succeeds. This wrapper status is recorded,
not represented as a passing shell command and not retried.

Read-only verification commands:

```powershell
.\.venv\Scripts\python.exe -m flybrain.turnflee.audit experiments/turnflee-2026-10-06
.\.venv\Scripts\python.exe scripts/audit_turnflee_preservation.py
```

Preservation passes: **2,510 original files, zero changed/missing**.
Targeted command (OMP/MKL threads one):

```powershell
.\.venv\Scripts\python.exe -m pytest tests -k 'flight3d or encounters3d or pathway or retreat3d or turnflee' -q --junitxml=artifacts/turnflee-2026-10-06/targeted.xml
```

**188 passed**, 567 deselected, zero skips, three upstream warning instances,
84.31 s. The task adds 35 numerical/control/lifecycle/record/render tests.

Full command (same environment):

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/turnflee-2026-10-06/full.xml
```

**755 passed**, zero failures/skips, three upstream warning instances, **196.34 s**.
Source files stayed frozen throughout the scientific run and final checks.
Gate-aware idle smoke and replay, `python -m flybrain.turnflee --headless --seed
1150020 --steps 10 --record artifacts/turnflee-2026-10-06/gate-aware-idle-actions.json`
then `--headless --replay-log` on that file, both report 0.20 s takeoff and escape
supported=False. This is original-model engineering smoke, not a neural escape
case. The historical motion-repair saved-record audit also passes unchanged.

The scientific audit passes: source/archived-source/receipt/old protocol hashes,
candidate order, lowest eligibility, recorded strong-event latch and exact ties,
signed scores, rolling wing vectors, paired baseline equality, sustained rates,
CSV/JSON interval correspondence, timing/sign classifications and stop/budget.
**Zero neural steps** in the audit. Offset validation is not called unseen;
43 historical config/protocol files were audited before lock, with all calibration
azimuths already present and no offset-angle match in those fields. Manual
interaction placements are not proven absent.

Preflight before onset: **34 passed** in 10.30 s, zero skips, one upstream warning;
then the extra visual-silent regression passed with the neural/receipt subset.
Initial RED checks precede neural/control/session/protocol/record/audit/viewer
implementation. Gain-one dynamics match historical voltage, phase, spikes,
histories and edge transmissions exactly. Gain masks preserve IDs, edges and
original counts; tests cover original motion inputs, delayed propagation,
rolling expiration, rate rejection, complete reset, spike-only gate, power-only
rejection, b1 sign, bilateral tie determinism, grounded turn/takeoff, neural gaps,
weak braking, IMU envelope hysteresis, config tampering, source/receipt validation,
old-model replay, endpoint pause and actual-spike display. Visual-silent neurons
retain motion activity but produce zero increments and zero escape triggers.

The selected configuration is bound to `calibration.json` by hash. Default
activation also requires the final gate receipt; calibration alone is insufficient.
Preflight candidate construction is explicitly experimental. New v2 action logs
include the response configuration and hash; old v1 logs replay in the original
session. A display episode ID changes on replay as designed; numerical snapshots
and all reconstructed spike histories match.

Separate rendering review: real offscreen GPU tests leave neural steps and
publications unchanged during drawing and replay the endpoint. Initial saved
frames exposed a five-line summary overlapping the anatomy heading. A failing
layout test reproduced it; the two-line summary and scroll-panel details pass.
`final-development-frames/initial-grounded.png`, `idle-cruise.png`,
`moving-predator.png` and `replay-endpoint.png` were saved before scientific lock;
the revised moving-predator frame was inspected. These development images are
UI evidence, not evaluated escape successes. Prefreeze action logs are development
artifacts and can fail subsequent strict source binding after source edits.

Native Computer Use was initialized via the supported `@oai/sky` package;
`sky.list_apps()` fails with missing native pipe/os error 2. No native input,
screenshot or human review is claimed. No unsafe custom helper/fallback protocol.

Local one-agent acceptance review follows the plan and user instructions. Physics,
paired neural stepping, sensory-only control, recorder and renderer are separated.
RGB has no path to turn side; voltage is passive diagnostic only. Named initial
closed-loop calibrations are explicit assumptions, with independent fixed physics
setup; they do not transfer actual body/target coordinates. Historical source and
data bytes are protected. No commit/push/subagent or resumed interrupted fit.

## 2026-10-06 frozen wrong-side review

Separate [review](../TURNFLEE-WRONG-SIDE-REVIEW.md) reconstructs all 10,800 intervals,
scores/gates/seeded ties, nine full timelines and 42 failure/comparison rows.
Audits rerun and pass; 2,510 original protected files and the 2,720-file broader
review input snapshot are unchanged. Runtime call guard confirms zero brain/
sensor simulation calls. Additive reviewer rejects all flybrain imports.
Review added zero presentations/seconds/steps; simulation suites not rerun.
Three failed ties/six unequal failures reconcile; all nine later favor left.
Figure visually inspected; no improved decoder/performance/escape claim.
One instrumentation proposal awaits separate authorization.
