# Relay display refinement — 2026-10-13

**Current Milestone:** Dashboard accepted in user feedback; relay color refinement complete.
**What Works:** Other relays changed from white to steel-blue #5B8FB9 in dashboard
branches and matching checkbox swatch. Description explains structural selection.
Six focused tests pass; zero-step palette check changes exactly 241 entries,
preserves original palette/model/delivery and all 3,010 protected files. Full 928
passed, zero skips, three upstream warnings, 414.38s; paused screenshot reviewed.
**In Progress:** None for this change. Comprehensive walkthrough remains next.
**Known Issues:** Existing scientific limits/missing skeletons unchanged. Native
automation unavailable; user reports the redesigned controls work well.
**Important Decisions:** Other relays is a mixed structural-path category, not one
biological type or proof of individual function. Dashboard-only color override;
no neuron/controller/group-membership/pulse timing change or training experiment.
**Next Task:** Begin learning-roadmap lesson 1 / preferred Sol Medium; trace launch,
worker, snapshot and display, keeping the standing comprehensive understanding goal.
**How To Run:** `.\.venv\Scripts\python.exe -m flybrain.escape_ui`; reopen for new color.
**How To Test:** `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_ui.py -q`.
Exact commands/evidence: [relay color verification](docs/verification/relay-color.md);
[guide](docs/ESCAPE-UI.md).

# Modern dark dashboard completed — 2026-10-13

**Current Milestone:** User-approved dashboard redesign implemented and regression-verified.
**What Works:** Ten real role checkboxes hide branches, selection and picking while
all neurons remain simulated. Persistent Groups/Inspect/Help sidebar, rounded
charcoal panels, readable 16px body text, 46px primary actions, opt-in technical
inspection and Fit for visible/full selected anatomy. Final focused 44 pass;
ten zero-step layouts reviewed at 1000x700 and 1440x900. Full 928 pass, zero skips,
three upstream warnings, 480.87s; all 3,010 originals/source-bound candidate valid.
**In Progress:** No remaining implementation/checks. Actual dashboard was opened
paused for user review (PID 26440, responding at launch); it is no longer running
at final handoff. Comprehensive walkthrough not started.
**Known Issues:** Native mouse/keyboard acceptance is unreviewed; native control
tools unavailable. Windows did not bring the new window forward; use Alt+Tab or
taskbar (title FlyBrain | Explore the brain). Four source skeletons missing as
before. Scientific/learning limits and original directional failures unchanged.
**Important Decisions:** User rejected the first visual polish despite passing
tests, selected modern dark dashboard and full group hiding, and approved this
plan. Filters are viewing controls, not neural silencing. One agent, original
source-bound candidate/receipts preserved; no new scientific/training experiment.
**Next Task:** Review the redesigned UI, then learning roadmap lesson 1: launch,
source verification, worker/snapshot/render and why hiding cells does not silence
them. Preferred Sol Medium. Standing comprehensive understanding/AI-assistance
honesty requirement remains in AGENTS.md and docs/LEARNING_ROADMAP.md.
**How To Run:** `.\.venv\Scripts\python.exe -m flybrain.escape_ui`.
**How To Test:** `.\.venv\Scripts\python.exe -m pytest tests/test_dashboard_ui.py -q`.
Exact commands/evidence: [dashboard verification](docs/verification/dashboard-redesign.md).
Controls: [guide](docs/ESCAPE-UI.md). Approved task: [plan](docs/plans/DASHBOARD-REDESIGN.md).

# UI polish and project understanding — 2026-10-12

**Current Milestone:** UI polish implemented and regression-verified.
**What Works:** Additive `escape_ui` uses the original verified candidate; larger
spawn/pause/new-trial/replay controls, collapsed diagnostics and retained neuron
inspection/settings. Initial combined 41 pass; final UI 3 pass; final full 925
pass, zero skips, three upstream warnings. Six layouts reviewed at 1000x700 and
1440x900, all captures zero-step. All 3,010 protected files/original delivery pass.
**In Progress:** None for implementation/checks; comprehensive walkthrough not yet
started. No topic is marked understood solely from agent explanations.
**Known Issues:** Native mouse/keyboard acceptance remains unreviewed because the
tools expose no native app control. Some image-tool previews omit headers that
are present/opaque in PNG pixel checks; no speculative renderer fix. Historical
neural/learning limits remain as documented below.
**Important Decisions:** User wants UI polish first, then full project/script/
algorithm/library understanding for honest AI-assisted interview discussion.
Standing requirement saved in AGENTS.md and docs/LEARNING_ROADMAP.md. One agent,
presentation only; no training experiment, scientific retry, receipt alteration or commit.
**Next Task:** Begin learning roadmap lesson 1: trace demo
launch, verification, viewer/worker/snapshot/render. Preferred Sol Medium.
**How To Run:** `.\.venv\Scripts\python.exe -m flybrain.escape_ui`.
**How To Test:** `.\.venv\Scripts\python.exe -m pytest tests/test_escape_ui.py -q`.
Full command, exact results and review limits: [verification](docs/verification/escape-ui.md).
Controls: [demo guide](docs/ESCAPE-UI.md). Next: [learning roadmap](docs/LEARNING_ROADMAP.md).

# Sustained edge escape verified and delivered — 2026-10-11

**Current Milestone:** Sustained edge flight and hover verified on the declared protocol.
**What Works:** 38 new tests pass: real motor flight to edge/20-second hover,
off-center/corner starts, weak-threat persistence, reset/replay and fault sealing.
Broader targeted 239 pass; final full 922 pass, zero skips. Zero-step viewer
frame reviewed using normal frame tasks; all 3,010 protected files unchanged.
Both measured batches pass: 48/48 intact escapes, zero triggers in 48 silent
controls; 96 cases/1,152 simulated seconds/2,304,000 individual neural steps.
Both saved-only audits pass; four exact replays match every interval field
(96,000 separate verification steps). Minimum stable hover seven seconds;
arrival 4.84–5.02 s; hover wall clearance 10.39–15.28 mm; no collisions.
**In Progress:** None for this task; verified launcher activated. Native viewer
API review passes through 12 s, separately 24,000 engineering steps, and the
responsive window remains open/paused at edge hover (window process 21420).
**Known Issues:** Historical 94/96 directional failures remain. Windows capture/input
helper has no native pipe; live mouse/keyboard review unavailable. Some labels
disappear transiently in saved native captures; flight/hover rendering reviewed.
**Important Decisions:** User replaced weak-threat recovery with continuous flight
to an endpoint 15 mm inside a map edge, then hover. One agent; frozen neurons and
readout; IMU/known-start calibration only. No learning, commit or publishing.
**Next Task:** One viewer task / preferred Sol High: reproduce intermittent
missing labels in saved native captures and complete live input review when the
native helper is available. Preserve the verified controller/receipts; do not
automatically begin another neural experiment or repair the historical readout.
**How To Run:** `./.venv/Scripts/python.exe -m flybrain.escape_edge` launches the
verified candidate. R/new trial, Spawn predator, Space/pause, Backspace/replay,
WASD/QE move predator. Historical launchers remain unchanged.
**How To Test:** See [exact commands/evidence](docs/verification/escape-edge.md).
[Guide and limits](docs/ESCAPE-EDGE.md); [measured results](docs/ESCAPE-EDGE-RESULTS.md).

# Escape completion repair in progress - 2026-10-08

The user now authorizes diagnosis, implementation and verification of escape.
One additive policy repair is being prepared; historical results stay closed.
Root causes reproduced: rectified pixel jitter prevents weak clearance; idle
recovery commands outward motion while turning home too slowly.
Motor regressions pass all 24 direction/start combinations and 20-second
recovery checks. Full neural evaluation has not started.
Active task: [escape repair](docs/plans/ESCAPE-REPAIR.md). One agent, no learning.

# FlyBrain Current Status

## Current Milestone - physical validation complete; gate failed, 2026-10-08

**What Works:** Frozen 589-cell/4,139-edge/(4,2) candidate actually flies in the
one measured protocol. All 24 intact sides timely (<=220 ms), all 24 actual signed
90 degree turns before flee, all 8 resting takeoffs; min separation 159.908 mm. Complete
48 cases /288 simulated seconds /576,000 individual neural steps /14,400 records.
Independent saved audit passes at zero new steps; exact replay passes all 48
including silent interventions (576,000 separately journaled verification steps).
117 targeted /847 final full tests pass, zero skips. All 4,858 protected files and
original 1,747-step diagnostic proof unchanged. [Results](docs/ESCAPE-FLIGHT-VALIDATION-RESULTS.md).

**In Progress:** None for this consumed experiment. Physical gate failed; the experiment is closed.
New prototype activation/launch withheld after failure;
live review remains pending/unavailable. No policy tuning, retry or learning.

**Known Issues:** Intact 19/24 below required 22; resting 6/8 and cruising 6/8 below required 7,
hovering 7/8. Three post-recovery boundary collisions; two cruising cases lack
weak-threat braking/recovery within 6 s. Silent triggers 0. Original directional
experiment still failed at 94/96, 86.25 degrees (4/6); known 1100018/1100021 unchanged.
Frozen raw audit CLI rejects NumPy vector repr in CSV; use the verified saved-only
format adapter below, preserving raw records and source archives.

**Important Decisions:** Prospective 94/96 prototype exception superseded the
previous next-readout request; it did not relabel the original failed gate.
Frozen readout/encoding/neuron rules/flight policy remain unchanged. No new
motion-only baseline. Science 576,000 steps; required exact replay adds 576,000
verification steps, zero new scientific cases. These 48 starts are now historical
benchmark cases, not unseen evaluation. No commit, push or automatic new task.

**Next Task:** One saved-record diagnosis and sensor-only flight-policy repair
proposal addressing boundary-safe recovery and weak-threat completion, with a
new predeclared budget before any experiment. Preferred Sol High, one agent;
keep accepted readout frozen. Do not automatically tune/retry or begin learning.

**How To Run:** `./.venv/Scripts/python.exe -m flybrain.escape_flight_validation`
refuses because physical activation failed. Historical launchers/defaults are
unchanged. Scientific worker/replay are consumed actions; do not rerun them.
Read-only audit: `./.venv/Scripts/python.exe artifacts/escape-flight-validation-2026-10-08/audit_serialized.py`.

**How To Test:** `./.venv/Scripts/python.exe -m pytest -q tests artifacts/escape-flight-validation-2026-10-08/test_serialized_audit.py -p flybrain.escape_recovery.pytest_plugin`.
Use OMP/MKL threads 1 and a fresh FLYBRAIN_TEST_LEDGERS directory to retain consumed
proofs. [Exact commands and evidence](docs/verification/escape-flight-validation.md).

## Temporal behavior repair implemented; gate failed - 2026-10-07

Implemented and evaluated one real controller replacement. Historical accuracy
improves **87/96 to 94/96**, correcting all nine previous failures. Two new
regressions at **86.25 degrees (4/6)** fail required >=5/6 per azimuth. The
single-candidate experiment closes at **96 cases /144 s /288,000 neural steps**.
No closed-loop cases or activation. Physical escape is not fixed/verified.

Remaining issue: valid left evidence lasts one sample in seed 1100018, so the
new two-sample rule suppresses it. Seed 1100021 has correct L=3/R=1 at .86 s,
then wrong L=1/R=2 at .88/.90; the rule waits and commits right. No retuning.
Independent saved-record audit passes, every unchanged neural/image record
matches history, archived sources/records preserved. Corrected targeted **27**,
final full **812** tests pass, zero skips, two/four upstream warnings. Source
readout/replay/trajectory gates implemented; saved regression figure reviewed.
One missing stdlib math binding failed at zero steps; additive corrected launch
preserves that receipt and unchanged behavior parameters. No scientific retry.
[Measured implementation/results](docs/ESCAPE-RECOVERY-RESULTS.md),
[execution ledger](docs/plans/ESCAPE-RECOVERY.md).
Next task / Sol High: agree one event-aware readout that retains brief phase
cues before any further experiment. User selected one candidate/honest closure;
no automatic fallback, altered threshold or unverified new launcher.

## Escape timing diagnosis complete - supported, 2026-10-07

All eighteen complete historical cases /1,350 intervals exactly match original
RGB, counts, diagnostics, rolling scores, threat and first side. All five
predeclared deficit-only cases show redistribution, exact expiration-generated
-1 b3 windows, positive final cumulative counts and prior right-DN delivery.
Observational support; causal necessity and other routes remain unresolved.

Final detached worker completed five cases /7.5 s /15,000 steps normally with
append-only journals. Aggregate complete episodes: 27 s /54,000 individual
steps. All executed attempts: 28.68 s /57,360 steps; 57,400 attempted reservations
include 40 unexecuted steps. 36 guarded tests pass, zero new engineering calls
(cumulative 1,747). Saved figures reviewed; no live review. Full suite excluded.
No neural decoder was evaluated or implemented; historical 87/96 unchanged,
escape disabled. All prior sources/records/keys preserved.
[Complete report](docs/ESCAPE-DIAGNOSTIC-COMPLETE.md),
[one fixed temporal readout proposal](docs/ESCAPE-TEMPORAL-READOUT-PROPOSAL.md).
Next task / Sol High: approve the concrete cumulative/positive/two-sample
spike-only readout and its single gated directional/closed-loop experiment.

## Thirteen cases verified; Windows journal repair tested - 2026-10-07

Detached worker completed six additional cases, then Windows denied replacing
progress.json during the next reservation. Failed receipt sealed normally.
505 intervals/chunks exactly match history and all recorded neural equations.
This run executed 10.10 s /20,200 steps; attempted counter is 20,240, with 40
not executed. Across attempts: 21.18 s /42,360 executed steps; 42,400 attempted.
No resume/retry or scientific mismatch. Thirteen of eighteen cases complete;
four of five deficit-only timing tests supported, fifth unstarted: unresolved.

Reader-held file replacement reproduces WinError 5. New append-only reservation
and heartbeat utility passes reader/fsync tests. 30 guarded saved-only tests pass,
zero added kernels (engineering still 1,747). Frozen consumed runner untouched.
[Report](docs/ESCAPE-DIAGNOSTIC-JOURNAL-RESULTS.md),
[one final-five-case proposal](docs/ESCAPE-DIAGNOSTIC-FINAL-FIVE-PROPOSAL.md).
Next task / Sol High: authorize five fresh cases /7.5 s /15,000 steps with tested
journals, retaining thirteen complete records. Escape disabled; 87/96 unchanged.

## Comparator fixed; diagnostic externally interrupted - 2026-10-07

21 saved-only tests pass, zero added engineering steps (cumulative 1,747).
Seven complete cases /552 flushed intervals match historical records exactly.
The process disappeared during case eight; no Python process/terminal exception
remained. External cause unresolved. 553 neural chunks verify all cell/edge
steps: 11.06 simulated seconds /22,120 individual steps. Both diagnostic attempts
now total 11.08 seconds /22,160 steps. The interrupted receipt is sealed/consumed;
no resume or automatic retry occurred.

Two completed deficit-only cases satisfy the four timing conditions, but three
were not reached: overall hypothesis unresolved. Comparator repair is fixed;
escape remains disabled, historical 87/96 unchanged. No neural repair/activation.
Saved partial figure reviewed separately; no live review. Full suite excluded.
[Measured report](docs/ESCAPE-DIAGNOSTIC-RECOVERY-RESULTS.md),
[one remaining-case completion proposal](docs/ESCAPE-DIAGNOSTIC-REMAINING-PROPOSAL.md).
Next task / Sol High: authorize exactly eleven fresh remaining cases, including
repeat of the interrupted eighth start, for 16.5 seconds /33,000 additional steps.

## Escape diagnostic ? closed incomplete, 2026-10-07

Passive instrumentation passed 10 synthetic preflight tests and final 6 checks
(1,747/2,000 engineering steps). The one scientific run stopped in the first
20 ms baseline interval: **40 individual steps, zero completed presentations,
zero threat images**. Runtime tuple diagnostics were compared with JSON lists;
the saved serialized interval matches the historical interval exactly.

The failed receipt is sealed/consumed. Guarded saved-only audit verifies all
589 cells/3,636 non-self edges for that interval, adding zero simulation calls.
**Hypothesis unresolved; no retry, neural repair, training or activation.**
Historical 87/96 remains unchanged; escape disabled. Saved baseline figure
reviewed separately; no live review. All 2,510 originals and 2,720 snapshot files
are unchanged. Full simulation suite deferred by explicit user instruction.
[Report](docs/ESCAPE-DIAGNOSTIC.md),
[one comparator-only recovery proposal](docs/ESCAPE-DIAGNOSTIC-RECOVERY-PROPOSAL.md).
Next task / Sol High: approve or reject that precise correction and one fresh
18-case diagnostic; additional budget and previous 40 steps are explicit.

## Frozen wrong-side review — complete, 2026-10-06

Saved-record reconstruction confirms three failed seeded ties and six unequal
scores. Five failures consist solely of a −1 left b3 rolling increment; all
nine later record left-favoring scores within their saved 1.50 s endpoint.
No replacement decoder or improved performance was tested. The frozen model
has 46 left/50 right LPLC4 cells with an engineered ID-ranked field assignment;
source side conventions show no global inversion. Causal attribution remains
limited by missing RGB, reference voltages and edge-resolved currents.

Zero new presentations/seconds/neural steps. Receipts/current and archived
sources pass; all 2,510 originals and the 2,720-file review input snapshot are
unchanged. Independent tables cover 10,800 intervals and 42 comparisons, with
a visually checked nine-panel figure. Runtime audit guard records zero brain/
sensor simulation calls. Full/targeted simulation suites were not rerun under
the zero budget. [Report and proposed instrumentation](docs/TURNFLEE-WRONG-SIDE-REVIEW.md).
One next task / Sol High: decide on that single unchanged-model, 18 historical
presentation proposal (27 s /54,000 paired steps). Separate authorization is
required; no instrumentation experiment has run. Escape remains disabled.

## Turn-and-flee — experiment closed at validation, 2026-10-06

Additive candidate preserves all 589 cells, 4,139 anatomical edges and original
motion/visual/LIF rules. Calibrates only 143 LPLC4→DNp03/DNp26 and 86 outgoing DN
effective weights; implements spike-only side control, 90° turn, braking/flee,
weak recovery, IMU home guard, immutable diagnostics and configuration-bound replay.
Historical launchers and all **2,510 original files are unchanged**.

Calibration: (2,1) **13/16**, (4,2) **16/16**, (8,4) **16/16**. Lowest eligible
**(4,2)** frozen. Validation **87/96**, below >=90/96 and >=5/6 per azimuth;
18.75°/63.75° groups are 3/6. Nine timely responses have the wrong side.
All 3,600 motion-baseline intervals match, state finite, max downstream 190 Hz.
The experiment closed at **144 cases /216 s /432,000 paired neural steps**.
**No closed-loop cases, default promotion, extra gains, tuning or retry.**

**188 targeted /755 full-suite passed**, zero skips, three upstream warning instances.
Independent saved-record audit passes. Saved development frames/GPU tests are
separate evidence; native Computer Use helper pipe fails, so live review remains
pending. The policy is implemented/tested; verified turn-and-flee was not achieved.
[Guide](docs/TURNFLEE.md), [plan](docs/plans/TURNFLEE.md),
[verification](docs/verification/turnflee.md).

## Idle encounter lab — partial delivery, 2026-10-06

Additive `python -m flybrain.retreat3d` starts without an object and performs
engineered seeded takeoff/cruise/landing/rest. Continuous exclusive spawning,
flower teacher IMU handoff, WASD/QE fixed-step predator movement, relative swept
collision and exact action-log replay to endpoint are implemented. Original
589-cell model, dynamics, diagnostics and historical launchers are preserved.

**Motion repair gate failed: 9/96 directional validation cases**, versus ≥90/96
and ≥5/6 each azimuth. The one experiment closed at 48 calibration + 96 validation
cases / 216 simulated seconds / 432,000 paired neural steps. No closed-loop
evaluation, neural escape/takeoff, stopping/recovery policy, tuning or retry.
The frozen failed spike decoder is diagnostic only; idle movement is engineered.

**153 targeted / 720 full-suite passed**, zero skips. One targeted/three full
upstream warning instances. The initial full run's provenance failure was caused
by a concurrent additive HUD edit; isolated and frozen-source rechecks pass.
Offline scientific audit passes. All **2,433 protected original files unchanged**.
Saved frames show idle/ground and moving predator; native helper pipe fails.
Human live review remains pending. [Guide](docs/RETREAT3D.md),
[plan](docs/plans/RETREAT3D.md), [verification](docs/verification/retreat3d.md).

## Brain-to-wing diagnostic — 2026-10-05

The encounter viewer now has a passive **Pathway diagnostics** section showing
visual phase inputs, DNp03/DNp26 voltage responses and wing-control cells.
Every completed interval records current sum/peak, discarded incoming current,
pre-reset voltage/phase peaks, counts and exact 1 ms first-spike times. Original
voltages, spikes, histories and edge transmissions are bit-identical in checks.
The user confirmed **“Panel and controls work”** in the paused native preview.

The one fixed-model investigation closed at **120 one-second presentations /
120,000 neural steps**. Intact totals: visual **13,568 spikes**, DNp03 **60**,
DNp26 **1**, wings **0**. Retained paths reach all 48 wings, but maximum wing
voltage is **.190239 / 1**. Near wing voltages distinguish eight directions;
the spike outputs collapse to identical zeros. DN interventions change individual
wing currents/voltages even though no condition produces wing spikes.

**Proposed next repair:** restore the existing body-local motion inputs alongside
vision in a separately agreed candidate, with a motion-only baseline and a
directional stimulus increment gate. This candidate is unimplemented/untested.
No decoder, retreat, cell addition, training, sweep or scientific retry occurred.
Historical sources/data/experiments are preserved; three encounter integration
files have hash-verified original-byte archives. [Report](docs/PATHWAY-DIAGNOSTICS.md),
[verification](docs/verification/pathway-diagnostics.md),
[active completed-task plan](docs/plans/PATHWAY-DIAGNOSTICS.md).

Fresh **111 targeted / 678 full-suite passed**, zero skips. Four panel tests
passed again after the final effective/discarded-current label clarification.
The full suite has three upstream warning instances (Pygame and TorchScript).
Saved-record audit and local one-agent acceptance review pass. No implementation
or verification work remains in this diagnostic task.

## Exclusive 3D encounters — 2026-10-05

Additive `python -m flybrain.encounters3d` provides Spawn flower / Spawn predator,
one stationary object per trial, unchanged teacher-controlled flowers, actual
recorded visual-circuit activity, R/new selected object and Backspace/exact replay.
Reset clears body/controller/neural/display history and retains pause/camera/
display preferences. The predator is a high-contrast cyan sphere in world and eyes.

**Avoidance gate failed:** all 16 intact physical image cases produced zero wing
endpoint spikes. No escape decoder or retreat/retrigger policy was selected.
Predator mode explicitly labels unavailable avoidance and IMU hover scaffolding.
Synthetic full-field drive: 1,838 intact / 1,476 DNp03-silent / zero visual-silent
wing spikes; DNp03 contributes but is not the sole route. All 24 five-second
closed-loop cases hovered with zero separation gain/collisions and absent retreat.
No training, optimizer updates, historical assay or interrupted fit continuation.

Fresh **100 targeted / 667 full-suite passed**, zero skips, two existing warnings.
All **2,465 historical protected files** are unchanged. Offscreen frames/GPU color
and injected controls pass; native helper access failed. A native preview is open
(seed 980000, reopened launcher PID 24840 / window PID 6692). The user confirms
all encounter controls work and observes placement-dependent brain activity with
no fly response. This verifies the encounter controls, not neural avoidance or
the remaining historical activity-view checks.
Verification and live-review limits are recorded in
[encounter verification](docs/verification/encounters3d.md).
[Guide / sourced recovery proposal](docs/ENCOUNTERS3D.md),
[active task](docs/plans/ENCOUNTERS3D.md).


## Smooth after-spike fade — verified 2026-10-05

Each neuron now eases its class color and alpha back to idle blue/violet over
600 ms after its last spike. The former fast exponential envelope/hard cutoff is
replaced by cubic smoothstep: full color initially, half strength at 300 ms and
zero at 600 ms, with smooth endpoints. New spikes refresh color; pause freezes
the fade. Explicit switch/scope exclusions remain immediate display choices.
The 200 ms rate average, complete spike recording and neural computation stay
unchanged. [Evidence](docs/verification/flight3d-activity-fade.md).

Fresh **87 targeted / 654 full-suite passed**, zero skips, two existing warnings.
Numerical/GPU fade checks and 30/60/120 Hz equality pass; rendered fade frames/GIF
were reviewed. All **2,370 protected files** are unchanged,
including cached geometry/scientific sources and the interrupted fit. No experiment
training or reserved assay ran. Updated teacher preview launcher PID 4408, seed
970000; native fade confirmation remains pending. The review instance is running
All activity with settings open; normal launches still start odor-only. Class-switch live acceptance
from October 4 is preserved. Work is complete within this display task; no next
milestone or experiment is started.

## Switchable activity classes — automatically verified 2026-10-04

The anatomical viewer now has ten independent display class switches, expandable
Sensory with off/on/mixed parent state, runtime counts, and Whole model/Selected
circuit scope. Fresh launches enable Odor inputs only. The user's first live
feedback found switches hard to locate; all ten now
appear at the top of settings in two columns with visible button backgrounds.
The user now confirms class switching works; other native checks remain pending.
All activity enables every class and whole scope; Activity off retains spike history. A two-column swatch
legend precedes navigation help; cyan inspection stays independent. Paused changes
recolor immediately without rebuilding geometry. Replay/new trial retain switches
and scope while resetting history. Clearing/background-only selection returns to
whole scope. Cord-only/missing geometry is explained with explicit cord action.

Fresh **83 targeted / 650 full-suite passed**, zero skips, two existing warnings.
Rendered GPU palette/RGBA and legend pixels pass; saved layouts at 1440×900 and
1000×700 were inspected. All sixteen isolated p95 frame-work values pass <33 ms;
maximum **14.10 ms**. All activity has **359/527** hidden/shown-cord pulses; odor
only has **69/69**, while the full model remains simulated and recorded. All
**2,368 protected existing files** are unchanged, including scientific sources,
cached geometry and the incomplete **1,050-update** fit. No experiment training or
reserved assay ran. [Verification](docs/verification/flight3d-activity-classes.md).

The user confirmed **"Yes, the switches work now"** in the revised native preview:
independent class color switching is live-verified. Other native interactions
(Sensory parent, selected scope/cyan, pause/replay/new trial, cord/navigation)
remain unconfirmed. Teacher preview seed 970000, launcher PID 27060 / window PID
25144, starts paused at two simulated seconds with Display settings open. Its
foreground warning did not prevent the user review; the native window is responding.
Automated/saved-frame evidence does not substitute for remaining live checks.

## Actual modeled firing - historical verification 2026-10-04

The default anatomical display now records every completed simulation interval
and colors every firing neuron with available visible branches. No twelve-cell
cap, outgoing-edge requirement or metadata-region exclusion remains. Static
owner-indexed batches use a compact explicit-RGBA GPU table. Cyan selection,
idle brightness, geometry and camera/navigation are preserved. Inspector values
show latest spikes, cumulative spikes, recent rate and time since activity;
missing geometry and absence of non-self inputs are explained separately.

Fresh **76 targeted / 643 full-suite passed**, zero full-suite skips, two existing
warnings. Passing isolated hidden/shown p95 frame work is at most **12.16 ms**
with **359/527 visible pulses** at 1440×900. An earlier shown-cord run failed;
timing variability remains documented rather than claiming uniformly low latency.
The two-second diagnostic reproduces **530 fired / 59 silent** (50 visual,
six other self-only, three with silent upstream cells), including the 56 firing
cells previously excluded despite having visible brain branches.
See [verification](docs/verification/flight3d-spike-activity.md).

The user says this view is good and wants it retained as an option alongside a
smaller **defined-circuit** view. That initial proposal is superseded by the
implemented class switches above. Individual native pause/reset/cord/navigation checks are
unconfirmed. Preview launched with seed 970000; foreground activation warned.
All **2,365 protected files**, including cached geometry, scientific sources and
the incomplete **1,050-update** fit, are unchanged. No training or assay ran.

## Clearer controls and brain navigation - 2026-10-02

Implemented the user's camera/control specification while preserving the accepted
appearance, teacher preview and all scientific/cache/experiment bytes. The brain
camera uses continuous quaternion rotation, pixel input, 60 ms rotation/pan and
unchanged wheel multipliers/120 ms smoothing. It reaches 0.03 mm with either cord
visibility. Cord toggles preserve every camera state/target and matrix exactly;
presets preserve pan/zoom; Fit view and Reset view explicitly frame anatomy.

Main toolbar: Fit view, presets (Views menu below 1200 px), Show nerve cord and
Display settings. Background anatomy defaults on and retains Tab; brightness is
in settings; N/C are removed. Selection is passive, with identity/model/source,
Focus selected, Clear selection and default-off retained connection inspection.

Fresh checks: **61 targeted / 628 full-suite passed**, zero full-suite skips, two
existing upstream warnings. Isolated p95 paused/running/orbit/select:
**4.44 / 5.87 / 3.91 / 9.05 ms**; paced display about 58.5 FPS; simulation throughput
0.97 s/wall s in that run. [Evidence](docs/verification/flight3d-controls.md).
Live teacher preview is open. The user confirmed panning and scrolling are good;
remaining live interactions are not yet confirmed. They clarified the intended
toggle: whole fly brain on, exact project subset off. Off currently isolates
project branches; on adds a registered surface/reference-point overview, not every
full-brain skeleton. Native automation helper is unavailable. The interrupted
**1,050-update** fit remains untouched.

Activity-display clarification: a bounded two-second seed-970000 teacher preview
recorded 530/589 cells firing and 59 silent in that window. Of the brain-only
highlight-eligible cells, 281 fired without ever lighting up. Warm highlights
rank outgoing delivered current and cap emphasis at 12; they do not display
every spike. Model membership does not depend on firing. See the verification
record above; no activity rule, neural dynamics or scientific result changed.

## Anatomical visual repair - verified 2026-10-02

The default 3D viewer again uses actual cached source skeletons, registered surface
and source-reference context on black. Both optic lobes and the central brain stay
recognizable; up to twelve modeled active neurons are warm, full selected branches
are cyan, and centroid connections appear only on explicit selected inspection.
Source/neural/sensor calculations and historical evidence are preserved. One serial
simulation worker keeps fixed numerical steps separate from display frames.

The September 30 teacher failure is historical. October 1 sensory development
passed 49/49, and its separately authorized experiment passed 48/48 teacher cases
before an execution interruption at **1,050 completed updates**. The partial fit
is **incomplete**, with consumed receipt, checkpoints and journals preserved.
No restart, further fitting or reserved evaluation occurred in visual repair.
See [guide](docs/FLIGHT3D.md) and
[repair ledger](docs/plans/FLIGHT3D-ANATOMICAL-REPAIR.md).

## Historical 3D flight and referenced anatomy — 2026-09-30

The user explicitly authorized the [3D plan](docs/plans/FLIGHT3D.md), overriding
the earlier 3D deferral. The isolated Panda3D lab, headless physics/sensors,
589-cell/4,139-edge BANC registry, motor-specific readouts and anatomical viewer
are implemented. The single bounded experiment **closed at its teacher gate**:
neural prerequisites passed, but seed 960000 produced no flower detections or
food contact in 30 seconds. **Zero optimizer updates; no 3D trained checkpoint;
validation/final not evaluated.** See [3D guide](docs/FLIGHT3D.md) and
[recorded outcome](experiments/flight3d-landing-2026-09-30/results.json).

## Approach-and-stop repair — 2026-09-29

The 85-cell candidate preserves 83 sourced BANC cells and adds two labeled
engineered contact cells. One 1,500-update fit changed only its 419 readout
parameters and passed 40/40 full training cases. Fresh validation passed
**16/16**. The first final process disappeared without a result; all its
reserved cases remain historical. After the user's explicit continuation, one
new final-only frozen assay passed **63/64** full 20-second cases versus
**0/64** for each matched control, with seven or eight successes in every
heading sector and exact paired p ≈ 1.08e-19. The one failure was seed 840044.
The separate contact-block diagnostic changed stopping (4/5 last intact
ticks versus 0/5 blocked). All 192 final episodes and the frozen parameters
reconcile with the [recorded result](experiments/food-stop-recovery-2026-09-29/results.json).
The passing controller is now the default `python -m flybrain`. The user
completed the live review: the HUD and contact pathway were visible, stopping
and dwell appeared, and R/Backspace worked. The historical arrival-only model
remains available with `--control banc-trained`.

## Historical reassessment — 2026-09-28

**At that date, project acceptance was open beyond a desktop check.**
The user requires varied random headings, learned speed control, and slowing to a
stop at food. The existing checkpoint was trained/evaluated for first arrival only.
All 15,864 target speeds are 0.7; the saved model cannot command a stationary stop.
Two historical starts continued for 20 seconds both passed through food. See
[behavior review and next task](docs/plans/BEHAVIOR-REVIEW.md) and
[recorded diagnosis](experiments/behavior-review-2026-09-28/results.json).

The current app now starts a fresh random-heading trial with **R**, replays the
current trial with **Backspace**, and shows trial seed, commanded/actual speed,
food contact and stationary-contact duration. `--seed` reproduces the sequence;
omitting it chooses a fresh launch seed. A local food-contact observation is
implemented at the user's request. The historical default model still ignores
that signal. A separate learned candidate now slows and stops in training and
validation, but lacks a completed final assay.

### Historical arrival evidence

Last verified: 2026-09-28. **FB-10's original frozen candidate remains accepted (15/16 validation, 31/32 final), and its extra covered-training failure remains preserved. A separate, capped 750-update same-method fit passed that extra gate and its own fresh assay (16/16 validation, 31/32 final).**

The earlier `python -m flybrain` app loaded the verified 750-update model;
that arrival-only controller remains selectable with `--control banc-trained`.
The historical 500-update launcher is preserved. Automated integration and saved-frame
review pass; live desktop keyboard/window review is pending because the native
inspection helper is unavailable. See [integration task](docs/plans/TRAINED-BANC-APP.md).

## Current Milestone

- Current 2026-10-07: Actual temporal controller implemented/evaluated; 94/96 but 86.25-degree group 4/6 fails. Physical escape remains unverified; app disabled.

- Frozen wrong-side review complete; failed 87/96 validation remains closed.
  One instrumentation hypothesis proposed for a later user decision.

- Turn-and-flee candidate implemented, but signed validation failed 87/96. Experiment closed; default escape remains unavailable. No closed-loop assay.

- Idle/continuous interaction lab implemented and automatically verified in the
  targeted suite. Neural escape is not delivered: motion-restoration directional
  gate closed at 9/96; grounded escape and all closed-loop tests were not run.

- Brain-to-wing diagnostics: implemented and human live-reviewed; the bounded
  fixed-model investigation closed with weak descending/wing electrical drive
  and loss of direction at the wing spike threshold. Repair proposal only;
  avoidance remains unavailable.

- Exclusive encounters: infrastructure and actual visual-circuit probe delivered;
  directional neural avoidance failed the characterization gate and remains
  unavailable. No retreat, retrigger or learning claim.

- 3D flight lab: implementation and anatomical integration delivered as a
  research preview. Actual firing, accepted appearance and revised camera/controls
  are implemented and automatically verified. User approves the full-activity view
  as an option; independent class switches and selected-circuit activity are now
  implemented and automatically/saved-frame verified, including the smooth
  after-spike fade. Class switches are also user-confirmed.
  Individual native interactions remain unconfirmed. Learned landing remains open
  after the October 1 interrupted fit. The historical accepted 2D result below is
  independent of this new task.

- Approach-and-stop candidate: sensory pathway, teacher, one fit, validation,
  new final and contact diagnostic gates verified; user-observed live review
  passed and the checkpoint is the default.
- Behavior reassessment: arrival-only acceptance was too narrow for the user's
  intended simulation. Random-trial controls, food-contact sensing and stationary
  dwell measurement are implemented. The food model passed its recorded
  approach-and-stop gates; broader center-start reliability remains unassessed.
- M0-M5: deterministic world, sensory inputs, synthetic spiking controller, closed-loop movement and read-only brain view implemented.
- M6 / FB-07 recovery: learned synthetic charge readout passed its recorded fresh assay. M7 / FB-08 interventions are complete; no downstream topology benefit established.
- M8 / FB-09: sourced 83-cell BANC subset, verified engineered dynamics, provenance and fixed BANC app mode complete under the recorded sustained-rate limits.
- FB-10: six historical 500-update fits and all failures preserved. Following the user's continuation of the explicit acceptance proposal, the exact frozen PN-input head was admitted using the original training gates, with its additional covered-data failure disclosed.
- New matched assay: validation **15/16 trained, 1/16 untrained, 1/16 fixed**; final **31/32, 4/32, 3/32**. All unchanged fresh arrival, paired-effect, progress and integrity gates pass. No new optimizer updates.
- Trained app integration: `flybrain.trained_app --control banc-trained` loads the exact accepted frozen head; existing modes delegate to the original app. No retraining or new scientific assay.
- Separate extra-training attempt: one seed-1010, 750-update fit on the same 15,864 rows, model and Adam objective passed the previously missed covered ceiling. Its independently locked fresh assay passed without fitting on validation/final outcomes. The 750-update head remains available as `--control banc-trained`; `flybrain.trained_app` retains the established 500-update head.

## What Works

- Current 2026-10-07: New cumulative side gate, retained flight integration, reset/rearm, intervention-bound replay, measured physical scoring, append-only supervision; 27 targeted/812 full tests pass.

- Reproducible saved-record wrong-side review: exact six-cell contributions,
  seeded tie choices, complete score timelines and exhaustive comparisons.
  See [review report](docs/TURNFLEE-WRONG-SIDE-REVIEW.md).

- Portfolio knowledge base updated on 2026-10-06 at
  `C:/Users/Lenovo/Desktop/Gabriel_Paunica_Portfolio_Knowledge_Base.md`.
  It separates verified 2D approach-and-stop, sourced 3D anatomy, the incomplete
  1,050-update landing fit and failed escape gates. Original-byte backup retained;
  document/result checks pass. No model, experiment or milestone changed.

- Immutable effective-gain response configuration, rolling 200 ms paired spike readout, tested spike-only turn/flee state machine, passive response diagnostics and old/new exact endpoint replay. These are experimental implementation capabilities, not successful physical escape evidence.

- Additive seeded idle cycles, safe ground landing/support, sensory-only estimate
  transfer to the retained flower teacher, continuous optional encounters,
  normalized/clamped moving predator, swept collision, reset and exact action
  replay, passive actual-brain display and failed-decoder diagnostic confidence.

- Read-only pathway panel, exact per-substep diagnostic measurements, atomic
  interval publication, reset/replay/pause and retained scene/camera/preferences.
  A full scientific trace and saved-record audit cover all 120 presentations.

- Exclusive flower/predator scene, RGB visibility/occlusion/expansion, source-ID
  preservation, teacher parity, deterministic reset/replay, pause barrier and
  passive rendering. Recorded DNp03 and wing signals are independent of motor
  commands; explicit predator color matches eye images in the framebuffer.

- Continuous 600 ms class-color fade after each last spike, independent per neuron,
  with smooth endpoints, refresh on firing, frozen pause and clean reset. Real GPU
  intermediate/tail checks and refresh-rate equality are verified.
- Ten independent activity classes with runtime membership, sensory parent state,
  class RGB endpoints, whole/selected scope and a two-column color legend.
  Same-frame paused recoloring, all/off, replay retention and class/scope masking
  are verified with complete spike recording. Odor-only is the launch default.
- Default anatomical viewer: continuous display-only quaternion camera, explicit
  fit/reset, passive inspector, deep zoom and independent fly camera.
- Default anatomical viewer: batched source branches, soft registered context,
  cached full selection, bounded progressive loading, paginated neuron browsing,
  passive uncapped spike activity and independent camera/panel controls.
- Display-only completed-step recorder and atomic immutable publication retain
  activity between refreshes; pause freezes pulses and same-seed replay clears
  history. Per-neuron GPU colors and 30/60/120 Hz matched-time equality are verified.
- UI scheduling preserves 20 ms controls, 5 ms physics and 1 ms neural steps.
  Serial/background snapshots and neural voltages match exactly in checks.

- `python -m flybrain.flight3d`: SI-unit vertical/horizontal flight, fixed pose,
  random flower position/height, local eyes/odor/motion/contact senses, physical
  landing/stem/ground checks, R/new and Backspace/replay.
- Source registry preserves string IDs, annotations, directed edges and
  coordinates. Actual brain/cord surfaces and progressively loaded skeletons
  use one coordinate transform; the viewer provides rotation/zoom, presets,
  whole/project modes, source inspection and simulation-time activity.
- Motor readouts are restricted to the matching wing-power, wing-steering and
  leg-extension endpoint groups. Checkpoint source binding and motor silence
  are tested. Neural signal/cut prerequisites passed in the recorded run.

- Candidate numerical checks cover odor-rate contrast in real spikes, delayed
  contact-to-relay propagation, reset, 22-feature histories, exactly zero
  learned stop, 419-parameter save/reload and frozen inference. The teacher
  passed 40/40 training cases; the candidate fit passed 40/40 matched training
  and 16/16 validation cases. The original pre-recovery focused checks were
  **19 passed**. The latest full suite passed **567 tests in 111.61 s**, zero
  skips, with one existing upstream Pygame deprecation warning. The default
  launcher assertion now checks the food mode; focused app/recovery checks
  passed **15 tests** after the launcher change.
- The one-use recovery receipt, per-episode fsynced journal, JSON/CSV and
  offline audit reconcile **192 episodes / 192,000 world ticks / 3,840,000
  neural steps / zero optimizer updates**. The app loader pins those records
  and the unchanged source/checkpoint bytes. Its focused integration tests
  cover reset, 20:1 timing, motor disable, rendering independence, R/replay
  and historical mode delegation. The accepted saved frame was inspected at
  2.6 simulated seconds with 2.48 seconds of stationary contact. Both
  120-tick headless smokes passed for default `food-trained` and selectable
  historical `banc-trained` modes.
- Historical validation/final headings were already random across the full
  circle, with 16/16 and 32/32 distinct orientations; matched controls shared cases.
- New current-app trials use distinct headings from a local recorded seed stream.
  Food contact supplies only a boolean alongside the two odor values. Dwell scoring
  cannot count a pass-through, wall stall, or continuing turn as a stationary stop.
- Historical reference integration: `python -m flybrain --control banc-trained --seed 7` loads the exact verified 750-update checkpoint, checks five evidence files plus 68 recorded source and four accepted data hashes, freezes all 370 parameters, and labels the model explicitly. Historical entry points remain intact.
- Focused old/new app tests: **32 passed**, zero skips. The 120-tick default-launcher smoke completed; the [saved frame](docs/verification/verified-app-frame.png) was visually inspected at neural step2400. All **321** pre-integration source/experiment files remain byte-identical. Full suite: **523 passed** in 101.56 s, zero skips, one existing upstream Pygame deprecation warning. Reports: `.verified-app-targeted.xml`, `.verified-app-full.xml`.

- Post-assay software verification: focused extra-training/evaluation tests **9 passed**; complete suite **504 passed** in 93.55 s, zero skips, one existing upstream Pygame warning. Current reports: `.verified-app-targeted.xml` and `.verified-app-full.xml`. Historical reports: `.fb10-extra-final-targeted.xml`, `.fb10-extra-final-full.xml`. Scientific and file audits are recorded separately.
- Extra fit: 750 updates; covered turn MSE **0.0213424152 ≤ 0.0222527315**, original-data MSE **0.0100253166**; original train arrivals **8/8 versus 0/8 untrained and 0/8 fixed**. All nine integrity checks pass; 370 changed/readout-gradient scalars. 25 training-case episodes, 10,778 ticks, 215,560 neural steps. Saved [fit record](experiments/fb10-extra-training/results.json).
- Extra fresh assay: validation **16/16 versus 0/16 and 2/16**; final **31/32 versus 2/32 and 2/32**. Final paired gains +29/+29, exact one-sided p=1.862645149230957e-9 for each. All arrival, progress and integrity gates pass. 144 episodes, 64,106 ticks, 1,282,120 neural steps, zero assay updates. [Result](experiments/fb10-extra-evaluation/results.json), [plot](experiments/fb10-extra-evaluation/extra_training_summary.png), and [25-check offline audit](experiments/fb10-extra-evaluation/audit.json).
- All 303 files present before the extra fit and all 312 before its fresh assay remain byte-identical. New 730000/731000 case identities are now historical. The single-use training and assay receipts are consumed.
- App verification: full-suite invocation **494 passed**, 263.41 s; final targeted integration suite **13 passed**, 18.10 s. The reset/exit event test was added after full-suite collection and passed in the final targeted run (495 unique tests covered across both runs). Zero skips; one existing upstream Pygame warning per run. Reports: `.trained-app-full.xml`, `.trained-app-targeted.xml`.
- Additive app/loader: pinned checkpoint, fit and assay records, recorded source/data hashes and strict model contract; 370 frozen parameters. Reset replay, 20:1 neural/world timing, motor gate and rendering independence verified. The 120-tick smoke and [saved frame](docs/verification/trained-app-frame.png) show step2400. All 300 pre-session source/data/experiment files are byte-identical.
- Python 3.12.10, Pygame 2.6.1, CPU PyTorch 2.14.0+cpu. Post-assay full suite: **482 passed**, 51.95 s, zero skips, one existing upstream warning. Pre-assay full suite also passed 482 tests in 56.92 s. Targeted new runner suite: **12 passed**, 33.54 s.
- Fixed 20 ms world ticks contain exactly twenty 1 ms neural steps. Reset replay, sensory-only controller input, motor disabling and rendering independence remain tested.
- BANC subset: 83 original IDs, 176 raw edges, 6,062 synapse counts; 145 effective nonself edges, row budget 3. Accepted source bundle, encoder and LIF dynamics are unchanged.
- Frozen 20->16->2 head: 370 previously learned scalars; sixteen endpoint histories plus four PN incoming-spike histories. No raw odor, external current, voltage, geometry or teacher label enters inference.
- Candidate SHA256: `117f7e8b759c8916b33cd66a814e471a11cfc7aac941f71c51495dabeeddafa1`. Strict loading reproduces original MSE **0.0108849243**, covered MSE **0.0235447047**, all nine original integrity checks, 370 changed parameters, exact seed-1010 untrained initialization and original episode summaries.
- Additive frozen-assay runner verifies complete provenance, scans JSON/JSONL/CSV history, blocks renamed retries through the central receipt, and records partial execution honestly. Astra reviewed the design, implementation and audit.
- Assay consumed **144 episodes / 63,599 world ticks / 1,271,980 neural steps / 0 optimizer updates**, within its locked budget. Final paired arrival gains are +27 and +28, with one-sided p-values 7.45e-9 and 3.73e-9.
- [Offline audit](experiments/fb10-projection-evaluation/audit.json) reconciles all results, CSVs, journal rows, counters, summaries and gates. All **63 pre-session source files, 219 prior experiment files and four accepted bundle files** remain byte-identical. Audit used zero simulation/optimizer steps.
- [Result plot](experiments/fb10-projection-evaluation/frozen_assay.png) visually inspected; SVG also saved. This is scientific figure review, not a live app/UI check.

## In Progress

- Current 2026-10-07: No active scientific run. Single candidate closed on per-direction failure; no fallback or activation.

- No scientific run. The completed frozen review awaits a decision on its one
  capped instrumentation proposal; historical experiment and defaults stay closed.

- Turn-and-flee implementation, closed experiment, regression verification and local handoff are complete. Native review remains blocked by the helper pipe. No scientific continuation.

- Diagnostic implementation/investigation/live review and final verification are
  complete. The separately authorized motion repair has now closed at its failed
  directional gate; no further repair experiment is automatically authorized.

- Full-suite verification of the partial idle/input lab passes (720 tests). Native
  idle/rest/predator-control review is blocked by the Computer Use helper pipe;
  saved-frame and injected-event checks are separate evidence. Escape live review
  is unavailable because its prerequisite failed.

- Encounter controls are now user-confirmed in the live window. The user observes
  different brain responses with predator placement and no flight reaction.
  Neural escape remains unavailable; no scientific recovery run is in progress.
  Earlier activity-view checks retain their own historical review status.

- Separate native live review of the implemented class switches, colors/legend,
  selected scope, pause/reset, cord and navigation. Automated/saved-frame work is
  complete; native computer automation is disabled. The user has confirmed class switches;
  other native interactions, including the revised fade, remain pending.
- 3D scientific acceptance is open. The October 1 experiment is incomplete
  after 1,050 updates and is not still training. Individual live interactions
  remain unconfirmed; saved
  frames and offscreen event/render checks are distinct evidence.

The approach-and-stop plan is complete. The user watched the live food app and
confirmed contact, zero-speed dwell, contact/relay activity, new-trial R, and
same-trial Backspace replay. `python -m flybrain` now launches that verified
food mode. The recorded FB-10 arrival experiments remain complete within their
original scope.

## Known Issues / Limits

- Current 2026-10-07: Two-sample rule suppresses one short valid cue and delays another into wrong preference. No closed-loop physical evaluation after failed direction gate.

- Local negative spike increments can reflect timing/window expiration; later
  score reversals do not repair first-side validation. Raw RGB, reference
  voltages/exact spike times and edge currents are absent from the frozen record.

- Turn-and-flee validation has nine wrong-side responses and two azimuth groups at 3/6. Increasing electrical drive yielded calibration responses without reliable signed generalization. Closed-loop collisions/stopping/recovery remain unassessed and default activation is disabled.

- New motion-plus-vision repair passed only 9/96 directional cases. No retreat,
  neural grounded takeoff, weak-threat termination/braking or closed-loop recovery
  is installed. Restoring tonic wing activity does not establish threat direction.
  The 48 closed-loop intact/visual-silent cases were not run. No automatic retry.

- Wing peak voltage reaches only 19.0% of threshold under intact diagnostic
  vision-only input. Near voltage patterns differ by direction, but wing spikes
  are all zero. Sparse DNp03/DNp26 firing and far/sequence voltage collisions
  remain. Existing motion-input restoration is a proposal, not a verified repair.

- Current predator circuitry does not support a verified spike-history escape
  decoder under declared inputs/dynamics; UI is a hover-only circuit probe.
  Retreat and reappearance/growth retrigger remain unavailable. Physiological
  signs/gains/retinotopy and a subthreshold decoder are not established.

- The former thin-flower aliasing failure was repaired with deterministic eight-ray
  RGB sampling and sensory altitude search; historical seed 960000 is a regression.
  The later partial fit has no accepted learned behavior or validation/final result.
- All/odor hidden/shown offscreen recordings meet p95 <33 ms, at most 14.10 ms.
  An earlier shown run failed and its variability has no established cause.
  Latest isolated paced display is 57.94–58.65 FPS rather than verified steady 60.
  Running simulation throughput was 1.013–1.015 s/wall s. Pause waits
  for one indivisible in-flight step.
- The user approves the full-activity appearance; individual native interaction
  checks remain unconfirmed. Saved PNGs and injected events
  are distinct checks. Cached 589-cell anatomy is not every full-brain skeleton.
- 3D neuron dynamics, sensory tuning, unsigned weights and flight mechanics
  are declared assumptions. Structural corridors are not demonstrated biological
  mediation. Source anatomy/IDs/edges are traceable; physiology is not a full
  reconstruction. One selected odor cell lacks retained DN routes.
- The atlas reports 18,707 missing root positions and 28,632 untyped records.
  Selected skeleton coverage is 585/589: four public exports return 404.
  Whole view initially shows reference points/region surfaces, not every full
  skeleton. Cached simulated cells load full morphology on selection. Native live desktop
  automation is unavailable in this session; offscreen controls were tested.

- The selectable historical `--control banc-trained` mode still lacks learned
  slowing/stopping. Its diagnostic seed-7 run arrived at 4.76 s
  at 68.99 units/s and ended 101.61 units from food after 20 s. Both diagnostic
  starts recorded zero stationary dwell. The saved head's analytical speed lower
  bound is about 7.4 units/s for all possible hidden activations.
- Food contact is an engineered sensory signal. The historical default ignores
  it; the passing food candidate uses it through a learned readout. Neither
  controller has an automatic geometry-based motor override.
- The candidate app's additional seed-7 start at the arena center did not
  contact food in 20 seconds. Seed 2 at the same center start reached food and
  remained stopped (15.68 s stationary dwell). These are app smoke examples,
  not a new success-rate estimate. The final 63/64 result applies to its
  recorded randomized case distribution, not every possible app trial.
- The original food-stop final process ended without a result or known final
  episode count. Its 830000–830063 identities remain historical. The new
  840000–840063 assay passed, with one noncontact failure at seed 840044.
  These cases are historical too. The separate live-app review passed by
  user observation and is recorded in the recovery plan below.
- Original evaluation stops at distance <=20 and does not score contact/dwell.
  New contact uses body/patch overlap (distance <=16 with the default radii).
- The historical default odor-rate encoder saturates within 54.42 units of each
  antenna. The candidate uses the separate documented rational map, and actual
  weak/strong neural activity and embodied teacher behavior were checked.
- The center app start is 200 units from the source; the new final starts
  covered 60–220 units, including that distance. Broader layouts and source
  strengths remain untested.
- The **earlier 500-update** covered fit remains closed negative: aggregate turn MSE 0.02354470 exceeds its extra ceiling 0.02225273. The later, separate 750-update fit passed that ceiling; it does not change the earlier result or its explicitly approved exception.
- Earlier original/repair fit failures, the first fresh validation failure (11/16), the exact-rounding replay miss, and the fixed range-encoder probe failure remain preserved.
- PN incoming-spike observation moves the readout upstream as well as downstream. Held-out improvement is established for this fixed internal-spike readout and recorded task distribution. Downstream mediation, BANC topology benefit and biological fidelity are not established.
- Original encoder clipping and simultaneous-window contrast limits remain. Sourced IDs/counts do not supply transmitter signs, receptor physiology, LIF rules or motor rules.
- All 720000/721000 assay identities are now historical, including the reserved protocol identities. Prior FB-07/08/09 and 620000 cases remain historical too.
- The extra-training assay's 730000/731000 validation/final identities are now historical. A new name cannot make them unseen again.
- App mode `banc` still uses the original fixed decoder. `flybrain.trained_app`
  uses the original accepted 500-update head and requires its preserved
  experiment files. The verified 750-update arrival model is selectable with
  `--control banc-trained`. The default `python -m flybrain` app uses the
  accepted food approach-and-stop checkpoint. The user confirmed the live HUD,
  stopping/dwell, contact pathway activity, R, and Backspace behavior; the
  native inspection helper remained unavailable to the agent.
- No Git repository is initialized; commits are null. No commit/push. Full connectome, fluid dynamics, biomechanics and large RL systems remain deferred. Exclusive stationary 3D predator trials are now explicitly authorized. The isolated 3D preview is explicitly authorized.

## Important Decisions

- Current 2026-10-07: Honor one candidate with honest closure; preserve original behavior/data, use no target geometry in side gate, no retuning after results. Corrected zero-step entrypoint dependency separately.

- The wrong-side review has a zero simulation budget. Reconstruct saved counts
  without production decoder calls; do not reclassify later activity as successes.
  One unchanged-model instrumentation proposal is pending separate authorization.

- The supplied turn-and-flee specification separately authorizes exactly three effective-gain pairs, lowest eligible selection, spike-only steering and sequential fail-closed gates. Source/cell/edge/count identity, neuron rules, inputs and historical modes stay unchanged. All gains/wing scores/flight policies are engineering assumptions; no learning or unseen-placement claim.

- The user's October 6 specification authorizes one additive idle/moving encounter
  lab and one bounded motion-restoration experiment, overriding the older 3D
  deferral. Preserve every old launcher/source/data/experiment byte. Freeze wing
  spike templates before validation; failed gates end scientific execution.
  Baseline subtraction and angular thresholds are engineering diagnostics, not
  biological fidelity or learning claims. Partial infrastructure remains usable.

- Diagnostic measurements cannot feed neural updates or brain colors. Visual
  cells use phase threshold; downstream cells use normalized voltage threshold.
  Keep the original kernel unchanged as a parity oracle. Record all scientific
  intervals on disk; bound the viewer's in-memory trace to five seconds.
- Close the single 120-presentation investigation without another method, sweep,
  decoder or source-cell extension. New development directions do not restore
  unseen status to historical placements. Any repair requires its separate task.

- User-authorized predator extension is exclusive/additive. Keep controllers
  sensory-only and close the failed neural gate without hidden steering or
  training. Source proposal: review LPLC4 -> DNp26/DNp57 -> saccade DN -> i1/b3
  routes and physiological calibration as a separate bounded task.

- The later fade request replaces the initial 200 ms exponential display envelope
  with a 600 ms cubic eased envelope. Recorded rate averaging remains 200 ms;
  neural/simulation time and data are unchanged. The existing expiry roundoff
  tolerance prevents invisible tails from being counted as colored pulses.
- Independent role-derived classes are display categories, not new biological
  classifications. A frozenset of switches intersects an immutable whole/selected
  membership mask; no neural or recorder filtering. Odor-only is the launch
  default; All activity selects whole scope. Replay retains choices but clears
  history. No preferences persist, and cord visibility never changes framing.
- Actual spike history is display-only and published with its original scientific
  snapshot. Full-neuron warm pulses are not anatomical propagation. The user wants
  All activity retained alongside a defined-circuit option; this follow-up changes
  display membership only, with no recorder or neural filtering.
- Default source anatomy replaces the rejected grouped network. Whole-branch light
  is modeled neuron activity, without measured conduction position/speed.
- A bounded 100 ms wall-time queue protects UI latency while every completed
  numerical step retains its original duration. Render FPS and simulation speed
  are measured separately.
- October 1 checkpoints/journals remain untouched; reconcile 1,050 actual updates
  externally because the parameter-change summary stops at 1,049. No resume/assay.

- Explicit user 3D authorization supersedes the previous scope deferral. Work is
  additive in `flight3d`; existing accepted source/experiment bytes are preserved.
- Experiment prerequisites/gates stop the run on failure. No automatic repair
  sweep, new fit, or reuse of reserved final cases follows the teacher failure.
- The 3D app defaults to a labeled hand-written teacher; neural activity in
  that mode is observational. Fixed neural mode is available explicitly.

- Keep the original one-use interrupted final receipt consumed and its
  potentially exposed cases historical. The user's continuation authorized
  exactly one new final-only frozen-checkpoint assay; its receipt is consumed
  and its new cases are now historical. The user-confirmed live review passed;
  the food checkpoint is promoted to the default launcher as specified in
  [the proposal](docs/plans/FOOD-STOP-RECOVERY.md).
- The user selected food-contact sensing. World geometry produces a local boolean;
  inference receives no coordinates, target distance, bearing or stopping command.
  The acceptance criterion is 2 s of consecutive contact at <=1 unit/s commanded
  and actual speed and <=0.05 rad/s absolute turn. These are engineering thresholds.
- Current-app R means a new orientation; Backspace means exact replay. Explicit
  episode seeds remain reproducible. Scientific comparisons must share identical
  recorded cases across controllers, with disjoint train/evaluation identities.
- Historical first-arrival results do not establish slowing, stopping or feeding.
  No new fit, threshold rewrite, or final-split reuse belongs to this reassessment.
- Expose the 750-update model through new `__main__.py`, `verified_app.py` and `neural/verified_banc.py` modules. Preserve the hash-bound original loaders/app. Reuse existing views and neural calculations; no monkeypatching, new fitting or assay replay. Runtime inference never updates learned weights.

- Only motor-readout parameters were learned in the prior fit. Source weights, graph, encoder, neuron dynamics, feature filters and normalization remain fixed. Forward evaluation and optimizer updates are separate operations.
- The explicit [acceptance record](docs/plans/FB-10-FROZEN-ACCEPTANCE.json) and [proposal update](docs/plans/FB-10-FROZEN-CANDIDATE-ASSAY.md) preserve the failed extra gate while admitting the frozen candidate under original eligibility. No threshold, historical result or checkpoint was changed.
- Fresh requirements stayed >=13/16 validation, >=26/32 final, paired gain >=8 and one-sided p<=.025 against each control, strict all-case progress and unchanged parameters/core/provenance. All passed.
- Preserve all experiment files and protocol-bound Python sources. Future changes need additive implementations or explicit compatibility handling, not silent hash bypasses. See [decisions](docs/decisions.md).
- App integration uses new Python modules and view subclasses because the original app/view bytes are protocol-bound. Loading pins both the failed fit record and the successful accepted assay; no failed gate is relabeled.
- The user's continuation authorized one separately predeclared same-method 750-update attempt, not a sweep. Its update-500 loss row exactly matches the historical candidate; its fit and fresh assay use distinct receipts, seeds and records. Passing this extra experiment does not establish BANC topology benefit or biological fidelity.

## Next Task

One task / preferred Sol High: agree a spike-event/timing readout that retains brief valid cues seen in seeds 1100018 and 1100021. Predeclare its hypothesis/budget before any further experiment. Current one-candidate run is closed, 94/96 with 4/6 in one azimuth group; no automatic fallback, tuning or physical activation. [Failure evidence](docs/ESCAPE-RECOVERY-RESULTS.md).

## How To Run

- Current 2026-10-07: New recovery app is receipt-gated and disabled after failed validation. Historical launchers remain available; experiment commands are consumed, not rerun instructions.

Frozen-record review only: `.\.venv\Scripts\python.exe scripts/review_turnflee_wrong_side.py`.
This regenerates additive review tables/figure from saved data, with no model imports.

The original `.\.venv\Scripts\python.exe -m flybrain.retreat3d` remains the idle lab with escape unavailable. The gate-aware `-m flybrain.turnflee` also falls back to that original lab after failed validation. Do not rerun the consumed experiment. Old v1 action logs keep the original model.

New partial idle/input lab: `.\.venv\Scripts\python.exe -m flybrain.retreat3d`.
WASD/QE move the predator; spawn buttons preserve flight; R clears to no object;
Backspace replays all recorded actions to endpoint and pauses. Neural escape is
unavailable. Add `--seed 1000000` for reproducible idle timing.

Pathway panel: `.\.venv\Scripts\python.exe -m flybrain.encounters3d --kind predator`.
Paused review: `.\.venv\Scripts\pythonw.exe scripts/preview_pathway_diagnostics.py`.
Scroll the lower-left panel for the retained legend and guide.

Exclusive encounters: `.\.venv\Scripts\python.exe -m flybrain.encounters3d`.
Use `--kind predator --seed 980000` for a reproducible hover-only circuit probe.
Spawn buttons select one object; R/new selected kind, Backspace/exact replay.
The existing launchers below are unchanged.

```powershell
.\.venv\Scripts\python.exe -m flybrain.flight3d
.\.venv\Scripts\python.exe -m flybrain.flight3d --control manual
.\.venv\Scripts\python.exe -m flybrain.flight3d --control fixed
```

See [3D controls and preparation](docs/FLIGHT3D.md). F toggles a close fly view;
B resets/fits the visible brain anatomy, V toggles nerve cord without moving
the camera, Tab toggles Background anatomy. Brightness is in Display settings;
connections are in the selected inspector. N/C are removed. Teacher stays default.
For the reviewed activity case add `--seed 970000`. Space freezes activity;
R and Backspace clear display history. Pause first to inspect cleared counters.
Display settings contains independent activity switches and whole/selected scope.
Fresh launches enable Odor inputs only; All activity restores every class. Activity
off hides glow without clearing history. Replay/new trials retain display choices.

The default command loads the behavior-gated frozen food candidate.
`--control banc-trained` selects the accepted historical arrival reference.

```powershell
.\.venv\Scripts\python.exe -m flybrain --seed 2
.\.venv\Scripts\python.exe -m flybrain.food_app --seed 2
.\.venv\Scripts\python.exe -m flybrain.food_app --headless --steps 120 --seed 2
.\.venv\Scripts\python.exe -m flybrain --control banc-trained --seed 7
.\.venv\Scripts\python.exe -m flybrain --control banc --seed 7
.\.venv\Scripts\python.exe -m flybrain --headless --steps 120 --seed 7
.\.venv\Scripts\python.exe -m flybrain.app --seed 7
.\.venv\Scripts\python.exe -m flybrain.app --seed 7 --control neural
.\.venv\Scripts\python.exe -m flybrain.app --seed 7 --control banc
.\.venv\Scripts\python.exe -m flybrain.app --headless --steps 120 --seed 7 --control banc
.\.venv\Scripts\python.exe -m flybrain.trained_app --seed 7 --control banc-trained
.\.venv\Scripts\python.exe -m flybrain.trained_app --headless --steps 120 --seed 7
```

The default `flybrain` launcher uses the 85-cell food approach-and-stop readout.
R starts a new random trial; Backspace replays the current trial; Esc exits. No
seed means a fresh random launch.
The original app defaults to MANUAL DEBUG; the separate trained app defaults to
`banc-trained`. Historical entry points keep their old R/replay behavior.
`banc` is still the fixed decoder. The
frozen-assay commands are consumed records, not rerun instructions. Results are
in [results.json](experiments/fb10-projection-evaluation/results.json). The
food-stop recovery `--prepare` and `--run` commands are consumed too; inspect
[their result](experiments/food-stop-recovery-2026-09-29/results.json) instead.

## How To Test

- Current 2026-10-07: Current full: 812 pass; corrected targeted: 27 pass. Use -p flybrain.escape_recovery.pytest_plugin to isolate frozen diagnostic ledgers. Saved audit: artifacts/escape-recovery-execution-2026-10-07/audit_run.py.

Review checks: `.\.venv\Scripts\python.exe scripts/review_turnflee_wrong_side.py`
(10,800 reconstructed intervals, 42 comparisons, zero added steps),
`.\.venv\Scripts\python.exe -m flybrain.turnflee.audit experiments/turnflee-2026-10-06`
(pass, zero audit steps), and `.\.venv\Scripts\python.exe scripts/audit_turnflee_preservation.py`
(2,510 unchanged). Review figure visually checked. Simulation suites were not
rerun because the explicit review budget forbids all brain/sensor stepping.
Guarded artifact acceptance: `.\.venv\Scripts\python.exe artifacts/turnflee-wrong-side-review-2026-10-06/verify_review.py`
passes the saved-record audit, one offline forged-window test, table/link checks
and 2,720 final hashes, with zero brain/sensor simulation calls.

Latest turn-and-flee targeted check: `.venv/Scripts/python.exe -m pytest tests -k "flight3d or encounters3d or pathway or retreat3d or turnflee" -q --junitxml=artifacts/turnflee-2026-10-06/targeted.xml`: **188 passed**, zero skips, three upstream warning instances (84.31 s). Full: `.venv/Scripts/python.exe -m pytest -q --junitxml=artifacts/turnflee-2026-10-06/full.xml`: **755 passed**, zero skips, three warning instances (196.34 s). OMP/MKL threads one. Read-only audit: `python -m flybrain.turnflee.audit experiments/turnflee-2026-10-06`; preservation: `python scripts/audit_turnflee_preservation.py` (**2,510 unchanged**). No rerun of the consumed experiment.

Latest additive verification: **153 targeted / 720 full-suite passed**, zero skips,
one targeted/three full upstream warnings. Final full XML:
`artifacts/retreat3d-2026-10-06/full-final.xml` (188.49 s).
See [exact commands](docs/verification/retreat3d.md). Offline scientific audit:
`python scripts/audit_retreat_records.py experiments/retreat3d-motion-repair-2026-10-06`.
Preservation: `python scripts/audit_retreat_preservation.py check` (2,433 unchanged).
Do not rerun the closed experiment into any directory.

Diagnostic checks: `.\.venv\Scripts\python.exe -m pytest tests/test_pathway_diagnostics.py tests/test_pathway_panel.py tests/test_pathway_protocol.py tests/test_pathway_records.py -q`.
Read-only scientific audit: `.\.venv\Scripts\python.exe scripts/audit_pathway_records.py experiments/brain-to-wing-diagnostic-2026-10-05`.
Do not rerun the closed investigation into a new directory without authorization.

Latest diagnostic verification: **111 targeted / 678 full-suite passed**, zero
skips; final panel checks **4 passed**. Full command: `python -m pytest -q
--junitxml=artifacts/pathway-diagnostics-2026-10-05/full.xml` in the venv with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Three upstream warning instances;
361.86 s. Evidence and exact targeted command:
[verification](docs/verification/pathway-diagnostics.md).

Historical prior encounter verification: **100 targeted / 667 full-suite passed**, zero
skips, two existing warnings. Commands: `python -m pytest -q tests -k
"encounters3d or flight3d"` then `python -m pytest -q` (venv; OMP/MKL threads 1).
Logs/XML: `.superpowers/encounters3d/targeted.*`, `full.*`. Native review remains
separate; no avoidance success is claimed.

Verified 2026-10-05: **87 targeted passed / 654 full-suite passed**, zero
full-suite skips, two existing upstream warnings; pytest durations 96.73/179.22 s.
Current logs/XML: `.superpowers/activity-fade-2026-10-04/final-targeted.*` and `final-full.*`.
Final commands below used the venv with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

3D targeted tests: `python -m pytest -q tests -k flight3d`.
Saved views:
`python -m flybrain.flight3d --capture .superpowers/flight3d/frames`.
Full suite: `python -m pytest -q`.

Verified 2026-09-30: **594 passed**, zero skips, two upstream deprecation
warnings. The full source/geometry audit passed; **374/374** historical files
are unchanged. Final viewer-only changes have their own targeted check recorded
in the latest project log and `.superpowers/flight3d/targeted-final.log`.

Food-candidate and interruption-record checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_food_contact_network.py tests/test_food_teacher.py tests/test_food_preflight.py tests/test_food_readout.py tests/test_food_fit.py tests/test_food_evaluation.py tests/test_food_fresh.py tests/test_food_brain_view.py tests/test_food_app.py tests/test_food_records.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_food_recovery.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_food_app.py
```

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_behavior.py tests/test_verified_app.py tests/test_trained_app.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_verified_app.py tests/test_trained_app.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_banc_frozen_evaluation.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_trained_app.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_banc_extra_training.py tests/test_banc_extra_evaluation.py
```

Current behavior reports: `.behavior-review-targeted.xml` and `.behavior-review-full.xml`.
Historical reports include `.verified-app-*.xml`, `.fb10-extra-final-*.xml` and
`.fb10-frozen-*.xml`. The capped behavior diagnostic and historical assay/audit
directories are consumed records; do not rerun commands into them.
