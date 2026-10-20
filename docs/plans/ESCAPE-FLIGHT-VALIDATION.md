# Frozen-candidate physical validation — 2026-10-08

Authority: user-pasted “FlyBrain: physical escape validation and prototype
launcher” request. One agent; no commit, tuning, retry or learning.

Prospective exception: candidate
`ea70e22d8879874f14df4299ba797170d7624e0b26c6e0baeddf720d3494dcab`
is accepted for this prototype at 94/96. Its original gate remains failed:
86.25° 4/6; failures 1100018 and 1100021 remain. No directional rerun.

Hypothesis: the accepted frozen directional controller supports reliable
turn, takeoff where required, flee, weak-threat braking and stable recovery.
Exactly the existing 48 closed cases in their existing order and seeds:
8 directions × resting/hovering/cruising × intact/visual-silent, six seconds.
Budget: 288 simulated seconds / 576,000 individual paired neural steps.
Stop immediately on source/receipt mismatch, missing measurements, nonfinite
state, downstream rolling rate >240 Hz, I/O failure or budget breach.
Otherwise finish all cases and require ≥22/24 intact, ≥7/8 each starting state,
zero silent triggers. Seal every terminal outcome; never resume consumption.

Intact criteria: correct side within 500 ms of first strong threat; actual signed
90° yaw within 3° before first flee, anchored at pre-command orientation;
resting contact false and height ≥8 mm for three consecutive intervals;
predator-center separation gain ≥20 mm before weak braking; no collision;
weak braking achieves horizontal and vertical speed <20 mm/s; recovered idle
cruise stable for 15 intervals, height 75–125 mm, |vz|<20 mm/s, no new trigger.
No newly measured motion-only baseline: bind previously verified direction data.

1. Preflight: source/receipt exception, adjacent package, ordering/budget,
   movement scoring, storage/entrypoint/supervisor failure tests.
2. Targeted and full suite with isolated historical diagnostic ledgers.
3. Lock source archives and preflight evidence; one detached closed-only run.
4. Independent saved temporal/physical audit, source preservation, exact replay
   including silent cases, separate saved visual review.
5. Only if all gates pass: source-bound activation and launch new prototype app,
   retaining known direction limitation. Otherwise leave activation disabled.
   Update status/log/decisions/experiments/queue with one concrete next task.

Pre-flight shared interfaces:
- Frozen CandidateConfig → acceptance/runner/app: exact recorded identity,
  complete Python inventory, circuit and calibration receipt; no new file in
  escape_recovery. Corrected execution module supplies stdlib math only.
- Runner → audit/replay: chronological JSON/CSV records, calibrated named start,
  bound original RecoveryLog, initial pose, actions and intervention. Old action
  format lacks named-start setup; adjacent envelope adds it without modifying
  frozen state or action semantics.
- Audit → activation: separately reconstruct source/seal, temporal decisions,
  physical metrics, gates and budget. Original directional gate is never bypassed
  in historical code; separate verifier accepts this one explicit exception.

Ruling: no Git repository exists, so execute in the existing workspace. User
requires one agent and no commit; Git-based skill scripts and review delegation
do not apply. Local acceptance-criteria review replaces delegated review.

Execution ledger (append verification and outcomes below):

Preflight evidence: final targeted 117 passed /zero skips (60.18 s), synthetic
48-case saved audit passes with zero neural calls, separate original direction
saved audit passes with zero added calls; offscreen new-viewer construction/render
and visually reviewed saved frame, world and paired brain steps remain zero.
Full suite in progress. No scientific consumption. Local code review corrected
first versus latest side in independent temporal audit, with failing then passing
five-tick synthetic reproduction. Activation delivery evidence must bind the
exact physical receipt, replay budget and saved visuals. No frozen source edit.

Preflight sealed: 117 targeted /841 full tests pass, zero skips, four upstream
warning instances each, 60.18/281.86 s. 4,858 protected original files unchanged;
original diagnostic proof stays 1,747 steps. Exactly one hidden detached
`python -m flybrain.escape_flight_validation.worker` launched. Its source-bound
48-case/288 s/576,000-step experiment is now consumed; do not launch another.

Scientific terminal outcome: closed-physical-failed, 48 cases /288 s /576,000
steps /14,400 records, normal worker exit0. Intact 19/24; resting 6/8, hovering 7/8,
cruising 6/8; silent triggers 0. No activation or policy/readout tuning/retry.
All 24 timely sides and actual turns, all 8 resting takeoffs, min separation159.908mm;
max downstream215Hz. Three post-recovery boundary collisions, two missing weak/
brake/recovery by six seconds. Saved figures separately reviewed.

Ruling: frozen raw CSV has NumPy scalar wrappers unlike JSON numeric lists.
Preserve source archives/raw records/seals; six failing-then-passing saved-only
adapter tests establish finite literal parsing and changed-value rejection.
The additive artifact adapter checks exact float64 values before the original
independent numerical auditor runs on a temporary canonical CSV view. No gate,
receipt or model mutation, zero audit neural steps. Original failed audit log
preserved. Corrected audit passes all 14,400 intervals. Replay and final suite
including adapter are in progress in separate journals/logs.

Exact replay: all 48 cases /14,400 intervals /576,000 separately journaled
verification steps pass every saved interval field including silent intervention.
Replay supervisor exits0; no new scientific cases. Scientific budget remains
288 s/576,000 steps, combined science+replay1,152,000. No policy or source change.
All four flight figures visually reviewed; live review pending, app disabled.
Final suite including six adapter tests is in progress with fresh output/ledgers.

Task complete: final847-test suite passes (414.03s), zero skips / four upstream
warnings. Final budget/source/proof/CLI refusal check passes; original4,858 files
and1,747 diagnostic steps unchanged. Exact replay all 48 passed with separately
journaled576,000 steps. Saved flight visuals reviewed; live review pending.
Prototype acceptance links direction and physical receipts and records denial;
activation/app launch withheld after physical failure. Updated status/log/queue/
decisions/experiments/report/verification. One next task: saved-only diagnosis and
one sensor-only flight-policy repair proposal (Sol High); no automatic new work.
