# Sustained escape to the map edge — 2026-10-11

User-approved implementation plan: one agent, no training, commits or publishing.
Replace weak-threat recovery with latched turn/flee/edge braking/edge hover in a
separate package. Preserve existing neuron/readout/physics sources and records.
IMU estimates and the authored fixed-start calibration determine a ray endpoint
15 mm inside the rectangular horizontal arena. No actual object/body geometry
enters the controller. Flee speed .12 m/s; altitude .10 m; no home-seeking guard.

Development: original 48 historical intact/silent cases, 12 seconds each,
576 simulated seconds / 1,152,000 individual paired neural steps per candidate.
Seed robustness: separately declared 48 cases with new seeds, same geometry,
duration and budget; do not call geometry unseen. All 24 intact must satisfy
retained timely/correct side, signed 90-degree turn and resting takeoff criteria,
collision freedom, arrival by 10 s and >=2 s edge hover. Hover: horizontal speed
<=.01 m/s, altitude .10 +/- .015 m, nearest wall distance .005–.025 m.
No premature stop or inward reversal. Silent triggers zero. Stop/seal on health,
source, I/O or budget fault; downstream ceiling 240 Hz. Failed attempts preserved;
user authorizes development iterations, not overwriting failures or lowering gates.

Four predetermined 12-second exact replays: development rest 0 intact/silent,
development cruise 0 intact, robustness cruise 180 intact: 96,000 additional
individual verification steps. Tests use separate engineering ledgers. Audit
saved records independently; review plots/offscreen frame separately from live UI.
Activate only after engineering, evaluation, audit/replay checks pass. Update all
project handoff docs; directional 94/96 historical limitation remains explicit.

## Execution ledger

- Plan reviewed; no Git repository. Work directly in the existing workspace.
- Existing recovery readout remains active throughout edge hover; no recovered
  callback is issued, avoiding reset/retrigger or mid-flight neural reset.
- Tasks: controller/motor regressions; session/log/viewer; evaluator/audit/replay;
  engineering checks; measured development and seed robustness; gated delivery.
- Controller/session implemented; 36 tests pass, including 24 real motor
  combinations through 20 s, off-center/corners, latch/reset/replay and fault seal.
- Local review: original readout active throughout edge hover; no neural reset
  or sensory/coordinate shortcut. Added viewer-only safe spawn placement with
  red/green regression, one pass. Existing sources/experiments inventoried:
  3,010 protected files. Targeted 239 pass; full running; science unstarted.
- Windows Computer Use native pipe absent twice. Use native app's own render
  evidence separately; do not claim mouse/keyboard UI automation review.
- Final focused 38/final full 922 pass, zero skips. Normal frame-task capture
  fixes missing text; Filename conversion fixes absolute Windows PNG writes.
  Proofs in preflight.json; source frozen, worker running. First development
  case passes at 4.92 s. No aggregate claim or delivery yet.
- Local acceptance review found the per-step 2 mm reversal tolerance does not
  exclude a slow cumulative inward retreat. Before finishing development, add
  a separate saved-only delivery check: peak-to-later-progress retreat <=5 mm
  while fleeing/braking, and every fleeing command has .12 m/s horizontal norm.
  No frozen source, recorded original gate or scientific steps are changed.
- Development and robustness both pass all 24 intact/zero silent triggers;
  2,304,000 scientific steps. Independent audits pass at zero added steps.
  Four exact replays pass (96,000 verification steps). Supplemental pre-hover
  retreat 0 m, full flee target .12 m/s. Both saved plots visually reviewed.
- Receipt-gated launcher activated. Native viewer API actual flight/12 s hover
  passes with 24,000 separately reserved engineering steps; native frames
  reviewed and window left open/paused. Native input helper still unavailable;
  intermittent label loss in saved native captures is a separate next UI task.
