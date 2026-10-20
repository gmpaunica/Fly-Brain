# Execution ledger: docs/plans/FLIGHT3D-SPIKE-ACTIVITY.md

One agent; no git repository. Implementation and automated handoff complete 2026-10-04.
Preflight: recorder frame is shared by pump, GPU table and inspector; one frame read per draw. Historical draw_snapshot calls will consume publication only; tests/capture use observed stepping.

Task 1: complete. RED 10 failures (missing activity module), GREEN 11 passed. Scientific snapshot/state and global RNG comparisons pass in teacher/fixed modes. Cache inventory supplemented before geometry edits.
Tasks 2/3: implemented and focused checks pass. GPU sentinel RGBA pixels, mixed/deferred ownership, 56 region-excluded firing cells, 530/59 two-second counts, no-outgoing inspector, same-seed reset and 30/60/120 matched history/colors verified. Follow-camera publication regression added RED then GREEN. Required targeted/full driver running.
2026-10-04 continuation: selection cache RED (unchanged frame reformatted 3 times), GREEN 1 passed. First required suites 76/643 passed; hidden performance gate failed at selection p95 38.54 ms. Profile found repeated edge scans/reformatting. Static edge counts and inspector publication/status token implemented. Final isolated timing rerun underway; preserve first evidence.

Final: fresh required targeted 76 passed /567 deselected /47.47 s; full 643 passed
/zero skips /145.88 s, exit 0 both, two existing warnings. Final hidden p95
4.65/9.39/4.61/12.16 ms; passing shown repeat 5.69/9.39/4.64/11.80 ms.
359/527 visible pulses; 75/138 batches. Earlier failed shown run retained;
cause of variability not established. Scientific/cache/experiment audit:
2,365 protected existing files unchanged. Final capture exit 0, 15 PNGs exist.

Tasks 1–3 complete. Task 4 automated verification, local review, timings, launch
and documentation complete; individual native interaction observations remain
unconfirmed. User approves full-activity view as an option and requested /plan
for selectable All activity and smaller defined-circuit views. Proposal saved in
docs/plans/FLIGHT3D-ACTIVITY-MODES.md; no follow-up product change. Status/log/
guide/queue/architecture/decisions/verification records updated. No training.
