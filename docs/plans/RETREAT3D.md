# Idle flight and interactive neural escape — 2026-10-06

Authority: the user's supplied `Idle flight and interactive neural escape` spec.
Single agent, inline execution; no commits, retries, tuning or neural changes.
The supplied design and explicit implementation request authorize execution.
No Git repository is present. Existing files are hash-protected; all implementation
is additive in `src/flybrain/retreat3d` and new tests.

## Locked hypothesis and protocol

Restoring motion input produces directional wing-spike responses sufficient for a
frozen decoder to support escape from an approaching predator.
Maximum 48 calibration + 96 validation + 48 closed-loop cases; 504 simulated
seconds and 1,008,000 combined actual/reference neural steps. Stop at first failed
gate, with no tuning, retry, different decoder, added cells or neural rule change.
Existing inspected placements remain historical development cases.

Calibration: azimuth 0:45:315 degrees, elevations -20/0/20 degrees,
approaches 17→8 and 12→6 cm. Validation: 11.25 + k*22.5 degrees, same elevations
and profiles. Each presentation is .5 s motion-only then 1 s linear approach.
Constant upright/rest IMU (0 angular velocity, specific force 0/0/9.81) in
sensory calibration/validation; closed-loop uses actual sensors in each state.
Use five nonoverlapping 200 ms post-onset signed wing count windows, pooling
their mean across the six presentations per calibration direction. Normalize
eight templates, then freeze/save their manifest-ordered 48 IDs and values.
Validation success requires at least one confident post-onset window and every
confident window within 22.5 degrees. Thresholds: norm ≥1, cosine ≥.8, margin ≥.1.
This conservative case aggregation is fixed before inspecting responses.
Validation gate: ≥90/96, ≥5/6 per azimuth. Only a pass authorizes closed-loop
evaluation (8 directions × resting/hovering/cruising × intact/visual-silent,
6 s each). Gate: ≥22/24 intact, ≥7/8 each state, zero silent triggers. Success:
away within .5 s, no collision, weak-threat condition by 6 s, stable recovery.

## Implementation tasks

1. Neural candidate/reference, immutable frozen decoder, RGB angular-size/growth
   monitor, source-bound fresh-only experiment runner and disk traces. Tests:
   exact original motion rates/visual parity, unchanged graph/dynamics/diagnostics,
   identical paired motion-only spikes, signed subtraction and confidence gates,
   seam/rear/vertical/occlusion/shrinking/reappearance, protocol counts and stop.
2. Environment-only predator input, optional encounter world, ground support and
   relative swept collision. Shared IMU estimate and engineered seeded idle
   takeoff/cruise/landing/rest; teacher inherits estimates at handoff. Tests:
   landing safety, bounded repeated idle, continuous spawn, diagonal clamp and
   moving collision, no scene geometry in observation/controller interfaces.
3. Fixed-step session/action log and worker-barrier viewer. Replay consumes all
   spawns and applied commands to recorded endpoint, then pauses. Preserve
   camera/preferences/pause/diagnostics, display actual brain only. R clears to
   fresh no-object trial. Tests: exact replay, full reset, timestamps and rendering
   independence, key release on pause/focus loss, real-renderer smoke/screenshots.
4. Run the locked experiment once; install escape only if gates pass. Any failed
   gate closes scientific work but finish and label independent idle/input lab.
5. Targeted 3D/encounter/new tests, full suite, preservation audit, local acceptance
   review, separate visual review and documented live-review limits. Update
   status/log/architecture/decisions/queue/experiment guide.

## Review focus

No physical predator distance reaches control; angular size never supplies
heading. Spawns preserve body and IMU state but reset paired brains and display
histories with a local neural-clock origin. Replay includes same-tick spawns even
at its endpoint. Grounded takeoff may only trigger through the neural gate.
No failed scientific result is labeled delivered retreat.

## Ledger

- Plan locked before any scientific presentation. Tasks 1–3 implemented with
  RED/GREEN behavior/numerical tests; all historical implementation remains intact.
- Task 4 closed: 48 calibration + 96 validation; gate fails 9/96. No closed-loop
  cases, tuning, retries or neural escape implementation. Frozen decoder is passive
  diagnostic only; unsupported retreat stays explicitly labeled.
- Task 5: 153 targeted / 720 full-suite pass; disk audit and 2,433-file preservation
  pass. Saved frames reviewed independently; native helper pipe fails. Partial lab
  verification/handoff complete; escape is not delivered and native review is pending.
- Ruling: no independent subagent/reviewer, worktree or commit — explicit user
  one-agent/no-auto-commit rules and absent Git override skill defaults.
- Ruling: first 100 ms size/growth smoothing uses fixed five-sample averages;
  equivalent circular diameter uses contrast-weighted pixel solid angle, without
  a controller range or heading estimate. Occlusion reduces measured visible size.
- Local review fixes: removed flower support, stale handoff contact, moving sphere
  double placement; tests reproduced each failure before the correction.
- Final HUD reduced to two lines to keep angular diagnostics above the scroll
  panel. First full suite: 719 passed, one provenance-guard failure caused by that
  concurrent source edit. Sole changed hash was additive viewer.py. Isolated test
  passes once source is frozen; final frozen-source suite passes 720/720 (188.49 s).
- Ten additional 35 s idle-only seed checks have no crashes (largest measured
  home-relative displacement .227280 m); zero neural steps or scientific cases.
