# Turn-and-flee — authorized specification and implementation, 2026-10-06

User supplied the full turn-and-flee specification. One agent, no Git repository,
no automatic commits. Routine choices are authorized by AGENTS.md. Historical
sources, launchers, records and circuit counts remain byte-preserved; implement
in `src/flybrain/turnflee/`. Only a passing final gate permits default promotion.

Hypothesis: multiplying existing LPLC4→DNp03/DNp26 edges and those four DNs'
outgoing effective weights yields timely, correctly signed wing-spike increments
that can initiate engineered evasive flight without identifying eight bearings.

Exactly (2,1), (4,2), (8,4), each 16 development presentations (0:45:315 degrees,
zero elevation, 17→8/12→6 cm), 1.5 s with .5 s motion baseline. Eligibility:
16/16 correctly signed gated responses during presentation, exact paired
baseline equality, finite state, no downstream rolling 200 ms rate >240 Hz.
Choose lowest eligible, otherwise close. Validation: 18.75+k*22.5 degrees,
-20/0/20 elevations, two profiles; >=90/96 and >=5/6 per azimuth. Either side
within 15 degrees front/rear. These placements receive a recorded historical
configuration audit before lock; no unseen claim without evidence.
Then only 48 closed-loop six-second cases: 8 azimuths × resting/hovering/cruising
× intact/visual-silent. >=22/24 intact, >=7/8 per state, zero silent triggers.
Maximum 192 cases, 504 simulated seconds, 1,008,000 paired 1 ms steps. Stop at
first failed gate; no tuning, extra candidates or learning.

1. Add immutable identity/hash/receipt-bound configuration, effective gain
   application, original motion/visual formulas and rolling 200 ms differences
   every 20 ms. Tests: gain-one bit parity, exact affected edges/counts, input
   parity, paired baseline, 1 ms directed propagation, reset and rate guard.
2. Add signed wing scores and sensory-only escape state machine: pending strong
   event <=500 ms, norm >=1 on six i1/b3/b1 endpoints, seeded bilateral exact
   ties, 90° target, braking during turn, .12 m/s flee at .1 m altitude,
   weak-condition brake/recovery <.02 m/s both axes, IMU home guard .22/.18 m.
   Test spike gate, ties, grounding, brief gaps, braking and guard.
3. Add session/snapshot/config-bound action logs and additive diagnostic viewer.
   Preserve old replay via original session, reset all new state and exact pause.
4. Lock source/config/history receipts, run scientific experiment once, save
   full paired counts/eyes hashes/timing/guard evidence and independently audit
   stored records. No test runs the full experiment. Gate final promotion.
5. Run targeted and full tests, preservation audit, saved-frame review and
   attempt live review separately; update status/log/decisions/queue/guide.

Review focus: RGB never supplies turn side, voltage never controls movement,
anatomy uses actual spikes, no historical benchmark is relabeled unseen,
no candidate is enabled by calibration alone, old logs remain original-model.

Ledger: plan preflight complete. .02 s control / .001 s neural, fixed manifest
order, independent paired brains, geometry restricted to evaluator/world/view.
Unsigned historical weights and neuron rules remain exactly as originally assumed.

- Tasks 1–3: numerical/control/session/GPU tests RED→GREEN. New preflight 34/34
  passed (10.30 s), followed by receipt and visual-silent additions. Original
  source preservation record contains 2,510 files. New logs are v2, old v1 logs
  replay through the original model. Source edits are complete before onset.
- Local review: corrected a too-strict calibration latency condition before any
  scientific presentation; the supplied specification puts the first-onset
  500 ms gate on validation. Calibration requires a correctly signed gated
  response during its presentation. No result-driven protocol change.
- Local visual regression: five-line top summary overlapped the anatomy heading;
  two-line summary plus scrolling details passes RED→GREEN and saved-frame review.
- Named closed starts assume fixed upright nominal hover/cruise calibration,
  initialized separately in controller and physics; no world-state handoff.
- Native Computer Use initialized; listing fails: missing native pipe/os error 2.
  Native review is unavailable; saved-frame and GPU checks are separate evidence.
- Final acceptance review is local, as required by user one-agent instructions.
  No Git/worktree/commit/subagent. No deferred findings before scientific lock.

- Task 4 closed: (2,1) 13/16, (4,2) and (8,4) 16/16; lowest (4,2) frozen.
  Validation 87/96, two groups 3/6; no closed-loop assay or default promotion.
  Exact budget 144 cases/216 s/432,000 paired steps. Disk audit passes 10,800
  intervals/3,600 equal baselines with zero new neural steps. The redirected
  PowerShell command reports exit 1 from the upstream warning NativeCommandError;
  closed result/receipts are finalized and audited; no execution retry.
- Task 5 complete: 188 targeted (84.31 s), 755 full (196.34 s), zero skips,
  three upstream warning instances each. All 2,510 protected originals unchanged.
  Revised saved frames and passive GPU/replay tests checked; native helper fails.
  Status/log/queue/architecture/decisions/experiment guide updated. Final local
  review: no outstanding implementation findings. Scientific failure is retained,
  closed-loop behavior unverified, native human review pending. No next experiment.

- Frozen-record review complete, 2026-10-06: all nine first responses reconciled,
  three failed seeded ties/six unequal scores; all later favor left in saved
  timelines. Zero presentations/seconds/steps, unchanged protected hashes.
  [Review and one proposed instrumentation hypothesis](../TURNFLEE-WRONG-SIDE-REVIEW.md).
  The proposed 18 historical presentations /27 s /54,000 paired steps require
  separate authorization. Original experiment closed; no automatic continuation.
