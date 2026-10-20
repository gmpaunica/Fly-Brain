# Fixed escape recovery execution - 2026-10-07

User authorizes implementation without routine approval pauses; one fixed
candidate, honest closure on failed scientific gate, new working launcher after
all behavioral/audit gates pass. One agent, no commit or historical source edits.
Spec: `docs/ESCAPE-TEMPORAL-READOUT-PROPOSAL.md` plus recorded trajectory criteria.

1. Implement cumulative six-cell readout: positive winner/margin >=1,
   two eligible samples after strong, fixed 25-tick deadline, weak rearm/reset.
2. Integrate retained physical policy/session, bound replay and receipt-gated UI.
3. Preflight targeted/full tests using isolated diagnostic engineering ledger;
   preserve all old records. Source-lock one candidate.
4. Exactly 96 historical 1.5 s direction cases; >=90 and >=5/6/azimuth within
   500 ms. Only on passage, 48 six-second closed cases; >=22/24 intact,
   >=7/8/state, zero silent triggers. Max 432 s /864,000 individual neural steps.
5. Independently audit, review saved visuals, activate/launch only passing new app,
   update docs/status/log/queue. No extra candidate or scientific retry.

Exact added evaluator criteria (fixed before results): actual signed yaw reaches
90 degrees within existing 3-degree tolerance before first flee; resting takes
off with foot contact false and z>=FOOT_LENGTH+.005 for >=3 control intervals;
predator-center distance gains >=.020 m between trigger and first weak braking;
no collision; braking achieves horizontal and vertical speed <.020 m/s;
recovery/idle cruise stable >=15 ticks with z .075..125 and |vz|<.020, no new
trigger/collision. Capture pose/orientation/contact/commands/predator position
for evaluator only. These engineering thresholds do not enter control.

Interfaces: Readout.update(actual589,reference589,strong,weak,active)->immutable
decision; controller owns IMU flight control; session owns stepping and signals
recovery to readout. No world geometry enters side gate. Readout state resets on
spawn/trial/replay, weak recovery clears cumulative evidence without neural reset.
Retained neuron/encoder topology/gains and numeric operation order unchanged.

Task 1: implemented/tested. Temporal counts, positive winner, two observations,
fixed deadline, weak rearm; tests reject deficit-only, tied and premature choices.
Task 2: implemented/tested. Retained physical policy, cumulative UI, bound action
logs, intervention-preserving replay, gated activation; actual trajectory metrics
include pre-command pose and require completed turn before first flee.
Task 3: first targeted 25/full 810 pass. Added silent-intervention replay and
premature-flee regression; pre-lock targeted 26/full 811 pass. Zero-step missing
stdlib math dependency reproduced/fixed via additive bound entrypoint; corrected
targeted 27 pass and final full 812 pass, no original ledger/source mutation.
Task 4: closed direction gate failed: 94/96, all nine old failures repaired;
two new 86.25 degree regressions leave 4/6. Exactly 96 cases/144 s/288,000 steps.
No closed-loop cases, parameter change, fallback or activation.
Task 5: independent audit passes, source preservation and saved regression plot
reviewed; docs/handoff updated. Physical escape remains unverified/unfixed.
User-selected one-candidate honest stop applies; no new experiment appended.
