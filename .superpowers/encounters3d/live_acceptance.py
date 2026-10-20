"""Record user's live feedback without changing code or scientific results."""
from pathlib import Path

status=Path('PROJECT_STATUS.md'); text=status.read_text(encoding='utf-8')
text=text.replace('for user review (seed 980000, reopened launcher PID 24840 / window PID 6692); live acceptance is pending.',
'''(seed 980000, reopened launcher PID 24840 / window PID 6692). The user confirms
all encounter controls work and observes placement-dependent brain activity with
no fly response. This verifies the encounter controls, not neural avoidance or
the remaining historical activity-view checks.''',1)
text=text.replace('''- Encounter native UI review is pending: the computer-use helper failed to
  connect. Offscreen event/render review does not substitute for live acceptance.
  No scientific recovery run is in progress.''','''- Encounter controls are now user-confirmed in the live window. The user observes
  different brain responses with predator placement and no flight reaction.
  Neural escape remains unavailable; no scientific recovery run is in progress.
  Earlier activity-view checks retain their own historical review status.''',1)
text=text.replace('''Native encounter controls and earlier activity checks
also remain pending user review; they are not scientific recovery experiments.''','''Native encounter controls are user-confirmed. Earlier activity checks retain
their separate pending status; they are not scientific recovery experiments.''',1)
status.write_text(text,encoding='utf-8')

queue=Path('docs/BUILD_QUEUE.md'); text=queue.read_text(encoding='utf-8')
text=text.replace('''  open; helper could not connect, so live feedback remains pending.''','''  user-reviewed: all encounter controls work. The user observes differing brain
  activity by predator placement and no fly reaction. Helper access still fails;
  this human review does not establish neural avoidance.''',1)
queue.write_text(text,encoding='utf-8')

feedback='''

## 2026-10-05 — User-confirmed live encounter controls

The user reports: “all controlls work, but the fly doesnt react. its interesting
that the brain reacts differently depending on where the predator spawns”.

This confirms the requested live spawn exclusivity, R/Backspace, pause/reset and
camera/display retention checks. Placement-dependent brain appearance and absent
fly response are user observations, consistent with the recorded sensory activity
and failed motor-spike gate. No directional decoding or neural escape success is
inferred. Current predator flight is controlled solely by the explicit hover
stabilizer. Historical activity-view acceptance remains separate. No code or
scientific result changed; no tests rerun, training or recovery experiment begun.
'''
for target in ('docs/verification/encounters3d.md','docs/plans/ENCOUNTERS3D.md'):
    with Path(target).open('a',encoding='utf-8') as stream: stream.write(feedback)

log='''

## 2026-10-05 — Encounter controls accepted; placement-dependent activity observed

### Goal
Record the user's completed live encounter review and explain the motor limit.

### Changes Made
Updated status, queue, plan and verification with the user's control confirmation
and observations. No product, neural, source-graph or experiment changes.

### Decisions
Confirm encounter controls only. Different neural activity patterns by placement
do not establish a usable motor direction code. Keep the failed escape gate closed.

### Tests / Verification
User: “all controlls work, but the fly doesnt react. its interesting that the brain
reacts differently depending on where the predator spawns”. This is live user
evidence for the requested controls; numerical evidence remains the existing
100 targeted / 667 full suite and recorded negative neural gate. No repeat suites
are needed for this documentation-only feedback record.

### Problems / Failed Attempts
The fly still has no neural escape response, as labeled. No new fix was attempted.
The native computer-use helper remains unavailable; user observation supplies review.

### Current State
Exclusive encounter controls are live-accepted. The user sees placement-dependent
brain activity and absent locomotor reaction. Hover remains the sole predator motor
controller; no escape decoder was installed after zero wing spikes in all physical
characterization trials. Earlier activity-view checks retain separate status.

### Next Step
One task / Sol High: specify the bounded sourced/physiological escape-path recovery,
tracing sensory-to-wing response and assessing directional information before
choosing any fixed decoder or extending the graph. No automatic scientific retry.

### Notes
Image position changes which assumed receptive fields drive existing visual cells;
source connectivity then changes downstream activity. Retinotopy and neuron gains
remain modeling assumptions. Forward PyTorch activity is not learned behavior.
'''
with Path('PROJECT_LOG.md').open('a',encoding='utf-8') as stream: stream.write(log)
