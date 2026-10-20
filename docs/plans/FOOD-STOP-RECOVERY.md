# Food-stop final-assay recovery proposal — 2026-09-29

## Recorded position

The single 85-cell, 419-parameter checkpoint passed its locked 40-case
training gate and 16/16 fresh validation gate. The one-use fresh process
disappeared after writing the complete validation record. It wrote no final
JSON/CSV, completion record, or traceback. The count of final episodes
actually executed is unknown. The original 830000–830063 final identities
are historical and will not be reused as unseen cases. The checkpoint was
not promoted. A separate fixed-input contact-block diagnostic changed its
stop response (last five ticks: 4 stops intact, 0 blocked), but it is not a
full-episode result.

## Proposed single recovery decision

Authorize **one new final-only assay of the exact frozen checkpoint**. No
additional optimization, normalization, feature, threshold, teacher, world,
or app-default change is included. The completed 16/16 validation result is
the eligibility gate; it is not called unseen again. All original final
attempt files and its interruption record remain untouched.

Before simulation, write a new immutable protocol and source/checkpoint
hashes. Reserve 64 previously unused seeds **840000–840063**, eight headings
per 45-degree sector. For each seed, use a local generator to draw a heading
within its assigned sector, a radius uniformly from 60–220 units, and a
source-centered angle uniformly over 360 degrees; reject positions that do
not fit the 800×600 arena with a radius-6 body. Record every pose and seed.
All three controllers receive the same cases in the same order.
The proposed exact [64-case manifest](FOOD-STOP-RECOVERY-CASES.json) was
generated without simulation or optimizer steps. Its SHA256 is
`1941e4436fa3b857faf26746560c43874df1cea352a8fe55e0ffa91072ac1d32`.
It contains eight cases per sector, 64 unique seeds, and actual start
distances from 62.56 to 217.31 units. No recorded experiment JSON used these
seed identities at proposal time. A later authorized protocol must bind
these exact bytes before evaluating any case.

Run 1,000 world ticks (20 s) per case for trained, identical-seed untrained,
and fixed BANC controls. Budget: 192 episodes, 192,000 world ticks,
3,840,000 neural steps, zero optimizer updates. Write and flush a durable
per-episode journal with case identity, controller, metrics, and counters;
the final summary must reconcile exactly 192 rows with the protocol and
checkpoint hashes. If the process stops or any integrity check fails, record
the completed journal and close this attempt without silently replaying
potentially exposed cases.

Use the unchanged full-episode success rule: contact by 17 s, mean commanded
speed at most 30 units/s in the last 0.5 s of approach, exact zero speed and
turn within 0.2 s of contact, and uninterrupted stationary contact through
20 s including at least 2 s of dwell. Record arrival, contact, approach
speed, stopping latency, overshoot, departures, and dwell separately.

Final acceptance requires at least 58/64 trained successes, at least 7/8 in
every heading sector, a paired gain of at least eight over each comparator,
and one-sided exact paired p ≤ 0.025 against each. Verify frozen parameters,
source/graph and checkpoint identity, full execution budget, and the
previously recorded contact-path intervention. A failed gate closes the
candidate; do not retune or run another split automatically. Promote the
checkpoint to the default app only after all gates, numerical/integration
tests, saved-frame review, and a separate live desktop check pass.

## Why a new decision is required

The original protocol explicitly allowed one use and forbade retries after
failure. Its consumed receipt and unknown final exposure mean rerunning
830000–830063 would not be a fresh final assay. The project instructions
also require a new decision before another experiment after a failed or
unresolved scientific gate. This proposal made the next action and limits
reviewable before execution; the later outcome is recorded below.

## Prepared implementation (2026-09-29)

The additive `flybrain.training.food_recovery` runner was implemented and
tested before evaluating a recovery case. `--prepare` checks and pins
the fit, completed validation, interruption, contact intervention, exact case
manifest, and relevant source bytes. `--run` requires that protocol, creates
an exclusive one-use receipt, and flushes/fsyncs a JSONL journal after each
of the 192 episodes. Its final audit requires every case and controller in
the declared order, exactly 1,000 world ticks and 20,000 neural steps per
episode, and unchanged model parameters before writing JSON/CSV and gate
results. A caught error records the completed row count and closes the
attempt; an abrupt process loss leaves the durable journal and consumed
receipt. Both commands were subsequently run once as recorded below.

After an explicit recovery decision, the two distinct commands are:

```powershell
.\.venv\Scripts\python.exe -m flybrain.training.food_recovery --prepare
.\.venv\Scripts\python.exe -m flybrain.training.food_recovery --run
```

`--prepare` and `--run` are single-use actions; do not use these commands
on the historical interrupted directory.

## Execution and handoff (2026-09-29)

After the user was told why a new final-only decision was needed, they replied
“if not continue.” The exact proposal was prepared and run once. The
[consumed protocol](../../experiments/food-stop-recovery-2026-09-29/protocol.json),
[journal](../../experiments/food-stop-recovery-2026-09-29/episodes.jsonl), and
[result](../../experiments/food-stop-recovery-2026-09-29/results.json) show
192/192 complete episodes, 192,000 world ticks, 3,840,000 neural steps, zero
updates, unchanged parameters, and **63/64** trained final successes against
**0/64** for each control. Sector counts are 8, 8, 8, 8, 7, 8, 8, 8; paired
gain is +63 and exact one-sided p is about 1.08e-19 against each. All frozen
behavior gates pass. Seed 840044 failed to contact food. All new case
identities are now historical; this receipt must not be rerun.

At the previous handoff, the pinned checkpoint loaded through
`flybrain.food_app`, automated app tests and both headless smoke modes passed,
and the saved frame was inspected. The native Computer Use pipe was
unavailable, so the live gate was still open at that point. The later user
observation and default promotion are recorded below.

## Live review and default promotion (2026-09-29)

The user observed the running food app and confirmed all requested live checks:
food contact and zero-speed dwell appeared with sensor/relay activity in the
brain panel; R started a new trial; Backspace replayed the same seed and
heading. This satisfies the separate live review gate. The default launcher
now loads the pinned food approach-and-stop controller. The historical
arrival controller remains selectable with `--control banc-trained`.

The agent's native Computer Use pipe remained unavailable, so the live visual
and keyboard evidence is the user's direct observation. Automated tests and
headless smokes are recorded separately and do not substitute for that review.
