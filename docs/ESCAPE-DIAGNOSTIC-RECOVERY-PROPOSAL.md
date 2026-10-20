# Comparator-only diagnostic recovery — approval requested

The diagnostic is consumed after a false `pathway` mismatch in the first
baseline interval. Its saved values equal the historical record exactly.
There is no threat-exposure evidence to justify choosing an escape decoder.
This is one concrete repair to the **diagnostic comparator**, followed by one
explicitly reauthorized unchanged-model measurement run. It is not implemented.

## Exact correction

In a new additive runner version, immediately before the historical equality
test, compute:

```python
canonical_original = json.loads(json.dumps(record['original'], allow_nan=False))
if canonical_original != historical_interval:
    # Seal failure immediately, exactly as before.
    raise ValueError('historical interval mismatch')
```

Compare the whole canonical dictionary, including every diagnostic field.
Use no numerical tolerance, rounding, removed field or alternate array test.
Counts, values, RGB hashes, timing, scores, threat events and first selection
must all still match exactly. Canonicalization changes only the tuple/list
representation of a comparison copy; it never feeds into neuron/controller state.

Preserve the consumed v1 package/source archive, failed records, parent lock,
launchers and defaults. Add a small comparator module and v2 runner with a fresh
explicitly authorized parent consumption key; do not edit/delete/reuse the old
key. Carry forward source hashes of the unchanged recorder kernels and frozen
production modules. Bind the new comparator, runner, tests and receipt together.

## Failure coverage, timing and tests

This fixes the verified false comparison stop. It **does not address any of the
nine wrong-side escape selections**, including the three seeded ties. Existing
readout latency and failure behavior stay exactly as recorded: 20 ms decision
intervals, 200 ms signed rolling counts, norm >=1, the original 500 ms pending
latch and seeded ties. No response performance is predicted from this correction.

Before consumption, use saved-only tests that accept all tuple/list diagnostic
fields after canonicalization and reject a one-value mutation in each of RGB
hash, counts, current, voltage, phase, timing, rolling vector, score, threat and
selected side. Test missing fields, nonfinite values, changed sources, consumed
receipt refusal and sealing on interruption. These tests require zero neural
steps and zero rendered presentations. Reuse the unchanged-kernel parity proof;
if wrapper confirmation is needed, cap **all additional synthetic checks at
163 individual kernel steps**, keeping cumulative engineering use <=1,910/2,000.
No unbudgeted full synthetic file or full simulation suite before this diagnostic.

## One fresh scientific budget and stops

Only after approval, execute exactly the same 18 historical identities, failure
first, one time each, in this order:

| Failure | Successful context | Elevation difference | Profile difference |
| --- | --- | ---: | --- |
| 1100000 | 1100004 | +40 degrees | Same |
| 1100001 | 1100003 | +20 degrees | Same |
| 1100002 | 1100005 | +20 degrees | P1 versus P0 |
| 1100010 | 1100008 | -20 degrees | Same |
| 1100012 | 1100014 | +20 degrees | Same |
| 1100013 | 1100015 | +20 degrees | Same |
| 1100017 | 1100016 | 0 degrees | P0 versus P1 |
| 1100032 | 1100033 | 0 degrees | P1 versus P0 |
| 1100036 | 1100037 | 0 degrees | P1 versus P0 |

Keep 589 cells, 4,139 anatomical edges, (4,2), visual-field assignment, motion
inputs, precision, neuron rules, independent actual/reference brains and every
original detector/readout calculation. Record all neurons and non-self edges.
No shadow brain, parameter search, ablation, extra presentation or automatic retry.

Additional hard budget: **18 x1.5 seconds =27 simulated seconds /54,000 individual
kernel steps**, including the 0.5-second baseline per case. Account for the
already consumed **0.02 seconds /40 steps** separately: cumulative diagnostic
science would be **27.02 seconds /54,040 steps**, with 19 presentation starts,
18 completed presentations if successful, and the prior start having no threat
image. Do not relabel the failed baseline as zero computation or unseen data.

Stop at the first true canonical historical mismatch, changed protected source,
missing measurement, nonfinite state, downstream rolling rate >240 Hz or budget
breach; otherwise stop at exactly 18. Seal success, failure or interruption and
never resume/retry automatically. The same four observational support conditions
must hold for **all five deficit-only failures**; otherwise classify unsupported
or unresolved and return for a new decision. Temporal order cannot establish
causal necessity. Produce guarded audits, tables and timelines from saved records.

## Replay, activation and later behavioral requirements

This correction has no interactive activation or physical trajectory replay.
Saved-only record verification checks seals, archives, tensor recurrence and
original endpoints. Preserve existing replay/launchers and keep escape disabled.
If the timing hypothesis receives support, return with exactly one concrete
spike-only temporal readout proposal and fixed parameters selected from engineering
timing assumptions, with another approval before its implementation/evaluation.

Carry the user's later repair gates forward unchanged: sensory/neural-only side
selection; one source-locked candidate; historical cases remain historical;
stop at the first failed gate. Require >=90/96 directionally correct and >=5/6
at every azimuth within 500 ms. Only then run the prescribed 48 six-second
closed-loop cases: eight directions x resting/hovering/cruising x intact/silent.
Require >=22/24 intact, >=7/8 per state, zero silent triggers, correctly signed
turning/takeoff when applicable, separation, collision-free flight, weak-threat
braking and stable recovery demonstrated in recorded trajectories. A side pass
alone is insufficient. Run targeted and small full suite, independent audits,
preservation and replay checks after that repair. Review saved visuals separately
from live interaction; mark unavailable live review pending. Activate only a new
receipt-bound recovery launcher after all gates pass **and** activation approval.

**Approval requested:** the comparator-only correction, the bounded preflight
above and exactly one fresh unchanged-model diagnostic under this new explicit
budget. No neural repair, training, closed-loop experiment or activation is
included in that approval.
