# FB-09 contrast diagnosis — saved-data analysis

2026-09-25. No new simulation, parameter change, candidate or training run.

## Evidence

Source: [closed recovery results](../../experiments/fb09-dynamics-recovery/results.json),
SHA-256 `8cfa725e30510469a38bf7b56d312967f0207ef3dd3a654bd07646f4980a5311`.
The table uses budget 3, center 0.25 and the final 750 ms pulse window.
Entries are recorded spike counts, not Hz. ORN/motor columns are group sums;
each PN and DNa02 column represents one cell.

| Odors (left, right) | ORN L | ORN R | PN L | PN R | DNa02 L | DNa02 R | Motors L | Motors R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25, 0.25 | 2042 | 2371 | 125 | 125 | 124 | 125 | 372 | 372 |
| 0.20, 0.30 | 1989 | 2401 | 125 | 125 | 124 | 125 | 372 | 372 |
| 0.30, 0.20 | 2072 | 2323 | 125 | 122 | 124 | 122 | 372 | 363 |

The right-strong input changes sensory counts by -53 on the left and +30 on
the right. Both PN counts are unchanged. This identifies the PN output as the
first measured layer where this particular contrast disappears from aggregate
counts; it does not prove identical spike timing or identical tick-EMA features.

Applying the declared weights to recorded presynaptic counts gives an additional
descriptive check (arbitrary summed jump units, not membrane voltage):

| Pattern | Left PN weighted presynaptic count | Right PN weighted presynaptic count |
| --- | ---: | ---: |
| Equal | 195.020867 | 193.763158 |
| Right stronger | 193.009631 | 193.990305 |
| Left stronger | 195.327448 | 192.801939 |

Calculation: sum `3 * edge.count / sum(retained nonself incoming count) *
recorded_presynaptic_spikes`, using the artifact's ordered root IDs. These
counts do not account for one-step arrival at the window boundaries or preserve
spike timing. Both weighted inputs move in the expected direction for the
right-strong case, yet their PN output counts do not. The right-PN change is
particularly small (+0.227147 summed units), consistent with compressed drive.

## What this establishes and what it does not

The encoder has not erased all contrast: sensory spike counts differ. The
failure is already visible before the motor decoder, at the PN count output.
At center 0.1, budget 3's right-strong case changes PN counts by -3/+2 and
DNa02 counts by -3/+2; at center 0.025 it changes PN counts by -2/+7.
The observed high-band plateau is therefore concentration-dependent.

Saved aggregate counts cannot separate timing-dependent summation, leak,
threshold/reset quantization and encoder compression. The LIF implementation
discards threshold overshoot; that is an existing documented assumption, not
yet a demonstrated cause here. Changing only the encoder would be premature.

## Approved diagnostic — execution started

The user approved this diagnostic and requested continued repair to FB-10
readiness on 2026-09-25. The earlier candidate experiment stays closed.

One instrumented diagnostic replay, with **no changed parameters**: budget 3
and exactly the three existing high-band patterns above, each 250 baseline,
1,000 pulse and 250 washout steps. Ceiling: **4,500 neural steps, zero updates**.
Record both PNs' received weighted spikes, pre-reset voltages, reset loss,
leak and spike times, with source hashes and the same 750 ms summary window.

Hypothesis: distinct but compressed presynaptic drive is mapped onto a PN
spike-count plateau by the current integration/reset dynamics. The diagnostic
must reproduce the recorded counts and identify where input differences go;
it does not select a new gain, encoder or neuron rule and cannot open FB-10.
Stop after those three cases. If it cannot isolate a cause, record that limit
without starting another candidate sweep. A subsequent repair needs its own
explicit hypothesis, fixed budget and unchanged acceptance gates.

## Execution ledger

- Tracer tests first failed for missing instrumentation, then passed (2/2).
  They check actual delayed input, reset overshoot, leak and unchanged dynamics.
- The observer uses existing `record_charge=True`; it never replaces neuron
  equations. No Git/worktree/commit is available or required by this project.
- Running exactly the approved three cases; results are written separately to
  `experiments/fb09-pn-diagnosis`, with original evidence protected by hashes.

## Diagnostic outcome — closed after 4,500 steps

All three cases exactly reproduce all 83 recorded steady-window spike counts.
For equal versus right-strong odor, both PNs also have identical spike times.
No candidate or parameter was changed. All prior files and measured code hashes
are preserved; see [results](../../experiments/fb09-pn-diagnosis/results.json)
and [per-step trace](../../experiments/fb09-pn-diagnosis/projection_trace.csv).

For the left PN, right-strong odor changes actual received jump sum from
195.020876 to 192.953450, leak loss from 5.585040 to 6.598094, and overshoot
discarded on spikes from 64.819267 to 61.888509. It still produces 125 spikes.
For the right PN, jumps change from 193.763152 to 193.952909, leak loss from
5.792901 to 5.188190 and discarded overshoot from 63.423924 to 64.223489,
again with the same 125 spikes at the same times.

Thus this waveform's presynaptic differences change membrane/loss terms without
changing PN output at all. Hard-reset overshoot is a measured information-loss
mechanism here, alongside input compression; these observations alone do not
prove that a soft reset or wider encoder will pass every scientific gate.
