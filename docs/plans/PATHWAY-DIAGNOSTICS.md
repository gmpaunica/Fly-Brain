# Brain-to-wing diagnostic implementation plan

User specification: 2026-10-05, Build the brain-to-wing diagnostic.
One agent; existing wiring, sensory rules, dynamics and motor behavior fixed.
The supplied specification authorizes execution without another design approval.
No Git repository; no automatic commit, source extension, sweep or training.

## Design and predeclared investigation

Add `diagnostics.py`: a read-only instrumented equivalent of the original
TorchScript stepping kernel, mixin subclasses for flower and predator brains,
immutable per-neuron interval records and graph reachability/stage summaries.
Visual inputs have phase threshold 1; downstream cells have voltage threshold 1.
Record effective incoming current sum/peak (normalized voltage increment per
1 ms), discarded current, pre-reset peak voltage, peak input phase, counts and
first spike on the exact 1 ms time grid. Preserve every original numerical operation.
Snapshot publication carries the complete diagnostic interval atomically.
Add `diagnostic_panel.py`: passive three-stage text panel with separate DNp03 and
DNp26 rows. Brain colors continue to use only original spike records.
Add `investigate.py`: fresh-directory-only, source-bound JSON/CSV traces and report.

Hypothesis: current sensory assignment and assumed dynamics fail to deliver a
usable directional signal to the wings. Budget: exactly 24 RGB presentations x
five interventions x 1 s = 120 presentations / 120,000 neural steps.
Cases: eight body-relative azimuths at .09/.17 m, offset 22.5 degrees from the
historical grid; eight growth/occlusion/reappearance sequences at those azimuths.
Sequence: .17 m (0-.20 s), linear growth toward .09 m (.20-.40 s), opaque
mask (.40-.60 s), reappearance at .09 m (.60-1 s). RGB only reaches the brain.
Fixed body pose; environment/evaluator alone uses coordinates. All cases are
development diagnostics, never an unseen behavioral assay.
Interventions: intact, visual-silent, DNp03-silent, DNp26-silent, both DN pairs
silent, applied before onset with all state reset. No saturation stimulus.
Stop after the budget, even if wing signals fail. No automatic scientific retry.
Directional evidence: distinct per-cell spike vectors and per-cell peak voltage
vectors across matched azimuths, including collisions/zero signals; distinct
vectors do not establish reliable decoding or behavior. Separate retained-path
absence, reachable-but-subthreshold drive, and non-distinguishing output.

## Tasks and checks

- [x] 1. Add numerical instrumentation. Test exact tensor/count/history/edge
  equality against original stepping for flower and predator, variable interval
  sizes, silencing, threshold/reset/leak/delay and first-spike timing.
- [x] 2. Publish interval diagnostics and add passive panel; test replay/reset,
  pause, rendering independence, flower parity, rejected background/flower RGB,
  preferences and two responsive layouts. Capture frames and review separately
  from a native live review attempt.
- [x] 3. Lock/run one investigation; audit 120 presentations, paired image hashes,
  complete traces and source hashes; classify weak stages and one concrete repair.
- [x] 4. Targeted encounter/3D suites then full suite; local acceptance review;
  update status/log/architecture/decisions/queue and explain one calculation.

Review focus: input phase versus ignored voltage; clamps during interventions;
exact pre-reset peaks; immutable atomic publication; no timing/stepping in panel;
opaque sequence removes RGB signal rather than passing an occlusion label;
complete reset and preserved camera/display preferences; no decoder promotion.

## Execution ledger

Tasks 1–3 complete: exact original/instrumented parity, passive UI and accepted
controls, one locked 120-presentation record and saved-record audit. User confirms
“Panel and controls work”. Initial targeted run:111 passed; final label regression
and real panel checks:4 passed. Historical preservation:2395 unchanged files,
three encounter integration originals archived byte-for-byte.

Rulings: user supplied execution scope overrides additional skill approval,
commit/worktree or subagent workflows; one agent/local acceptance review.
No Git repository. Additive diagnostic modules plus minimal encounter integration
changes; preserve those original integration bytes in artifacts. Interactive trace
ring bounded to five seconds, with full disk traces for the investigation. The
occlusion condition is an explicit evaluator-only binocular shutter. No parameter
sweep, training, decoder promotion or second scientific attempt.

Task 4 complete: 111 targeted, 4 final panel checks, 678 full-suite tests passed;
zero skips, three upstream warning instances in full suite (361.86 s). Author local
acceptance review and documentation complete. First full-suite process yielded
no output/XML and is not counted; successful test-only retry used single-thread
OMP/MKL settings. Closed scientific record was never rerun.
