# FB-10 conditioned endpoint-readout recovery

2026-09-27. The user's renewed instruction to fix the failures and finish FB-10
authorizes this bounded repair and, only after training gates pass, its separate
fresh matched evaluation. Preserve the original failed fit. Work with one agent;
no Git repository or automatic commits. Existing scientific gates stay intact.

## Diagnosis and design

The offline, zero-update/tick [diagnosis](../../experiments/fb10-conditioning-diagnosis/results.json)
found common-mode std about 72.5 Hz and bilateral-contrast std about 4.2 Hz.
Original normalized singular values span 89.75 to 1.14 over four effective
dimensions. Both sides were saturated in 249/1,008 samples, where residual
error was lower than in the unsaturated samples; saturation alone does not
explain this failure. Conditioning is a measured limitation, not yet a proven
complete cause.

Keep all eight original endpoint EMA rates and their fixed 100 ms filter.
For each anatomical pair (DNa02 and three matching motor slots), form its mean
and right-minus-left contrast. Independently standardize those eight values
using training population mean/std, floor 1 Hz. This invertible arithmetic
re-expression of endpoint rates adds no sensory information or geometry.
Keep seed 1010, 8->16->2, 178 learned scalars, Adam lr .01 and 500 updates.
Use separate modules/checkpoint schema so historical implementations, source
graph, bundle and evidence remain byte-identical.

For left/right rates 140/144 Hz, common activity is 142 Hz and contrast 4 Hz.
If contrast mean is 0 and scale 4 Hz, it becomes 1; independent 72 Hz rate
scales previously made that difference about .056. PyTorch uses [1008,8]
inputs, [16,8] then [2,16] matrices. Only their weights/biases learn; the fixed
transform, normalization, endpoint EMA and spiking circuit do not.

## Implementation and verification

- [ ] Write failing tests in `tests/test_banc_conditioning.py`: numerical pair
  orientation and conditioning, bounds, deterministic initialization, artificial
  gradients, normalization, checkpoint integrity, neural-only exact filter/reset,
  two-tick replay, protocol binding and single-use behavior.
- [ ] Add `neural/banc_conditioning.py` for the fixed transform/readout and strict
  checkpoint contract. Run targeted tests: expected numerical tests pass.
- [ ] Add `training/banc_conditioning.py` to prepare/validate and consume exactly
  one recovery protocol, fit and record three matched training controls plus
  replay, using the existing numerical fit and gate helpers. Run targeted/full
  suite, then freeze a protocol before any scientific updates.
- [ ] Execute one candidate at `experiments/fb10-conditioned-readout`. Hypothesis:
  conditioning exposes endpoint contrasts sufficiently to pass the original
  turn-error and embodied gates. Budget: 500 updates, 25 episodes /15,000 world
  ticks /300,000 neural steps. Stop on nonfinite numerics; close on any failed
  scientific gate. No sweep or unrecorded follow-up candidate.
- [ ] If all training gates pass, implement/test and freeze a separate evaluation
  protocol with 16 fresh validation and 32 fresh final starts, three matched
  conditions, zero updates and at most 86,400 ticks. Validation >=13/16 and
  strictly more arrivals than both controls; final >=26/32 and paired improvement
  >=8 with one-sided exact p<=.025 against each control. Stop before final if
  validation fails. Cases generated independently from historical seed ranges,
  recorded before execution; no candidate changes after locking.
- [ ] Audit JSON/CSV/checkpoints, preserve historical bytes, inspect result plots,
  update status/log/decisions/queue and report achieved gates and limits. A failed
  recovery returns to a documented diagnosis/decision, not an automatic rerun.

## Local review focus

Signed contrasts must be derived only from spikes, with correct right-minus-left
orientation. Validation/final information must not enter normalization or fitting.
Copied/resealed protocols must not reopen consumed attempts. Checkpoint loading
must reject wrong transforms, feature order, core, normalization or protocol.
Rendering and auditing must consume no simulation time. Training success alone
does not complete FB-10; fresh matched behavior is the finish gate.

## Progress

Offline diagnosis complete with zero optimizer/world/neural steps. Renewal of
authorization overrides the previous preparation-only scope and permits carrying
the repair through the scientific gates; it does not reopen old protocols.

### Conditioning outcome and debugging reset

The conditioning-only candidate completed 500 updates and 25 episodes,
13,850 ticks /277,000 neural steps. Turn MSE .06408428 and 2/8 arrivals failed
the original gates. All integrity checks passed; full suite 413 passed.
Close this candidate and preserve its entire implementation/protocol/checkpoints.
Conditioning is insufficient, so extending or repeating its fit is rejected.

A minimal equal/mirrored odor reproduction ran exactly 9,000 neural steps with
zero optimizer updates/world ticks. Equal odor caused contrast std 3.88796 Hz;
a 10% odor difference produced only about 1.44 Hz mean change. A causal 1 ms
delay barely reduced noise (3.79060 Hz); that approach is rejected. A 500 ms EMA
reduced it to .815565 Hz. This isolates short-window spike-count noise as a
measured interface problem, while preserving the source graph.

One train-only 500 ms refiltering readiness report recovered the original
integer counts from saved 100 ms EMAs with maximum error 1.78e-15. Episode
resets and all 1,008 labels were preserved. Contrast/target correlation rose
from .34580 to .67369 without any extra neural/world steps or optimization.
These results do not prove navigation or authorize lowering its gates.

### Revised measured repair

The user explicitly renewed the broad instruction to fix failures and finish
FB-10. Choose one dual-timescale readout candidate from the measured diagnosis:
the same eight endpoint spikes feed independent causal 100 ms and 500 ms EMAs.
Apply common/contrast conditioning to each, producing 16 readout inputs. Keep
16 tanh hidden units and two bounded outputs (306 trainable scalars), seed 1010,
Adam lr .01, 500 updates and all original scientific gates/budgets. Fast history
retains response speed; slow history reduces counted-spike noise. No direct odor,
voltage/charge, teacher input, future alignment or world geometry enters inference.

Add separate `neural/banc_memory.py` and `training/banc_memory.py` plus behavioral
tests before implementation. Do not modify any of the 50 now-recorded Python
sources. The new contract binds both EMA time constants, 16 feature IDs/order,
conditioning, source graph and train-only normalization. Record one experiment
at `experiments/fb10-memory-readout`; close it on any failed gate and do not
automatically append another candidate. A pass permits the separately frozen
fresh matched evaluation described above, within the renewed user scope.

### Temporal training passed; fresh evaluation locked next

The temporal candidate passed all original gates in exactly 500 updates:
turn MSE .00837231336133627, trained 8/8 arrivals versus matched untrained 0/8
and fixed BANC 0/8. All 306 parameters received nonzero gradients and changed.
Exact tensor/embodied replay and frozen core/normalization/provenance passed.
25 episodes used 10,775 ticks /215,500 neural steps. Full suite 426 passed.
The temporary refiltering input-alias bug was reproduced/fixed before fitting;
the dataset now preserves every original fast rate and teacher label exactly.

Frozen checkpoint SHA256:
`06d289e5bb7f7886b88ef3e08f0493f7f3ed16b981617b0043fa12614f5dca72`.
No changes to the candidate, its filters, normalization or optimizer follow.

Fresh assay: validation seeds 620000-620015, final seeds 621000-621031, drawn
with the same area-uniform 60-180 world-unit source annulus and independent
uniform headings. Verify their disjointness from recorded prior seed identities
before locking. Three matched conditions, same world/horizon, zero updates,
144 episodes maximum /86,400 ticks /1,728,000 neural steps. Validation needs
>=13/16 and strict arrival/progress improvement over both controls; stop before
final if it fails. Final needs >=26/32, paired arrival difference >=8 and exact
one-sided p<=.025 against each control, plus strict all-case mean progress and
frozen source/parameter/history checks. These gates are declared before outcomes.

Add/test `training/banc_memory_evaluation.py`, then prepare and consume one
destination-bound protocol at `experiments/fb10-banc-evaluation`. A central
receipt in the candidate directory blocks a second evaluation at any destination.
The broad renewed user request authorizes this separate frozen evaluation;
it does not authorize tuning on its results or relabeling its cases as new.

### First fresh assay closed at validation

Protocol SHA256 `ca39d7f03e23b9957b46ed792e84cf7a7d3d3ccc28f58aa0f8ae9ac61d071e31`.
Exactly 48 episodes /24,519 ticks /490,380 neural steps /zero updates.
Validation trained 11/16, untrained 0/16, fixed 1/16. Paired comparisons and
all-case progress passed, but minimum arrivals missed 13/16. Close the assay;
its central receipt is consumed, and no final episode ran. Preserve all seeds
as reserved/historical identities for any later protocol; do not rename/reuse.

Stable full suite 433 passed in 34.37s. A concurrent full-suite run had one
test-fixture provenance failure because it captured the actively growing
evaluation journal; actual scientific protocol integrity remained valid.
The failed fixture named only `episodes.jsonl` as changed. Rerunning after
the journal stopped passed without source changes or weakened integrity checks.

All five failed student episodes ended near 72 world units from food, with
zero collisions. A new bounded historical-case diagnosis records exact replay,
last-200-tick teacher/readout disagreement and matching teacher outcomes.
Budget 10 episodes /6,000 ticks /120,000 neural steps /zero updates. No new
candidate or held-out case is inspected during that diagnosis.

### Measured coverage repair decision

Exact replay reproduced all five failures; the sensory teacher arrived 5/5.
Last-200-tick student means were about -.27 to -.34 versus teacher -.93 to -.996.
The original 1,008 labels contain zero turns below -.8; minimum -.46347 versus
maximum +.89150. Failed-rollout tails need mean -.98112 and lie on average
7.20477 normalized units from their nearest training feature row. This is a
measured coverage gap, not a reason to adjust the success radius or turn gain.
Diagnosis used 3,445 world ticks /68,900 neural steps /zero updates.

Under the renewed fix-and-finish request, choose one dataset-coverage repair,
keeping the 306-parameter temporal model, graph, filters, seed/lr/500 updates
and scientific gates. Reserve 32 new training starts, seeds 700000-700031,
same 60-180 unit annulus/headings. Record all 32 sensory-teacher trajectories
and all 32 frozen-student trajectories, labelling the latter from actual sensory
observations without changing their body commands. Include all cases/outcomes,
not just strong-left failures. Append the original 1,008 training samples.
Historical validation cases 620000-620015 and reserved prior finals are excluded.

Predeclare collection budget 64 episodes /38,400 ticks /768,000 neural steps
and zero updates; require teacher 32/32, finite endpoint-only features/labels,
exact original data and fixed parameters/core. Declare a train-only constant
predictor before fitting. One new full-batch 500-update candidate then uses
25 original matched training episodes including replay /15,000 ticks maximum.
Require aggregate turn MSE <=.02225273146561865 AND <=25% of the new train-only
constant baseline, original-eight teacher-data turn MSE <=.02225273146561865,
>=7/8 original training arrivals and both strict matched improvements.

Add separately tested `training/banc_coverage.py` and a separate evaluation
runner; preserve all 53 now-recorded Python sources and all consumed experiments.
Collection and fit have independent receipts/protocols. Close on failure;
no automatic dataset sweep, additional updates or learning-method change.
A passing fit allows a new locked 16/32 matched assay, validation seeds
720000-720015 and final 721000-721031, with the same 13/16 and 26/32 gates,
paired gain/p thresholds and zero-update integrity/budget controls. Scan all
prior seed records, including the failed assay, before locking the new cases.

### Covered-state collection and fit outcome; one capacity test

Collection passed: teacher 32/32, frozen student 22/32; exactly 14,856 samples,
14,856 world ticks /297,120 neural steps /zero updates. Combined training data
is [15864,16], 72 recorded trajectories, mean turn -.27600, constant turn MSE
.31523325, 4,954 strong-left labels and 231 strong-right labels. No failed
validation case entered training. The original 1,008 samples remain exact.

The fixed 16-hidden-unit covered fit completed 500 updates, original training
8/8 versus 0/8 and 0/8, all numerical/replay/integrity checks, at 11,009 ticks /
220,180 steps. Aggregate MSE .03579290 and original MSE .02694830 missed the
frozen .02225273 absolute ceilings. Preserve and close it; do not relax gates
or append optimizer updates. The new constant-fraction gate passed and does
not replace those failures.

Measured residuals are not limited to unobservable initial ticks: teacher later
MSE .01975, student later MSE .03974 over 10,031 rows. Final loss still declines
with finite gradients. Choose one bounded capacity test under the user scope:
16 inputs ->64 tanh hidden units ->2 outputs, 1,218 readout scalars. Keep all
data, fixed 100/500 ms filters, conditioning, seed 1010, Adam .01, 500 updates,
original and aggregate gates/budgets. Add separate neural/training modules and
strict checkpoint schema; preserve all 54 existing sources. No architecture
sweep or automatic next candidate. This is extra readout capacity, not extra
spiking neurons or trainable source wiring. Output: `experiments/fb10-capacity-readout`.

### Precision diagnosis and one projection-input readout

The capacity test closed after 500 updates: aggregate turn MSE .03413672 still
missed .02225273. No width sweep. A training-only replay probe measured the
actual weighted previous-step spikes entering the two projection neurons,
without changing their dynamics or inputs. Over 20,160 neural steps, zero
world ticks/updates, all original endpoint rates replayed exactly (maximum
error zero). Projection-input contrast correlation with teacher turn was
.87787458 at 100 ms and .82013066 at 500 ms, compared with endpoint .34580/.67369.

Choose one information-preserving readout revision under the renewed repair
scope. Add four fixed histories of actual PN incoming spike events to the
existing 16 endpoint histories. At each real neural step, observe
`W[PN,pre] @ previous_spikes`; sum over the interval and divide by the unchanged
row budget 3 and elapsed seconds. This measures internal synaptic events, not
external sensory current, voltage, reconstructed odor, coordinates or teacher
commands. It changes the readout boundary: downstream endpoint mediation is
no longer a supported claim. Wiring, encoding, LIF rules and all 83 cells stay
fixed; only a 20->16->2 head (370 scalars) learns. PN pairs receive the same
100/500 ms filters and fixed train-only mean/contrast conditioning.

Reproduce all 64 already-declared coverage trajectories with their exact old
body policies; append original 1,008 training observations replayed with no
world stepping. Require bit-exact old 16 features, labels and all episode
outcomes; do not collect different cases or select rows. Collection maximum
64 episodes /14,856 world ticks /317,280 neural steps /zero updates, with
1,008 extra file-only neural intervals. Preserve all 56 existing source files.
Fit one 500-update seed-1010 Adam-.01 candidate on all [15864,20] rows with
unchanged original/aggregate error ceilings and matched navigation gates.
Close on failure. A passing fit permits the separately reserved 720000/721000
16/32-case assay, unchanged arrival/progress/statistical gates, zero updates,
one central receipt. No tuning on held-out outcomes or silent repeated fits.

The replay's exact-feature gate closed false: all 64 body outcomes, labels,
coverage features and original fast rates were exact, but original slow EMA
rates differed by at most 1.27898e-13 Hz. The original data used `exp(-dt/tau)`;
the live decoder uses `1-(-expm1(-dt/tau))`. This float64 arithmetic difference
is measured, isolated to reconstructed original slow rows, and must remain
recorded as a failed collection gate. Do not rerun world/neural collection.
Choose an explicitly frozen zero-step canonicalization: require every other
collection check, bit-exact coverage/fast rows and labels, original slow error
<=1e-12 Hz; retain all 16 original canonical values and append the four measured
PN features unchanged. Write separate canonical data/protocol/evidence and a
separate fit runner, preserving both collection source modules unchanged.

### Covered precision fit closed; clipping reproduced; fixed encoder repair

Canonical data passed with zero steps/updates. The 370-scalar precision head
completed 500 updates, original MSE .01088492, original arrivals 8/8 vs0/8 vs0/8,
all integrity checks; aggregate MSE .02354470 missed .02225273. Close it.
Residual partition: PN inputs both >90Hz contribute .01479074 of total MSE,
versus .00875397 from all other rows. A minimal historical training replay of
student seed700015 (one episode /600ticks /12,000steps /zero updates) was exact
and proved both encoder rates equal96Hz for133ticks, with58 strong-left and60
strong-right labels, ranging from-1 to+1. This erases current sensory contrast.

Under the renewed fix-whatever instruction, make a separate fixed sensory
scaffolding revision, leaving 83-node wiring, weights, LIF dynamics and time
unchanged. Preserve the original sqrt rule through c=.03 (30.484Hz), then
use `r0 + (96-r0)*log(c/.03)/log(2.5/.03)`, capped96 above the arena's maximum
c=2.5. This preserves contrast throughout the finite arena range. Parameters
come from the existing low-band boundary and world maximum, not fitted labels.
Keep distributed phases/pulses and PN input/endpoint readout semantics. This
is a new engineered encoder, not the original FB-09 encoder or physiology.
The original app, accepted package, sources and all failures remain unchanged.

Before collection/fitting run one entry assay: five centers .025,.1,.25,1,2
with the original five patterns plus zero (26pulses), original transient,
finite-rate/washout gates, source-derived aligned direction windows and the
same >=1Hz/2% thresholds; then four original causal cuts. Budget45,000neural
steps /zero world/optimizer. Close on any failure; no encoder slope sweep.
If it passes, replay all original+covered training sensory trajectories using
the old frozen body policy and a separate new-encoder shadow circuit. Labels
and all64body outcomes must be exact, all new features finite, graph/neurons
fixed. Budget64episodes /14,856worldticks /614,400totalneuralsteps /zero updates
(two20-step brains perworldtick plus1,008file-only shadowintervals). Record odors
for audit but explicitly load only20neural rates as features. No fresh seeds.

Fit one 370-scalar head, same500/1010/.01, all unchanged error/arrival/integrity
gates. Matched fixed/untrained controls use the same revised encoder/core.
A passing fit permits one fresh720000/721000 assay, unchanged16/32case gates,
central receipt and zero updates. Do not run it if entry, data or fit fails.

Range entry closed negative at39,000steps, zero world/optimizer: all26transient
checks passed, but relative endpoint contrast missed2% at .1,.25,1,2. No cuts,
collection or range fit followed; no slope sweep. Range software prototypes
are explicitly blocked from collection/fitting. A file-only calibration check
found PN side mixing has singular values1.00143/.26372, but the covered-data
PN contrast stds already exceed1Hz (1.30786/1.26407). Unmixing followed by
standardization is effectively the same contrast representation, so no
calibration candidate is justified or fitted.

The frozen original-encoder precision candidate passes every original FB-10
gate (original MSE .01088492,8/8 vs0/8 vs0/8,all9integrity checks), but its
additional covered absolute gate remains failed and its fit remains closed.
An explicit scientific acceptance decision is pending: evaluate that frozen
candidate under the original gates while reporting the extra failure, or retain
the additional gate and stop. No dependent fresh episode runs before the user
answers. Existing16/32fresh cases/gates and all failures remain unchanged.
