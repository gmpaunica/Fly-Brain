# Turn-and-flee on the retained BANC circuit

**Experiment closed at validation: 87/96**, required >=90/96 and >=5/6 per
azimuth. At 18.75 and 63.75 degrees, only 3/6 responses had the correct side.
All nine failures turned right against right-side threats. No closed-loop cases,
default promotion, extra candidates or retry. The engineered policy is implemented
and tested, but successful turn-and-flee behavior has not been established.

Calibration: (2,1) 13/16; (4,2) and (8,4) both 16/16 eligible. The lowest pair,
**(4,2)**, was frozen. The run consumed **144 cases / 216 simulated seconds /
432,000 paired neural steps**. All 3,600 motion-baseline intervals match exactly;
state is finite and maximum downstream 200 ms rate is 190 Hz. Nine validation
failures reflect incorrect side rather than absent/late output. Calibration
maximum was 175 Hz for all pairs. No optimizer updates.

Ten validation triggers used an exact bilateral score tie; three of the nine
failures were tie-break outcomes. The other six failures had unequal scores.
Trigger strength ranged from 1 to approximately 3.464 steering spikes in norm.

[Result](../experiments/turnflee-2026-10-06/results.json),
[plan](plans/TURNFLEE.md), [verification](verification/turnflee.md).

One separately authorized engineering calibration on the existing 589 cells
and 4,139 anatomical edges. No new cells, anatomical counts, optimizer,
voltage-to-steering readout or RGB-derived steering direction.

The [flight-circuit study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977095/)
supports LPLC4 input into DNp03/DNp26 and DNp03 outputs onto flight/saccade
pathways. i1/b3 reduce same-side wing amplitude; b1 can increase it. Our equal
per-spike score weights, unsigned LIF physiology, effective gains, initial-state
calibration and flight servo are engineering assumptions. This is a retained
source slice with modeled physiology, not a reconstructed functioning fly brain.

`turnflee.neural.ResponseConfig` binds gain pair, affected directed edge IDs,
circuit/source hashes and calibration receipt. Effective weights are multiplied
after the original count normalization. Original anatomy, 100 ms leak, threshold
one, subtractive reset, 1 ms propagation, visual encoding and 16 motion rates
are unchanged. Both actual and motion-only reference receive the same gains.

Each 20 ms interval adds the signed actual-minus-reference counts to a rolling
ten-interval / 200 ms vector. Only six i1/b3/b1 steering endpoints enter the side
scores and response norm. i1/b3 add to the same-side score, b1 to the opposite
side. A norm of at least one and a strong threat event are required. An event
can wait at most 500 ms for spikes. Exactly tied, nonzero bilateral responses
choose left/right using a resettable seeded RNG: engineering symmetry breaking.

The experimental policy brakes horizontal velocity during a 90 degree turn,
then approaches .12 m/s and .1 m altitude. The chosen turn survives neural gaps.
The existing weak-threat continuity starts braking; horizontal and vertical
speeds must each fall below .02 m/s before idle cruise resumes. IMU displacement
above .22 m enables a chosen-side tangent with an inward component until below
.18 m. RGB supplies only apparent size/growth, never bearing or range. Actual
spikes continue to color anatomy. Neural differences and scores are labeled
control calculations.

For the specified closed-loop starts, the controller has a named-state initial
calibration: upright, heading zero; resting at the original leg height; hovering
at nominal .1 m; cruising at nominal .1 m and .06 m/s forward. Physics is set
independently to those constants. No measured world pose/velocity is passed to
control. This assumption does not establish general initialization robustness.

The formal run is `experiments/turnflee-2026-10-06`. The parent `turnflee-once.json`
is consumed before any onset and refuses renamed-directory retries. Config,
source byte archives, prior-record audit, full paired spike counts, image hashes,
pathway diagnostics, signed vectors, scores, rates and gate receipts are saved.
Development unit/render checks are separate from the scientific protocol.

Exactly three pairs (2,1), (4,2), (8,4), each 16 historical presentations at zero
elevation and two approach profiles. Require 16 correct gated responses, paired
baseline equality, finite state and no downstream sustained rate above 240 Hz
over 200 ms. The lowest eligible pair is frozen before 96 offset-angle validation
cases, requiring >=90/96 and >=5/6 per azimuth within 500 ms of first strong onset.
Within 15 degrees of front/rear either turn side is accepted. The old protocols
were audited before lock; offset validation is not claimed unseen.

Only a validation pass permits 48 six-second resting/hovering/cruising intact and
visual-silent cases. Intact gate: >=22/24, >=7/8 per start; silent gate: zero neural
triggers. Success also requires no collision, weak threat and stable recovery.
Maximum 192 cases / 504 s / 1,008,000 paired 1 ms steps. Stop at first failed gate.
Default promotion needs every gate; opening a viewer never runs the experiment.

Read-only audit:

```powershell
.\.venv\Scripts\python.exe -m flybrain.turnflee.audit experiments/turnflee-2026-10-06
```

The gate-aware `python -m flybrain.turnflee` launcher uses the original idle lab
when no final verified response exists. `python -m flybrain.retreat3d` and all
historical launchers retain their original behavior unless the final promotion
condition is met. Do not rerun the consumed scientific experiment.

New candidate interaction logs contain the full immutable response configuration
and identity. Tampered or mismatched sources/configs are refused. Historical
`retreat3d-actions-v1` logs use the original model; Backspace resets added state
and reconstructs each recorded interval, including endpoint spawns, then pauses.

Teaching example: PyTorch stores a voltage vector of shape `[589]`, with aligned
edge source/destination/weight vectors. A source spike delivers a weight .05 to
its destination one millisecond later; an effective gain four makes that .20.
Destination voltage leaks, receives incoming contributions, then spikes at one
and subtracts one. In a toy calculation, old voltage .85 leaks to .84154, receives
.20, reaches 1.04154, emits one spike and retains .04154 after reset.
The sparse computation is `weights * previous[pre]`, followed by accumulation
into `voltage[post]`. The vectors have shape `[E]` for non-self edges and `[589]`
for neurons. An equivalent dense matrix would have destination rows and source
columns; the actual implementation uses indexed accumulation instead.
Three extra left i1 spikes and one extra right i1 spike yield
left/right scores 3/1 and choose left. A left b1 increment adds to right instead.
These are forward tensor calculations under our fixed rules; no autograd,
gradients, optimizer updates or learned parameters are involved.
