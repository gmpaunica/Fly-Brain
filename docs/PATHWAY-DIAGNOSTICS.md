# Brain-to-wing diagnostic — 2026-10-05

The existing circuit carries visual information, but its sparse electrical drive
does not produce a wing spike code. The fly also has **no escape decoder**: its
predator commands come solely from the explicit hover stabilizer. This diagnostic
changes neither fact. Brain colors remain tied to actual spikes.

## What the panel means

The lower-left **Pathway diagnostics** section shows visual neurons, the separate
DNp03/DNp26 pairs, and wing-control neurons. Scroll below it for the retained color
legend and guide. Each row reports spikes in the last completed 20 ms interval,
cumulative spikes, maximum incoming current per cell per 1 ms, maximum pre-reset
voltage or phase, and the first episode spike on the exact 1 ms grid.

Visual entry cells use a rate-to-phase counter with threshold 1. They do not use
the voltage threshold; incoming electrical current is discarded by the original
input clamp. Downstream cells use voltage threshold 1. Current is a normalized
voltage increment, **not amperes**; voltage is also normalized, not millivolts.
Maxima are individual-cell maxima, not stage averages or measured physiology.
The panel reads the same immutable completed snapshot as the brain and body.
Pause freezes it; replay/spawn clear it. The interactive trace ring retains five
seconds; the scientific record retains every completed interval.

## Single bounded investigation

Hypothesis and stop condition were locked before onset in
[config](../experiments/brain-to-wing-diagnostic-2026-10-05/config.json).
Exactly **120 presentations, 1 s each / 120,000 neural steps** completed once:
eight directions at .17/.09 m plus eight growth/shutter/reappearance sequences,
each under intact, visual-silent, DNp03-silent, DNp26-silent and both-DN-silent.
Azimuths are 22.5° + 45° × sector, offset from the historical grid. These are new
development diagnostics, not an unseen behavioral assay. Historical placements
remain historical.

The body pose is fixed. The evaluator produces physical RGB eye images from a
stationary spherical proxy at each presentation sample. Sequence growth changes
distance in the evaluator; occlusion uses an explicitly opaque binocular image
shutter for .40–.60 s. Only RGB and the unchanged local observation fields reach
the neural encoder. The shutter is experimental masking, not a sourced visual
occlusion neuron or an added world object.

All wiring, sensory formulas, weights, leak, threshold, subtractive reset and
1 ms propagation delay stayed fixed. There were no sweeps, fitting, parameter
updates, automatic retries, cell additions or motor policies.

## Weak stages and directional information

Across the 24 intact cases:

| Stage | Total spikes | Cases with spikes | Highest pre-reset voltage | Earliest spike |
|---|---:|---:|---:|---:|
| Visual inputs, 96 cells | 13,568 | 24/24 | Voltage is clamped; phase controls firing | .006 s |
| DNp03, 2 cells | 60 | 14/24 | 1.043590 | .089 s |
| DNp26, 2 cells | 1 | 1/24 | 1.000250 | .786 s |
| Wing control, 48 cells | 0 | 0/24 | .190239 | None |

**Missing retained connections:** the current registry contains 87 direct visual
edges into DNp03 and 56 into DNp26. Every intact presentation's driven visual
cells have retained paths to both pairs and all 48 wing endpoints. A separate
graph check that stops at voltage-clamped sensory cells also reaches all 48.
[Functional path check](../experiments/brain-to-wing-diagnostic-2026-10-05/functional_paths.json).
Missing graph routes therefore do not explain these intact wing failures.
This does not establish completeness of the biological circuit: LC4, LPLC1 and
DNp57 are absent from this subset. Their addition is not justified by reachability
alone and requires sourced identities/connections and a separate agreement.

**Insufficient electrical drive:** DNp26 is especially weak (one spike); DNp03
also converts much of its direction-varying input into subthreshold voltage.
All 48 reachable wing cells remain below threshold in every intact case; the
maximum reaches only **19.0% of threshold**. The .9 total incoming-count scaling,
100 ms leak, zero initial voltage and vision-only sensory input are engineered
assumptions. Synapse counts are source structure, not measured electrical gains.

**Direction lost at the spike threshold:** visual per-cell spike patterns
distinguish all eight directions at each distance/sequence. Endpoint spike
patterns are identical zeros everywhere. Endpoint *voltage* contains eight
distinct near patterns, only two far patterns and seven sequence patterns
(values compared at 12 decimal places). Near DNp03 temporal spike patterns also
distinguish eight directions; near DNp26 spike patterns distinguish only two.
Distinct patterns describe this deterministic development set; they do not prove
a robust decoder, unseen generalization or a biologically correct turning sign.
Far/sequence voltage collisions are an additional limitation beyond weak drive.

The per-presentation `failure_categories` in
[results](../experiments/brain-to-wing-diagnostic-2026-10-05/results.json) distinguish
route absence under intervention, reachable-but-silent wings, and directional
collisions. Visual silencing removes every spike and every downstream voltage;
its missing paths are intentional blocking, not absent source wiring.

Silencing DNp03 removes 33.710810 units of aggregate incoming wing current across
the 24 matched cases, and changes individual wing peaks by up to .153771.
DNp26 silencing removes .219485 current units and changes peaks by up to .069767.
Both pairs together remove 33.930296 current units, but other retained routes
still provide small endpoint voltages. All interventions still yield zero wing
spikes. The unchanged *global maximum* .190239 under DN silencing hides these
per-cell changes; it must not be interpreted as no DN contribution.
[Postprocessing evidence](../experiments/brain-to-wing-diagnostic-2026-10-05/postprocessing.json).

## One concrete repair proposal — not implemented

**Restore the existing body-local motion sensory drive alongside cyan vision in
a separately agreed candidate.** Keep the 589 cells, retained edges, original
weights/dynamics and visual formula fixed. Reuse only the original
`FlightBrain` motion-role rate calculation for the existing 16 motion input
cells, then combine those rates with the unchanged predator visual rates. Do not
enable flower/odor/taste inputs or pass coordinates, scene identity or steering.

The current predator encoder zeroes all nonvisual input, although its physical
fly is already flying under the hover stabilizer. The original motion encoder
supplies baseline phase drive (120 Hz for a zero-valued motion component).
Existing retained routes from these 16 cells reach both descending pairs and
all 48 wing endpoints. Restoring this modeled flight-state input is a concrete,
small candidate for bringing weak visual currents closer to firing threshold
without adding arbitrary source cells or lowering thresholds after seeing data.
This is an inference, not a demonstrated remedy: baseline activity could saturate
the circuit or obscure direction, so it requires its own predeclared test.

The biological motivation is limited: Croke et al. report that DNp03's looming
response changes with flight state (Figure 2F). They also describe a broader
DNp03/DNp26/DNp57 network (Figure 5G). Neither result validates our IMU-to-rate
formula, source-ID receptive fields or unsigned gains.
[Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977095/).

The next task should implement this single sensory-drive candidate and declare
a separate budget with a motion-only baseline, stimulus-minus-baseline directional
spike measures, matched interventions and sensory-leakage checks. Success must
require a directional *increment*, not just tonic wing firing. Fix that budget
before running; do not start another experiment from this report. No escape
decoder or retreat behavior should be promoted from the present evidence.

## One small sensory-to-neuron calculation

For an ideal uniform cyan patch with RGB (.05,.95,.95), contrast is .9. Without
image growth, peak and mean are both .9, so the original sensory rule produces
`240 × (.75 × .9 + .25 × .9) = 216 Hz`. This rate fills one entry of a `(589,)`
float64 tensor. Each 1 ms step adds `.216` to that cell's phase: after five steps
it reaches `1.080`, emits a spike and subtracts 1, leaving `.080`. This visual
spike can illuminate its source branches while downstream neurons remain silent.

Directed edge arrays index the *previous* `(589,)` spike tensor. Each retained
edge computes `weight × previous[pre]`; `index_add_` sums those increments into
its destination. Thus that input spike first arrives at its DN destination at
the next 1 ms step, after voltage leak. For one retained LPLC4 → DNp03 edge,
29 of the destination's 618 incoming synapses gives assumed weight
`.9 × 29 / 618 = .042233`. That single arriving spike adds only .042233 to
the DN's voltage, about 4.2% of threshold before other contributions. The DN fires only when its accumulated
normalized voltage reaches 1. If it does not fire, its outgoing edges transmit
zero despite a visible visual response. These operations are PyTorch tensor
calculations; phase encoding, leak, threshold and reset are our hand-written
rules. No autograd, gradients or optimizer updates occur here.
In dense-matrix notation the same orientation would be `W[post, pre]` and
`voltage += W @ previous`; the implementation stores directed edge vectors
instead of allocating a dense 589×589 matrix.

## Records and verification

[JSONL trace](../experiments/brain-to-wing-diagnostic-2026-10-05/traces.jsonl.gz)
contains all 589 cells, input rates, exact first-spike times, current/voltage/phase
measurements and sparse edge transmissions in manifest order for all 6,000
intervals. [CSV trace](../experiments/brain-to-wing-diagnostic-2026-10-05/traces.csv.gz)
contains 888,000 pathway cell intervals; [responses](../experiments/brain-to-wing-diagnostic-2026-10-05/responses.csv)
summarize stages. Config includes software/device, source hashes and seeds;
[receipt](../experiments/brain-to-wing-diagnostic-2026-10-05/receipt.json) closes the
budget and binds the scientific files. The saved-record audit reconciles all
intervals/counts/currents/peaks/times and paired image hashes without another run.

Launch: `.venv/Scripts/python.exe -m flybrain.encounters3d --kind predator`.
The review preview was paused at one development second, seed 990000. The user
confirmed **“Panel and controls work”** in this session. Native automation still
has the missing-pipe error; the human review supplies live acceptance. Saved
1440×900 and 1000×700 frames were reviewed separately.

Exact commands, suite results and preservation audit are in
[verification](verification/pathway-diagnostics.md).
Final results: **111 targeted / 678 full-suite passed**, zero skips; four final
panel checks passed after clarifying effective/discarded-current labels. No
neural or behavior change was made by that clarification.
