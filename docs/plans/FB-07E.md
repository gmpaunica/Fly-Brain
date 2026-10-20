# FB-07E plan: weak-signal sensory probe

## Question and one candidate

Can a zero-preserving saturating encoder increase weak-odor spike activity and
reduce exact seven-feature aliasing on both locked train and validation
observation sequences? Test one hand-written rational current rule:
`I(c) = 2.5*c/(c+0.005)`, with `I(0)=0`. Concentration and current are in the
same arbitrary simulation units used by the synthetic encoder. The 0.005
half-saturation concentration and 2.5 current limit are fixed before the
trajectory replay; they are not fitted to validation outcomes or biology.

Replay the same FB-07D fixed-controller observations (200 ticks per train case,
600 per validation case) through fresh networks at original scale 0.025,
wider scale 0.05, and the rational candidate. Reset network and seven-feature
EMA at every case boundary. Record per-case and split-level raw/feature variety,
aliased ticks, teacher-turn MSE floor, and weak-input sensory spike activity.
Check a small reversed-input probe below concentration 0.025, zero input,
monotonic current, and fixed topology. Do not train or evaluate embodied
controllers or touch final starts in this task.

The gate to another bounded readout attempt is: the candidate must reduce
aliased ticks below the **better** FB-07D scale on both splits, avoid worsening
the turn-only MSE floor relative to the better scale on either split, and
increase weak-input sensory spike activity on both splits. If any condition
fails, record the result and identify the remaining bottleneck without tuning
another candidate this session.

## Implementation and verification

1. Add tests for the rational encoding formula, finite/zero input, bilateral
   spike discrimination and unchanged default behavior/topology.
2. Add tests for same-observation replay, episode reset, split restriction,
   weak-input activity counting and the gate decision.
3. Record one full train/validation run under `experiments/fb07e-weak-signal/`
   with config, software/device, seeds, per-case CSV and JSON metrics.
4. Audit artifacts, run focused and full suites, and update experiment,
   decision, queue, status and log records.

Only raw left/right concentrations cross the neural boundary. The candidate is
synthetic scaffolding; a more informative feature probe alone is not learning
or improved embodied navigation.

## Verification record

On 2026-09-21, the one full probe reproduced all FB-07D original/wider
train and validation alias metrics exactly. It recorded 36 case/encoder rows,
unchanged topology and no training/final episodes. The candidate passed all
validation gate criteria but failed train aliasing: 627 ticks versus the
wider scale's 452. Forty-four focused tests and the 212-case full suite
passed; `compileall` passed. The suite retained one upstream Pygame warning.
See [experiments](../experiments.md) for the full table.
