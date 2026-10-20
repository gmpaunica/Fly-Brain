# Sustained escape to the edge

The requested behavior is turn, take off if needed, flee to the map edge, brake,
and hover there. A shrinking or absent threat does not end a committed escape.
The old radius/home guard and weak-threat recovery are preserved in historical
controllers, but not used by this new policy. Both declared neural batches passed:
48/48 intact escapes and zero triggers in 48 silent controls. See
[measured results](ESCAPE-EDGE-RESULTS.md) and [verification](verification/escape-edge.md).

## Controller and sensing

The existing 589-cell PyTorch forward simulation and temporal paired-spike
readout are unchanged. Actual spikes color the brain; subtracting the reference
counts is a separate engineered control calculation. For example, left score
3 and right score 1 on two consecutive 20 ms samples commits a left turn.
Those six steering counts are selected from the manifest-ordered `(589,)`
count tensor. The readout sums integer counts; no optimizer or autograd update
occurs. Wing signs, neuron equations and stabilization are modeling assumptions.

The neural decision commands a signed 90-degree turn. IMU acceleration/angular
velocity then update the existing orientation/velocity/displacement estimate.
The authored calibration is a fixed centered start in a .6 x .6 m arena. The
flee ray meets a rectangle 15 mm inside its walls, equivalent to relative bounds
[-.285,.285] m. These are fixed modeling constants, not live coordinates.
World geometry and predator placement belong only to physics, evaluation and UI.

Outside the final 60 mm, the flight target is .12 m/s. Inside that approach
region, the target is twice the remaining endpoint displacement per second,
capped at .12 m/s. The retained velocity servo handles inertia. Hover latches
when estimated endpoint error <=5 mm and speed <=.01 m/s; it continues endpoint
correction and .10 m altitude stabilization. No landing timer or home bias runs
during active escape. A new encounter mode or reset clears the latch.

## Verification contract

Motor tests use real physics and IMU/contact but a synthetic neural impulse;
they do not establish neural success. Each measured batch uses 48 cases x 12 s,
1,152,000 individual actual/reference neural steps. Development geometry and
seeds are historical. A separate batch uses new seeds on the same geometry;
this is seed robustness, not unseen geometry. All 24 intact cases must pass;
silent visual controls must never trigger. Saved-only audit and four exact
replays are separate evidence. Tests, rendering and live interaction have their
own engineering accounting. Historical 94/96 directional failures at seeds
1100018 and 1100021 remain; no universal escape or learning claim is made.

## Run after activation

`.\.venv\Scripts\python.exe -m flybrain.escape_edge`

The launcher refuses without passing source-bound delivery evidence. Spawn
predator to observe the new escape, WASD/QE to move it, Space to pause, Backspace
to replay the interaction, R for a fresh no-object trial, Esc to exit. Historical
launchers stay unchanged. Opening the app never runs training or evaluation.
