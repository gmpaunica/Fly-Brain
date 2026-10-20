# Sustained edge escape — measured result, 2026-10-11

The separate edge controller now keeps fleeing through weak/absent threat until
it reaches a map-edge endpoint, then brakes and hovers. Every declared intact
case passes. The verified native app is open and paused at edge hover.

| Evidence | Recorded result |
|---|---|
| Historical development | 24/24 intact successes; 24 silent controls, zero triggers |
| New-seed robustness, same geometry | 24/24 intact successes; 24 silent controls, zero triggers |
| Episodes | 96 x 12 s = 1,152 simulated seconds |
| Individual actual/reference neural steps | 2,304,000 scientific; 96,000 exact replay |
| Edge-hover arrival | 4.84–5.02 s, required <=10 s |
| Minimum stable edge hover | 7.00 s, required >=2 s |
| Hover wall clearance | 10.39–15.28 mm, required 5–25 mm |
| Maximum horizontal hover speed | .008245 m/s, required <=.01 m/s |
| Hover altitude | .09999976–.10001321 m, required .085–.115 m |
| Minimum separation gain | .214862 m, required >=.02 m |
| Maximum downstream rate | 215 Hz, ceiling 240 Hz |
| Motion before edge hover | Zero cumulative inward retreat; every fleeing target .12 m/s |
| Engineering checks | 38 focused, 239 broader targeted, 922 final full; zero skips |
| Original source/experiment preservation | 3,010 protected files unchanged |

Both independent saved-only audits pass at zero added neural steps. Four
predetermined exact replays match every interval field: development rest-0
intact/silent, development cruise-0 intact, robustness cruise-180 intact. They
consume their separate 96,000-step verification budget, not new scientific cases.

The native viewer API review separately consumes 24,000 engineering steps. Its
real session is turning at 1 s, fleeing at 3 s and in edge hover at 6 and 12 s;
final position approximately (.01500009, .30000003, .10000002) m. Actual neural
state, frame publication and viewer spawning are exercised. Saved native frames
and movement plots were visually reviewed. Mouse/keyboard automation is
unavailable because the native helper pipe is missing. Some labels disappear in
saved native captures; this capture/UI issue remains separate from verified
motion. The responding viewer window is left open/paused.

## Records and scientific limits

Full config/case seeds, sources, environment, budgets, JSON/CSV results, actions,
traces and immutable receipts are under
`experiments/escape-edge-2026-10-11/candidate-01/{development,robustness}/`.
Audits, exact replay, numerical summary, plots and delivery are under
`artifacts/escape-edge-2026-10-11/`. The recorded environment is CPU/one thread,
PyTorch 2.14.0+cpu, NumPy 2.5.3, Panda3D 1.10.16, Pygame 2.6.1; exact Python/OS
are in each config. No Git repository, checkpoint, optimizer update or training.

These are static-predator cases at eight historical angles and three calibrated
start states. The second batch varies seeds, not geometry; neither geometry nor
the inspected final cases are unseen for future experiments. The original 94/96
directional result remains failed under its original gate, including seeds
1100018 and 1100021. The older 19/24 weak-recovery experiment remains failed.
This user-selected edge-hover contract is a new behavior, not a relabeling of
those results. The source graph, neuron/gain/encoding/readout rules and physics
are unchanged. Spikes select a side; IMU flight and edge hover are engineered.

## Run

`.\.venv\Scripts\python.exe -m flybrain.escape_edge`

The launcher verifies source-bound delivery evidence and never runs an experiment.
R starts fresh; Spawn predator triggers an encounter; Space pauses; Backspace
replays; WASD/QE moves the predator. Current review is paused at the edge.

Next task / Sol High: diagnose intermittent capture label loss and complete
live input review when the native helper is available. No automatic neural retry.
