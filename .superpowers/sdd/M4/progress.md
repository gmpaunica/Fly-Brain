# SDD ledger — plan: docs/plans/M4.md

Execution: inline, one agent. The project has no Git repository, so the
plan-authorized local review checkpoints replace worktree, BASE, commit, and
review-package operations. Cost if wrong: there is no atomic Git history for
task-by-task rollback.

Pre-flight: Task 1 produces `MotorDecoder.decode/reset`; Task 2 consumes the
same interface. No conflict.

Pre-flight: Task 2 produces `ClosedLoopSimulation.step/advance/reset`; Task 3
consumes the same interface after an explicit lazy neural-mode import. No
conflict.

Pre-flight: Task 2 exposes its contained `FixedStepSimulation` as `world`; Task
3 passes that world object to the existing renderer. No conflict.

Task 2: Ruling: expose `NeuralController.set_motor_enabled(bool)` and gate the
decoded command after the real network and decoder advance — this proves motor
silencing/restoration without a mock while retaining causal neural state — cost
if wrong: the test-only diagnostic seam becomes an unnecessary public API.

Task 1: complete (local checkpoint; tests:
`.\.venv\Scripts\python.exe -m pytest tests/test_motor.py -q -p no:cacheprovider`
→ 15 passed; full suite → 156 passed, one existing upstream Pygame warning).

Task 2: complete (local checkpoint; tests:
`.\.venv\Scripts\python.exe -m pytest tests/test_motor.py tests/test_closed_loop.py tests/test_simulation.py tests/test_neural_network.py -q -p no:cacheprovider`
→ 63 passed; full suite → 171 passed, one existing upstream Pygame warning).

Task 3: complete (local checkpoint; focused integration/app suite → 43 passed;
fresh full suite → 176 passed in 18.45 s with one existing upstream Pygame
warning; exact 120-tick manual and neural smokes exited 0; `pip check` reported
no broken requirements).

Final review: self-review (project instructions prohibit review subagents and no
Git review range exists). No Critical, Important, or Minor findings. Acceptance
mapping checked: motor sign/range/EMA/full counts; forward movement; output
silencing/restoration; sensory-only boundary; no oracle path; exact 20 substeps;
pre-movement observation; frame partitions; reset/replay; changed future sensory
input; lazy Torch import; explicit manual/neural modes; read-only dual-mode draw.

Branch finish: no Git repository exists, confirmed from inside the project
folder. No merge, push, commit, repository initialization or cleanup performed.
