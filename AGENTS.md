# FlyBrain agent instructions

Build a Python/Pygame 2D fly simulation, then a small PyTorch spiking brain and live brain view, then measured learning. This is an educational research/portfolio project. The complete original brief is preserved in [docs/reference/PROJECT_BRIEF.md](docs/reference/PROJECT_BRIEF.md); its scientific and product requirements still apply. Read relevant sections when needed, not the entire archive each session. Current operating instructions and task scope are here and in status.

## Start each session

1. Read `PROJECT_STATUS.md`.
2. Read the latest relevant dated entry in `PROJECT_LOG.md` (start with the last 90 lines; expand only if needed).
3. Open the active task linked from status and relevant source/tests. Use `rg` for focused discovery.
4. Inspect Git status if a repository exists. Preserve unrelated changes. Continue working components rather than rebuilding them.

## Scope and model use

- Preferred model: **gpt-5.6-sol**, explicitly set to **medium** for ordinary implementation, **high** for neural math, learning, scientific evaluation, or a difficult root cause. Task recommendations are in `docs/BUILD_QUEUE.md`.
- Model choice is a user/client setting. These instructions cannot switch the running model. Never claim to have changed it. See `SOL_WORKFLOW.md` for launch commands and prompts.
- Work with one agent by default. Do not launch subagents or use Astra unless the user explicitly requests it. Review the changed code locally against the acceptance criteria.
- One task per session by default. Finish its implementation, necessary checks, and handoff; do not drift into another milestone. A broader explicit user request overrides this default.
- A request to build the active task authorizes routine implementation choices and reversible fixes within that task. Do not repeatedly request design approval already given by the brief/task. Ask for genuinely missing decisions that materially change scope or scientific assumptions.
- After two unsuccessful fixes to the same failure, gather a minimal reproduction and state the observed cause or remaining uncertainty. Narrow the problem; recommend Sol High if needed. This is a debugging reset, not a reason to abandon authorized work.
- An implemented experiment is not an achieved milestone. Predeclare one
  hypothesis, budget and stop condition. A failed scientific gate closes that
  experiment; do not automatically append another FB-07 letter or switch the
  learning method. Record the evidence and return to the agreed recovery plan.
- Keep plans and updates short. Load only relevant skills/references, batch independent reads, summarize test output, and avoid repeated passing checks without a reason.

## Scientific invariants

- The controller receives simulated sensory signals only. Never pass it fly/food coordinates, heading, target bearing/distance, world objects, or a precomputed steering answer. Environment physics, rendering, and evaluation may use coordinates; keep those paths separate.
- Clearly distinguish hand-written scaffolding, assumed neuron dynamics, trainable parameters, and sourced biological structure. The initial graph is synthetic. Do not describe it as a real connectome or a reconstructed brain.
- Start with Milestones 0 and 1 only: world, fly, deterministic stepping, temporary manual controls. Then odor; then 50-200 neurons. No neural dependency until the world is verified.
- Keep world, neural computation, rendering, training, and data import separate. Rendering must never advance neural state, consume simulation RNG, or control simulation time.
- Use explicit time units, documented coordinate conventions, seeded local random generators, and episode reset of all state. Reproducibility claims apply to the recorded software/device environment.
- Before adding neural complexity, test leak, threshold, reset, and directed propagation. Before claiming learning, verify gradients/parameter changes, a fixed topology, save/reload, a baseline, and evaluation on unseen starts.
- Before another imitation run, check the teacher's embodied behavior and
  compare readout turn error with a train-only mean-command predictor. Reduced
  combined loss alone does not show learned steering. Compare trained and
  untrained versions of the same controller as well as the fixed baseline.
- A final split already inspected for an earlier experiment is a historical
  benchmark. A new task name or validation gate does not make it unseen again.
- Never invent results. Training and evaluation must be separate. Store experiment configs and JSON/CSV results under `experiments/`, including seed, software/device, commit if available, objective, steps, learning rate, network size, evaluation cases, metrics, and checkpoint. Summarize recorded results in `docs/experiments.md`.
- Defer predators/vision, full connectome, 3D, fluid dynamics, complex biomechanics, large RL systems, cloud infrastructure, and distributed training.

## Teaching alongside implementation

- Standing user priority (2026-10-12): finish usability/UI polish first, then
  teach the project comprehensively so the user can explain it in interviews.
  This applies to every future project session, not only neural milestones.
- Work toward understanding every distinct project script, its purpose,
  dependencies, inputs/outputs, algorithms, state and tests. Cover Python,
  Pygame (historical 2D), Panda3D (current 3D), PyTorch, tensor calculations,
  neural assumptions, training/autograd/optimizers, evaluation and limitations.
  Explain active code first, then historical experiments and supporting tools;
  identify archived copies without implying that they are separate algorithms.
- Use docs/LEARNING_ROADMAP.md as the teaching sequence. Teach in manageable
  sessions with concrete code traces, numerical examples and small exercises.
  Ask the user to explain or predict behavior; never mark a topic understood
  solely because an agent wrote an explanation or the tests passed.
- Describe the project honestly as AI-assisted. Help the user distinguish their
  decisions, AI contributions, what they can personally explain, sourced biology,
  modeling assumptions, engineered control and actually measured learning.
  Do not imply sole authorship or understanding that has not been demonstrated.
- The user wants to understand what PyTorch does and how it works as the project
  develops. Explain new neural concepts in plain language alongside the relevant
  code, in small increments, without turning routine work into approval gates.
- For neural milestones, trace a concrete calculation from sensory input through
  tensors to neural activity/output. Explain tensor shapes, matrix orientation,
  and state changes with a tiny numerical example before generalizing.
- Distinguish PyTorch's tensor calculations from our hand-written neuron rules
  and graph. Clearly separate forward simulation from autograd, gradients and
  optimizer updates; a network running in PyTorch does not establish learning.

## Implementation and completion

- Python 3.11+, minimal dependencies, type hints, clear public docstrings. Favor readable educational code and small testable steps.
- Use behavior tests for simulation/neural correctness; do not replace numerical checks with mocks. Verify bug fixes with a reproduction. UI smoke tests and visual checks serve different purposes; report each honestly.
- Follow contracts in `docs/architecture.md`; change a meaningful assumption only with an explicit decision record and matching tests.
- At task completion run targeted tests and the small full suite; broaden checks only when justified. Report exact commands and observed results, including skipped/blocked checks. Never equate a headless smoke run with a visual review.
- Update `PROJECT_STATUS.md`: Current Milestone, What Works, In Progress, Known Issues, Important Decisions, Next Task, How To Run, How To Test.
- Append a dated `PROJECT_LOG.md` entry with: Goal; Changes Made; Decisions; Tests / Verification; Problems / Failed Attempts; Current State; Next Step; Notes. Preserve all historical entries.
- Update `docs/decisions.md` for meaningful architecture/scientific choices. Update the task queue when a task is actually verified. Leave one concrete next task and its recommended effort.
- Final handoff: what changed, verification and limits, one learning takeaway, next task. Keep it concise. Do not automatically commit, push, publish, or begin long training runs without the appropriate user authorization.
