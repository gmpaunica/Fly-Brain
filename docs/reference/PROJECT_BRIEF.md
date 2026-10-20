# FlyBrain Project — Codex Project Brief

## 1. Project Summary

Build an interactive simulation of a **digital fruit fly embodied in a 2D world**, controlled by a **PyTorch-based spiking neural network** whose architecture is inspired by, and later partially derived from, the real *Drosophila melanogaster* connectome.

The project has two equally important visual components:

1. **World view** — a 2D arena containing the fly, odor sources, food, hazards, predators, obstacles, etc.
2. **Brain view** — a live neural visualization showing neurons/regions lighting up as spikes propagate through the network.

The long-term research question is:

> Can a neural network constrained by real fruit-fly connectivity learn useful embodied behavior such as odor navigation, danger avoidance, and stimulus-response associations?

The project must be useful as a **CV / portfolio project**, but its primary purpose is also to teach:
- PyTorch
- neural-network fundamentals
- spiking neural networks
- training loops
- differentiable parameters
- reinforcement / reward-based learning
- simulation
- basic computational neuroscience
- experiment design
- evaluation and visualization

---

# 2. Core Design Principle

The world may be 2D, but the fly must **not receive privileged world coordinates** as sensory input.

The fly should only receive simulated sensory signals.

Examples:

- left antenna odor concentration
- right antenna odor concentration
- simple retinal / angular visual input
- touch / collision input
- internal reward signal

The brain/network must convert those sensory signals into motor activity.

Motor outputs should then control:
- turn left
- turn right
- move forward
- later: stop, escape, feed, etc.

Closed-loop architecture:

```text
2D WORLD
   |
   +--> left antenna
   +--> right antenna
   +--> simple visual field
   +--> touch / collision
             |
             v
      SENSORY ENCODERS
             |
             v
      PYTORCH SNN
  connectome-constrained
             |
      +------+------+
      |             |
      v             v
BRAIN VISUALIZER   MOTOR OUTPUT
                    |
              +-----+-----+
              |     |     |
            LEFT FORWARD RIGHT
                    |
                    v
                 FLY BODY
                    |
                    +----> changes WORLD
```

---

# 3. Important Scientific Distinction

A connectome is primarily a **wiring diagram**, not a complete functioning brain.

It tells us information such as:
- which neurons connect
- direction of connection
- approximate synapse count / connectivity strength
- neuron identity/type
- anatomical location, depending on dataset

It does NOT fully specify:
- exact synaptic strength
- exact firing thresholds
- all neuron-specific electrical properties
- plasticity rules
- neuromodulation
- spontaneous activity
- the complete mapping from sensory input to behavior

Therefore this project must distinguish clearly between:

### Biological structure
What comes from published fly connectivity data.

### Assumed neural dynamics
Leaky integrate-and-fire or another simplified neuron model.

### Trainable parameters
Unknown values learned with PyTorch.

### Hand-written scaffolding
Sensory conversion, motor decoding, environment physics, etc.

Never imply that the simulation is a complete biological reconstruction of a real fly brain.

---

# 4. PyTorch's Role

PyTorch should be central to the project.

The neural state should be represented using tensors.

Example conceptual state:

```python
V = torch.zeros(num_neurons)
spikes = torch.zeros(num_neurons)
```

Connections should eventually be represented efficiently, preferably as a sparse matrix when the network becomes large.

Conceptual update:

```python
incoming = W @ spikes.float()

V = decay * V + incoming + external_input

new_spikes = V >= threshold

V[new_spikes] = reset_voltage
```

This is only illustrative. Use a proper implementation with clear state management and tests.

PyTorch should eventually be used for:
- tensors
- network state
- sparse operations
- GPU acceleration where useful
- trainable parameters via `torch.nn.Parameter`
- optimizers
- autograd where appropriate
- checkpointing
- experiment reproducibility

---

# 5. Machine Learning Component

Simply simulating the connectome is NOT enough.

The ML component should investigate whether selected neural parameters can be learned while keeping the biological topology constrained.

Preferred setup:

```text
fixed biological / connectome-inspired topology
                    +
       selected trainable parameters
                    |
                    v
              learning signal
                    |
                    v
            improved behaviour
```

Examples of trainable quantities:
- selected synaptic gains
- selected pathway weights
- neuron thresholds
- sensory gain parameters
- motor readout parameters

Do NOT make every aspect trainable initially.

The project should eventually compare:

1. **Random network**
2. **Fixed connectome-inspired network**
3. **Connectome-inspired network with learned parameters**

Potential learning approaches:
- supervised optimization for simple early experiments
- reward-based optimization
- reinforcement learning
- biologically inspired plasticity
- reward-modulated STDP

Start simple.

---

# 6. Version 1 Scope

Do NOT begin with the full ~100k+ neuron fly connectome.

Version 1 should use a small toy network, approximately:
- 50–200 neurons initially
- possibly scaling to 1,000+ once stable

The goal of V1 is to prove the complete loop:

```text
odor source
   ->
antenna sensing
   ->
sensory neurons spike
   ->
activity propagates through PyTorch network
   ->
motor neurons spike
   ->
fly turns / moves
   ->
new sensory readings
```

The brain visualization must update live.

---

# 7. V1 World

Create a simple 2D top-down arena.

Minimum entities:

### Fly
State:
- x
- y
- heading
- forward speed
- angular speed

### Food / odor source
State:
- x
- y
- odor identity
- odor strength

### Environment
- bounded arena
- simple collision handling
- deterministic seed support

The fly should not receive the food's coordinates directly.

---

# 8. Odor Model

The first sensory modality should be smell.

The fly has two antenna sample points:
- left antenna
- right antenna

Each antenna measures odor concentration independently.

Simple initial concentration model:

```text
concentration ~= source_strength / (distance^2 + constant)
```

A more realistic diffusion model can be added later.

The left/right difference gives the neural system directional information without giving it coordinates.

Example:

```text
left antenna  = 0.41
right antenna = 0.67
```

These values should be converted into sensory neural activity.

Possible initial encoding:
- rate coding
- Poisson spike generation

Example idea:

```text
odor concentration -> firing rate -> spikes
```

---

# 9. Brain / Neural Network V1

Initial toy topology can contain:

```text
LEFT OLFACTORY SENSORS
          \
           -> INTERNEURONS -> MOTOR LEFT
          /
RIGHT OLFACTORY SENSORS
          \
           -> INTERNEURONS -> MOTOR RIGHT

additional neurons -> FORWARD DRIVE
```

The implementation should support:
- neuron groups / labels
- connection matrix
- excitatory/inhibitory signs if included
- spike state
- membrane potential
- reset
- refractory period if useful
- external sensory current

Use clear abstractions so the toy graph can later be replaced with real connectome-derived data.

---

# 10. Brain Visualization — Major "Wow Factor"

This is a core feature, not an afterthought.

Main app layout should eventually resemble:

```text
+----------------------------+----------------------------+
|                            |                            |
|       WORLD VIEW           |       BRAIN VIEW           |
|                            |                            |
|        FOOD                |      o---o---o             |
|         ~~~                |       \ / \                |
|      ~~~~~                 |        o   o               |
|              FLY ->        |       / \ /                |
|                            |      o---o                  |
|                            |                            |
+----------------------------+----------------------------+
| Left antenna: 0.41         | Active neurons: 47         |
| Right antenna: 0.67        | Spikes/sec: ...            |
| Reward: ...                | Motor: TURN RIGHT          |
+----------------------------+----------------------------+
```

Requirements:
- neurons light up when they spike
- recently active neurons fade gradually
- connection lines may be shown selectively
- sensory, interneuron, and motor groups should be visually distinguishable
- motor output should be visible
- avoid rendering every edge once the graph gets large

Later, if real neuron coordinates are available, brain layout should approximate biological anatomical positions.

---

# 11. Predator / Vision Later

The environment may remain top-down 2D.

The fly should receive a simulated angular visual field rather than the top-down image.

Example visual representation:

```text
left                              right

....##.......................
    ^
 predator occupies this region
```

As a predator gets closer, its angular size increases.

Conceptual angular size:

```text
theta = 2 * atan(object_radius / distance)
```

This allows a looming visual stimulus in a 2D world.

Later this can drive looming-sensitive neural populations.

Do NOT implement predator vision in V1.

---

# 12. Future Stimuli / Experiments

Once odor navigation works, add experiments incrementally.

Possible experiments:

## Odor attraction
Train or test navigation toward food odor.

## Odor avoidance
Associate another odor with a negative reward.

## Novel source positions
Train in one set of layouts and test in unseen layouts.

## Predator / looming
Test escape responses.

## Neural ablation
Disable selected neurons or pathways and measure behavioral effect.

Example:

```text
Normal network success: 82%
After olfactory pathway ablation: 18%
```

## Weight perturbation
Increase/decrease selected pathway gains.

## Conflicting stimuli
Food odor + predator.

## Multiple odors
Attractive odor vs aversive odor.

---

# 13. Real Connectome Integration — Later Phase

Only attempt after the toy network works.

Potential sources:
- published Drosophila connectome datasets
- FlyWire / neuPrint-derived connectivity
- publicly available neuron annotations

Desired data pipeline:

```text
connectome dataset
      |
      v
filter relevant neurons
      |
      v
map neuron IDs to contiguous indices
      |
      v
build sparse adjacency / weight matrix
      |
      v
map sensory populations
      |
      v
map output / descending / motor populations
```

Start with a biologically relevant SUBSET.

Do not immediately load the entire fly CNS.

Suggested first real subset:
- olfactory receptor neurons
- projection neurons
- selected interneuron populations
- selected descending/motor-related populations

Keep provenance of all biological data.

---

# 14. Training / Evaluation

Training and testing must be separate.

Use deterministic seeds where possible.

Track metrics such as:
- success rate
- time to food
- distance traveled
- path efficiency
- cumulative reward
- collision count
- neural spike count
- energy / activity proxy
- success under ablation

Always evaluate on unseen starting positions.

Possible later evaluation:

```text
Fixed topology, no learning:
success = 31 / 100

Trainable connectome-inspired network:
success = 82 / 100

Ablated trained network:
success = 19 / 100
```

Do not present made-up metrics in the final project. All numbers must come from recorded experiments.

---

# 15. Recommended Technology Stack

Prefer a simple Python-first stack.

Core:
- Python 3.11+
- PyTorch
- NumPy
- Pygame initially

Useful later:
- Matplotlib for experiment plots
- pandas for results
- networkx only for offline graph analysis if needed
- PyTorch sparse tensors for larger neural graphs

Avoid adding a heavy game engine unless the Python implementation becomes a real limitation.

The learning objective is PyTorch/ML, not game-engine development.

---

# 16. Suggested Repository Structure

```text
flybrain/
|
|-- AGENTS.md
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|
|-- src/
|   |-- flybrain/
|       |
|       |-- world/
|       |   |-- arena.py
|       |   |-- fly.py
|       |   |-- odor.py
|       |   `-- entities.py
|       |
|       |-- neural/
|       |   |-- lif.py
|       |   |-- network.py
|       |   |-- sensory.py
|       |   |-- motor.py
|       |   `-- plasticity.py
|       |
|       |-- visualization/
|       |   |-- world_view.py
|       |   |-- brain_view.py
|       |   `-- hud.py
|       |
|       |-- training/
|       |   |-- trainer.py
|       |   |-- rewards.py
|       |   `-- evaluation.py
|       |
|       |-- data/
|       |   `-- connectome.py
|       |
|       `-- app.py
|
|-- tests/
|   |-- test_odor.py
|   |-- test_neurons.py
|   |-- test_network.py
|   `-- test_motor.py
|
|-- experiments/
|   |-- configs/
|   `-- results/
|
`-- docs/
    |-- architecture.md
    `-- experiments.md
```

This structure can evolve, but keep world, neural logic, visualization, training, and biological data handling separate.

---

# 17. Milestones

## Milestone 0 — Skeleton

Goal:
- repository structure
- Python environment
- basic tests
- basic executable app

Acceptance:
- `python -m flybrain.app` runs
- blank/simple arena displays
- tests run successfully

---

## Milestone 1 — Fly in 2D World

Goal:
- fly moves in arena
- heading visible
- controls can temporarily be manual for debugging

Acceptance:
- fly position/heading update correctly
- bounds handled
- deterministic stepping possible

---

## Milestone 2 — Odor and Two Antennae

Goal:
- food emits odor
- left/right antenna readings differ spatially

Acceptance:
- concentration decreases with distance
- each antenna samples independently
- sensor values visible on HUD

---

## Milestone 3 — PyTorch Spiking Network

Goal:
- implement small leaky integrate-and-fire network
- sensory activity drives spikes

Acceptance:
- network runs entirely from PyTorch tensors
- unit tests verify leak, threshold, reset, and propagation
- no world coordinates are given directly to the network

---

## Milestone 4 — Neural Motor Control

Goal:
- network motor populations control the fly

Acceptance:
- left/right network output changes heading
- forward activity changes movement
- fully closed-loop simulation works

This is the first major project milestone.

---

## Milestone 5 — Live Brain Visualizer

Goal:
- display network alongside world

Acceptance:
- spiking neurons visibly light up
- neural activity fades over time
- motor population activity visible
- world and neural simulation remain synchronized

This is the major demo / wow-factor milestone.

---

## Milestone 6 — Learning

Goal:
- selected parameters become trainable
- behavior improves with training

Acceptance:
- explicit train / evaluation modes
- saved checkpoints
- learning curve
- evaluation on unseen starts
- measurable performance difference before vs after training

---

## Milestone 7 — Experiments

Goal:
- build reproducible scientific experiments

Examples:
- food navigation
- odor preference
- ablation
- parameter perturbation

Acceptance:
- experiment configs stored
- result files stored
- graphs generated from recorded results

---

## Milestone 8 — Real Connectome Subset

Goal:
- replace toy topology with a real Drosophila-derived subset

Acceptance:
- biological source documented
- neuron IDs preserved in metadata
- connectivity transformation reproducible
- subset drives the same embodied simulation interface

---

# 18. Coding Guidelines for Codex

When modifying this project:

1. Prefer small, testable steps.
2. Do not jump directly to the full connectome.
3. Keep the simulation deterministic when a seed is provided.
4. Keep neural logic independent from rendering.
5. Keep training logic independent from rendering.
6. Avoid hidden "cheats" such as feeding target coordinates directly to the neural network.
7. Clearly label hand-written scaffolding versus learned or connectome-derived behavior.
8. Add tests for neural dynamics before increasing network complexity.
9. Avoid premature optimization.
10. Prefer readable educational code over clever abstractions.
11. Use type hints.
12. Add docstrings to non-trivial public functions/classes.
13. Do not silently change the scientific assumptions of the project.
14. Do not claim biological realism beyond what is actually implemented.
15. Keep dependencies minimal unless a new dependency solves a concrete problem.

---

# 19. Things NOT To Do Yet

Do NOT initially:
- build a full 3D environment
- simulate all 176k+ neurons
- use Unity/Unreal
- implement a full compound-eye optical simulator
- simulate fluid dynamics for odor
- build realistic fly biomechanics
- train a huge deep RL policy
- add unnecessary cloud infrastructure
- optimize for distributed training
- over-engineer the UI
- claim that connectome wiring alone recreates cognition

Focus first on completing the sensory -> neural -> motor -> world feedback loop.

---

# 20. Definition of the First Successful Prototype

The first prototype is successful when:

1. A food odor source exists in a 2D arena.
2. A fly has a position and heading.
3. Two antennae independently measure odor concentration.
4. Those measurements stimulate sensory neurons.
5. A PyTorch spiking neural network processes the signal.
6. Motor neuron activity controls left/right/forward movement.
7. The fly's movement changes future sensory inputs.
8. A second panel shows the neurons firing in real time.

At this stage the fly does NOT need to reliably find food.

The purpose is proving the entire embodied neural loop.

---

# 21. Definition of the First ML Prototype

The first true ML milestone is successful when:

1. At least some neural parameters are trainable.
2. Training uses an explicit objective/reward.
3. PyTorch optimization changes those parameters.
4. Training performance improves measurably.
5. Evaluation is performed on unseen starting positions.
6. The trained model can be saved and reloaded.
7. A baseline exists for comparison.

Example question:

> Can a connectome-inspired network with trainable synaptic gains learn odor-guided navigation better than the same fixed network?

---

# 22. Long-Term Portfolio Goal

The final demo should make the project understandable within seconds:

```text
WORLD                         LIVE BRAIN

food odor                     sensory neurons fire
   |                                   |
   v                                   v
digital fly  <------------- motor neurons activate
   |
   v
changes position
   |
   +-----------------------> new sensory input
```

The viewer should be able to:
- see the fly react
- see neural activity propagate
- toggle experiments
- compare trained/untrained behavior
- disable neural populations
- see quantitative results

The final project should support a concise portfolio statement such as:

> Built an embodied PyTorch spiking neural network constrained by Drosophila connectome structure, with simulated olfactory/visual sensory input, live neural activity visualization, learned synaptic parameters, and reproducible behavioral/ablation experiments.

Only use wording like this once the corresponding features actually exist.

---

# 23. Immediate Next Task

Start with **Milestone 0 and Milestone 1 only**.

Create:
- project structure
- environment configuration
- simple Pygame window
- 2D arena
- fly object with x/y/heading
- deterministic update loop
- keyboard controls only for temporary debugging
- basic tests for fly movement

Do not implement PyTorch neural logic until the world/fly loop is clean and testable.

After Milestone 1 is stable, proceed to odor sensing.

# 24. Persistent Project Logging — REQUIRED

Codex must maintain a persistent project log at:

```text
PROJECT_LOG.md
```

This file is the project's running memory and must be updated after every meaningful implementation session or milestone.

## Purpose

The log exists so that:
- a future Codex session can immediately understand the current state
- previous decisions are not repeatedly reconsidered
- bugs, failed approaches, and fixes are preserved
- experiments remain reproducible
- the next task is always obvious
- the user can see actual project progress

## Required Log Entry Format

Append a new dated entry after meaningful work.

Use this format:

```markdown
## YYYY-MM-DD — Short Session Title

### Goal
What this session attempted to accomplish.

### Changes Made
- files created
- files modified
- important implementation details

### Decisions
- architectural or scientific decisions made
- assumptions introduced
- why an approach was chosen

### Tests / Verification
- commands run
- tests passed/failed
- manual checks performed
- relevant measurements

### Problems / Failed Attempts
- errors encountered
- approaches that did not work
- known limitations

### Current State
A concise description of what currently works.

### Next Step
The single most important next task.

### Notes
Any useful context for the next session.
```

Do not write vague entries such as "worked on neural network."

Log concrete facts.

Example:

```markdown
### Changes Made
- Added `Fly` class with position, heading and velocity.
- Added deterministic `step(dt)` movement.
- Added arena boundary clamping.
- Added three movement tests.

### Tests / Verification
- `pytest tests/test_fly.py`
- 3/3 tests passed.
- Manually confirmed keyboard steering in Pygame.

### Current State
The fly can move and rotate deterministically inside the arena.

### Next Step
Implement left/right antenna positions and odor concentration sampling.
```

---

# 25. Project Status File — REQUIRED

Maintain:

```text
PROJECT_STATUS.md
```

This is different from the chronological log.

`PROJECT_STATUS.md` should always describe the CURRENT project state and be short enough to read in under two minutes.

Keep these sections:

```markdown
# FlyBrain Current Status

## Current Milestone
Milestone X — ...

## What Works
- ...

## In Progress
- ...

## Known Issues
- ...

## Important Decisions
- ...

## Next Task
...

## How To Run
...

## How To Test
...
```

Whenever a milestone meaningfully changes, update this file.

A future coding agent should read, in this order:

1. `AGENTS.md`
2. `PROJECT_STATUS.md`
3. recent entries in `PROJECT_LOG.md`
4. relevant source files/tests

---

# 26. Experiment Logging — REQUIRED Once ML Work Begins

When experiments/training begin, do NOT rely only on `PROJECT_LOG.md`.

Each experiment must have machine-readable results stored under:

```text
experiments/results/
```

Prefer formats such as:
- JSON
- CSV

Every experiment should record relevant metadata such as:

```text
experiment name
timestamp
git commit if available
random seed
configuration
network size
training steps/episodes
learning rate
reward definition
evaluation conditions
metrics
checkpoint path
```

Human-readable conclusions should be added to:

```text
docs/experiments.md
```

Never manually invent result numbers in documentation.

Results must come from actual recorded runs.

---

# 27. Decision Log — Important Architecture Decisions

Important decisions should also be summarized in:

```text
docs/decisions.md
```

Examples:
- why Pygame was chosen
- why the first network uses 100 neurons
- why a particular learning method was chosen
- why a biological pathway was simplified
- why specific connectome data was filtered

Suggested format:

```markdown
## Decision: Use a 2D world for V1

### Context
...

### Decision
...

### Reason
...

### Consequences
...
```

Do not create an entry for trivial implementation choices.

---

# 28. Codex Session Rules

At the START of a new coding session:

1. Read `AGENTS.md`.
2. Read `PROJECT_STATUS.md` if it exists.
3. Read the latest relevant entries in `PROJECT_LOG.md`.
4. Inspect the code/tests relevant to the requested task.
5. Continue from the existing state rather than rebuilding working components.

At the END of meaningful work:

1. Run relevant tests.
2. Update `PROJECT_STATUS.md`.
3. Append to `PROJECT_LOG.md`.
4. Update experiment/decision documentation when applicable.
5. Clearly state unresolved issues.
6. Set one concrete `Next Step`.

Never claim something works unless it was actually tested or verified.

Never delete historical log entries merely because the implementation changed later.

