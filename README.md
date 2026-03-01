# WalkAI — 2D ragdoll Walking Trainer

A real-time 2D simulation that trains a ragdoll character to walk using a genetic algorithm and a sparse neural network. Watch ten stick figures flail, fall, and slowly, painfully learn to move forward — all without ever being told what "walking" means.

![Python 3.13+](https://img.shields.io/badge/Python-3.13%2B-blue)

---

## What it does

A population of ragdoll agents each run their own isolated physics simulation. Every agent is controlled by a feedforward neural network whose weights are evolved across generations using a genetic algorithm. No reinforcement learning, no gradients — just selection pressure, mutation, and a lot of falling over.

After enough generations the agents figure out something resembling locomotion. Sometimes it's elegant. Usually it isn't.

### Key features

- **Genetic algorithm** — tournament selection, uniform crossover, Gaussian weight mutation
- **Sparse neural network** — connections have masks that grow over generations but never get pruned; the network topology literally evolves
- **Explorer agent** — one slot per generation is always an aggressively mutated copy of the best genome, trying to break out of local optima
- **Death laser** — optional vertical wall of death that sweeps right at configurable speed, forcing agents to keep moving or die
- **42 inputs / 18 outputs** — body pose, joint velocities, foot contacts, head height, proprioception (previous motor commands), body acceleration
- **Real-time ImGui UI** — tweak every parameter while the simulation runs; no restart needed
- **Neural network visualizer** — live per-neuron activation colors and active-connection overlay
- **Save / Load** — serialize the entire population to `data/save.npz` and resume later

---

## Architecture

```
src/
  main.py                  — entry point
  core/app.py              — window, main loop, ImGui panels, population management
  physics_engine/__init__.py — RagdollAgent: pymunk spaces, joints, NN I/O, fitness
  engine/__init__.py       — OpenGL renderer: SVG textures, camera, laser, debug overlays
  ai/__init__.py           — NeuralNetwork (masked weights) + GeneticAlgorithm

assets/person/             — SVG body part sprites + model.json (skeleton definition)
assets/ui/                 — platform tile SVG
data/config.json           — all tuneable parameters
data/save.npz              — training checkpoint (auto-generated)
```

Each agent lives in its own `pymunk.Space`. All spaces step simultaneously each frame. The speed multiplier runs N physics steps per rendered frame — so ×200 is 200 physics steps per frame with one render.

### Ragdoll

10 body segments: `head`, `body`, `upper_arm_L/R`, `hand_L/R`, `upper_leg_L/R`, `foot_L/R`.
Connected by 9 motor joints: `neck`, `shoulder_L/R`, `elbow_L/R`, `hip_L/R`, `knee_L/R`.

Each joint = `PivotJoint` + `RotaryLimitJoint` + `SimpleMotor`. Self-collisions are suppressed via `ShapeFilter(group=1)`.

### Neural network

```
42 inputs → [32 → 32] hidden (tanh) → 18 outputs
```

Outputs: 9 motor speeds + 9 torque multipliers (scaled from tanh to [0, 1]).

Genome layout per layer: `[weights_flat | mask_flat | biases]`. Masks are binary (0 = dormant, 1 = active). Mutations can flip dormant connections to active; they never prune active ones.

### Fitness

```
fitness = 1.0 × distance_x
        + 1.4 × avg_speed
        + 0.47 × height_bonus      (body above 0.5 m)
        + 0.1  × survival_time
        + 0.5  × avg_upright       (cos of body tilt angle)
```

All weights are editable in `data/config.json`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

**Requirements:** `pyopengl pyopengl-accelerate imgui-bundle pymunk numpy cairosvg pillow`

Python 3.13 is required. The libraries were chosen specifically because they have 3.13 wheels (pybox2d and pyimgui do not).

---

## Configuration

### `data/config.json`

| Section      | Key                          | Description                                                 |
|--------------|------------------------------|-------------------------------------------------------------|
| `simulation` | `population_size`            | Agents per generation                                       |
| `simulation` | `simulation_time`            | Max seconds per generation                                  |
| `simulation` | `fatal_ground_parts`         | Body parts that kill the agent on ground contact            |
| `simulation` | `laser_enabled`              | Enable the death laser                                      |
| `simulation` | `laser_speed`                | Laser advance speed in m/s                                  |
| `ai`         | `hidden_layers`              | Hidden layer sizes, e.g. `[32, 32]`                         |
| `ai`         | `mutation_rate`              | Per-gene mutation probability                               |
| `ai`         | `connection_add_rate`        | Probability of activating a dormant connection per mutation |
| `ai`         | `explorer_mutation_strength` | Mutation scale for the explorer agent                       |
| `fitness`    | `weight_*`                   | Per-component fitness weights                               |

### `assets/person/model.json`

Defines the skeleton: per-segment dimensions, masses, collision shapes, and per-joint anchor points, angle limits, and motor torques. **Change joint limits here, not in code.**

---

## Controls (runtime ImGui panel)

| Control            | Effect                                        |
|--------------------|-----------------------------------------------|
| Pause              | Freeze simulation                             |
| Speed ×1–200       | Physics steps per rendered frame              |
| Show all agents    | Toggle ghost rendering for non-watched agents |
| Death laser        | Toggle + speed slider                         |
| New population     | Reset everything, randomize genomes           |
| Restart generation | Respawn same genomes                          |
| Save / Load        | Checkpoint to `data/save.npz`                 |
