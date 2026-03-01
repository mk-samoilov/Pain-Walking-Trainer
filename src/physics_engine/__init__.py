"""Ragdoll physics simulation built on pymunk (Chipmunk2D)."""

import math

import numpy as np
import pymunk


JOINT_ORDER: list[str] = [
    "neck",
    "shoulder_L", "shoulder_R",
    "elbow_L",    "elbow_R",
    "hip_L",      "hip_R",
    "knee_L",     "knee_R",
]

N_JOINTS = len(JOINT_ORDER)

# NN input/output sizes — keep in sync with App constants.
NN_INPUTS = 42   # 26 original + 4 foot positions + 1 head y + 2 accel + 9 prev motors
NN_OUTPUTS = 18  # 9 motor speeds + 9 torque multipliers


class RagdollAgent:
    """Single ragdoll agent with its own isolated pymunk space."""

    SPAWN_HEIGHT: float = 1.2

    def __init__(self, model_cfg: dict, physics_cfg: dict, sim_cfg: dict) -> None:
        """Create space, ground plane and ragdoll from config dicts."""

        self._model_cfg = model_cfg
        self._physics_cfg = physics_cfg
        self._sim_cfg = sim_cfg

        self.space = pymunk.Space()
        self.space.gravity = tuple(physics_cfg["gravity"])

        self.parts: dict[str, pymunk.Body] = {}
        self._joints: dict[str, tuple[pymunk.SimpleMotor, pymunk.RotaryLimitJoint]] = {}
        self._ground_contacts: set[str] = set()
        self._fatal_parts: frozenset[str] = frozenset(sim_cfg.get("fatal_ground_parts", []))

        self._create_ground()
        self._create_ragdoll()

        self.alive = True
        self.age = 0.0
        self._initial_x = float(self.parts["body"].position.x)
        self._height_sum = 0.0
        self._upright_sum = 0.0
        self._height_n = 0

        # State needed for extended inputs.
        body = self.parts["body"]
        self._prev_vel = (float(body.velocity.x), float(body.velocity.y))
        self._accel = (0.0, 0.0)
        self._prev_motor_outputs = np.zeros(N_JOINTS, dtype=np.float32)

    def _create_ground(self) -> None:
        """Add a static ground segment to the space."""

        cfg = self._physics_cfg
        seg = pymunk.Segment(self.space.static_body, (-500, 0), (500, 0), 0)
        seg.friction = cfg["ground_friction"]
        seg.elasticity = cfg["ground_restitution"]
        self.space.add(seg)

    def _create_ragdoll(self) -> None:
        """Instantiate all body parts and joints from model config."""

        for name, part in self._model_cfg["parts"].items():
            ox, oy = part["offset"]
            w, h = part["width"], part["height"]
            mass = part["mass"]

            if part.get("collision_shape") == "circle":
                r = w / 2
                moment = pymunk.moment_for_circle(mass, 0, r)
                body = pymunk.Body(mass, moment)
                shape = pymunk.Circle(body, r)
            else:
                moment = pymunk.moment_for_box(mass, (w, h))
                body = pymunk.Body(mass, moment)
                shape = pymunk.Poly.create_box(body, (w, h))

            body.position = (ox, oy + self.SPAWN_HEIGHT)
            shape.friction = 0.8 if name.startswith("foot") else 0.2
            shape.elasticity = 0.05
            shape.filter = pymunk.ShapeFilter(group=1)  # no self-collisions

            self.parts[name] = body
            self.space.add(body, shape)

        for jname, jcfg in self._model_cfg["joints"].items():
            parent = self.parts[jcfg["parent"]]
            child = self.parts[jcfg["child"]]
            ap, ac = jcfg["anchor_on_parent"], jcfg["anchor_on_child"]

            pivot = pymunk.PivotJoint(parent, child, (ap[0], ap[1]), (ac[0], ac[1]))
            pivot.collide_bodies = False

            limit = pymunk.RotaryLimitJoint(parent, child, jcfg["angle_min"], jcfg["angle_max"])
            limit.collide_bodies = False

            motor = pymunk.SimpleMotor(parent, child, rate=0.0)
            motor.max_force = jcfg["motor_max_torque"]
            motor.collide_bodies = False

            self._joints[jname] = (motor, limit)
            self.space.add(pivot, limit, motor)

    def _update_ground_contacts(self) -> None:
        """Detect ground contact via world-space bounding box (pymunk 7 compatible)."""

        self._ground_contacts.clear()
        for name, body in self.parts.items():
            for shape in body.shapes:
                if shape.bb.bottom <= 0.02:
                    self._ground_contacts.add(name)
                    break

    def step(self) -> None:
        """Advance physics one timestep and evaluate death conditions."""

        body = self.parts["body"]
        vx, vy = float(body.velocity.x), float(body.velocity.y)
        self._accel = (vx - self._prev_vel[0], vy - self._prev_vel[1])
        self._prev_vel = (vx, vy)

        dt = self._physics_cfg["time_step"]
        self.space.step(dt)
        self._update_ground_contacts()
        self.age += dt

        self._height_sum += float(self.parts["body"].position.y)
        self._upright_sum += math.cos(float(self.parts["body"].angle))
        self._height_n += 1

        # Death from model.json fatal_ground_contact flags.
        fatal_from_model = {
            n for n, p in self._model_cfg["parts"].items()
            if p.get("fatal_ground_contact")
        }
        if self._ground_contacts & fatal_from_model:
            self.alive = False

        # Death when configured parts touch ground.
        if self._ground_contacts & self._fatal_parts:
            self.alive = False

    def get_nn_inputs(self) -> np.ndarray:
        """Build the 42-element input vector for the neural network."""

        body = self.parts["body"]
        jcfg = self._model_cfg["joints"]
        foot_L = self.parts["foot_L"]
        foot_R = self.parts["foot_R"]
        head = self.parts["head"]

        inputs: list[float] = [
            # Body state (5)
            body.angle,
            body.angular_velocity,
            body.velocity.x,
            body.velocity.y,
            body.position.y,
        ]

        # Joint angles relative to parent (9)
        for jname in JOINT_ORDER:
            p = self.parts[jcfg[jname]["parent"]]
            c = self.parts[jcfg[jname]["child"]]
            inputs.append(c.angle - p.angle)

        # Joint angular velocities relative to parent (9)
        for jname in JOINT_ORDER:
            p = self.parts[jcfg[jname]["parent"]]
            c = self.parts[jcfg[jname]["child"]]
            inputs.append(c.angular_velocity - p.angular_velocity)

        # Ground contacts (2)
        inputs.append(1.0 if "foot_L" in self._ground_contacts else 0.0)
        inputs.append(1.0 if "foot_R" in self._ground_contacts else 0.0)

        # Body x position (1)
        inputs.append(body.position.x)

        # Foot positions relative to body (4)
        inputs.append(foot_L.position.x - body.position.x)
        inputs.append(foot_L.position.y - body.position.y)
        inputs.append(foot_R.position.x - body.position.x)
        inputs.append(foot_R.position.y - body.position.y)

        # Head height (1)
        inputs.append(head.position.y)

        # Body acceleration from previous step (2)
        inputs.append(self._accel[0])
        inputs.append(self._accel[1])

        # Previous motor speed outputs — proprioception (9)
        inputs.extend(self._prev_motor_outputs.tolist())

        return np.array(inputs, dtype=np.float32)

    def apply_nn_outputs(self, outputs: np.ndarray) -> None:
        """Set motor speeds and torque limits from the 18-element network output."""

        jcfg = self._model_cfg["joints"]
        for i, jname in enumerate(JOINT_ORDER):
            motor, _ = self._joints[jname]
            motor.rate = float(outputs[i]) * jcfg[jname]["motor_speed"]
            # Second half: torque multiplier, tanh → [0, 1].
            torque_mult = (float(outputs[N_JOINTS + i]) + 1.0) * 0.5
            motor.max_force = torque_mult * jcfg[jname]["motor_max_torque"]

        self._prev_motor_outputs = outputs[:N_JOINTS].astype(np.float32)

    def compute_fitness(self, fitness_cfg: dict) -> float:
        """Return a weighted fitness score based on distance, speed, height, uprightness and age."""

        body = self.parts["body"]
        dist_x = float(body.position.x) - self._initial_x
        age = max(self.age, 1e-6)
        avg_speed = dist_x / age
        avg_h = (self._height_sum / self._height_n) if self._height_n else 0.0
        h_bonus = max(0.0, avg_h - 0.5)
        avg_upright = (self._upright_sum / self._height_n) if self._height_n else 0.0

        score = (
            fitness_cfg["weight_distance"] * dist_x
            + fitness_cfg["weight_speed"]    * avg_speed
            + fitness_cfg["weight_height"]   * h_bonus
            + fitness_cfg["weight_survival"] * self.age
            + fitness_cfg["weight_upright"]  * avg_upright
        )
        return max(0.0, float(score))

    def get_render_state(self) -> list[dict]:
        """Return position, angle and mirror flag for every body part."""

        return [
            {
                "name":     name,
                "position": (float(body.position.x), float(body.position.y)),
                "angle":    float(body.angle),
                "mirror":   self._model_cfg["parts"][name].get("mirror", False),
            }
            for name, body in self.parts.items()
        ]
