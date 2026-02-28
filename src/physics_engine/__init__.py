import numpy as np
import pymunk

JOINT_ORDER = [
    "neck",
    "shoulder_L", "shoulder_R",
    "elbow_L",    "elbow_R",
    "hip_L",      "hip_R",
    "knee_L",     "knee_R",
]


class RagdollAgent:
    SPAWN_HEIGHT = 1.2

    def __init__(self, model_cfg: dict, physics_cfg: dict, sim_cfg: dict) -> None:
        self._model_cfg   = model_cfg
        self._physics_cfg = physics_cfg
        self._sim_cfg     = sim_cfg

        self.space = pymunk.Space()
        self.space.gravity = tuple(physics_cfg["gravity"])

        self.parts:  dict[str, pymunk.Body]  = {}
        self._joints: dict[str, tuple[pymunk.SimpleMotor, pymunk.RotaryLimitJoint]] = {}
        self._ground_contacts: set[str] = set()

        self._create_ground()
        self._create_ragdoll()

        self.alive = True
        self.age   = 0.0
        self._initial_x  = float(self.parts["body"].position.x)
        self._height_sum = 0.0
        self._height_n   = 0

    def _create_ground(self) -> None:
        cfg = self._physics_cfg
        seg = pymunk.Segment(self.space.static_body, (-500, 0), (500, 0), 0)
        seg.friction   = cfg["ground_friction"]
        seg.elasticity = cfg["ground_restitution"]
        self.space.add(seg)

    def _create_ragdoll(self) -> None:
        for name, p in self._model_cfg["parts"].items():
            ox, oy = p["offset"]
            w, h   = p["width"], p["height"]
            mass   = p["mass"]

            if p.get("collision_shape") == "circle":
                r      = w / 2
                moment = pymunk.moment_for_circle(mass, 0, r)
                body   = pymunk.Body(mass, moment)
                shape  = pymunk.Circle(body, r)
            else:
                moment = pymunk.moment_for_box(mass, (w, h))
                body   = pymunk.Body(mass, moment)
                shape  = pymunk.Poly.create_box(body, (w, h))

            body.position  = (ox, oy + self.SPAWN_HEIGHT)
            shape.friction = 0.8 if name.startswith("foot") else 0.2
            shape.elasticity = 0.05
            shape.filter = pymunk.ShapeFilter(group=1)  # no self-collisions

            self.parts[name] = body
            self.space.add(body, shape)

        for jname, jc in self._model_cfg["joints"].items():
            parent = self.parts[jc["parent"]]
            child  = self.parts[jc["child"]]
            ap, ac = jc["anchor_on_parent"], jc["anchor_on_child"]

            pivot = pymunk.PivotJoint(parent, child, (ap[0], ap[1]), (ac[0], ac[1]))
            pivot.collide_bodies = False

            limit = pymunk.RotaryLimitJoint(parent, child, jc["angle_min"], jc["angle_max"])
            limit.collide_bodies = False

            motor = pymunk.SimpleMotor(parent, child, rate=0.0)
            motor.max_force      = jc["motor_max_torque"]
            motor.collide_bodies = False

            self._joints[jname] = (motor, limit)
            self.space.add(pivot, limit, motor)

    def _update_ground_contacts(self) -> None:
        """Detect ground contact via world-space bounding box (avoids pymunk 7 handler API)."""
        self._ground_contacts.clear()
        for name, body in self.parts.items():
            for shape in body.shapes:
                if shape.bb.bottom <= 0.02:
                    self._ground_contacts.add(name)
                    break

    def step(self) -> None:
        dt = self._physics_cfg["time_step"]
        self.space.step(dt)
        self._update_ground_contacts()
        self.age += dt

        self._height_sum += float(self.parts["body"].position.y)
        self._height_n   += 1

        fatal = {n for n, p in self._model_cfg["parts"].items() if p.get("fatal_ground_contact")}
        if self._ground_contacts & fatal:
            self.alive = False

        if self._sim_cfg.get("head_death", False) and "head" in self._ground_contacts:
            self.alive = False

        if "upper_arm_L" in self._ground_contacts:
            self.alive = False

        if "hand_L" in self._ground_contacts:
            self.alive = False

        if "upper_arm_R" in self._ground_contacts:
            self.alive = False

        if "hand_R" in self._ground_contacts:
            self.alive = False

    def get_nn_inputs(self) -> np.ndarray:
        body  = self.parts["body"]
        jcfg  = self._model_cfg["joints"]
        inputs = [
            body.angle,
            body.angular_velocity,
            body.velocity.x,
            body.velocity.y,
            body.position.y,
        ]
        for jname in JOINT_ORDER:
            p = self.parts[jcfg[jname]["parent"]]
            c = self.parts[jcfg[jname]["child"]]
            inputs.append(c.angle - p.angle)
        for jname in JOINT_ORDER:
            p = self.parts[jcfg[jname]["parent"]]
            c = self.parts[jcfg[jname]["child"]]
            inputs.append(c.angular_velocity - p.angular_velocity)
        inputs.append(1.0 if "foot_L" in self._ground_contacts else 0.0)
        inputs.append(1.0 if "foot_R" in self._ground_contacts else 0.0)
        inputs.append(body.position.x)
        return np.array(inputs, dtype=np.float32)

    def apply_nn_outputs(self, motor_speeds: np.ndarray) -> None:
        jcfg = self._model_cfg["joints"]
        for i, jname in enumerate(JOINT_ORDER):
            motor, _ = self._joints[jname]
            motor.rate = float(motor_speeds[i]) * jcfg[jname]["motor_speed"]

    def compute_fitness(self, fitness_cfg: dict) -> float:
        body      = self.parts["body"]
        dist_x    = float(body.position.x) - self._initial_x
        age       = max(self.age, 1e-6)
        avg_speed = dist_x / age
        avg_h     = (self._height_sum / self._height_n) if self._height_n else 0.0
        h_bonus   = max(0.0, avg_h - 0.5)
        score = (
            fitness_cfg["weight_distance"] * dist_x
            + fitness_cfg["weight_speed"]    * avg_speed
            + fitness_cfg["weight_height"]   * h_bonus
            + fitness_cfg["weight_survival"] * self.age
        )
        return max(0.0, float(score))

    def get_render_state(self) -> list[dict]:
        return [
            {
                "name":     name,
                "position": (float(body.position.x), float(body.position.y)),
                "angle":    float(body.angle),
                "mirror":   self._model_cfg["parts"][name].get("mirror", False),
            }
            for name, body in self.parts.items()
        ]
