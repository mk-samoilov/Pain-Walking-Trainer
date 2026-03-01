"""Main application: GLFW window, ImGui panels, population loop."""

import json

from pathlib import Path

import glfw
import numpy as np

from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from src.ai import GeneticAlgorithm, NeuralNetwork
from src.engine import Renderer
from src.physics_engine import RagdollAgent


_PROJECT_ROOT = Path(__file__).parent.parent.parent


class App:
    """Top-level application: owns the window, population and render loop."""

    NN_INPUTS = 42
    NN_OUTPUTS = 18

    _NN_INPUT_LABELS: list[str] = [
        # Body state (5)
        "body.angle", "body.ω", "vel.x", "vel.y", "pos.y",
        # Joint angles (9)
        "∠neck",  "∠sh_L",  "∠sh_R",  "∠el_L",  "∠el_R",
        "∠hip_L", "∠hip_R", "∠kn_L",  "∠kn_R",
        # Joint angular velocities (9)
        "ω neck",  "ω sh_L", "ω sh_R", "ω el_L", "ω el_R",
        "ω hip_L", "ω hip_R","ω kn_L", "ω kn_R",
        # Ground contacts (2)
        "ft_L↓", "ft_R↓",
        # Body x (1)
        "pos.x",
        # Foot positions relative to body (4)
        "ftL.rx", "ftL.ry", "ftR.rx", "ftR.ry",
        # Head height (1)
        "head.y",
        # Body acceleration (2)
        "acc.x", "acc.y",
        # Previous motor outputs / proprioception (9)
        "p.neck", "p.sh_L", "p.sh_R", "p.el_L", "p.el_R",
        "p.hip_L","p.hip_R","p.kn_L", "p.kn_R",
    ]
    _NN_OUTPUT_LABELS: list[str] = [
        # Motor speeds (9)
        "spd.neck", "spd.sh_L", "spd.sh_R",
        "spd.el_L", "spd.el_R",
        "spd.hip_L","spd.hip_R",
        "spd.kn_L", "spd.kn_R",
        # Torque multipliers (9)
        "trq.neck", "trq.sh_L", "trq.sh_R",
        "trq.el_L", "trq.el_R",
        "trq.hip_L","trq.hip_R",
        "trq.kn_L", "trq.kn_R",
    ]

    def __init__(self) -> None:
        """Load configs, create window, initialize population."""

        self._cfg = self._load_json(_PROJECT_ROOT / "data" / "config.json")
        self._model = self._load_json(_PROJECT_ROOT / "assets" / "person" / "model.json")

        self._win_cfg = self._cfg["window"]
        self._phys_cfg = self._cfg["physics"]
        self._sim_cfg = self._cfg["simulation"]
        self._ai_cfg = self._cfg["ai"]
        self._fit_cfg = self._cfg["fitness"]
        self._ren_cfg = self._cfg["rendering"]

        self._window, self._imgui_renderer = self._init_window()
        self._renderer = Renderer(
            str(_PROJECT_ROOT / "assets"),
            self._model,
            self._ren_cfg["pixels_per_meter"],
        )

        self._layer_sizes = [self.NN_INPUTS] + self._ai_cfg["hidden_layers"] + [self.NN_OUTPUTS]
        self._ga = GeneticAlgorithm(self._ai_cfg, self._layer_sizes)
        self._generation = 0
        self._best_ever = 0.0
        self._gen_best_history: list[float] = []

        self._paused = False
        self._speed = int(self._sim_cfg["speed_multiplier"])
        self._show_all = bool(self._ren_cfg["show_all_agents"])
        self._show_collisions = bool(self._ren_cfg["show_collisions"])
        self._ghost_alpha = float(self._ren_cfg["ghost_alpha"])
        self._sim_time = float(self._sim_cfg["simulation_time"])
        self._pop_size = int(self._sim_cfg["population_size"])
        self._show_nn_window = True

        self._agents: list[RagdollAgent] = []
        self._networks: list[NeuralNetwork] = []
        self._genomes: list[np.ndarray] = []
        self._fitnesses: list[float] = []
        self._last_activations: list[np.ndarray] | None = None
        self._watched_idx = 0
        self._explorer_idx = 0

        self._new_generation()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: Path | str) -> dict:
        """Load and return a JSON file as a dict."""

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _init_window(self) -> tuple:
        """Initialise GLFW and imgui-bundle, return (window, renderer)."""

        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        window = glfw.create_window(
            self._win_cfg["width"],
            self._win_cfg["height"],
            self._win_cfg["title"],
            None, None,
        )
        if not window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(window)
        glfw.swap_interval(1 if self._win_cfg.get("vsync", True) else 0)

        imgui.create_context()
        impl = GlfwRenderer(window)
        return window, impl

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _new_generation(self) -> None:
        """Evolve (or randomly initialize) genomes and spawn fresh agents."""

        if not self._genomes:
            conn_rate = self._ai_cfg.get("initial_connection_rate", 1.0)
            self._genomes = [
                NeuralNetwork(self._layer_sizes, conn_rate).get_genome()
                for _ in range(self._pop_size)
            ]
            self._explorer_idx = len(self._genomes) - 1
        else:
            self._genomes, self._explorer_idx = self._ga.evolve(self._genomes, self._fitnesses)

        self._networks = [NeuralNetwork.from_genome(self._layer_sizes, g) for g in self._genomes]
        self._agents = [RagdollAgent(self._model, self._phys_cfg, self._sim_cfg) for _ in self._genomes]
        self._fitnesses = [0.0] * len(self._genomes)
        self._generation += 1
        self._watched_idx = 0

    def _best_idx(self) -> int:
        """Return the index of the agent with the highest fitness."""

        return int(np.argmax(self._fitnesses)) if self._fitnesses else 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the main application loop until the window is closed."""

        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            self._imgui_renderer.process_inputs()

            if not self._paused:
                for _ in range(self._speed):
                    if self._step_physics():
                        break

            self._update_watched()

            sw, sh = glfw.get_framebuffer_size(self._window)
            self._renderer.begin_frame(sw, sh)
            self._renderer.draw_ground(sw, sh)
            self._draw_agents(sw, sh)

            if self._agents[self._watched_idx].alive:
                inputs = self._agents[self._watched_idx].get_nn_inputs()
                self._last_activations = self._networks[self._watched_idx].forward_with_activations(inputs)

            imgui.new_frame()
            self._draw_imgui()
            if self._show_nn_window:
                self._draw_nn_window()
            imgui.render()
            self._imgui_renderer.render(imgui.get_draw_data())

            glfw.swap_buffers(self._window)

        self._shutdown()

    def _update_watched(self) -> None:
        """Switch camera to the best living agent when the current one dies."""

        target_x = float(self._agents[self._watched_idx].parts["body"].position.x)

        if not self._agents[self._watched_idx].alive:
            living = [i for i, a in enumerate(self._agents) if a.alive]
            if living:
                self._watched_idx = max(living, key=lambda i: self._fitnesses[i])
                target_x = float(self._agents[self._watched_idx].parts["body"].position.x)

        self._renderer.update_camera(target_x, self._ren_cfg["camera_smoothing"])

    def _step_physics(self) -> bool:
        """Step all living agents forward one timestep. Returns True when generation ends."""

        all_done = True
        for i, (agent, network) in enumerate(zip(self._agents, self._networks)):
            if not agent.alive or agent.age >= self._sim_time:
                continue
            outputs = network.forward(agent.get_nn_inputs())
            agent.apply_nn_outputs(outputs)
            agent.step()
            self._fitnesses[i] = agent.compute_fitness(self._fit_cfg)
            if agent.alive and agent.age < self._sim_time:
                all_done = False

        if all_done:
            best = max(self._fitnesses)
            self._best_ever = max(self._best_ever, best)
            self._gen_best_history.append(best)
            self._new_generation()
            return True
        return False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw_agents(self, sw: int, sh: int) -> None:
        """Render all living agents: watched in full color, others as ghosts."""

        if self._show_all:
            for i, agent in enumerate(self._agents):
                if i == self._watched_idx or not agent.alive:
                    continue
                states = agent.get_render_state()
                color = (1.0, 0.75, 0.0, self._ghost_alpha) if i == self._explorer_idx \
                    else (0.7, 0.7, 0.8, self._ghost_alpha)
                self._renderer.draw_agent(states, sw, sh, color=color)
                if self._show_collisions:
                    self._renderer.draw_collision_shapes(states, sw, sh)

        if self._agents[self._watched_idx].alive:
            states = self._agents[self._watched_idx].get_render_state()
            self._renderer.draw_agent(states, sw, sh, color=(1.0, 1.0, 1.0, 1.0))
            if self._show_collisions:
                self._renderer.draw_collision_shapes(states, sw, sh)

    # ------------------------------------------------------------------
    # ImGui panels
    # ------------------------------------------------------------------

    def _draw_imgui(self) -> None:
        """Draw the main control panel."""

        imgui.set_next_window_pos((10, 10), imgui.Cond_.once)
        imgui.set_next_window_size((290, 0), imgui.Cond_.once)
        imgui.begin(
            "WalkAI",
            flags=imgui.WindowFlags_.no_resize | imgui.WindowFlags_.always_auto_resize,
        )

        alive = sum(1 for a in self._agents if a.alive)
        imgui.text(f"Generation : {self._generation}")
        imgui.text(f"Alive      : {alive} / {len(self._agents)}")
        imgui.text(f"Best ever  : {self._best_ever:.2f}")
        if self._fitnesses:
            imgui.text(f"Best now   : {max(self._fitnesses):.2f}")
            imgui.text(f"Avg now    : {sum(self._fitnesses) / len(self._fitnesses):.2f}")
            imgui.text(f"Explorer   : {self._fitnesses[self._explorer_idx]:.2f}")

        imgui.separator()

        _, self._paused = imgui.checkbox("Pause", self._paused)
        _, self._speed = imgui.slider_int("Speed x", self._speed, 1, 200)
        _, self._show_all = imgui.checkbox("Show all agents", self._show_all)
        _, self._show_collisions = imgui.checkbox("Show collisions", self._show_collisions)
        _, self._show_nn_window = imgui.checkbox("Show NN window", self._show_nn_window)

        imgui.separator()
        imgui.text("Next generation settings:")
        _, self._pop_size = imgui.slider_int("Population", self._pop_size, 2, 100)
        changed, new_sim_t = imgui.slider_float("Sim time (s)", self._sim_time, 5.0, 60.0)
        if changed:
            self._sim_time = new_sim_t

        imgui.separator()

        if imgui.button("New population"):
            self._genomes = []
            self._fitnesses = []
            self._generation = 0
            self._best_ever = 0.0
            self._gen_best_history.clear()
            self._new_generation()

        imgui.same_line()

        if imgui.button("Restart generation"):
            self._agents = [RagdollAgent(self._model, self._phys_cfg, self._sim_cfg) for _ in self._genomes]
            self._fitnesses = [0.0] * len(self._genomes)

        if imgui.button("Save"):
            self._save()

        imgui.same_line()

        if imgui.button("Load"):
            self._load()

        if len(self._gen_best_history) > 1:
            imgui.separator()
            imgui.text("Fitness history (best):")
            imgui.plot_lines(
                "##fit",
                np.array(self._gen_best_history, dtype=np.float32),
                graph_size=(265, 60),
            )

        imgui.end()

    def _draw_nn_window(self) -> None:
        """Draw the neural network visualization for the watched agent."""

        imgui.set_next_window_pos((310, 10), imgui.Cond_.once)
        imgui.set_next_window_size((700, 420), imgui.Cond_.once)
        watched_nn = self._networks[self._watched_idx]
        active = watched_nn.active_connections
        total = sum(w.size for w in watched_nn.weights)
        imgui.begin(f"Neural Network — {active}/{total} connections")

        draw_list = imgui.get_window_draw_list()
        origin = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        imgui.dummy(avail)

        if self._last_activations is None:
            imgui.end()
            return

        ox, oy = float(origin[0]), float(origin[1])
        aw, ah = float(avail[0]), float(avail[1])
        acts = self._last_activations
        sizes = self._layer_sizes
        n_layers = len(sizes)

        pad_left, pad_right, pad_y = 90.0, 72.0, 12.0

        def layer_x(li: int) -> float:
            """Screen X for layer li."""
            return ox + pad_left + li * (aw - pad_left - pad_right) / (n_layers - 1)

        def neuron_y(li: int, ni: int) -> float:
            """Screen Y for neuron ni in layer li."""
            n = sizes[li]
            if n == 1:
                return oy + ah / 2
            return oy + pad_y + ni * (ah - 2 * pad_y) / (n - 1)

        def im_col(r: int, g: int, b: int, a: int) -> int:
            """Pack RGBA bytes into an ImGui ImU32 colour."""
            return (a << 24) | (b << 16) | (g << 8) | r

        def activation_color(val: float) -> int:
            """Map tanh activation [-1, 1] to a blue→gray→red color."""
            v = max(-1.0, min(1.0, float(val)))
            if v >= 0:
                return im_col(80 + int(175 * v), 80 + int(50 * v), max(0, 80 - int(60 * v)), 255)
            return im_col(max(0, 80 + int(20 * v)), max(0, 80 + int(20 * v)), 80 - int(175 * v), 255)

        nr = max(3, min(7, int(ah / max(sizes[0], 1) / 2.2)))
        font_h = imgui.get_font_size()
        text_col = im_col(210, 210, 210, 200)
        border = im_col(200, 200, 200, 180)

        # Connections — only active (mask > 0.5)
        for li in range(n_layers - 1):
            W = watched_nn.weights[li]
            M = watched_nn.masks[li]
            max_w = float(np.abs(W * M).max()) + 1e-6
            for j in range(sizes[li + 1]):
                yj = neuron_y(li + 1, j)
                for i in range(sizes[li]):
                    if M[j, i] < 0.5:
                        continue
                    w = float(W[j, i])
                    norm = abs(w) / max_w
                    if norm < 0.08:
                        continue
                    alpha = int(25 + 160 * norm)
                    col = im_col(50, int(180 * norm), 50, alpha) if w > 0 \
                        else im_col(int(180 * norm), 50, 50, alpha)
                    draw_list.add_line(
                        (layer_x(li), neuron_y(li, i)),
                        (layer_x(li + 1), yj),
                        col, max(0.4, norm * 1.5),
                    )

        # Neurons
        for li, layer_acts in enumerate(acts):
            x = layer_x(li)
            for ni in range(len(layer_acts)):
                y = neuron_y(li, ni)
                draw_list.add_circle_filled((x, y), nr, activation_color(layer_acts[ni]))
                draw_list.add_circle((x, y), nr, border, 0, 1.0)

        # Input labels (right-aligned)
        x0 = layer_x(0)
        for ni, label in enumerate(self._NN_INPUT_LABELS):
            y = neuron_y(0, ni)
            tw = imgui.calc_text_size(label)[0]
            draw_list.add_text((x0 - nr - 5 - tw, y - font_h * 0.5), text_col, label)

        # Output labels (left-aligned)
        xl = layer_x(n_layers - 1)
        for ni, label in enumerate(self._NN_OUTPUT_LABELS):
            y = neuron_y(n_layers - 1, ni)
            draw_list.add_text((xl + nr + 5, y - font_h * 0.5), text_col, label)

        imgui.end()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    _SAVE_PATH = _PROJECT_ROOT / "data" / "save.npz"

    def _save(self) -> None:
        """Serialize full training state to data/save.npz."""

        np.savez(
            self._SAVE_PATH,
            genomes=np.array(self._genomes),
            fitnesses=np.array(self._fitnesses),
            gen_best_history=np.array(self._gen_best_history),
            layer_sizes=np.array(self._layer_sizes),
            generation=np.array(self._generation),
            best_ever=np.array(self._best_ever),
            explorer_idx=np.array(self._explorer_idx),
        )

    def _load(self) -> None:
        """Restore training state from data/save.npz, if it exists and is compatible."""

        if not self._SAVE_PATH.exists():
            return

        data = np.load(self._SAVE_PATH)

        if list(data["layer_sizes"].tolist()) != self._layer_sizes:
            return  # network architecture mismatch — silently skip

        self._genomes = list(data["genomes"])
        self._fitnesses = list(data["fitnesses"].tolist())
        self._gen_best_history = list(data["gen_best_history"].tolist())
        self._generation = int(data["generation"])
        self._best_ever = float(data["best_ever"])
        self._explorer_idx = int(data["explorer_idx"])
        self._pop_size = len(self._genomes)
        self._last_activations = None

        self._networks = [NeuralNetwork.from_genome(self._layer_sizes, g) for g in self._genomes]
        self._agents = [RagdollAgent(self._model, self._phys_cfg, self._sim_cfg) for _ in self._genomes]
        self._watched_idx = 0

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        """Release all GPU and windowing resources."""

        self._renderer.cleanup()
        self._imgui_renderer.shutdown()
        glfw.terminate()
