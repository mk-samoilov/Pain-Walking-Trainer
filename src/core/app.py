import json
import os
import sys
import time

import glfw
import OpenGL.GL as gl
import numpy as np

# imgui-bundle: different import paths vs pyimgui
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

# Project root = two levels above this file (src/core/app.py → project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from ai import NeuralNetwork, GeneticAlgorithm
from physics_engine import RagdollAgent
from engine import Renderer


class App:
    NN_INPUTS  = 26
    NN_OUTPUTS = 9

    def __init__(self) -> None:
        self._cfg      = self._load_json(os.path.join(_PROJECT_ROOT, "data", "config.json"))
        self._model    = self._load_json(os.path.join(_PROJECT_ROOT, "assets", "person", "model.json"))
        self._assets   = os.path.join(_PROJECT_ROOT, "assets")

        self._win_cfg  = self._cfg["window"]
        self._phys_cfg = self._cfg["physics"]
        self._sim_cfg  = self._cfg["simulation"]
        self._ai_cfg   = self._cfg["ai"]
        self._fit_cfg  = self._cfg["fitness"]
        self._ren_cfg  = self._cfg["rendering"]

        self._window, self._imgui_renderer = self._init_window()
        self._renderer = Renderer(self._assets, self._model, self._ren_cfg["pixels_per_meter"])

        self._ga          = GeneticAlgorithm(self._ai_cfg)
        self._layer_sizes = [self.NN_INPUTS] + self._ai_cfg["hidden_layers"] + [self.NN_OUTPUTS]
        self._generation  = 0
        self._best_ever   = 0.0
        self._gen_best_history: list[float] = []
        self._gen_avg_history:  list[float] = []

        self._paused          = False
        self._speed           = int(self._sim_cfg["speed_multiplier"])
        self._show_all        = bool(self._ren_cfg["show_all_agents"])
        self._show_collisions = bool(self._ren_cfg["show_collisions"])
        self._ghost_alpha     = float(self._ren_cfg["ghost_alpha"])
        self._sim_time        = float(self._sim_cfg["simulation_time"])
        self._pop_size        = int(self._sim_cfg["population_size"])

        self._agents:    list[RagdollAgent]  = []
        self._networks:  list[NeuralNetwork] = []
        self._genomes:   list[np.ndarray]    = []
        self._fitnesses: list[float]         = []
        self._last_activations: list[np.ndarray] | None = None
        self._show_nn_window = True
        self._watched_idx = 0

        self._new_generation()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        w = glfw.create_window(
            self._win_cfg["width"],
            self._win_cfg["height"],
            self._win_cfg["title"],
            None, None,
        )
        if not w:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(w)
        glfw.swap_interval(1 if self._win_cfg.get("vsync", True) else 0)

        imgui.create_context()
        impl = GlfwRenderer(w)
        return w, impl

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def _new_generation(self) -> None:
        if not self._genomes:
            self._genomes = [
                NeuralNetwork(self._layer_sizes).get_genome()
                for _ in range(self._pop_size)
            ]
        else:
            self._genomes = self._ga.evolve(self._genomes, self._fitnesses)

        self._networks  = [NeuralNetwork.from_genome(self._layer_sizes, g) for g in self._genomes]
        self._agents    = [RagdollAgent(self._model, self._phys_cfg, self._sim_cfg) for _ in range(self._pop_size)]
        self._fitnesses = [0.0] * self._pop_size
        self._generation += 1
        self._watched_idx = 0

    def _best_idx(self) -> int:
        return int(np.argmax(self._fitnesses)) if self._fitnesses else 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        while not glfw.window_should_close(self._window):
            glfw.poll_events()
            self._imgui_renderer.process_inputs()

            if not self._paused:
                for _ in range(self._speed):
                    if self._step_physics():
                        break

            # Switch camera to best living agent if current one died
            if not self._agents[self._watched_idx].alive:
                living_idxs = [i for i, a in enumerate(self._agents) if a.alive]
                if living_idxs:
                    self._watched_idx = max(living_idxs, key=lambda i: self._fitnesses[i])
            target_x = float(self._agents[self._watched_idx].parts["body"].position.x)
            self._renderer.update_camera(target_x, self._ren_cfg["camera_smoothing"])

            sw, sh = glfw.get_framebuffer_size(self._window)
            self._renderer.begin_frame(sw, sh)
            self._renderer.draw_ground(sw, sh)
            self._draw_agents(sw, sh)

            # Update activations for NN window (watched agent)
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

    def _step_physics(self) -> bool:
        """Returns True when generation ends."""
        all_done = True
        for i, (agent, network) in enumerate(zip(self._agents, self._networks)):
            if not agent.alive or agent.age >= self._sim_time:
                continue
            inputs  = agent.get_nn_inputs()
            outputs = network.forward(inputs)
            agent.apply_nn_outputs(outputs)
            agent.step()
            self._fitnesses[i] = agent.compute_fitness(self._fit_cfg)
            if agent.alive and agent.age < self._sim_time:
                all_done = False

        if all_done:
            best = max(self._fitnesses)
            avg  = sum(self._fitnesses) / max(len(self._fitnesses), 1)
            self._best_ever = max(self._best_ever, best)
            self._gen_best_history.append(best)
            self._gen_avg_history.append(avg)
            self._new_generation()
            return True
        return False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw_agents(self, sw: int, sh: int) -> None:
        if self._show_all:
            for i, agent in enumerate(self._agents):
                if i == self._watched_idx or not agent.alive:
                    continue
                states = agent.get_render_state()
                self._renderer.draw_agent(states, sw, sh, color=(0.7, 0.7, 0.8, self._ghost_alpha))
                if self._show_collisions:
                    self._renderer.draw_collision_shapes(states, sw, sh)

        if self._agents[self._watched_idx].alive:
            states = self._agents[self._watched_idx].get_render_state()
            self._renderer.draw_agent(states, sw, sh, color=(1.0, 1.0, 1.0, 1.0))
            if self._show_collisions:
                self._renderer.draw_collision_shapes(states, sw, sh)

    # ------------------------------------------------------------------
    # ImGui  (imgui-bundle API: enum-style constants, Vec2 positions)
    # ------------------------------------------------------------------

    def _draw_imgui(self) -> None:
        imgui.set_next_window_pos((10, 10), imgui.Cond_.once)
        imgui.set_next_window_size((290, 0), imgui.Cond_.once)
        imgui.begin(
            "WalkAI",
            flags=imgui.WindowFlags_.no_resize | imgui.WindowFlags_.always_auto_resize,
        )

        alive = sum(1 for a in self._agents if a.alive)
        total = len(self._agents)
        imgui.text(f"Generation : {self._generation}")
        imgui.text(f"Alive      : {alive} / {total}")
        imgui.text(f"Best ever  : {self._best_ever:.2f}")
        if self._fitnesses:
            imgui.text(f"Best now   : {max(self._fitnesses):.2f}")
            imgui.text(f"Avg now    : {sum(self._fitnesses) / len(self._fitnesses):.2f}")

        imgui.separator()

        _, self._paused       = imgui.checkbox("Pause", self._paused)
        _, self._speed        = imgui.slider_int("Speed x", self._speed, 1, 100)
        _, self._show_all     = imgui.checkbox("Show all agents", self._show_all)
        _, self._show_collisions = imgui.checkbox("Show collisions", self._show_collisions)
        _, self._show_nn_window  = imgui.checkbox("Show NN window", self._show_nn_window)
        changed_hd, hd = imgui.checkbox("Head death", self._sim_cfg.get("head_death", False))
        if changed_hd:
            self._sim_cfg["head_death"] = hd

        imgui.separator()
        imgui.text("Next generation settings:")
        _, self._pop_size = imgui.slider_int("Population", self._pop_size, 2, 100)
        changed, new_sim_t = imgui.slider_float("Sim time (s)", self._sim_time, 5.0, 60.0)
        if changed:
            self._sim_time = new_sim_t

        imgui.separator()

        if imgui.button("New population"):
            self._genomes        = []
            self._fitnesses      = []
            self._generation     = 0
            self._best_ever      = 0.0
            self._gen_best_history.clear()
            self._gen_avg_history.clear()
            self._new_generation()

        imgui.same_line()

        if imgui.button("Restart generation"):
            self._agents    = [RagdollAgent(self._model, self._phys_cfg, self._sim_cfg) for _ in range(len(self._genomes))]
            self._fitnesses = [0.0] * len(self._genomes)

        if len(self._gen_best_history) > 1:
            imgui.separator()
            imgui.text("Fitness history (best):")
            imgui.plot_lines(
                "##fit",
                np.array(self._gen_best_history, dtype=np.float32),
                graph_size=(265, 60),
            )

        imgui.end()

    _NN_INPUT_LABELS = [
        "body.angle", "body.ω", "vel.x", "vel.y", "pos.y",
        "∠neck", "∠sh_L", "∠sh_R", "∠el_L", "∠el_R",
        "∠hip_L", "∠hip_R", "∠kn_L", "∠kn_R",
        "ω neck", "ω sh_L", "ω sh_R", "ω el_L", "ω el_R",
        "ω hip_L", "ω hip_R", "ω kn_L", "ω kn_R",
        "ft_L↓", "ft_R↓", "pos.x",
    ]
    _NN_OUTPUT_LABELS = [
        "neck", "sh_L", "sh_R", "el_L", "el_R",
        "hip_L", "hip_R", "kn_L", "kn_R",
    ]

    def _draw_nn_window(self) -> None:
        imgui.set_next_window_pos((310, 10), imgui.Cond_.once)
        imgui.set_next_window_size((700, 420), imgui.Cond_.once)
        imgui.begin("Neural Network (best)")

        draw_list = imgui.get_window_draw_list()
        origin = imgui.get_cursor_screen_pos()
        avail  = imgui.get_content_region_avail()
        imgui.dummy(avail)

        if self._last_activations is None:
            imgui.end()
            return

        ox, oy   = float(origin[0]), float(origin[1])
        aw, ah   = float(avail[0]), float(avail[1])
        acts     = self._last_activations
        sizes    = self._layer_sizes
        n_layers = len(sizes)

        pad_left, pad_right, pad_y = 90.0, 72.0, 12.0

        def lx(li: int) -> float:
            usable = aw - pad_left - pad_right
            return ox + pad_left + li * usable / (n_layers - 1)

        def ny(li: int, ni: int) -> float:
            n = sizes[li]
            if n == 1:
                return oy + ah / 2
            return oy + pad_y + ni * (ah - 2 * pad_y) / (n - 1)

        def im_col(r: int, g: int, b: int, a: int) -> int:
            return (a << 24) | (b << 16) | (g << 8) | r

        def act_col(val: float) -> int:
            v = max(-1.0, min(1.0, float(val)))
            if v >= 0:
                return im_col(80 + int(175 * v), 80 + int(50 * v), max(0, 80 - int(60 * v)), 255)
            else:
                return im_col(max(0, 80 + int(20 * v)), max(0, 80 + int(20 * v)), 80 - int(175 * v), 255)

        nr       = max(3, min(7, int(ah / max(sizes[0], 1) / 2.2)))
        font_h   = imgui.get_font_size()
        text_col = im_col(210, 210, 210, 200)
        best_nn  = self._networks[self._best_idx()]

        # Connections
        for li in range(n_layers - 1):
            W = best_nn.weights[li]
            max_w = float(np.abs(W).max()) + 1e-6
            for j in range(sizes[li + 1]):
                yj = ny(li + 1, j)
                for i in range(sizes[li]):
                    w = float(W[j, i])
                    norm = abs(w) / max_w
                    if norm < 0.08:
                        continue
                    alpha = int(25 + 160 * norm)
                    col = im_col(50, int(180 * norm), 50, alpha) if w > 0 \
                        else im_col(int(180 * norm), 50, 50, alpha)
                    draw_list.add_line((lx(li), ny(li, i)), (lx(li + 1), yj),
                                       col, max(0.4, norm * 1.5))

        # Neurons
        border = im_col(200, 200, 200, 180)
        for li, layer_acts in enumerate(acts):
            x = lx(li)
            for ni in range(len(layer_acts)):
                y = ny(li, ni)
                draw_list.add_circle_filled((x, y), nr, act_col(layer_acts[ni]))
                draw_list.add_circle((x, y), nr, border, 0, 1.0)

        # Input labels (right-aligned, left of first layer)
        x0 = lx(0)
        for ni, label in enumerate(self._NN_INPUT_LABELS):
            y = ny(0, ni)
            tw = imgui.calc_text_size(label)[0]
            draw_list.add_text((x0 - nr - 5 - tw, y - font_h * 0.5), text_col, label)

        # Output labels (left-aligned, right of last layer)
        xl = lx(n_layers - 1)
        for ni, label in enumerate(self._NN_OUTPUT_LABELS):
            y = ny(n_layers - 1, ni)
            draw_list.add_text((xl + nr + 5, y - font_h * 0.5), text_col, label)

        imgui.end()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        self._renderer.cleanup()
        self._imgui_renderer.shutdown()
        glfw.terminate()
