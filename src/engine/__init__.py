"""OpenGL 2D renderer: SVG texture loading, sprite drawing, ground tiling."""

import io
import math
import os

import cairosvg
import OpenGL.GL as gl

from PIL import Image


_DRAW_ORDER: list[str] = [
    "foot_L",     "foot_R",
    "upper_leg_L","upper_leg_R",
    "hand_L",     "hand_R",
    "upper_arm_L","upper_arm_R",
    "body",
    "head",
]


def _svg_to_texture(svg_path: str, width_px: int, height_px: int) -> int:
    """Rasterize an SVG file into an OpenGL RGBA texture and return its ID."""

    png  = cairosvg.svg2png(url=svg_path, output_width=max(width_px, 4), output_height=max(height_px, 4))
    img  = Image.open(io.BytesIO(png)).convert("RGBA")
    data = img.tobytes("raw", "RGBA")

    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, gl.GL_RGBA,
        img.width, img.height, 0,
        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data,
    )
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tex


class Renderer:
    """OpenGL 2D renderer using legacy immediate mode."""

    def __init__(self, assets_dir: str, model_cfg: dict, ppm: float) -> None:
        """Load all SVG textures and initialize camera state."""

        self._ppm = ppm
        self._model_cfg = model_cfg
        self._textures: dict[str, int] = {}
        self._platform_tex: int | None = None

        self.cam_x = 0.0
        self.cam_y = 1.0
        self._cam_target_x = 0.0

        self._load_textures(assets_dir, model_cfg, ppm)

    def _load_textures(self, assets_dir: str, model_cfg: dict, ppm: float) -> None:
        """Rasterize and upload every unique SVG referenced in the model."""

        person_dir = os.path.join(assets_dir, "person")
        ui_dir = os.path.join(assets_dir, "ui")
        loaded: set[str] = set()

        for part in model_cfg["parts"].values():
            svg_name = part["svg"]
            if svg_name in loaded:
                continue
            w_px = max(int(part["width"]  * ppm), 4)
            h_px = max(int(part["height"] * ppm), 4)
            self._textures[svg_name] = _svg_to_texture(os.path.join(person_dir, svg_name), w_px, h_px)
            loaded.add(svg_name)

        tile_px = max(int(0.5 * ppm), 4)
        self._platform_tex = _svg_to_texture(
            os.path.join(ui_dir, "platform.svg"),
            tile_px,
            max(int(0.15 * ppm), 4),
        )

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def begin_frame(screen_w: int, screen_h: int) -> None:
        """Set up orthographic projection and clear the framebuffer."""

        gl.glViewport(0, 0, screen_w, screen_h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, screen_w, screen_h, 0, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_TEXTURE_2D)

        gl.glClearColor(0.08, 0.10, 0.14, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def update_camera(self, target_x: float, smoothing: float) -> None:
        """Smoothly interpolate the camera towards target_x."""

        self._cam_target_x = target_x
        self.cam_x += (self._cam_target_x - self.cam_x) * smoothing

    def _world_to_screen(self, wx: float, wy: float, sw: int, sh: int) -> tuple[float, float]:
        """Convert world coordinates to screen pixels."""

        sx = (wx - self.cam_x) * self._ppm + sw * 0.4
        sy = sh * 0.55 - (wy - self.cam_y) * self._ppm
        return sx, sy

    # ------------------------------------------------------------------
    # Ground
    # ------------------------------------------------------------------

    def draw_ground(self, screen_w: int, screen_h: int) -> None:
        """Draw a scrolling tiled platform strip at world y=0."""

        _, gy      = self._world_to_screen(0, 0, screen_w, screen_h)
        tile_px    = max(int(0.5 * self._ppm), 4)
        ground_h   = max(int(0.15 * self._ppm), 6)

        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glColor4f(0.07, 0.08, 0.10, 1.0)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(0,        gy + ground_h)
        gl.glVertex2f(screen_w, gy + ground_h)
        gl.glVertex2f(screen_w, float(screen_h))
        gl.glVertex2f(0,        float(screen_h))
        gl.glEnd()

        world_tile       = 0.5
        tile_world_offset = math.fmod(self.cam_x, world_tile)
        start_x          = -tile_px * (tile_world_offset / world_tile) - tile_px

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._platform_tex)
        gl.glColor4f(1, 1, 1, 1)

        x = start_x
        while x < screen_w + tile_px:
            gl.glBegin(gl.GL_QUADS)
            gl.glTexCoord2f(0, 0); gl.glVertex2f(x,           gy)
            gl.glTexCoord2f(1, 0); gl.glVertex2f(x + tile_px, gy)
            gl.glTexCoord2f(1, 1); gl.glVertex2f(x + tile_px, gy + ground_h)
            gl.glTexCoord2f(0, 1); gl.glVertex2f(x,           gy + ground_h)
            gl.glEnd()
            x += tile_px

    # ------------------------------------------------------------------
    # Agent rendering
    # ------------------------------------------------------------------

    def draw_agent(
        self,
        render_state: list[dict],
        screen_w: int,
        screen_h: int,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        """Draw all body parts of an agent in back-to-front Z-order."""

        by_name   = {s["name"]: s for s in render_state}
        parts_cfg = self._model_cfg["parts"]

        for name in _DRAW_ORDER:
            if name not in by_name:
                continue
            seg = by_name[name]
            pcfg = parts_cfg[name]
            sx, sy = self._world_to_screen(seg["position"][0], seg["position"][1], screen_w, screen_h)
            w_px = pcfg["width"] * self._ppm
            h_px = pcfg["height"] * self._ppm
            tex = self._textures[pcfg["svg"]]
            self._draw_sprite(tex, sx, sy, w_px, h_px, seg["angle"], seg["mirror"], color)

    @staticmethod
    def _draw_sprite(
            tex_id: int,
        cx: float,
        cy: float,
        w: float,
        h: float,
        angle: float,
        mirror: bool,
        color: tuple[float, float, float, float],
    ) -> None:
        """Draw a single textured quad centred at (cx, cy) with rotation."""

        hw, hh = w * 0.5, h * 0.5
        u0, u1 = (1.0, 0.0) if mirror else (0.0, 1.0)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glColor4f(*color)

        gl.glPushMatrix()
        gl.glTranslatef(cx, cy, 0)
        gl.glRotatef(math.degrees(-angle), 0, 0, 1)

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(u0, 0.0); gl.glVertex2f(-hw, -hh)
        gl.glTexCoord2f(u1, 0.0); gl.glVertex2f( hw, -hh)
        gl.glTexCoord2f(u1, 1.0); gl.glVertex2f( hw,  hh)
        gl.glTexCoord2f(u0, 1.0); gl.glVertex2f(-hw,  hh)
        gl.glEnd()

        gl.glPopMatrix()

    # ------------------------------------------------------------------
    # Debug overlays
    # ------------------------------------------------------------------

    def draw_collision_shapes(
        self,
        render_state: list[dict],
        screen_w: int,
        screen_h: int,
    ) -> None:
        """Draw wireframe collision shapes for every body part."""

        gl.glDisable(gl.GL_TEXTURE_2D)
        parts_cfg = self._model_cfg["parts"]

        for seg in render_state:
            name   = seg["name"]
            pcfg   = parts_cfg[name]
            sx, sy = self._world_to_screen(seg["position"][0], seg["position"][1], screen_w, screen_h)
            w_px = pcfg["width"] * self._ppm
            h_px = pcfg["height"] * self._ppm
            hw, hh = w_px * 0.5, h_px * 0.5

            gl.glColor4f(0.0, 1.0, 0.3, 0.6)
            gl.glPushMatrix()
            gl.glTranslatef(sx, sy, 0)
            gl.glRotatef(math.degrees(-seg["angle"]), 0, 0, 1)

            if pcfg.get("collision_shape") == "circle":
                segs = 24
                gl.glBegin(gl.GL_LINE_LOOP)
                for i in range(segs):
                    a = 2 * math.pi * i / segs
                    gl.glVertex2f(hw * math.cos(a), hw * math.sin(a))
                gl.glEnd()
            else:
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glVertex2f(-hw, -hh)
                gl.glVertex2f( hw, -hh)
                gl.glVertex2f( hw,  hh)
                gl.glVertex2f(-hw,  hh)
                gl.glEnd()

            gl.glPopMatrix()

        gl.glEnable(gl.GL_TEXTURE_2D)

    def cleanup(self) -> None:
        """Delete all OpenGL textures."""

        for tex in self._textures.values():
            gl.glDeleteTextures(1, [tex])
        if self._platform_tex is not None:
            gl.glDeleteTextures(1, [self._platform_tex])
