"""Pick an OpenGL backend for MuJoCo offscreen rendering (videos, env.render()).

Must run before ``mujoco`` is imported. Order: keep a user choice (MUJOCO_GL),
use the window system if there is a display (WSLg / desktop Linux), otherwise
OSMesa (software, always works) and finally EGL (headless GPU).
"""

from __future__ import annotations

import ctypes.util
import os


def choose_gl_backend() -> str:
    if os.environ.get("MUJOCO_GL"):
        return os.environ["MUJOCO_GL"]
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return "glfw"
    backend = "osmesa" if ctypes.util.find_library("OSMesa") else "egl"
    os.environ["MUJOCO_GL"] = backend
    if backend == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    return backend
