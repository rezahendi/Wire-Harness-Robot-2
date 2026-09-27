# Pick an OpenGL backend for MuJoCo before any test module imports it (camera tests
# render offscreen: OSMesa or EGL without a display, GLFW under WSLg / a desktop).
from harness_core.render_util import choose_gl_backend

choose_gl_backend()
