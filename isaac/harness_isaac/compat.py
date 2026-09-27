"""Version-tolerant access to the Isaac Sim APIs this package needs.

Isaac Sim renamed its Python packages between 4.x (``omni.isaac.core``) and 5.x
(``isaacsim.core.api`` / ``isaacsim.core.prims``). Everything version dependent is
resolved here so the rigs read the same on either release.
"""

from __future__ import annotations

from typing import Any, Optional

_APP = None


def start_app(headless: bool = True, width: int = 1280, height: int = 720) -> Any:
    """Boot the Isaac Sim kit application. Must happen before any omni/pxr import."""
    global _APP
    if _APP is not None:
        return _APP
    SimulationApp = None
    for module, name in (("isaacsim.simulation_app", "SimulationApp"),
                         ("isaacsim", "SimulationApp"),
                         ("omni.isaac.kit", "SimulationApp")):
        try:
            SimulationApp = getattr(__import__(module, fromlist=[name]), name)
            break
        except Exception:
            continue
    if SimulationApp is None:
        raise RuntimeError(
            "Isaac Sim is not importable. Install it into this interpreter "
            "(pip install 'isaacsim[all,extscache]==5.1.0' on Python 3.11) or run this "
            "script with Isaac Sim's own python (python.sh / python.bat).")
    _APP = SimulationApp({"headless": headless, "width": width, "height": height})
    return _APP


def stop_app() -> None:
    global _APP
    if _APP is not None:
        _APP.close()
        _APP = None


def world_class():
    for module, name in (("isaacsim.core.api", "World"), ("omni.isaac.core", "World")):
        try:
            return getattr(__import__(module, fromlist=[name]), name)
        except Exception:
            continue
    raise RuntimeError("Isaac Sim core World class not found")


def rigid_prim_view():
    """Batched rigid-body view (poses and velocities straight from PhysX)."""
    for module, name in (("isaacsim.core.prims", "RigidPrim"),
                         ("omni.isaac.core.prims", "RigidPrimView")):
        try:
            return getattr(__import__(module, fromlist=[name]), name)
        except Exception:
            continue
    raise RuntimeError("Isaac Sim RigidPrim view class not found")


def articulation_view():
    for module, name in (("isaacsim.core.prims", "Articulation"),
                         ("omni.isaac.core.articulations", "ArticulationView")):
        try:
            return getattr(__import__(module, fromlist=[name]), name)
        except Exception:
            continue
    raise RuntimeError("Isaac Sim articulation view class not found")


def measured_joint_forces(articulation) -> Optional[Any]:
    """Reaction forces at each link's incoming joint, if this build exposes them."""
    for attr in ("get_measured_joint_forces", "get_measured_joint_efforts"):
        fn = getattr(articulation, attr, None)
        if fn is not None:
            try:
                return fn()
            except Exception:
                return None
    return None
