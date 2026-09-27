"""Disturbances for the recovery benchmark: things that go wrong on a real line.

The planner can neither cause nor see these directly. It only meets their effect in the
state that the next skill (or get_status) reports, the way a cell supervisor would:

    pop_fork         someone snags the wire and it comes out of a fork it was seated in
    slip_on_insert   the connector slips out of the fingers above its holder and lands
                     on the holder, usually on a rail or a wall

(A connector knocked onto its end needs no injecting: it happens on its own in some
builds, and the planner has a skill for it.)

A scenario is a list of triggers ("after route_fork F2 succeeds, pop F2") that the ToolBox
fires between tool calls. ``SCENARIOS`` lists the ones the benchmark runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def _step_holding(session, n: int, before_step: Optional[Callable[[int], None]] = None) -> None:
    for k in range(n):
        if before_step is not None:
            before_step(k)
        session.env.step(session.expert._hold())
        session.expert._last_obs = session.obs
        session._grab_frame()


def pop_fork(session, fork_id: str, open_force: float = 4.0, lift: float = 0.014, side: float = 0.035,
             max_force: float = 0.25) -> Dict[str, Any]:
    """Pull the wire out of a fork the way a snag would: the jaws are pushed open, the few
    wire segments at the fork are lifted just clear of the lips and moved sideways, then
    let go. The forces are servoed (clipped at ``max_force`` per segment), so only the
    wire at this fork moves; the forks before it keep holding."""
    import mujoco
    sim = session.env.cell.sim
    m, d = sim.model, sim.data
    cfg = session.cfg
    i = session.route.index(fork_id)
    dofs = [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"fork{i}_hinge_{s}")])
            for s in ("l", "r")]
    slot = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"fork{i}_slot")].copy()
    _, _, _, yaw = session.obs["forks"][i]
    y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])  # across the slot
    wires = [b for b in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"wire_{k}") for k in range(m.nbody))
             if b >= 0]
    near = sorted(wires, key=lambda b: float(np.linalg.norm(d.xpos[b] - slot)))[:3]
    z_top = slot[2] + cfg.fork.prong_height + lift            # just above the lips
    goal = {b: d.xpos[b].copy() for b in near}
    dt = cfg.sim.policy_dt
    kp, kd = 25.0, 2.0

    def servo(k: int, sideways: bool) -> None:
        for dof in dofs:
            d.qfrc_applied[dof] = open_force
        for b in near:
            target = goal[b].copy()
            target[2] = z_top
            if sideways:
                target[:2] = goal[b][:2] + side * y_axis[:2]
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
            f = kp * (target - d.xpos[b]) - kd * vel[3:6]
            if not sideways:
                f[:2] = kp * (goal[b][:2] - d.xpos[b][:2]) - kd * vel[3:5]
            d.xfrc_applied[b, :3] = np.clip(f, -max_force, max_force)

    try:
        _step_holding(session, int(round(0.8 / dt)), lambda k: servo(k, False))
        _step_holding(session, int(round(0.6 / dt)), lambda k: servo(k, True))
    finally:
        d.qfrc_applied[dofs] = 0.0
        d.xfrc_applied[near] = 0.0
    _step_holding(session, int(round(0.8 / dt)))
    truth = session.truth()["forks_routed"]
    return {"fork_still_holds": bool(truth[fork_id]),
            "other_forks_holding": [f for f in session.route if f != fork_id and truth[f]]}


@dataclass
class Trigger:
    """Fire ``action(session)`` once, right after a matching tool call returns."""
    label: str
    after: str                                   # tool name
    action: Callable[[Any], Dict[str, Any]]
    arg: Optional[str] = None                    # e.g. the fork id of a route_fork
    only_ok: bool = True                         # only after a successful call
    fired: bool = False

    def matches(self, name: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> bool:
        if self.fired or name != self.after:
            return False
        if self.arg is not None and str((arguments or {}).get("fork_id")) != self.arg:
            return False
        return bool(result.get("ok")) or not self.only_ok


@dataclass
class Scenario:
    name: str
    triggers: List[Trigger] = field(default_factory=list)
    faults: List[str] = field(default_factory=list)        # session-level faults, e.g. slip_on_insert
    description: str = ""


SCENARIOS = ("nominal", "popped_wire", "slip_on_insert", "both")


def make_scenario(name: str, route: List[str]) -> Scenario:
    """The benchmark's scenarios for a route (fork ids in order)."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; use one of {SCENARIOS}")
    mid = route[min(1, len(route) - 1)]
    pop = Trigger(f"wire pulled out of {mid}", "route_fork", lambda s, f=mid: pop_fork(s, f), arg=mid)
    if name == "nominal":
        return Scenario("nominal", description="no disturbance")
    if name == "popped_wire":
        return Scenario("popped_wire", [pop],
                        description=f"right after {mid} is routed, the wire is pulled out of it again")
    if name == "slip_on_insert":
        return Scenario("slip_on_insert", faults=["slip_on_insert"],
                        description="the first insertion loses the connector above the holder")
    return Scenario("both", [pop], ["slip_on_insert"], description="both, in one build")
