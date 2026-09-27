"""Slow end-to-end check of the scripted expert (about a minute of CPU)."""

import os

import numpy as np
import pytest

from harness_core.actions import ActionInterface, ActionSpec
from harness_core.cell import HarnessCell
from harness_core.config import CellConfig
from harness_core.expert import HarnessExpert
from harness_core.geometry import tool_yaw


@pytest.mark.skipif(os.environ.get("HARNESS_SLOW_TESTS", "0") != "1",
                    reason="set HARNESS_SLOW_TESTS=1 to run the full routing episode")
def test_expert_routes_nominal_cell():
    cfg = CellConfig()
    cell = HarnessCell(cfg, seed=1, randomize=True)
    spec = ActionSpec()
    iface = ActionInterface(spec, cell.sim.instance.board_z)
    p, R = cell.sim.tcp_pose()
    iface.reset(p, tool_yaw(R), cell.sim.gripper_opening())
    expert = HarnessExpert(cfg, spec)
    while not expert.done and cell.sim.time < 160.0:
        obs = cell.observe()
        obs["target_pos"] = iface.target_pos.copy()
        obs["target_yaw"] = np.array([iface.target_yaw])
        a = expert.step(obs)
        p, R = cell.sim.tcp_pose()
        iface.apply(a, p, tool_yaw(R))
        cell.set_pose_target(iface.target_pos, yaw=iface.target_yaw)
        cell.set_gripper(iface.gripper)
        cell.step_time(cfg.sim.policy_dt)
    assert cell.sim.task_status()["success"], expert.log[-5:]
