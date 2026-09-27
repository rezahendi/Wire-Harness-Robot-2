import mujoco
import numpy as np

from harness_core.config import CellConfig
from harness_core.geometry import homogeneous, tool_down_rotation
from harness_core.sim import HarnessSim
from harness_core.ur_kinematics import URKinematics


def test_dh_fk_matches_mujoco():
    cfg = CellConfig()
    sim = HarnessSim(cfg, seed=0, randomize=False)
    d = mujoco.MjData(sim.model)
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(-3.0, 3.0, 6)
        d.qpos[sim.arm_qadr] = q
        mujoco.mj_kinematics(sim.model, d)
        T = sim.kin.fk(q)
        assert np.allclose(T[:3, 3], d.site_xpos[sim.sid["tcp"]], atol=1e-9)
        assert np.allclose(T[:3, :3], d.site_xmat[sim.sid["tcp"]].reshape(3, 3), atol=1e-9)


def test_jacobian_matches_finite_differences():
    kin = URKinematics(CellConfig().robot)
    q = np.array([0.2, -1.4, 1.7, -1.8, -1.5, 0.4])
    J = kin.jacobian(q)
    eps = 1e-6
    T0 = kin.fk(q)
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        T1 = kin.fk(q + dq)
        assert np.allclose((T1[:3, 3] - T0[:3, 3]) / eps, J[:3, i], atol=1e-5)


def test_ik_reaches_board_poses():
    cfg = CellConfig()
    kin = URKinematics(cfg.robot)
    for pos, yaw in [((0.45, 0.0, 0.05), 0.0), ((0.6, -0.2, 0.03), 1.0), ((0.35, 0.3, 0.1), -2.0)]:
        T = homogeneous(tool_down_rotation(yaw), np.array(pos))
        q = kin.ik(T, cfg.robot.home_q, iters=500)
        assert np.allclose(kin.fk(q)[:3, 3], pos, atol=1e-5)
