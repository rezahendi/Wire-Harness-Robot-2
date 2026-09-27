import numpy as np

from harness_core.cell import HarnessCell
from harness_core.config import CellConfig
from harness_core.layout import sample_instance
from harness_core.sim import HarnessSim


def test_random_cells_build_and_settle():
    cfg = CellConfig()
    for seed in range(4):
        sim = HarnessSim(cfg, seed=seed, randomize=True)
        pts = sim.cable_points()
        assert np.all(np.isfinite(pts))
        # the free part of the wire rests on the board after settling
        assert np.all(pts[3:, 2] > sim.instance.board_z - 0.002)
        st = sim.task_status()
        assert st["n_routed"] == 0 and not st["connector_seated"]


def test_holder_latch_starts_open_and_can_be_disabled():
    sim = HarnessSim(CellConfig(), seed=0, randomize=False)
    assert sim.latch_eq >= 0 and not sim.latched
    assert sim.data.eq_active[sim.latch_eq] == 0
    assert not sim.task_status()["connector_latched"]
    cfg = CellConfig()
    cfg.holder.latch = False
    assert HarnessSim(cfg, seed=0, randomize=False).latch_eq == -1


def test_instance_is_deterministic():
    cfg = CellConfig()
    a = sample_instance(cfg, randomize=True, seed=11)
    b = sample_instance(cfg, randomize=True, seed=11)
    assert np.allclose(a.initial_wire, b.initial_wire)
    assert a.to_dict() == b.to_dict()


def test_ft_sensor_is_payload_compensated():
    sim = HarnessSim(CellConfig(), seed=0, randomize=False)
    w = sim.ft_wrench(noise=False) - sim.ft_bias
    assert np.linalg.norm(w[:3]) < 0.05


def test_compliance_force_control_on_board():
    cell = HarnessCell(CellConfig(), seed=0, randomize=False)
    bz = cell.sim.instance.board_z
    p = cell.sim.tcp_pose()[0]
    cell.set_pose_target(np.array([p[0], p[1], bz + 0.02]), yaw=0.0)
    cell.step_time(1.5)
    # pure force control along z: press the open fingers on the board with 5 N
    cell.set_pose_target(np.array([p[0], p[1], bz + 0.02]), yaw=0.0,
                         wrench=np.array([0, 0, -5.0, 0, 0, 0]), selection=np.array([0, 0, 1.0, 0, 0, 0]))
    cell.step_time(2.0)
    fz = cell.wrench_world()[2]
    assert 3.5 < fz < 6.0          # environment pushes up with ~5 N (minus dead band)
    assert not cell.ctrl.protective_stop
