"""The URDF used by robot_state_publisher / RViz must match the MuJoCo model."""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from harness_core.config import CellConfig
from harness_core.geometry import rot_x, rot_y, rot_z
from harness_core.sim import HarnessSim

xacro = pytest.importorskip("xacro")


def _find_xacro() -> str:
    here = os.path.dirname(os.path.realpath(__file__))
    cand = os.path.join(here, "..", "..", "harness_description", "urdf", "harness_cell.urdf.xacro")
    if os.path.exists(cand):
        return cand
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("harness_description"), "urdf", "harness_cell.urdf.xacro")


def _origin(el):
    o = el.find("origin")
    xyz = np.zeros(3) if o is None else np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
    rpy = np.zeros(3) if o is None else np.array([float(v) for v in o.get("rpy", "0 0 0").split()])
    T = np.eye(4)
    T[:3, :3] = rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])
    T[:3, 3] = xyz
    return T


def _urdf_fk(root, q_by_name, target):
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain = []
    link = target
    while link in joints:
        j = joints[link]
        chain.append(j)
        link = j.find("parent").get("link")
    T = np.eye(4)
    for j in reversed(chain):
        T = T @ _origin(j)
        if j.get("type") in ("revolute", "continuous"):
            axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            ang = q_by_name.get(j.get("name"), 0.0)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
            M = np.eye(4)
            M[:3, :3] = R
            T = T @ M
    return T


def test_urdf_tcp_matches_mujoco():
    doc = xacro.process_file(_find_xacro())
    root = ET.fromstring(doc.toxml())
    cfg = CellConfig()
    sim = HarnessSim(cfg, seed=0, randomize=False)
    d = mujoco.MjData(sim.model)
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.uniform(-2.5, 2.5, 6)
        d.qpos[sim.arm_qadr] = q
        mujoco.mj_kinematics(sim.model, d)
        qn = dict(zip(cfg.robot.joint_names, q))
        for link, site in (("tcp", "tcp"), ("tool0", "tool0"), ("ft_frame", "ft_site")):
            T = _urdf_fk(root, qn, link)
            sid = sim.sid[site]
            assert np.allclose(T[:3, 3], d.site_xpos[sid], atol=1e-6), link
            assert np.allclose(T[:3, :3], d.site_xmat[sid].reshape(3, 3), atol=1e-6), link
