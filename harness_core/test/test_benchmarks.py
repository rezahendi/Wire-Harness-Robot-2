import math

import numpy as np

from harness_core import benchmarks as B
from harness_core.backend import missing_attributes
from harness_core.config import CellConfig
from harness_core.sim import HarnessSim


def test_elastica_matches_small_deflection_for_a_stiff_short_beam():
    p = B.wire_properties(0.003, 2500.0, 2.0e6)
    r = B.elastica_cantilever(0.02, p["EI"], p["weight_per_length"])
    # far below the gravity/bending length: the linear formula w L^4 / (8 EI) holds
    assert abs(r["tip_drop"] - r["small_deflection_tip_drop"]) < 0.02 * r["small_deflection_tip_drop"]
    assert r["tip_reach"] < 0.0201


def test_elastica_saturates_for_a_long_floppy_wire():
    p = B.wire_properties(0.003, 2500.0, 2.0e6)
    r = B.elastica_cantilever(0.25, p["EI"], p["weight_per_length"])
    assert r["tip_drop"] < 0.25                     # cannot droop further than its length
    assert r["tip_drop"] > 0.5 * 0.25               # but it does hang nearly straight down
    assert r["small_deflection_tip_drop"] > 1.0     # the linear formula is nonsense here


def test_catenary_sag_is_consistent():
    sag = B.catenary_sag(0.30, 0.36)
    assert 0.05 < sag < 0.12
    assert B.catenary_sag(0.30, 0.30) == 0.0        # taut cable does not sag
    assert B.catenary_sag(0.30, 0.42) > sag         # more slack, more sag


def test_swing_analysis_recovers_a_known_oscillation():
    t = np.linspace(0.0, 4.0, 2001)
    f, zeta = 2.0, 0.05
    z = 0.1 * np.exp(-zeta * 2 * math.pi * f * t) * np.cos(2 * math.pi * f * t)
    m = B.analyse_swing(t, z)
    assert abs(m["frequency"] - f) < 0.05
    assert abs(m["damping_ratio"] - zeta) < 0.02


def test_mujoco_backend_satisfies_the_cell_backend_protocol():
    sim = HarnessSim(CellConfig(), seed=0, randomize=False)
    assert missing_attributes(sim) == ()


def test_mujoco_wire_is_within_a_few_percent_of_the_catenary():
    from harness_core.rigs_mujoco import MujocoRigs
    rigs = MujocoRigs(CellConfig())
    spec = B.SagSpec(settle_time=2.5)
    tr = rigs.sag(spec.length, 0.30, spec.segment_length, spec)
    m = B.analyse_sag(tr["points"], tr["span"], spec.length)
    assert tr["finite"]
    assert abs(m["sag_rel_error"]) < 0.10
