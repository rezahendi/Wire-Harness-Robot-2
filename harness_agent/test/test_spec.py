import os

import numpy as np
import pytest

from harness_agent.drawing import render_drawing
from harness_agent.spec import (HarnessSpec, board_to_robot, has_errors, robot_to_board,
                                to_cell_config, validate)
from harness_core.config import CellConfig

SPECS = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "specs")


def spec(name):
    return HarnessSpec.from_yaml(os.path.join(SPECS, name + ".yaml"))


def test_board_coordinates_round_trip():
    cfg = CellConfig()
    for uv in ((0.0, 0.0), (260.0, 80.0), (999.0, 439.0)):
        back = robot_to_board(board_to_robot(uv, cfg), cfg)
        assert back == pytest.approx(uv, abs=1e-9)


def test_nominal_spec_reproduces_the_cell_layout():
    cfg = to_cell_config(spec("demo_3fork"))
    ref = CellConfig().layout
    assert cfg.layout.anchor_xy == pytest.approx(ref.anchor_xy)
    assert np.allclose(np.array(cfg.layout.fork_xy), np.array(ref.fork_xy))
    assert cfg.layout.holder_xy == pytest.approx(ref.holder_xy)


@pytest.mark.parametrize("name", ["demo_3fork", "demo_4fork", "demo_2fork_stiff"])
def test_demo_specs_are_valid(name):
    assert not has_errors(validate(spec(name)))


def test_infeasible_spec_explains_every_problem():
    codes = {i.code for i in validate(spec("demo_infeasible")) if i.severity == "error"}
    assert {"slack", "off_board", "reach", "spacing", "sharp_turn"} <= codes


def test_spec_round_trips_through_yaml(tmp_path):
    s = spec("demo_3fork")
    path = tmp_path / "s.yaml"
    s.to_yaml(str(path))
    assert HarnessSpec.from_yaml(str(path)).to_dict() == s.to_dict()


def test_drawing_renders(tmp_path):
    out = render_drawing(spec("demo_3fork"), str(tmp_path / "d.png"))
    assert os.path.getsize(out) > 10_000
