import pytest

from harness_core.config import CellConfig


def test_yaml_roundtrip(tmp_path):
    cfg = CellConfig()
    cfg.wire.bend_modulus = 3.3e6
    cfg.layout.fork_xy = [(0.5, -0.1), (0.6, 0.0)]
    path = tmp_path / "cell.yaml"
    cfg.to_yaml(str(path))
    back = CellConfig.from_yaml(str(path))
    assert back.to_dict() == cfg.to_dict()
    assert isinstance(back.layout.fork_xy[0], tuple)


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        CellConfig.from_dict({"wire": {"no_such_key": 1.0}})
