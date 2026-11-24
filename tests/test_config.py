import pathlib

import pytest

from quadrucat.config import apply_overrides, load_config

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_inheritance_and_overrides():
    cfg = load_config(ROOT / "configs/experiments/no_domain_rand.yaml",
                      ["env.overrides.reward_config.scales.feet_slip=0.0", "seed=3"])
    assert cfg["env"]["domain_randomization"] is False
    assert cfg["env"]["name"] == "Go1JoystickFlatTerrain"  # inherited from base
    assert cfg["env"]["overrides"]["reward_config"]["scales"]["feet_slip"] == 0.0
    assert cfg["seed"] == 3


def test_all_configs_load():
    for path in list((ROOT / "configs").rglob("*.yaml")):
        cfg = load_config(path)
        assert "env" in cfg and "experiment_name" in cfg, path


def test_apply_overrides_strict():
    target = {"reward_config": {"scales": {"feet_slip": -0.1}}}
    apply_overrides(target, {"reward_config": {"scales": {"feet_slip": 0}}})
    assert target["reward_config"]["scales"]["feet_slip"] == 0.0
    assert isinstance(target["reward_config"]["scales"]["feet_slip"], float)
    with pytest.raises(KeyError):
        apply_overrides(target, {"reward_config": {"scales": {"feet_slipp": 0.0}}})
