"""Go1CatWalk / Go1CatFall: new reward terms and their bookkeeping.

Skipped when mujoco_playground is not installed. Run on CPU (impl=jax).
"""
import pathlib

import numpy as np
import pytest

pytest.importorskip("mujoco_playground")
jax = pytest.importorskip("jax")
import jax.numpy as jp  # noqa: E402

from quadrucat import cat_envs  # noqa: E402
from quadrucat.config import apply_overrides, load_config  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _env(name, **overrides):
    from mujoco_playground import registry

    cat_envs.register()
    cfg = registry.get_default_config(name)
    cfg.impl = "jax"
    apply_overrides(cfg, overrides)
    return registry.load(name, config=cfg)


# ---------------- pure helpers ----------------

def test_register_reward_and_front_print():
    hind = jp.array([[0.0, -0.1], [0.5, 0.1]])
    prints = jp.array([[0.0, -0.1], [0.4, 0.1]])
    r, d = cat_envs.register_reward(hind, prints, jp.array([1.0, 1.0]), sigma=0.05)
    np.testing.assert_allclose(d, [0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(r, 1.0 + np.exp(-4.0), atol=1e-6)
    r, _ = cat_envs.register_reward(hind, prints, jp.array([0.0, 0.0]), sigma=0.05)
    assert float(r) == 0.0
    # print follows the foot in stance, freezes in swing (= the spot the foot vacated)
    p, v = cat_envs.update_front_print(prints, jp.array([False, True]), jp.array([[1.0, 1.0], [2.0, 2.0]]),
                                       jp.array([True, False]))
    np.testing.assert_allclose(p, [[1.0, 1.0], [0.4, 0.1]])
    assert list(np.array(v)) == [True, True]


def test_random_tilt_distribution():
    keys = jax.random.split(jax.random.PRNGKey(0), 2000)
    quat, tilt = jax.vmap(lambda k: cat_envs.random_tilted_quat(k, 1.0, 0.3))(keys)
    tilt = np.array(tilt)
    assert tilt.min() >= 0.3 - 1e-5 and tilt.max() <= 1.0 + 1e-5
    # the tilt is the angle between body z and world z: R[2,2] = cos(tilt)
    w, x, y, z = np.array(quat).T
    np.testing.assert_allclose(1 - 2 * (x * x + y * y), np.cos(tilt), atol=1e-5)


def test_trunk_lowest_z_matches_brute_force():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(0)
    for _ in range(20):
        rot = Rotation.random(random_state=rng).as_matrix()
        pos = np.array([0.0, 0.0, 0.5])
        # sample the surface of both trunk cylinders (axis = body x)
        th = np.linspace(0, 2 * np.pi, 361)
        s = np.linspace(-cat_envs._TRUNK_CYL_HALF_LEN, cat_envs._TRUNK_CYL_HALF_LEN, 101)
        pts = []
        for yc in cat_envs._TRUNK_CYL_Y:
            S, TH = np.meshgrid(s, th)
            body = np.stack([S, yc + cat_envs._TRUNK_CYL_RADIUS * np.cos(TH),
                             cat_envs._TRUNK_CYL_RADIUS * np.sin(TH)], -1).reshape(-1, 3)
            pts.append(body @ rot.T + pos)
        brute = np.concatenate(pts)[:, 2].min()
        assert abs(float(cat_envs.trunk_lowest_z(jp.array(pos), jp.array(rot))) - brute) < 2e-4


def test_trunk_geometry_constants_match_model():
    env = _env("Go1CatFall")
    m = env.mj_model
    trunk = m.body("trunk").id
    cyl = [g for g in range(m.ngeom) if m.geom_bodyid[g] == trunk and m.geom_type[g] == 5 and m.geom_contype[g]]
    assert len(cyl) == 2
    np.testing.assert_allclose(sorted(m.geom_pos[g][1] for g in cyl), cat_envs._TRUNK_CYL_Y, atol=1e-9)
    for g in cyl:
        np.testing.assert_allclose(m.geom_size[g][:2], [cat_envs._TRUNK_CYL_RADIUS, cat_envs._TRUNK_CYL_HALF_LEN])


# ---------------- Go1CatWalk ----------------

def test_catwalk_with_cat_terms_off_equals_joystick():
    """All new scales are 0 by default: same rewards and actor obs as the base task."""
    cat = _env("Go1CatWalk", register_in_privileged=False)
    base = _env("Go1JoystickFlatTerrain")
    s1, s2 = jax.jit(cat.reset)(jax.random.PRNGKey(1)), jax.jit(base.reset)(jax.random.PRNGKey(1))
    st1, st2 = jax.jit(cat.step), jax.jit(base.step)
    rng = np.random.default_rng(0)
    for _ in range(40):
        a = jp.array(rng.uniform(-0.5, 0.5, 12), dtype=jp.float32)
        s1, s2 = st1(s1, a), st2(s2, a)
        np.testing.assert_allclose(s1.reward, s2.reward, atol=1e-5)
        np.testing.assert_allclose(s1.obs["state"], s2.obs["state"], atol=1e-5)
    assert cat.observation_size["state"] == base.observation_size["state"] == (48,)


def test_catwalk_register_bookkeeping():
    env = _env("Go1CatWalk", reward_config={"scales": {"direct_register": 1.0}})
    assert env.observation_size["privileged_state"][0] == base_priv_size() + 6
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state.info["command"] = jp.array([0.5, 0.0, 0.0])
    step = jax.jit(env.step)
    rng = np.random.default_rng(1)
    n_td = 0
    for _ in range(60):
        state = step(state, jp.array(rng.uniform(-1, 1, 12), dtype=jp.float32))
        state.info["command"] = jp.array([0.5, 0.0, 0.0])
        td = np.array(state.info["hind_touchdown"])
        if td.any():
            n_td += 1
            d = np.array(state.info["register_dist"])
            expected = np.sum(np.exp(-(d / 0.05) ** 2) * td)
            np.testing.assert_allclose(float(state.metrics["reward/direct_register"]), expected, atol=1e-5)
        else:
            assert float(state.metrics["reward/direct_register"]) == 0.0
    assert n_td > 0, "random actions should produce some hind touchdowns"
    # front prints are where the front feet are while they are on the ground
    feet = np.array(state.data.site_xpos[env._feet_site_id])
    contact = np.array(state.info["last_contact"])
    for i in range(2):
        if contact[i]:
            np.testing.assert_allclose(np.array(state.info["front_print"])[i], feet[i, :2], atol=1e-6)


def base_priv_size():
    return _env("Go1JoystickFlatTerrain").observation_size["privileged_state"][0]


# ---------------- Go1CatFall ----------------

def _drop(env, n=16, steps=60, seed=0):
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    state = reset(jax.random.split(jax.random.PRNGKey(seed), n))
    zero = jp.zeros((n, env.action_size))
    landed_on_feet = np.zeros(n)
    trunk = np.zeros(n)
    for _ in range(steps):
        state = step(state, zero)
        landed_on_feet += np.array(state.info["feet_only_touch"])
        trunk += np.array(state.info["trunk_touch"])
    return state, landed_on_feet, trunk

def test_catfall_upright_drop_lands_on_feet():
    # action_mode="default": zero action holds the home pose (in "delta" mode zero action = limp legs)
    env = _env("Go1CatFall", max_tilt=0.0, init_angvel=0.0, init_linvel=0.0, joint_init="default",
               drop_height=[0.6, 0.6], noise_config={"level": 0.0}, action_mode="default")
    state, feet, trunk = _drop(env, n=4)
    assert np.all(feet == 1), feet          # exactly one first touch, and it was feet only
    assert np.all(trunk == 0)
    assert np.all(np.array(state.info["landed"]))
    assert np.all(np.array(state.info["touch_up_z"]) > 0.95)


def test_catfall_upside_down_drop_hits_trunk():
    env = _env("Go1CatFall", max_tilt=3.1416, min_tilt=3.1, init_angvel=0.0, init_linvel=0.0,
               joint_init="default", drop_height=[0.6, 0.6], noise_config={"level": 0.0},
               action_mode="default")  # zero action = hold the home pose (in "delta" mode it is limp)
    state, feet, trunk = _drop(env, n=4)
    assert np.all(feet == 0) and np.all(trunk > 0)
    assert np.all(np.array(state.info["touch_up_z"]) < -0.9)


def test_catfall_observation_options():
    blind = _env("Go1CatFall", observe_height=False, observe_accelerometer=False)
    full = _env("Go1CatFall", observe_height=True, observe_accelerometer=True)
    assert blind.observation_size["state"] == (42,)
    assert full.observation_size["state"] == (46,)
    s = jax.jit(full.reset)(jax.random.PRNGKey(0))
    assert s.obs["state"].shape == (46,) and s.obs["privileged_state"].shape == full.observation_size["privileged_state"]


# ---------------- configs ----------------

@pytest.mark.parametrize("path", sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").rglob("*.yaml")))
def test_config_env_overrides_are_valid(path):
    """Every config's env overrides exist in its env (catches typos in reward names, etc.)."""
    from quadrucat import envs

    cfg = load_config(ROOT / path)
    cfg["env"].setdefault("overrides", {})["impl"] = "jax"
    env, _ = envs.make_env(cfg)
    assert env.action_size == 12
