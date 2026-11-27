"""Go1Terrain with JAX: height lookup, curriculum, reset and auto-reset relocation.

Skipped without JAX / mujoco_playground (runs on Kaggle and in a Linux/WSL env). CPU: impl=jax.
"""
import numpy as np
import pytest

from quadrucat import terrain
from test_terrain import SMALL

jax = pytest.importorskip("jax", exc_type=ImportError)
import jax.numpy as jp  # noqa: E402


def test_sample_heights_bilinear():
    tm = terrain.build_map(SMALL)
    i, j = 100, 150
    x, y = tm.x0 + j * tm.resolution, tm.y0 + i * tm.resolution
    pts = jp.array([[x, y], [x + 0.02, y], [x + 0.02, y + 0.02]])
    h = np.asarray(terrain.sample_heights(tm.heights, tm.x0, tm.y0, tm.resolution, pts))
    H = tm.heights
    assert h[0] == pytest.approx(H[i, j], abs=1e-6)
    assert h[1] == pytest.approx(0.5 * (H[i, j] + H[i, j + 1]), abs=1e-6)
    assert h[2] == pytest.approx(0.25 * (H[i, j] + H[i, j + 1] + H[i + 1, j] + H[i + 1, j + 1]), abs=1e-6)


def test_curriculum_update():
    from quadrucat.terrain_env import curriculum_update

    k = jax.random.PRNGKey(0)
    f = lambda lvl, fell, dist, cdist: int(curriculum_update(jp.int32(lvl), jp.bool_(fell), jp.float32(dist),
                                                            jp.float32(cdist), k, 10, 1.0, 0.5, 0.25))
    assert f(3, False, 3.0, 4.0) == 4      # walked 75% of the command: up
    assert f(3, True, 3.0, 4.0) == 2       # fell: down
    assert f(3, False, 0.5, 4.0) == 2      # walked 12%: down
    assert f(3, False, 1.5, 4.0) == 3      # 37%: stay
    assert f(3, False, 0.0, 0.5) == 3      # (almost) no command: stay
    assert f(0, True, 0.0, 4.0) == 0       # clipped at 0
    assert 0 <= f(9, False, 4.0, 4.0) <= 9  # passed the last level: restart at a random level


pytest.importorskip("mujoco_playground", exc_type=ImportError)


@pytest.fixture(scope="module")
def env():
    from mujoco_playground import registry

    from quadrucat import cat_envs
    from quadrucat.config import apply_overrides

    cat_envs.register()
    cfg = registry.get_default_config("Go1Terrain")
    cfg.impl = "jax"
    apply_overrides(cfg, {"terrain": {k: v for k, v in SMALL.items()}, "episode_length": 20})
    return registry.load("Go1Terrain", config=cfg)


def test_lookup_matches_mujoco_heightfield(env):
    import mujoco

    m, d = env.mj_model, mujoco.MjData(env.mj_model)
    mujoco.mj_forward(m, d)
    rs = np.random.RandomState(0)
    hx, hy = env._tmap.half_extent
    pts = np.stack([rs.uniform(-hx + 0.5, hx - 0.5, 200), rs.uniform(-hy + 0.5, hy - 0.5, 200)], -1)
    mine = np.asarray(env.ground(jp.asarray(pts)))
    gid = np.zeros(1, np.int32)
    ray = np.array([5.0 - mujoco.mj_ray(m, d, np.array([x, y, 5.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
                    for x, y in pts])
    assert np.median(np.abs(ray - mine)) < 1e-3


def test_reset_on_tile_and_obs(env):
    state = jax.jit(env.reset_at)(jax.random.PRNGKey(1), 4, 2)      # beam, hardest level
    cx, cy = env._tmap.tile_center(4, 2)
    q = np.asarray(state.data.qpos)
    assert abs(q[1] - cy) < 1e-4 and abs(q[0] - cx) <= 1.5 + 1e-4
    assert q[2] == pytest.approx(0.298, abs=0.01)                    # beam top is at 0
    n_scan = 13 * 7
    assert state.obs["state"].shape == (48 + n_scan,)
    np.testing.assert_allclose(state.obs["state"][45:48], state.info["command"], atol=1e-6)
    assert float(state.info["command"][1]) == 0.0 and 0.2 <= float(state.info["command"][0]) <= 0.6


def test_auto_reset_moves_robot_and_updates_level(env):
    from mujoco_playground import wrapper

    wenv = wrapper.wrap_for_brax_training(env, episode_length=20)
    reset, step = jax.jit(wenv.reset), jax.jit(wenv.step)
    state = reset(jax.random.split(jax.random.PRNGKey(0), 2))
    act = jp.zeros((2, 12))
    for _ in range(21):          # time limit at 20 steps -> cached first state -> relocated on the next step
        state = step(state, act)
    lvl = np.asarray(state.info["terrain_level"])
    assert np.all((lvl >= 0) & (lvl <= 2))
    assert np.all(np.asarray(state.data.time) > 0)
    # robot is on a tile of its (row, level), not at the cached start
    for e in range(2):
        cx, cy = env._tmap.tile_center(int(state.info["terrain_row"][e]), int(lvl[e]))
        xy = np.asarray(state.data.qpos[e, :2])
        assert np.all(np.abs(xy - [cx, cy]) <= 2.0 + 0.3)
