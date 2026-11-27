"""Terrain map, height lookup, curriculum and the Go1Terrain env.

The map/summary tests need only NumPy; the rest is skipped without JAX / mujoco_playground.
"""
import pathlib
import sys

import numpy as np
import pytest

from quadrucat import terrain

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SMALL = dict(types=["rough", "stairs_up", "stairs_down", "boxes", "beam"], type_weights=[1, 1, 1, 1, 1],
             n_levels=3, tile_size=4.0, resolution=0.04, border=1.0, seed=0,
             rough_amplitude=[0.0, 0.1], rough_cell=0.15, step_height=[0.03, 0.14], step_width=0.3,
             platform_size=1.0, box_height=[0.02, 0.14], box_size=[0.4, 1.0], boxes_per_tile=12,
             beam_width=[0.6, 0.15], beam_pit_depth=0.4)


def _h(tm, x, y):
    j = int(round((x - tm.x0) / tm.resolution))
    i = int(round((y - tm.y0) / tm.resolution))
    return float(tm.heights[i, j])


def test_map_layout_and_tiles():
    tm = terrain.build_map(SMALL)
    hx, hy = tm.half_extent
    assert hx == pytest.approx(3 * 4 / 2 + 1, abs=0.05) and hy == pytest.approx(5 * 4 / 2 + 1, abs=0.05)
    # rough level 0 is flat ground
    cx, cy = tm.tile_center(0, 0)
    assert np.abs(tm.heights[np.abs(np.arange(tm.heights.shape[0]) * 0.04 + tm.y0 - cy) < 1.5][
        :, np.abs(np.arange(tm.heights.shape[1]) * 0.04 + tm.x0 - cx) < 1.5]).max() < 1e-6
    # stairs: top platform = k_max * step (k_max = floor((2 - 0.5) / 0.3) = 5), edges at 0
    for level, step in [(0, 0.03), (2, 0.14)]:
        cx, cy = tm.tile_center(1, level)
        assert _h(tm, cx, cy) == pytest.approx(5 * step, abs=1e-5)
        assert _h(tm, cx - 1.98, cy) == pytest.approx(0.0, abs=1e-5)
        cx, cy = tm.tile_center(2, level)
        assert _h(tm, cx, cy) == pytest.approx(-5 * step, abs=1e-5)
    # beam: 0 on the beam, pit beside it; narrower at the hard level
    for level, width in [(0, 0.6), (2, 0.15)]:
        cx, cy = tm.tile_center(4, level)
        assert _h(tm, cx, cy) == 0.0
        assert _h(tm, cx, cy + width / 2 - 0.03) == 0.0
        assert _h(tm, cx, cy + width / 2 + 0.05) == pytest.approx(-0.4)
    # border is flat
    assert _h(tm, tm.x0 + 0.2, tm.y0 + 0.2) == 0.0


def test_hfield_normalization_roundtrip():
    tm = terrain.build_map(SMALL)
    asset, geom, data = terrain.hfield_xml(tm)
    assert data.min() == 0.0 and data.max() == pytest.approx(1.0)
    zrange = float(asset.split('size="')[1].split()[2])
    zmin = float(geom.split('pos="')[1].split('"')[0].split()[2])
    np.testing.assert_allclose(zmin + data * zrange, tm.heights, atol=1e-5)


def test_summary_max_level():
    import evaluate_terrain as et

    rows = [{"type": "beam", "level": lv, "success_rate": s, "fall_rate": 1 - s}
            for lv, s in [(0, 1.0), (3, 0.9), (6, 0.5), (9, 0.9)]]
    out = et.summarize(rows, ["beam"])
    assert out["max_level_beam"] == 3.0       # level 9 does not count: level 6 failed
    assert out["success_beam"] == pytest.approx(0.825)


def test_gazebo_world_matches_map(tmp_path):
    from quadrucat import gazebo_terrain

    flat = ROOT / "ros2" / "quadrucat_deploy" / "worlds" / "flat.sdf"
    world, origin, z0 = gazebo_terrain.write_world(flat, SMALL, "stairs_up", 2, tmp_path)
    text = world.read_text()
    assert "terrain_tile" in text and "<world name=\"flat\">" in text and "Physics" in text
    tm = terrain.build_map(SMALL)
    verts, faces = gazebo_terrain.tile_mesh(tm, "stairs_up", 2)
    # every mesh vertex sits on the training map, shifted by the tile origin
    h = terrain.sample_heights_np(tm.heights, tm.x0, tm.y0, tm.resolution, verts[:, :2] + np.array(origin))
    np.testing.assert_allclose(h, verts[:, 2], atol=1e-5)
    assert verts[:, 0].min() == pytest.approx(-0.4) and faces.max() < len(verts)
    # the robot starts at the Gazebo origin, 0.4 m into the tile: on the first stair (0.14 m)
    assert z0 == pytest.approx(0.14, abs=1e-5)


def test_gazebo_crossing_metrics():
    import analyze_gazebo_terrain as agt

    tm = terrain.build_map(SMALL)
    n = 500
    log = {"segment": np.zeros(n), "fallen": np.zeros(n), "px": np.linspace(0, 3.5, n), "py": np.zeros(n),
           "pz": np.full(n, 0.3), "vx": np.full(n, 0.4)}
    reached, fell, progress, _ = agt.crossing(log, tm, "beam", 2)
    assert reached and not fell and progress == 1.0
    log["py"] = np.where(np.arange(n) > 100, 0.5, 0.0)   # off the beam, trunk at 0.3 over a 0.4 m pit...
    log["pz"] = np.where(np.arange(n) > 120, 0.05, 0.3)  # ...then drops below the beam top
    reached, fell, progress, _ = agt.crossing(log, tm, "beam", 2)
    assert not reached and fell and progress < 0.3


def test_terrain_report(tmp_path):
    import json

    import compare_sims as cs

    rows = [{"type": "beam", "level": 0, "success_rate": 1.0, "fall_rate": 0.0},
            {"type": "beam", "level": 9, "success_rate": 0.25, "fall_rate": 0.75}]
    for version, stem, seed in [("terrain", "eval_terrain_default", 0), ("terrain", "eval_terrain_default", 1),
                                ("v3", "eval_terrain_flat_policy", 0)]:
        d = tmp_path / "versions" / version / "mujoco" / f"s{seed}"
        d.mkdir(parents=True)
        (d / f"{stem}.json").write_text(json.dumps({"per_tile": rows, "mean": {"terrain_success": 0.625}}))
    text = cs.build_report("terrain", tmp_path)
    assert "| beam | 100% (0%) | 25% (75%) |" in text
    assert "Terrain crossing, MuJoCo MJX (n=2)" in text and "baseline: v3 flat cat walk" in text
    cs.build_summary(tmp_path)
    assert "62%" in (tmp_path / "versions" / "summary.md").read_text()
