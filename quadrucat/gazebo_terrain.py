"""Gazebo world with one terrain tile of the training map, for the terrain crossing test.

The tile is exported as a triangle mesh with exactly the vertices of the MuJoCo heightfield
(MuJoCo also collides with two triangles per grid cell), so both simulators see the same ground.
Layout, in the Gazebo world frame (the robot spawns at the origin, facing +x):

    flat run-up (z = 0)  |  tile: x in [-0.4, T - 0.4], y in [-T/2, T/2]  |  flat run-out (z = 0)

Everything else is a ground plane at the lowest point of the tile (the beam pits, the bottom of the
stairs_down pyramid), so a robot that leaves the tile sideways falls there.
`tile_origin` (map xy of the Gazebo origin) is what the policy node adds to the odometry position
to compute the height scan on the same map as in training (quadrucat.terrain.height_scan_np).

Pure NumPy + string templates (no ROS), called by ros2/.../launch/gazebo.launch.py when
terrain_type is set, and testable offline.
"""
from __future__ import annotations

import pathlib
import re
from typing import Mapping

import numpy as np

from quadrucat import terrain

START_INSET = 0.4   # the robot starts this far inside the tile (as scripts/evaluate_terrain.py)


def tile_origin(tmap: terrain.TerrainMap, type_name: str, level: int) -> tuple[float, float]:
    """Map xy of the robot's start point = Gazebo world origin."""
    row = list(tmap.types).index(type_name)
    cx, cy = tmap.tile_center(row, level)
    return cx - tmap.tile_size / 2 + START_INSET, cy


def tile_mesh(tmap: terrain.TerrainMap, type_name: str, level: int):
    """(vertices [n, 3], faces [m, 3]) of the tile in the Gazebo frame."""
    row = list(tmap.types).index(type_name)
    cx, cy = tmap.tile_center(row, level)
    T, r = tmap.tile_size, tmap.resolution
    j0 = int(round((cx - T / 2 - tmap.x0) / r))
    i0 = int(round((cy - T / 2 - tmap.y0) / r))
    n = int(round(T / r))
    H = tmap.heights[i0:i0 + n + 1, j0:j0 + n + 1].astype(np.float64)
    ox, oy = tile_origin(tmap, type_name, level)
    xs = tmap.x0 + r * (j0 + np.arange(n + 1)) - ox
    ys = tmap.y0 + r * (i0 + np.arange(n + 1)) - oy
    X, Y = np.meshgrid(xs, ys)
    verts = np.stack([X.ravel(), Y.ravel(), H.ravel()], -1)
    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, d], -1), np.stack([a, d, c], -1)])   # counter-clockwise from above
    return verts, faces


def write_stl(path: pathlib.Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Binary STL (facet normals included: DART's ODE mesh loader crashes on an OBJ without normals)."""
    tri = verts[faces].astype(np.float32)                                # [m, 3, 3]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(b"quadrucat terrain tile".ljust(80, b" "))
        f.write(np.uint32(len(faces)).tobytes())
        f.write(rec.tobytes())


def _box(name: str, x: float, y: float, z_top: float, sx: float, sy: float, mu: float) -> str:
    sz = 1.0
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z_top - sz / 2:.4f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz}</size></box></geometry>
          <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz}</size></box></geometry>
          <material><ambient>0.75 0.75 0.72 1</ambient><diffuse>0.75 0.75 0.72 1</diffuse></material>
        </visual>
      </link>
    </model>"""


def terrain_models(mesh_path: pathlib.Path, tmap: terrain.TerrainMap, type_name: str, level: int,
                   mu: float = 0.6) -> str:
    T = tmap.tile_size
    verts, _ = tile_mesh(tmap, type_name, level)
    floor = float(verts[:, 2].min()) - 0.02
    width = 2.0   # run-up/out platforms; the map border in training is flat ground too
    mesh = f"""
    <model name="terrain_tile">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><mesh><uri>file://{mesh_path.as_posix()}</uri></mesh></geometry>
          <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><mesh><uri>file://{mesh_path.as_posix()}</uri></mesh></geometry>
          <material><ambient>0.7 0.68 0.62 1</ambient><diffuse>0.7 0.68 0.62 1</diffuse></material>
        </visual>
      </link>
    </model>"""
    run_up = _box("run_up", -START_INSET - 1.5, 0.0, 0.0, 3.0, width, mu)
    run_out = _box("run_out", T - START_INSET + 1.5, 0.0, 0.0, 3.0, width, mu)
    ground = f"""
    <model name="ground_plane">
      <static>true</static>
      <pose>0 0 {floor:.4f} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><ambient>0.5 0.5 0.5 1</ambient><diffuse>0.5 0.5 0.5 1</diffuse></material>
        </visual>
      </link>
    </model>"""
    return mesh + run_up + run_out + ground


def write_world(flat_sdf: pathlib.Path, terrain_params: Mapping, type_name: str, level: int,
                out_dir: pathlib.Path) -> tuple[pathlib.Path, tuple[float, float], float]:
    """World file = flat.sdf (same physics, plugins, world name) with the ground replaced by the tile.

    Returns (world path, tile_origin, ground height at the origin: add it to the spawn height).
    """
    tmap = terrain.build_map(terrain_params)
    if type_name not in tmap.types:
        raise ValueError(f"terrain_type {type_name!r} not in {list(tmap.types)}")
    level = int(min(max(level, 0), tmap.n_levels - 1))
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / f"terrain_{type_name}_l{level}.stl"
    write_stl(mesh_path, *tile_mesh(tmap, type_name, level))
    sdf = pathlib.Path(flat_sdf).read_text(encoding="utf-8")
    models = terrain_models(mesh_path.resolve(), tmap, type_name, level)
    new, n = re.subn(r"\s*<model name=\"ground_plane\">.*?</model>", models, sdf, count=1, flags=re.S)
    if n != 1:
        raise ValueError(f"no ground_plane model in {flat_sdf}")
    world = out_dir / f"terrain_{type_name}_l{level}.sdf"
    world.write_text(new, encoding="utf-8")
    origin = tile_origin(tmap, type_name, level)
    z0 = float(terrain.sample_heights_np(tmap.heights, tmap.x0, tmap.y0, tmap.resolution, np.array(origin)))
    return world, origin, z0
