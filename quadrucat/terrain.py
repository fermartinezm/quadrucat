"""Terrain map for the rough-terrain task (Go1Terrain), as one MuJoCo heightfield.

The map is a grid of square tiles (the "terrain curriculum" of Rudin et al., 2022):

        level 0 (easy)  level 1  ...  level L-1 (hard)      -> x
  row 0   rough           rough          rough
  row 1   stairs_up       ...
  row 2   stairs_down
  row 3   boxes
  row 4   beam            (one continuous beam along x, narrower at every level)
  |
  v y

Every environment owns one terrain type (its row) for the whole training and a difficulty level
(its column) that the curriculum moves up when the robot walks well and down when it falls.
The robots do not see each other (one simulation per env), so thousands share the same tiles.

Pure NumPy (build) + jax.numpy (height lookup), no MuJoCo import: it can be tested without a GPU.
The env (quadrucat/terrain_env.py) copies `heights` into the MuJoCo heightfield and uses
`sample_heights` for the height scan, the terrain-relative foot heights and the termination.
"""
from __future__ import annotations

import dataclasses
from typing import Mapping, Sequence

import numpy as np

TYPES = ("rough", "stairs_up", "stairs_down", "boxes", "beam")

# Default `terrain` section of the Go1Terrain config (quadrucat/terrain_env.py documents each value;
# configs/cat_terrain.yaml repeats them with comments). Pure data so that the Gazebo side can rebuild
# the same map for a policy trained on flat ground.
DEFAULT_PARAMS = dict(
    types=list(TYPES), type_weights=[1.0] * len(TYPES), n_levels=10, tile_size=4.0, resolution=0.04,
    border=3.0, seed=0, rough_amplitude=[0.0, 0.10], rough_cell=0.15, step_height=[0.03, 0.14],
    step_width=0.30, platform_size=1.0, box_height=[0.02, 0.14], box_size=[0.4, 1.0], boxes_per_tile=12,
    beam_width=[0.60, 0.15], beam_pit_depth=0.4,
)

# Go1Terrain env overrides that put every episode on flat ground (rough row, level 0) with the
# commands left as given (no heading controller): the flat evaluations, comparable with v3.
FLAT_TERRAIN = {"terrain": {"eval_type": "rough", "eval_level": 0}, "heading_command": {"enable": False}}
FLAT_TERRAIN_SET = ["env.overrides.terrain.eval_type=rough", "env.overrides.terrain.eval_level=0",
                    "env.overrides.heading_command.enable=false"]


@dataclasses.dataclass(frozen=True)
class TerrainMap:
    heights: np.ndarray          # [nrow (y), ncol (x)] ground height in metres (0 = flat ground)
    resolution: float            # [m] grid spacing
    x0: float                    # world x of heights[:, 0]
    y0: float                    # world y of heights[0, :]
    tile_size: float             # [m]
    n_levels: int
    types: tuple                 # terrain type of each row
    border: float                # [m] flat margin around the tiles

    @property
    def half_extent(self) -> tuple[float, float]:
        """(half size in x, half size in y) of the whole map [m]."""
        nrow, ncol = self.heights.shape
        return (ncol - 1) * self.resolution / 2, (nrow - 1) * self.resolution / 2

    def tile_center(self, row: int, level: int) -> tuple[float, float]:
        return tile_center(row, level, len(self.types), self.n_levels, self.tile_size)


def tile_center(row, level, n_rows: int, n_levels: int, tile_size: float):
    """World xy of the centre of tile (row, level). Works on ints and on jax arrays."""
    return (level - (n_levels - 1) / 2) * tile_size, (row - (n_rows - 1) / 2) * tile_size


def _lerp(rng: Sequence[float], d: float) -> float:
    """Value of a [easy, hard] range at difficulty d in [0, 1]."""
    return float(rng[0] + d * (rng[1] - rng[0]))


# ---------------- one tile per type; local coords u (along x), v (along y) in [-T/2, T/2] ----------------

def _rough(u, v, d, p, rs):
    amp = _lerp(p["rough_amplitude"], d)
    cell = p["rough_cell"]
    n = int(np.ceil(p["tile_size"] / cell)) + 2
    coarse = rs.uniform(-amp / 2, amp / 2, size=(n, n))
    gx = (u + p["tile_size"] / 2) / cell
    gy = (v + p["tile_size"] / 2) / cell
    i0, j0 = np.clip(gy.astype(int), 0, n - 2), np.clip(gx.astype(int), 0, n - 2)
    fy, fx = gy - i0, gx - j0
    return ((1 - fy) * ((1 - fx) * coarse[i0, j0] + fx * coarse[i0, j0 + 1])
            + fy * ((1 - fx) * coarse[i0 + 1, j0] + fx * coarse[i0 + 1, j0 + 1]))


def _pyramid(u, v, d, p, sign):
    step = _lerp(p["step_height"], d)
    w = p["step_width"]
    half = p["tile_size"] / 2
    k_max = int(np.floor((half - p["platform_size"] / 2) / w))
    k = np.clip(np.floor((half - np.maximum(np.abs(u), np.abs(v))) / w), 0, k_max)
    return sign * k * step


def _boxes(u, v, d, p, rs):
    h = np.zeros_like(u)
    hmax = _lerp(p["box_height"], d)
    half = p["tile_size"] / 2
    for _ in range(p["boxes_per_tile"]):
        sx, sy = rs.uniform(*p["box_size"], size=2)
        cx, cy = rs.uniform(-half + sx / 2, half - sx / 2), rs.uniform(-half + sy / 2, half - sy / 2)
        inside = (np.abs(u - cx) < sx / 2) & (np.abs(v - cy) < sy / 2)
        h = np.where(inside, rs.uniform(-hmax, hmax), h)
    pad = p["platform_size"] / 2   # flat spawn platform in the middle
    h = np.where((np.abs(u) < pad) & (np.abs(v) < pad), 0.0, h)
    # tile edges at 0 so neighbouring tiles join without a cliff
    edge = 0.1
    return np.where((np.abs(u) > half - edge) | (np.abs(v) > half - edge), 0.0, h)


def _beam(u, v, d, p):
    width = _lerp(p["beam_width"], d)
    return np.where(np.abs(v) <= width / 2, 0.0, -p["beam_pit_depth"])


def build_map(params: Mapping, seed: int | None = None) -> TerrainMap:
    """Build the whole map from the `terrain` section of the env config (a dict or ConfigDict)."""
    p = {k: params[k] for k in params.keys()}
    types = tuple(p["types"])
    for t in types:
        if t not in TYPES:
            raise ValueError(f"unknown terrain type {t!r}; known: {TYPES}")
    L, T, res, border = int(p["n_levels"]), float(p["tile_size"]), float(p["resolution"]), float(p["border"])
    R = len(types)
    rs = np.random.RandomState(p["seed"] if seed is None else seed)
    half_x, half_y = L * T / 2 + border, R * T / 2 + border
    ncol, nrow = int(round(2 * half_x / res)) + 1, int(round(2 * half_y / res)) + 1
    xs = -half_x + res * np.arange(ncol)
    ys = -half_y + res * np.arange(nrow)
    X, Y = np.meshgrid(xs, ys)               # [nrow, ncol]
    H = np.zeros((nrow, ncol))
    for row, t in enumerate(types):
        for level in range(L):
            d = level / max(L - 1, 1)
            cx, cy = tile_center(row, level, R, L, T)
            sel = (np.abs(X - cx) <= T / 2) & (np.abs(Y - cy) <= T / 2)
            u, v = X[sel] - cx, Y[sel] - cy
            if t == "rough":
                h = _rough(u, v, d, p, rs)
            elif t == "stairs_up":
                h = _pyramid(u, v, d, p, +1.0)
            elif t == "stairs_down":
                h = _pyramid(u, v, d, p, -1.0)
            elif t == "boxes":
                h = _boxes(u, v, d, p, rs)
            else:
                h = _beam(u, v, d, p)
            H[sel] = h
    return TerrainMap(H.astype(np.float32), res, float(xs[0]), float(ys[0]), T, L, types, border)


# ---------------- height lookup (jax.numpy, also works with numpy arrays) ----------------

def sample_heights(heights, x0: float, y0: float, resolution: float, xy):
    """Bilinear ground height at world points xy [..., 2]. Outside the map: the nearest edge value.

    MuJoCo draws each grid cell as two triangles, so at sharp edges (stairs, beams) the physics and
    this lookup differ by at most one grid cell horizontally.
    """
    import jax.numpy as jp

    return _bilinear(jp, heights, x0, y0, resolution, xy)


def sample_heights_np(heights, x0: float, y0: float, resolution: float, xy):
    """NumPy version of sample_heights (deployment: Gazebo / ROS node, no JAX)."""
    return _bilinear(np, heights, x0, y0, resolution, np.asarray(xy, dtype=np.float64))


def _bilinear(jp, heights, x0, y0, resolution, xy):
    H = jp.asarray(heights)
    nrow, ncol = H.shape
    gx = (xy[..., 0] - x0) / resolution
    gy = (xy[..., 1] - y0) / resolution
    j0 = jp.clip(jp.floor(gx), 0, ncol - 2).astype(jp.int32)
    i0 = jp.clip(jp.floor(gy), 0, nrow - 2).astype(jp.int32)
    fx = jp.clip(gx - j0, 0.0, 1.0)
    fy = jp.clip(gy - i0, 0.0, 1.0)
    h00, h01 = H[i0, j0], H[i0, j0 + 1]
    h10, h11 = H[i0 + 1, j0], H[i0 + 1, j0 + 1]
    return (1 - fy) * ((1 - fx) * h00 + fx * h01) + fy * ((1 - fx) * h10 + fx * h11)


def scan_grid(x_range: Sequence[float], y_range: Sequence[float]) -> np.ndarray:
    """Height-scan points in the yaw-aligned body frame, [n, 2]. Ranges are [min, max, count]."""
    xs = np.linspace(x_range[0], x_range[1], int(x_range[2]))
    ys = np.linspace(y_range[0], y_range[1], int(y_range[2]))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], -1).astype(np.float32)


def height_scan_np(tmap: "TerrainMap", trunk_pos, yaw: float, scan_pts, offset: float) -> np.ndarray:
    """Same values as Go1Terrain.height_scan (clean, no noise), from the trunk pose: for deployment."""
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.asarray(trunk_pos[:2]) + np.asarray(scan_pts) @ np.array([[c, -s], [s, c]]).T
    h = sample_heights_np(tmap.heights, tmap.x0, tmap.y0, tmap.resolution, pts)
    return np.clip(trunk_pos[2] - h - offset, -1.0, 1.0)


def hfield_xml(tmap: TerrainMap, name: str = "terrain") -> tuple[str, str, np.ndarray]:
    """(<hfield> asset line, <geom> line, normalized data) for the MuJoCo scene.

    MuJoCo stores elevation in [0, 1] times size_z, starting at the geom's z. Row 0 of the data is
    the row with the smallest y (MuJoCo convention for user data), like `heights`.
    """
    H = tmap.heights
    hmin, hmax = float(H.min()), float(H.max())
    zrange = max(hmax - hmin, 1e-3)
    data = (H - hmin) / zrange
    sx, sy = tmap.half_extent
    nrow, ncol = H.shape
    asset = (f'<hfield name="{name}" nrow="{nrow}" ncol="{ncol}" '
             f'size="{sx:.6f} {sy:.6f} {zrange:.6f} 0.1"/>')
    cx, cy = tmap.x0 + sx, tmap.y0 + sy
    geom = (f'<geom name="floor" type="hfield" hfield="{name}" pos="{cx:.6f} {cy:.6f} {hmin:.6f}" '
            f'material="groundplane" contype="1" conaffinity="0" priority="1" friction="1.0" condim="3"/>')
    return asset, geom, data.astype(np.float32)
