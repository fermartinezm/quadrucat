"""Terrain crossing in Gazebo: same metrics as scripts/evaluate_terrain.py, from policy-node logs.

Each log is one crossing of one tile, run with
  ros2 launch quadrucat_deploy gazebo.launch.py policy_path:=... headless:=true pd:=gazebo \
      terrain_type:=beam terrain_level:=3 eval_commands:="[0.4,0,0]" eval_segment_s:=12 log_csv:=/tmp/beam3_r0.csv
(or all tiles at once with ros2/terrain_eval.sh). The tile and level are read from the file name
`<type>_l<level>_*.csv`, or given with --type/--level for every log.

  python scripts/analyze_gazebo_terrain.py /tmp/gz_terrain/*.csv --run-dir runs/<run>
  python scripts/compare_sims.py collect runs/<run>

A crossing succeeds when the trunk gets `goal` metres forward (tile minus 0.8 m, as in MuJoCo) while
upright; it is a fall if the robot flips (the node's `fallen`) or its trunk comes closer than
termination_height to the ground under it (same rule as the training env), before reaching the goal.
Writes runs/<run>/eval_terrain/gazebo_terrain.json (per_tile rows like evaluate_terrain.py).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_gazebo_log import load_log  # noqa: E402
from evaluate_terrain import summarize  # noqa: E402
from quadrucat import gazebo_terrain, terrain  # noqa: E402
from quadrucat.config import load_config  # noqa: E402

NAME = re.compile(r"(?P<type>[a-z_]+?)_l(?P<level>\d+)")


def crossing(log: dict, tmap: terrain.TerrainMap, type_name: str, level: int, termination_height: float = 0.05,
             beam_fall_height: float = 0.12):
    """(reached, fell, progress, vx) of one log."""
    keep = log["segment"] >= 0
    log = {k: v[keep] for k, v in log.items()}
    ox, oy = gazebo_terrain.tile_origin(tmap, type_name, level)
    goal = tmap.tile_size - 0.8 - 0.1
    xy = np.stack([log["px"] + ox, log["py"] + oy], -1)
    window = np.array([[0, 0], [0.25, 0.15], [0.25, -0.15], [-0.25, 0.15], [-0.25, -0.15],
                       [0.15, 0], [-0.15, 0], [0, 0.1], [0, -0.1]])
    highest = np.max(terrain.sample_heights_np(tmap.heights, tmap.x0, tmap.y0, tmap.resolution,
                                               xy[:, None, :] + window[None]), axis=1)
    down = (log["fallen"] > 0) | (log["pz"] - highest < termination_height)
    if type_name == "beam":     # same rule as the training env: trunk below the beam top + margin
        down |= log["pz"] < beam_fall_height
    dx = log["px"] - log["px"][0]
    first_down = int(np.argmax(down)) if down.any() else len(dx)
    alive_dx = dx[:first_down]
    reached = bool(alive_dx.size and alive_dx.max() >= goal)
    progress = float(np.clip(alive_dx.max() / goal, 0, 1)) if alive_dx.size else 0.0
    vx = log["vx"][:first_down]
    return reached, (not reached) and down.any(), progress, vx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--run-dir", required=True, help="the run whose policy was evaluated (its config gives the map)")
    ap.add_argument("--type")
    ap.add_argument("--level", type=int)
    ap.add_argument("--tag", default="gazebo_terrain")
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    params = dict(terrain.DEFAULT_PARAMS)
    params.update(((cfg["env"].get("overrides") or {}).get("terrain") or {}))
    tmap = terrain.build_map(params)
    over = cfg["env"].get("overrides") or {}
    term_h = float(over.get("termination_height", 0.05))
    beam_h = float(over.get("beam_fall_height", 0.12))
    speed = None
    groups: dict[tuple, list] = {}
    for path in map(pathlib.Path, args.csv):
        m = NAME.search(path.stem)
        t = args.type or (m and m.group("type"))
        lv = args.level if args.level is not None else (m and int(m.group("level")))
        if t is None or lv is None:
            raise SystemExit(f"{path}: give --type/--level or name the file <type>_l<level>_*.csv")
        log = load_log(path)
        speed = float(log["cmd_vx"][log["segment"] >= 0][0])
        groups.setdefault((t, int(lv)), []).append(crossing(log, tmap, t, int(lv), term_h, beam_h))
    rows = []
    for (t, lv), res in sorted(groups.items(), key=lambda kv: (list(tmap.types).index(kv[0][0]), kv[0][1])):
        vx = np.concatenate([r[3] for r in res]) if res else np.zeros(0)
        rows.append({"type": t, "level": lv, "episodes": len(res),
                     "success_rate": float(np.mean([r[0] for r in res])),
                     "fall_rate": float(np.mean([r[1] for r in res])),
                     "progress": float(np.mean([r[2] for r in res])),
                     "lin_vel_rmse": float(np.sqrt(np.mean((vx[50:] - speed) ** 2))) if vx.size > 50 else float("nan")})
        r = rows[-1]
        print(f"[gazebo-terrain] {t:12s} level {lv}: success {r['success_rate']:.2f}  falls {r['fall_rate']:.2f}  "
              f"progress {r['progress']:.2f}  (n={len(res)})")
    types = list(dict.fromkeys(r["type"] for r in rows))
    out = {"tag": args.tag, "speed": speed, "per_tile": rows, "mean": summarize(rows, types),
           "source": [pathlib.Path(c).name for c in args.csv]}
    out_dir = run_dir / "eval_terrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"[gazebo-terrain] -> {out_dir / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
