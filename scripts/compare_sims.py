"""MuJoCo vs Gazebo comparison per policy version (versions/<v>/{mujoco,gazebo}/s<seed>/).

  python scripts/compare_sims.py collect runs/<run> [runs/<run> ...]   # copy eval results, rebuild reports
  python scripts/compare_sims.py report [v3 ...]                       # rebuild reports only

`collect` reads the run's config.yaml (experiment_name, seed) to find the version, copies
runs/<run>/eval/*.json and eval_cat/*.json (gazebo*.json -> gazebo/, the rest -> mujoco/), and writes
versions/<v>/report.md plus versions/summary.md. Gazebo results come from
`scripts/analyze_gazebo_log.py <csv> --run-dir runs/<run> --tag gazebo` (or --tag gazebo_pushes).
Every number is mean ± std over the seeds that have it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import statistics
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

# experiment_name -> version folder (and what the version is)
VERSIONS = {
    "go1_ppo_base": ("base", "Go1 velocity tracking (configs/base.yaml)"),
    "go1_catwalk": ("v1", "cat walk v1 (configs/cat_walk.yaml)"),
    "go1_catwalk_v2": ("v2", "cat walk v2 (configs/cat_walk_v2.yaml)"),
    "go1_catwalk_v3": ("v3", "cat walk v3, the cat walk baseline (configs/cat_walk_v3.yaml)"),
    "go1_cat_terrain": ("terrain", "cat walk on rough ground, stairs, drops and beams (configs/cat_terrain.yaml)"),
    "go1_cat_terrain_blind": ("terrain_blind", "terrain ablation: no height scan (experiments/terrain_blind.yaml)"),
    "go1_cat_terrain_no_curriculum": ("terrain_no_curriculum",
                                      "terrain ablation: no curriculum (experiments/terrain_no_curriculum.yaml)"),
    "go1_cat_terrain_no_cat": ("terrain_no_cat", "terrain ablation: no cat gait terms (experiments/terrain_no_cat.yaml)"),
    "go1_cat_terrain_no_leg_collision": ("terrain_no_leg_collision",
                                         "terrain ablation: no leg collision penalty "
                                         "(experiments/terrain_no_leg_collision.yaml)"),
    "go1_cat_terrain_no_dr": ("terrain_no_dr", "terrain ablation: no domain randomization (experiments/terrain_no_dr.yaml)"),
    "go1_catfall": ("fall", "cat fall: land on the feet and stand up (configs/cat_fall.yaml)"),
}

# Terrain crossing (scripts/evaluate_terrain.py, Gazebo: analyze_gazebo_log.py --tag gazebo_terrain)
# (label, version folder or None = this one, simulator, file stem). The v3 row is the baseline: the
# flat-ground cat walk on the same map (`evaluate_terrain.py runs/<v3 run> --tag flat_policy`).
TERRAIN_SOURCES = [("MuJoCo MJX", None, "mujoco", "eval_terrain_default"),
                   ("Gazebo", None, "gazebo", "gazebo_terrain"),
                   ("baseline: v3 flat cat walk, MuJoCo MJX", "v3", "mujoco", "eval_terrain_flat_policy")]
TERRAIN_TYPES = ["rough", "stairs_up", "stairs_down", "boxes", "beam"]
FALL_METRICS = [("land_on_feet_rate", "lands on its feet", "{:.0%}"),
                ("feet_first_rate", "feet touch first", "{:.0%}"),
                ("trunk_hit_rate", "trunk hits the ground", "{:.0%}"),
                ("peak_acc_g", "peak acceleration [g]", "{:.1f}"),
                ("upright_end_rate", "standing at the end", "{:.0%}"),
                ("time_to_stand_s", "time to stand [s]", "{:.2f}")]

# (column, simulator folder, file stem in s<seed>/) -> which file holds that condition
CONDITIONS = [
    ("MuJoCo MJX", "mujoco", ["eval_default", "eval_cat_default"]),
    ("MuJoCo C (sim2sim)", "mujoco", ["eval_sim2sim_effort"]),
    ("Gazebo", "gazebo", ["gazebo_pd"]),
    ("Gazebo, PD over ROS", "gazebo", ["gazebo"]),
    ("Gazebo, PD over ROS, 1/4 speed", "gazebo", ["gazebo_slow"]),
]
PUSH_CONDITIONS = [
    ("MuJoCo MJX", "mujoco", ["eval_pushes", "eval_cat_pushes"]),
    ("Gazebo", "gazebo", ["gazebo_pd_pushes"]),
    ("Gazebo, PD over ROS", "gazebo", ["gazebo_pushes"]),
    ("Gazebo, PD over ROS, 1/4 speed", "gazebo", ["gazebo_pushes_slow"]),
]
GAZEBO_NOTE = (
    "'Gazebo': the policy node (ROS 2, 50 Hz) sends joint targets and Gazebo runs the PD (Kp 35, "
    "Kd 0.5, torque limits as in training) at every physics step, like the Go1's motor drivers "
    "(launch `pd:=gazebo`). The 'PD over ROS' columns instead compute the PD torques in the Python "
    "node at 500 Hz and send them through ros2_control. With the physics as fast as the PC allows "
    "(0.6-0.8x real time on the WSL laptop) the nodes compete with Gazebo for CPU and those torques "
    "arrive late; the '1/4 speed' columns cap the physics (`sim_rate_hz:=250`) so the same nodes keep "
    "up, which separates the physics gap from the communication delay. Each Gazebo seed is 3 "
    "repetitions at full speed (1 at 1/4 speed); the robot is not reset between commands, so after a fall the "
    "remaining commands count as falls. Feet count as on the ground when the foot centre is < 2.8 cm high "
    "(forward kinematics), MuJoCo uses its contact sensors, so the gait rows compare close but not "
    "identical definitions.")
METRICS = [
    ("fall_rate", "falls", "{:.0%}"),
    ("lin_vel_rmse", "lin. vel. error [m/s]", "{:.3f}"),
    ("yaw_rate_rmse", "yaw rate error [rad/s]", "{:.3f}"),
    ("register_err_median", "register error [cm]", "{:.1f}", 100.0),
    ("register_hit_rate", "register hits < 5 cm", "{:.0%}"),
    ("three_foot_frac", "3 feet on the ground", "{:.0%}"),
    ("duty_factor", "duty factor", "{:.2f}"),
]
FOOTFALLS = [("clean_hit_rate", "clean hits (front foot already up)", "{:.0%}"),
             ("overlap_at_td", "feet overlapping at touchdown", "{:.0%}"),
             ("stride_m", "stride [m]", "{:.2f}"),
             ("short_swing_frac", "swings of <= 2 steps", "{:.0%}")]


def version_of(run_dir: pathlib.Path) -> tuple[str, int]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    name = cfg.get("experiment_name")
    if name not in VERSIONS:
        raise SystemExit(f"{run_dir}: experiment '{name}' is not a version (known: {sorted(VERSIONS)})")
    return VERSIONS[name][0], int(cfg.get("seed", 0))


def collect(run_dir: pathlib.Path, root: pathlib.Path) -> str:
    version, seed = version_of(run_dir)
    n = 0
    for sub in ("eval", "eval_cat", "eval_terrain"):
        for src in sorted((run_dir / sub).glob("*.json")):
            if src.suffix != ".json":
                continue
            sim = "gazebo" if src.stem.startswith("gazebo") else "mujoco"
            stem = src.stem if sim == "gazebo" else f"{sub}_{src.stem}"
            dst = root / "versions" / version / sim / f"s{seed}" / f"{stem}.json"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            n += 1
    (root / "versions" / version / "mujoco" / f"s{seed}" / "run.txt").write_text(run_dir.name + "\n")
    print(f"[compare] {run_dir.name} -> versions/{version} seed {seed} ({n} files)")
    return version


def _seeds(vdir: pathlib.Path) -> list[int]:
    return sorted({int(p.name[1:]) for p in vdir.glob("*/s*") if p.is_dir() and p.name[1:].isdigit()})


def _value(vdir: pathlib.Path, sim: str, seed: int, stems: list[str], key: str):
    for stem in stems:
        p = vdir / sim / f"s{seed}" / f"{stem}.json"
        if p.exists():
            mean = json.loads(p.read_text(encoding="utf-8")).get("mean", {})
            if key in mean and mean[key] == mean[key]:   # skip NaN
                return float(mean[key])
    return None


def _fmt(values: list[float], fmt: str, scale: float = 1.0) -> str:
    if not values:
        return "–"
    vals = [v * scale for v in values]
    pct = fmt.endswith("%}")
    m = fmt.format(statistics.mean(vals))
    if len(vals) == 1:
        return m
    sd = fmt.format(statistics.stdev(vals))
    return f"{m[:-1]} ± {sd}" if pct else f"{m} ± {sd}"


def build_report(version: str, root: pathlib.Path) -> str | None:
    vdir = root / "versions" / version
    seeds = _seeds(vdir)
    if not seeds:
        return None
    desc = next(d for v, d in VERSIONS.values() if v == version)
    lines = [f"# {version}: {desc}", "",
             f"Seeds: {', '.join(map(str, seeds))}. Mean ± std over the seeds that have each result;"
             " '–' = not run yet. Generated by `scripts/compare_sims.py`, do not edit by hand.", ""]
    def n_seeds(c):
        return sum(1 for s in seeds if any((vdir / c[1] / f"s{s}" / f"{st}.json").exists() for st in c[2]))

    def table(conditions):
        cols = [c for c in conditions if n_seeds(c)]
        out = ["| metric | " + " | ".join(f"{c[0]} (n={n_seeds(c)})" for c in cols) + " |",
               "|---" * (len(cols) + 1) + "|"]
        for key, label, fmt, *scale in METRICS:
            cells = []
            for _, sim, stems in cols:
                vals = [v for s in seeds if (v := _value(vdir, sim, s, stems, key)) is not None]
                cells.append(_fmt(vals, fmt, scale[0] if scale else 1.0))
            if any(c != "–" for c in cells):
                out.append(f"| {label} | " + " | ".join(cells) + " |")
        return out

    gz_missing = not any(n_seeds(c) for c in CONDITIONS + PUSH_CONDITIONS if c[1] == "gazebo")
    lines += ["## No pushes", ""] + table(CONDITIONS)
    if any(n_seeds(c) for c in PUSH_CONDITIONS):
        lines += ["", "## Random pushes (Playground pert_config: kicks up to 3 m/s every 1-3 s)", ""]
        lines += table(PUSH_CONDITIONS)
    lines += _terrain_section(vdir, seeds) + _fall_section(vdir, seeds)
    ff = [[v for s in seeds if (v := _value(vdir, "mujoco", s, ["eval_cat_footfalls"], k)) is not None] for k, *_ in FOOTFALLS]
    if any(ff):
        lines += ["", "Footfalls in MuJoCo MJX (`scripts/analyze_footfalls.py`):", "",
                  "| metric | MuJoCo MJX |", "|---|---|"]
        for (key, label, fmt), vals in zip(FOOTFALLS, ff):
            lines.append(f"| {label} | {_fmt(vals, fmt)} |")
    if not gz_missing:
        lines += ["", GAZEBO_NOTE]
    if gz_missing:
        lines += ["", "No Gazebo result yet: run the policy in Gazebo, then "
                  "`python scripts/analyze_gazebo_log.py <csv> --run-dir runs/<run> --tag gazebo` and "
                  "`python scripts/compare_sims.py collect runs/<run>`."]
    runs = sorted(p.read_text().strip() for p in vdir.glob("mujoco/s*/run.txt"))
    if runs:
        lines += ["", "Runs: " + ", ".join(f"`{r}`" for r in runs)]
    (vdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def _terrain_section(vdir: pathlib.Path, seeds: list[int]) -> list[str]:
    """Success rate per terrain type and level, per simulator (mean over seeds)."""
    out = []
    own = any((vdir / "mujoco" / f"s{s}" / "eval_terrain_default.json").exists() for s in seeds)
    for label, version, sim, stem in TERRAIN_SOURCES:
        if version is not None and not own:
            continue          # the baseline row only goes into the terrain reports
        src = vdir.parent / version if version else vdir
        per = {}
        for p in sorted(src.glob(f"{sim}/s*/{stem}.json")):
            for r in json.loads(p.read_text(encoding="utf-8")).get("per_tile", []):
                per.setdefault((r["type"], r["level"]), []).append(r)
        if not per:
            continue
        levels = sorted({lv for _, lv in per})
        types = [t for t in TERRAIN_TYPES if any(t == k[0] for k in per)]
        n = max(len(v) for v in per.values())
        out += ["", f"## Terrain crossing, {label} (n={n}): success rate (falls)", "",
                "Cross a 4 m tile at 0.4 m/s from edge to edge (`scripts/evaluate_terrain.py`).", "",
                "| terrain | " + " | ".join(f"level {lv}" for lv in levels) + " |",
                "|---" * (len(levels) + 1) + "|"]
        for t in types:
            cells = []
            for lv in levels:
                rs = per.get((t, lv), [])
                cells.append(f"{statistics.mean(r['success_rate'] for r in rs):.0%} "
                             f"({statistics.mean(r['fall_rate'] for r in rs):.0%})" if rs else "–")
            out.append(f"| {t} | " + " | ".join(cells) + " |")
    return out


def _fall_section(vdir: pathlib.Path, seeds: list[int]) -> list[str]:
    rows = []
    for key, label, fmt in FALL_METRICS:
        vals = [v for s in seeds if (v := _value(vdir, "mujoco", s, ["eval_cat_default"], key)) is not None]
        upside = [v for s in seeds if (v := _value(vdir, "mujoco", s, ["eval_cat_upside_down"], key)) is not None]
        high = [v for s in seeds if (v := _value(vdir, "mujoco", s, ["eval_cat_high"], key)) is not None]
        if vals or upside or high:
            rows.append(f"| {label} | {_fmt(vals, fmt)} | {_fmt(upside, fmt)} | {_fmt(high, fmt)} |")
    if not rows:
        return []
    return ["", "## Falling (`scripts/evaluate_cat.py`)", "",
            "| metric | random drop | upside down (> 135°) | high drop (1.5-2 m) |", "|---|---|---|---|"] + rows


def build_summary(root: pathlib.Path) -> None:
    rows = ["# Versions", "", "One folder per policy version: `mujoco/` and `gazebo/` hold the evaluation"
            " results per seed, `report.md` compares the simulators. The code is shared (quadrucat/, ros2/,"
            " scripts/); a version is a config in configs/.", "",
            "| version | what | seeds | falls MJX + pushes | register error MJX [cm] | falls Gazebo "
            "| terrain crossings MJX |",
            "|---|---|---|---|---|---|---|"]
    for name, (v, desc) in VERSIONS.items():
        vdir = root / "versions" / v
        seeds = _seeds(vdir)
        if not seeds:
            continue
        get = lambda sim, stems, key: [x for s in seeds if (x := _value(vdir, sim, s, stems, key)) is not None]
        rows.append(f"| [{v}]({v}/report.md) | {desc} | {len(seeds)} | "
                    f"{_fmt(get('mujoco', ['eval_pushes'], 'fall_rate'), '{:.0%}')} | "
                    f"{_fmt(get('mujoco', ['eval_cat_default'], 'register_err_median'), '{:.1f}', 100.0)} | "
                    f"{_fmt(get('gazebo', ['gazebo_pd'], 'fall_rate'), '{:.0%}')} | "
                    f"{_fmt(get('mujoco', ['eval_terrain_default', 'eval_terrain_flat_policy'], 'terrain_success'), '{:.0%}')} |")
    (root / "versions" / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT), help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("runs", nargs="+")
    r = sub.add_parser("report")
    r.add_argument("versions", nargs="*")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    if args.cmd == "collect":
        todo = sorted({collect(pathlib.Path(p), root) for p in args.runs})
    else:
        todo = args.versions or sorted(p.name for p in (root / "versions").iterdir() if p.is_dir())
    for v in todo:
        text = build_report(v, root)
        print(text if text else f"[compare] versions/{v}: no results")
    build_summary(root)


if __name__ == "__main__":
    sys.exit(main())
