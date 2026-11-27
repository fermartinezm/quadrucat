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
}

# (column, simulator folder, file stem in s<seed>/) -> which file holds that condition
CONDITIONS = [
    ("MuJoCo MJX", "mujoco", ["eval_default", "eval_cat_default"]),
    ("MJX + pushes", "mujoco", ["eval_pushes", "eval_cat_pushes"]),
    ("MuJoCo C (sim2sim)", "mujoco", ["eval_sim2sim_effort"]),
    ("Gazebo", "gazebo", ["gazebo"]),
    ("Gazebo at 1/4 speed", "gazebo", ["gazebo_slow"]),
    ("Gazebo + pushes", "gazebo", ["gazebo_pushes"]),
]
GAZEBO_NOTE = (
    "Gazebo runs through ROS 2 exactly as a robot would: the policy (50 Hz) and the PD loop (500 Hz) are "
    "Python nodes talking to Gazebo over topics. 'Gazebo' runs the physics as fast as the PC allows "
    "(0.6-0.8x real time on the WSL laptop), so the nodes compete with Gazebo for CPU and the PD torques "
    "arrive late; 'Gazebo at 1/4 speed' caps the physics (`sim_rate_hz:=250`) so the same nodes keep "
    "up, which separates the physics gap from the communication delay. Each Gazebo seed is 3 "
    "real-time repetitions (1 at 1/4 speed); the robot is not reset between commands, so after a fall the "
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
    for sub in ("eval", "eval_cat"):
        for src in sorted((run_dir / sub).glob("*.json")):
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

    cols = [c for c in CONDITIONS if n_seeds(c)]
    gz_missing = not any(c[1] == "gazebo" for c in cols)
    if gz_missing:
        cols.append(next(c for c in CONDITIONS if c[1] == "gazebo"))
    lines.append("| metric | " + " | ".join(f"{c[0]} (n={n_seeds(c)})" for c in cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for key, label, fmt, *scale in METRICS:
        cells = []
        for _, sim, stems in cols:
            vals = [v for s in seeds if (v := _value(vdir, sim, s, stems, key)) is not None]
            cells.append(_fmt(vals, fmt, scale[0] if scale else 1.0))
        if any(c != "–" for c in cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
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


def build_summary(root: pathlib.Path) -> None:
    rows = ["# Versions", "", "One folder per policy version: `mujoco/` and `gazebo/` hold the evaluation"
            " results per seed, `report.md` compares the simulators. The code is shared (quadrucat/, ros2/,"
            " scripts/); a version is a config in configs/.", "",
            "| version | what | seeds | falls MJX + pushes | register error MJX [cm] | falls Gazebo |",
            "|---|---|---|---|---|---|"]
    for name, (v, desc) in VERSIONS.items():
        vdir = root / "versions" / v
        seeds = _seeds(vdir)
        if not seeds:
            continue
        get = lambda sim, stems, key: [x for s in seeds if (x := _value(vdir, sim, s, stems, key)) is not None]
        rows.append(f"| [{v}]({v}/report.md) | {desc} | {len(seeds)} | "
                    f"{_fmt(get('mujoco', ['eval_pushes'], 'fall_rate'), '{:.0%}')} | "
                    f"{_fmt(get('mujoco', ['eval_cat_default'], 'register_err_median'), '{:.1f}', 100.0)} | "
                    f"{_fmt(get('gazebo', ['gazebo'], 'fall_rate'), '{:.0%}')} |")
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
