"""Turn the policy node's Gazebo CSV log into the same metrics as scripts/evaluate.py.

  python scripts/analyze_gazebo_log.py /tmp/gz_eval.csv --run-dir runs/<run> --tag gazebo
  python scripts/plot_results.py runs/<run>       # table: MJX vs MuJoCo C vs Gazebo

Each eval segment (one command) is treated as one episode. The robot is not reset
between segments in Gazebo, so once it falls the remaining segments count as fallen.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadrucat import metrics  # noqa: E402
from quadrucat.kinematics import FootKinematics  # noqa: E402
from quadrucat.observation import quat_to_rotmat  # noqa: E402

CONTACT_KEYS = ("c_FR", "c_FL", "c_RR", "c_RL")
GAIT_KEYS = ("three_foot_frac", "two_foot_frac", "duty_factor", "register_err_median",
             "register_hit_rate", "hind_steps_per_s")


def load_log(path: pathlib.Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def gait_row(log: dict[str, np.ndarray], idx: np.ndarray, alive: np.ndarray, dt: float, warmup: int) -> dict:
    """Cat-gait metrics of one segment (same definitions as scripts/evaluate_cat.py)."""
    contact = np.stack([log[k][idx] for k in CONTACT_KEYS], -1)
    q = np.stack([log[f"q{i}"][idx] for i in range(12)], -1)
    pos = np.stack([log[k][idx] for k in ("px", "py", "pz")], -1)
    quat = np.stack([log[k][idx] for k in ("qx", "qy", "qz", "qw")], -1)
    rot = np.stack([quat_to_rotmat(x, "xyzw") for x in quat])
    feet = FootKinematics().feet_world(q, pos, rot)
    dist, td = metrics.footprint_signals(feet[..., :2], contact)
    return {**metrics.gait_metrics(contact, alive, dt, warmup),
            **metrics.register_metrics(dist, td, alive, dt, warmup=warmup)}


def analyze(log: dict[str, np.ndarray], warmup_s: float = 1.0) -> dict:
    t = log["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    per_command = []
    for seg in np.unique(log["segment"]).astype(int):
        idx = log["segment"] == seg
        cmd = [float(log[k][idx][0]) for k in ("cmd_vx", "cmd_vy", "cmd_wz")]
        alive = 1.0 - log["fallen"][idx]
        v = np.stack([log["vx"][idx], log["vy"][idx]], -1)
        row = metrics.tracking_metrics(cmd, v, log["wz"][idx], alive, dt, warmup=int(round(warmup_s / dt)))
        row["segment"] = int(seg)
        if all(k in log for k in CONTACT_KEYS):
            row.update(gait_row(log, idx, alive, dt, int(round(warmup_s / dt))))
        per_command.append(row)
        print("[gazebo]", metrics.format_row(row))
    return {"dt": dt, "per_command": per_command, "mean": metrics.summarize(per_command)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--run-dir", help="write <run>/eval/<tag>.json next to the MuJoCo results")
    ap.add_argument("--tag", default="gazebo")
    ap.add_argument("--warmup-s", type=float, default=1.0)
    args = ap.parse_args()

    res = analyze(load_log(pathlib.Path(args.csv)), args.warmup_s)
    res["tag"] = args.tag
    res["source"] = str(args.csv)
    print(f"[gazebo] mean: {res['mean']}")
    if args.run_dir:
        out = pathlib.Path(args.run_dir) / "eval"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
        print(f"[gazebo] -> {out / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
