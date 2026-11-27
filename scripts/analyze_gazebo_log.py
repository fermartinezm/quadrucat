"""Turn the policy node's Gazebo CSV log into the same metrics as scripts/evaluate.py.

  python scripts/analyze_gazebo_log.py /tmp/gz_eval.csv --run-dir runs/<run> --tag gazebo
  python scripts/analyze_gazebo_log.py rep0.csv rep1.csv rep2.csv --run-dir runs/<run> --tag gazebo
  python scripts/plot_results.py runs/<run>       # table: MJX vs MuJoCo C vs Gazebo

Each eval segment (one command) of each log is one episode. The robot is not reset
between segments in Gazebo, so once it falls the remaining segments count as fallen.
Gait/register metrics need the full-state columns written by the policy node.
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


FOOT_RADIUS = 0.023   # [m] Go1 foot sphere


def foot_contacts(log: dict[str, np.ndarray], contact_source: str = "height",
                  contact_height: float = FOOT_RADIUS + 0.005) -> tuple[np.ndarray, np.ndarray]:
    """(feet_world [T,4,3], contact [T,4]) for one log.

    contact_source="height": a foot is in contact when its centre (URDF forward kinematics + trunk
    pose) is below contact_height. Default, because the bridged Gazebo contact-sensor messages arrive
    in bursts (whole 20 ms windows with no message while all four feet rest on the floor), which
    would count as fake touchdowns. "sensor": use the c_* columns logged by the policy node.
    """
    q = np.stack([log[f"q{i}"] for i in range(12)], -1)
    pos = np.stack([log[k] for k in ("px", "py", "pz")], -1)
    quat = np.stack([log[k] for k in ("qx", "qy", "qz", "qw")], -1)
    rot = np.stack([quat_to_rotmat(x, "xyzw") for x in quat])
    feet = FootKinematics().feet_world(q, pos, rot)
    if contact_source == "height":
        return feet, feet[..., 2] < contact_height
    return feet, np.stack([log[k] for k in CONTACT_KEYS], -1) > 0


def analyze(logs: list[dict[str, np.ndarray]], warmup_s: float = 1.0, **contact_kw) -> dict:
    """Each log is one repetition of the eval; each segment of a log is one episode of its command.

    The robot is not reset between segments, so once it falls the remaining segments of that log
    count as fallen. Logs where the robot was not upright when the policy started are dropped.
    """
    if isinstance(logs, dict):
        logs = [logs]
    good = []
    for n, log in enumerate(logs):
        keep = log["segment"] >= 0          # segment -1 = stand-up phase before the policy starts
        log = {k: v[keep] for k, v in log.items()}
        if log["gravity_z"][0] < -0.9:
            good.append(log)
        else:
            print(f"[gazebo] WARNING: log {n}: the robot was not upright when the policy started "
                  "(bad start-up), dropped")
    if not good:
        raise SystemExit("no usable log")
    t = good[0]["t"]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    warmup = int(round(warmup_s / dt))
    gait = all(k in good[0] for k in ("q0", "px", "qw"))
    if gait:
        contacts = [foot_contacts(log, **contact_kw) for log in good]
    per_command = []
    for seg in np.unique(good[0]["segment"]).astype(int):
        idx = [log["segment"] == seg for log in good]
        T = min(int(i.sum()) for i in idx)

        def stack(key):
            return np.stack([log[key][i][:T] for log, i in zip(good, idx)])

        cmd = [float(stack(k)[0, 0]) for k in ("cmd_vx", "cmd_vy", "cmd_wz")]
        alive = 1.0 - stack("fallen")
        v = np.stack([stack("vx"), stack("vy")], -1)
        row = metrics.tracking_metrics(cmd, v, stack("wz"), alive, dt, warmup=warmup)
        row["segment"] = int(seg)
        if gait:
            feet = np.stack([f[i][:T] for (f, _), i in zip(contacts, idx)])
            contact = np.stack([c[i][:T] for (_, c), i in zip(contacts, idx)])
            sig = [metrics.footprint_signals(f[..., :2], c) for f, c in zip(feet, contact)]
            dist, td = np.stack([d for d, _ in sig]), np.stack([x for _, x in sig])
            row.update(metrics.gait_metrics(contact, alive, dt, warmup))
            row.update(metrics.register_metrics(dist, td, alive, dt, warmup=warmup))
        per_command.append(row)
        print("[gazebo]", metrics.format_row(row))
    mean = metrics.summarize(per_command)
    for k in GAIT_KEYS:
        if all(k in r for r in per_command):
            mean[k] = float(np.nanmean([r[k] for r in per_command]))
    out = {"dt": dt, "episodes_per_command": len(good), "logs_dropped": len(logs) - len(good),
           "per_command": per_command, "mean": mean}
    rtf = [(log["t"][-1] - log["t"][0]) / max(log["wall"][-1] - log["wall"][0], 1e-9)
           for log in good if "wall" in log and len(log["t"]) > 1]
    if rtf:
        out["real_time_factor"] = float(np.mean(rtf))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="one or more logs (repetitions of the same eval)")
    ap.add_argument("--run-dir", help="write <run>/eval/<tag>.json next to the MuJoCo results")
    ap.add_argument("--tag", default="gazebo")
    ap.add_argument("--warmup-s", type=float, default=1.0)
    ap.add_argument("--contact-source", choices=["height", "sensor"], default="height")
    ap.add_argument("--contact-height", type=float, default=FOOT_RADIUS + 0.005,
                    help="[m] foot-centre height below which a foot counts as in contact")
    args = ap.parse_args()

    res = analyze([load_log(pathlib.Path(c)) for c in args.csv], args.warmup_s,
                  contact_source=args.contact_source, contact_height=args.contact_height)
    res["contact"] = {"source": args.contact_source, "height_m": args.contact_height}
    res["tag"] = args.tag
    res["source"] = [pathlib.Path(c).name for c in args.csv]
    print(f"[gazebo] mean: {res['mean']}")
    if args.run_dir:
        out = pathlib.Path(args.run_dir) / "eval"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
        print(f"[gazebo] -> {out / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
