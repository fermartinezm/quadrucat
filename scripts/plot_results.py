"""Compare runs: diagnostics figure, per-reward-term breakdown and eval tables.

  python scripts/plot_results.py runs/go1_ppo_base_* runs/ablate_feet_slip_*      # all outputs
  python scripts/plot_results.py runs/* --metric eval/episode_reward/feet_slip    # one curve

Outputs (in --out-dir, default figures/):
  diagnostics.png   learning curves: return, episode length, tracking terms, PPO health
                    (KL, entropy, value loss, action std). Use it to explain WHY a run fails.
  reward_terms.png  final contribution of every reward term per run (what the agent optimizes).
  <metric>.png      only with --metric.
and prints the eval/*.json table (MJX evaluate, sim2sim, Gazebo, robustness tags).
Runs whose name differs only in the seed (…_s0_…, …_s1_…) are grouped: mean +- std band.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DIAGNOSTICS = [
    ("eval/episode_reward", "episode return (eval)"),
    ("eval/avg_episode_length", "episode length (eval, max = no falls)"),
    ("eval/episode_reward/tracking_lin_vel", "tracking_lin_vel term"),
    ("eval/episode_reward/tracking_ang_vel", "tracking_ang_vel term"),
    ("training/kl_mean", "KL(old || new) per update"),
    ("training/entropy_loss", "entropy loss (more negative = more random)"),
    ("training/v_loss", "value loss"),
    ("training/policy_dist_mean_std", "mean action std (exploration)"),
]


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_name(run: pathlib.Path) -> str:
    """'ablate_feet_slip_s1_20261124-120000' -> 'ablate_feet_slip'."""
    return re.sub(r"_s\d+_\d{8}-\d{6}$", "", run.name)


def curves(runs, metric):
    """{group: (steps, mean, std)} using the steps common to all seeds of a group."""
    groups = defaultdict(list)
    for run in runs:
        path = run / "metrics.jsonl"
        if not path.exists():
            continue
        rows = [r for r in read_jsonl(path) if metric in r]
        if rows:
            groups[group_name(run)].append({r["step"]: r[metric] for r in rows})
    out = {}
    for g, series in groups.items():
        steps = sorted(set.intersection(*(set(s) for s in series)))
        if not steps:
            continue
        vals = np.array([[s[k] for k in steps] for s in series])
        out[g] = (np.array(steps), vals.mean(0), vals.std(0), len(series))
    return out


def plot_metric(ax, runs, metric, title=None):
    for g, (x, m, s, n) in curves(runs, metric).items():
        line, = ax.plot(x, m, marker="o" if len(x) < 6 else None, label=f"{g} (n={n})")
        if n > 1:
            ax.fill_between(x, m - s, m + s, alpha=0.2, color=line.get_color())
    ax.set_title(title or metric, fontsize=9)
    ax.set_xlabel("env steps", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)


def reward_terms(runs):
    """Final eval contribution of each reward term, averaged over seeds."""
    groups = defaultdict(list)
    for run in runs:
        path = run / "metrics.jsonl"
        if path.exists():
            rows = [r for r in read_jsonl(path) if "eval/episode_reward" in r]
            if rows:
                last = rows[-1]
                groups[group_name(run)].append({k.split("/")[-1]: v for k, v in last.items()
                                                if k.startswith("eval/episode_reward/") and not k.endswith("_std")})
    return {g: {k: float(np.mean([d[k] for d in ds if k in d])) for k in ds[0]} for g, ds in groups.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--metric", help="also plot this single metric")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    runs = [pathlib.Path(r) for r in args.runs if pathlib.Path(r).is_dir()]
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for ax, (metric, title) in zip(axes.flat, DIAGNOSTICS):
        plot_metric(ax, runs, metric, title)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "diagnostics.png", dpi=130)
    print(f"[plot] {out / 'diagnostics.png'}")

    terms = reward_terms(runs)
    if terms:
        names = sorted({k for d in terms.values() for k in d})
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(names) * max(1, len(terms)) / 2), 4.5))
        width = 0.8 / len(terms)
        for i, (g, d) in enumerate(terms.items()):
            ax.bar(np.arange(len(names)) + i * width, [d.get(n, 0.0) for n in names], width, label=g)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2, names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("episode sum (scaled term)")
        ax.set_title("Final reward breakdown (eval)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "reward_terms.png", dpi=130)
        print(f"[plot] {out / 'reward_terms.png'}")

    if args.metric:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_metric(ax, runs, args.metric)
        ax.legend(fontsize=7)
        fig.tight_layout()
        name = args.metric.replace("/", "_") + ".png"
        fig.savefig(out / name, dpi=150)
        print(f"[plot] {out / name}")

    table = []
    for run in runs:
        for ev in sorted((run / "eval").glob("*.json")) if (run / "eval").exists() else []:
            res = json.loads(ev.read_text(encoding="utf-8"))
            if "mean" in res:
                table.append((run.name, res.get("tag", ev.stem), res["mean"]))
    if table:
        print(f"\n{'run':45s} {'tag':22s} {'lin_rmse':>9s} {'yaw_rmse':>9s} {'falls':>6s} {'surv_s':>7s}")
        for name, tag, m in table:
            print(f"{name[:45]:45s} {tag[:22]:22s} {m['lin_vel_rmse']:9.3f} {m['yaw_rate_rmse']:9.3f} "
                  f"{m['fall_rate']:6.2f} {m['survival_s']:7.1f}")

    # Cat tasks (scripts/evaluate_cat.py writes runs/<run>/eval_cat/<tag>.json)
    cat_rows = []
    for run in runs:
        for ev in sorted((run / "eval_cat").glob("*.json")) if (run / "eval_cat").exists() else []:
            res = json.loads(ev.read_text(encoding="utf-8"))
            cat_rows.append((run.name, res.get("tag", ev.stem), res.get("kind"), res["mean"]))
    for kind in ("walk", "fall"):
        rows = [r for r in cat_rows if r[2] == kind]
        if not rows:
            continue
        keys = list(rows[0][3].keys())
        print(f"\n[{kind}] {'run':45s} {'tag':14s} " + " ".join(f"{k[:14]:>14s}" for k in keys))
        for name, tag, _, m in rows:
            print(f"       {name[:45]:45s} {tag[:14]:14s} " + " ".join(f"{m.get(k, float('nan')):14.3f}" for k in keys))


if __name__ == "__main__":
    main()
