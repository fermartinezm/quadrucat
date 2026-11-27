"""Is a "direct register" real? Looks at every hind touchdown of a walking run, not only at its distance.

  python scripts/analyze_footfalls.py runs/<run> [--episodes 16] [--steps 1000]

The Go1 "feet only" model has no foot-foot collisions, so a policy can put the hind foot INSIDE a
front foot that is still standing there. Reported per command (pushes off):
  hit_rate            hind touchdowns closer than 5 cm to the same-side front print
  front_down_at_td    fraction of hind touchdowns with that front foot still on the ground
  clean_hit_rate      hits with the front foot already off the ground (a real direct register)
  overlap_at_td       touchdowns with the two foot centres closer than one foot diameter (4.6 cm, 3D)
  overlap_time        fraction of time the same-side feet overlap
  stride_m            hind stride between touchdowns (swing >= 3 steps), median
  short_swing_frac    touchdowns after a swing of <= 2 control steps (contact chatter / tapping)
  front_lift_to_td_s  time between the front foot lifting and the hind foot landing (clean touchdowns)
  clearance_m         peak foot height during hind swings, median
Writes runs/<run>/eval_cat/footfalls.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FOOT_DIAMETER = 0.046


def footfall_stats(feet: np.ndarray, contact: np.ndarray, alive: np.ndarray, dist: np.ndarray, td: np.ndarray,
                   dt: float, warmup: int = 50, tol: float = 0.05) -> dict:
    """feet [E,T,4,3], contact [E,T,4], alive [E,T], dist/td [E,T,2] (order FR FL RR RL; hind RR RL)."""
    feet, contact, alive = np.asarray(feet, float), np.asarray(contact).astype(bool), np.asarray(alive) > 0
    E, T = alive.shape
    d3 = np.linalg.norm(feet[:, :, 2:] - feet[:, :, :2], axis=-1)      # same-side hind-front distance
    rows, strides = [], []
    for e in range(E):
        for s in range(2):
            h, f = contact[e, :, 2 + s], contact[e, :, s]
            h_off, last_td, f_lift = None, None, None
            peak = 0.0
            for t in range(1, T):
                if not alive[e, t]:
                    break
                if f[t - 1] and not f[t]:
                    f_lift = t
                if h[t - 1] and not h[t]:
                    h_off, peak = t, feet[e, t, 2 + s, 2]
                if not h[t]:
                    peak = max(peak, feet[e, t, 2 + s, 2])
                if td[e, t, s] > 0 and t >= warmup:
                    swing = t - h_off if h_off is not None else 0
                    stride = np.nan if last_td is None else np.linalg.norm(feet[e, t, 2 + s, :2] - feet[e, last_td, 2 + s, :2])
                    rows.append((dist[e, t, s] < tol, f[t], d3[e, t, s] < FOOT_DIAMETER, swing <= 2,
                                 np.nan if f[t] or f_lift is None else (t - f_lift) * dt, peak))
                    if swing >= 3:
                        strides.append(stride)
                        last_td = t
    if not rows:
        return {"n_touchdowns": 0}
    hit, down, over, short, lag, peak = np.array(rows, float).T
    m = alive[:, warmup:]
    return {"n_touchdowns": len(rows), "hit_rate": hit.mean(), "front_down_at_td": down.mean(),
            "clean_hit_rate": ((hit > 0) & (down == 0)).mean(), "overlap_at_td": over.mean(),
            "overlap_time": float((d3[:, warmup:] < FOOT_DIAMETER)[m].mean()),
            "stride_m": float(np.nanmedian(strides)) if len(strides) else float("nan"),
            "short_swing_frac": short.mean(),
            "front_lift_to_td_s": float(np.nanmedian(lag)) if np.isfinite(lag).any() else float("nan"),
            "clearance_m": float(np.median(peak))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    from quadrucat.config import deep_merge, find_params, load_config

    run_dir = pathlib.Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    train_cfg = deep_merge(cfg, {})
    if cfg["env"]["name"] != "Go1CatWalk":
        cfg = deep_merge(cfg, {"env": {"name": "Go1CatWalk"}})
    cfg = deep_merge(cfg, {"env": {"overrides": {"pert_config": {"enable": False}}}})

    import jax
    import jax.numpy as jp
    from brax.io import model

    from quadrucat import agents, envs

    env, _ = envs.make_env(cfg)
    policy_env, _ = envs.make_env(train_cfg)
    params = model.load_params(str(find_params(run_dir)))
    policy = agents.make_inference_fn(train_cfg, policy_env)(
        params[:2] if isinstance(params, (tuple, list)) else params, deterministic=True)

    def set_cmd(state, cmd):
        return state.replace(info={**state.info, "command": cmd})

    def run_episode(k, cmd):
        k, k0 = jax.random.split(k)
        state = set_cmd(env.reset(k0), cmd)

        def body(carry, _):
            state, k, alive = carry
            k, k1 = jax.random.split(k)
            action, _ = policy(state.obs, k1)
            ns = set_cmd(env.step(set_cmd(state, cmd), action), cmd)
            alive = alive * (1.0 - ns.done)
            return (ns, k, alive), (ns.data.site_xpos[env._feet_site_id], ns.info["last_contact"], alive,
                                    ns.info["register_dist"], ns.info["hind_touchdown"])
        return jax.lax.scan(body, (state, k, jp.float32(1.0)), None, length=args.steps)[1]

    run_batch = jax.jit(jax.vmap(run_episode, in_axes=(0, None)))
    rng, rows = jax.random.PRNGKey(args.seed), []
    for cmd in cfg["eval"]["commands"]:
        rng, k = jax.random.split(rng)
        out = jax.device_get(run_batch(jax.random.split(k, args.episodes), jp.array(cmd, dtype=jp.float32)))
        row = {"command": cmd, **{k: float(v) for k, v in footfall_stats(*out, dt=float(env.dt)).items()}}
        rows.append(row)
        print("[footfalls] cmd=%s  " % cmd + "  ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "command"), flush=True)
    (run_dir / "eval_cat").mkdir(exist_ok=True)
    keys = [k for k in rows[0] if k != "command"]
    mean = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}   # plot_results.py reads "mean" (and skips this kind)
    (run_dir / "eval_cat" / "footfalls.json").write_text(
        json.dumps({"tag": "footfalls", "kind": "footfalls", "per_command": rows, "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
