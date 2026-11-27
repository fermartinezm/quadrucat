"""Terrain crossing test: can the policy walk across each tile type at each difficulty level?

For every (terrain type, level) the robot starts at the -x edge of the tile, faces +x and is
commanded `eval.terrain_speed` forward (heading held at 0). Per tile:
  success_rate   reached the far edge of the tile without falling
  fall_rate      the episode ended early (fell over, off a beam, trunk on the ground, left the map)
  progress       fraction of the tile crossed (mean)
  lin_vel_rmse   forward-speed tracking while alive [m/s]
Per type: `max_level` = highest level with success >= 80% and every easier evaluated level too.

Works on any walking run: a policy trained on flat ground (e.g. cat walk v3) is evaluated blind
(no height scan) on the same map, which gives the "before" numbers.

  python scripts/evaluate_terrain.py runs/<run>
  python scripts/evaluate_terrain.py runs/<v3 run> --tag flat_policy
  python scripts/evaluate_terrain.py runs/<run> --tag no_scan_noise --set env.overrides.height_scan.noise=0.0

Writes runs/<run>/eval_terrain/<tag>.json (and videos with --video).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadrucat.config import deep_merge, find_params, load_config, parse_overrides, set_dotted  # noqa: E402

SUCCESS_FOR_LEVEL = 0.8


def summarize(rows: list[dict], types: list[str]) -> dict:
    """Per-type and overall summary of the per-tile rows (pure, tested)."""
    out = {}
    succ_all, fall_all = [], []
    for t in types:
        rs = sorted((r for r in rows if r["type"] == t), key=lambda r: r["level"])
        if not rs:
            continue
        max_level = -1
        for r in rs:
            if r["success_rate"] >= SUCCESS_FOR_LEVEL:
                max_level = r["level"]
            else:
                break
        out[f"success_{t}"] = sum(r["success_rate"] for r in rs) / len(rs)
        out[f"max_level_{t}"] = float(max_level)
        succ_all += [r["success_rate"] for r in rs]
        fall_all += [r["fall_rate"] for r in rs]
    out["terrain_success"] = sum(succ_all) / max(len(succ_all), 1)
    out["terrain_fall_rate"] = sum(fall_all) / max(len(fall_all), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides for the EVAL env/config")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--levels", type=int, nargs="*")
    ap.add_argument("--speed", type=float)
    ap.add_argument("--video", action="store_true", help="one crossing per terrain type at --video-level")
    ap.add_argument("--video-level", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    train_cfg = load_config(run_dir / "config.yaml")
    cfg = deep_merge(train_cfg, {})
    blind = cfg["env"]["name"] != "Go1Terrain"
    if blind:   # a flat-ground policy: same terrain map, no height scan in its observation
        cfg = deep_merge(cfg, {"env": {"name": "Go1Terrain", "overrides": {"height_scan": {"in_actor": False}}}})
    cfg = deep_merge(cfg, {"env": {"overrides": {
        "terrain": {"eval_start": "edge", "curriculum": False},
        "heading_command": {"enable": True},
        "pert_config": {"enable": False}}}})
    for key, value in parse_overrides(args.set):
        set_dotted(cfg, key, value)

    import jax
    import jax.numpy as jp
    import numpy as np
    from brax.io import model

    from quadrucat import agents, envs, metrics

    env, env_cfg = envs.make_env(cfg)
    policy_env, _ = envs.make_env(train_cfg)
    priv_size = policy_env.observation_size["privileged_state"]
    params = model.load_params(str(find_params(run_dir, args.checkpoint)))
    make_policy = agents.make_inference_fn(train_cfg, policy_env)
    policy = make_policy(params[:2] if isinstance(params, (tuple, list)) else params, deterministic=True)

    def act(obs, key):   # the policy only reads "state"; the critic input is not used here
        return policy({"state": obs["state"], "privileged_state": jp.zeros(priv_size)}, key)

    ecfg = cfg.get("eval", {})
    types = list(env_cfg.terrain.types)
    n_levels = int(env_cfg.terrain.n_levels)
    levels = args.levels or ecfg.get("terrain_levels", [0, 3, 6, 9])
    levels = sorted({min(max(int(lv), 0), n_levels - 1) for lv in levels})
    speed = args.speed or float(ecfg.get("terrain_speed", 0.4))
    n_eps = args.episodes or int(ecfg.get("terrain_episodes", 16))
    tile = float(env_cfg.terrain.tile_size)
    goal = tile - 0.8 - 0.1           # start 0.4 m inside the tile, finish 0.4 m before its far edge
    dt = float(env.dt)
    ep_len = int(math.ceil(goal / speed / dt)) + 150
    cmd = jp.array([speed, 0.0, 0.0], dtype=jp.float32)

    def set_cmd(state):
        return state.replace(info={**state.info, "command": cmd})

    def run_episode(k, row, level):
        k, k0 = jax.random.split(k)
        state = set_cmd(env.reset_at(k0, row, level))
        x0 = state.data.qpos[0]

        def body(carry, _):
            state, k, alive, reached = carry
            k, k1 = jax.random.split(k)
            action, _ = act(state.obs, k1)
            ns = set_cmd(env.step(set_cmd(state), action))
            alive = alive * (1.0 - ns.done)
            reached = jp.maximum(reached, alive * (ns.data.qpos[0] - x0 >= goal))
            out = (env.get_local_linvel(ns.data), env.get_gyro(ns.data), alive, ns.data.qpos[0] - x0)
            return (ns, k, alive, reached), out

        (_, _, _, reached), (lin, gyro, alive, dx) = jax.lax.scan(
            body, (state, k, jp.float32(1.0), jp.float32(0.0)), None, length=ep_len)
        return reached, lin, gyro, alive, dx

    run_batch = jax.jit(jax.vmap(run_episode, in_axes=(0, None, None)))
    rng = jax.random.PRNGKey(args.seed)
    rows = []
    for row, t in enumerate(types):
        if t in types[:row]:          # repeated rows have the same terrain parameters
            continue
        for level in levels:
            rng, k = jax.random.split(rng)
            reached, lin, gyro, alive, dx = jax.device_get(
                run_batch(jax.random.split(k, n_eps), jp.int32(row), jp.int32(level)))
            track = metrics.tracking_metrics([speed, 0.0, 0.0], lin[..., :2], gyro[..., 2], alive, dt, 50)
            progress = np.clip(np.max(np.where(alive > 0, dx, -np.inf), axis=1) / goal, 0.0, 1.0)
            # a robot that reached the goal and fell afterwards counts as a success, not as a fall
            fell = (alive[:, -1] < 0.5) & (reached < 0.5)
            r = {"type": t, "level": level, "success_rate": float(np.mean(reached)),
                 "fall_rate": float(np.mean(fell)),
                 "progress": float(np.mean(np.where(np.isfinite(progress), progress, 0.0))),
                 "lin_vel_rmse": track["lin_vel_rmse"]}
            rows.append(r)
            print(f"[terrain] {t:12s} level {level}: success {r['success_rate']:.2f}  falls {r['fall_rate']:.2f}  "
                  f"progress {r['progress']:.2f}  lin_rmse {r['lin_vel_rmse']:.3f}")

    seen_types = list(dict.fromkeys(types))
    mean = summarize(rows, seen_types)
    results = {"tag": args.tag, "blind": blind, "speed": speed, "levels": levels, "episodes": n_eps,
               "overrides": args.set, "per_tile": rows, "mean": mean}
    out_dir = run_dir / "eval_terrain"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.tag}.json").write_text(json.dumps(results, indent=2))
    print("[terrain] " + "  ".join(f"{k}={v:.2f}" for k, v in mean.items()))
    print(f"[terrain] -> {out_dir / (args.tag + '.json')}")

    if args.video:
        import mediapy

        jit_reset = jax.jit(env.reset_at)
        jit_step, jit_act = jax.jit(env.step), jax.jit(act)
        level = min(args.video_level, n_levels - 1)
        for row, t in enumerate(types):
            if t in types[:row]:
                continue
            rng, k = jax.random.split(rng)
            state = set_cmd(jit_reset(k, jp.int32(row), jp.int32(level)))
            traj = [state]
            for _ in range(ep_len):
                rng, k = jax.random.split(rng)
                action, _ = jit_act(state.obs, k)
                state = set_cmd(jit_step(set_cmd(state), action))
                traj.append(state)
                if bool(state.done):
                    break
            frames = env.render(traj, camera=ecfg.get("camera", "track"), height=480, width=640)
            path = out_dir / f"{args.tag}_{t}_level{level}.mp4"
            mediapy.write_video(str(path), frames, fps=int(round(1.0 / dt)))
            print(f"[terrain] video -> {path}")


if __name__ == "__main__":
    main()
