"""Evaluate the cat tasks: gait/register metrics for walking, landing metrics for drops.

The task kind is taken from the run's env name (Go1CatFall / Go1Getup -> 'fall', else 'walk').

Walking (Go1CatWalk, but also works on base Go1Joystick runs, so you can compare):
  python scripts/evaluate_cat.py runs/<run>                       # per eval command
  -> lin_vel_rmse, falls, register_err_median [m], register_hit_rate, three_foot_frac, duty_factor
  For a non-cat run the register distance is still measured (with the Go1CatWalk bookkeeping),
  so the baseline gives you the "before" number.

Falling (Go1CatFall):
  python scripts/evaluate_cat.py runs/<run> --episodes 512
  python scripts/evaluate_cat.py --config configs/cat_fall.yaml --zero-policy   # do-nothing baseline
  -> land_on_feet_rate (overall and per initial-tilt bin), touch_up_z, trunk_hit_rate,
     peak_acc_g, upright_end_rate, time_to_stand_s

Eval conditions reuse --set on the eval env, e.g. harder drops:
  python scripts/evaluate_cat.py runs/<run> --tag high --set env.overrides.drop_height=[1.5,2.0]

Writes runs/<run>/eval_cat/<tag>.json (plot_results.py prints these as a table).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadrucat.config import deep_merge, find_params, load_config, parse_overrides, set_dotted  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--config", help="evaluate a config without a run (only with --zero-policy)")
    ap.add_argument("--zero-policy", action="store_true", help="action = 0 (passive robot) baseline")
    ap.add_argument("--checkpoint")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if args.run_dir:
        run_dir = pathlib.Path(args.run_dir)
        cfg = load_config(run_dir / "config.yaml")
    elif args.config and args.zero_policy:
        cfg = load_config(args.config)
        run_dir = ROOT / "runs" / f"passive_{cfg['experiment_name']}"
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        ap.error("give a run_dir, or --config together with --zero-policy")
    train_cfg = deep_merge(cfg, {})
    for key, value in parse_overrides(args.set):
        set_dotted(cfg, key, value)

    import jax
    import jax.numpy as jp
    import numpy as np

    from quadrucat import agents, cat_envs, envs, metrics

    name = cfg["env"]["name"]
    kind = cat_envs.task_kind(name)
    if kind == "walk" and name not in cat_envs.CATWALK_LIKE:
        # Measure register on a non-cat run: same physics/obs, cat bookkeeping, cat terms at 0.
        cfg = deep_merge(cfg, {"env": {"name": "Go1CatWalk"}})
    env, _ = envs.make_env(cfg)

    if args.zero_policy:
        def policy(obs, key):
            return jp.zeros(env.action_size), {}
    else:
        from brax.io import model

        params = model.load_params(str(find_params(run_dir, args.checkpoint)))
        # obs sizes exactly as in training; only the physics backend follows the eval --set
        impl = (cfg["env"].get("overrides") or {}).get("impl")
        policy_cfg = deep_merge(train_cfg, {"env": {"overrides": {"impl": impl}}}) if impl else train_cfg
        policy_env, _ = envs.make_env(policy_cfg)
        make_policy = agents.make_inference_fn(train_cfg, policy_env)
        policy = make_policy(params[:2] if isinstance(params, (tuple, list)) else params, deterministic=True)

    ecfg = cfg.get("eval", {})
    dt = float(env.dt)
    torso = env.mj_model.body("trunk").id
    rng = jax.random.PRNGKey(args.seed)
    results = {"tag": args.tag, "kind": kind, "env": name, "overrides": args.set,
               "zero_policy": args.zero_policy}

    if kind == "walk":
        n_eps = args.episodes or ecfg.get("num_episodes", 8)
        ep_len = ecfg.get("episode_length", 1000)
        warmup = ecfg.get("warmup_steps", 50)
        tol = float(env._config.reward_config.register_hit_tol)

        def set_cmd(state, cmd):
            info = dict(state.info)
            info["command"] = cmd
            return state.replace(info=info)

        def run_episode(k, cmd):
            k, k0 = jax.random.split(k)
            state = set_cmd(env.reset(k0), cmd)

            def body(carry, _):
                state, k, alive = carry
                k, k1 = jax.random.split(k)
                action, _ = policy(state.obs, k1)
                ns = set_cmd(env.step(set_cmd(state, cmd), action), cmd)
                alive = alive * (1.0 - ns.done)
                out = (env.get_local_linvel(ns.data), env.get_gyro(ns.data), alive,
                       ns.info["last_contact"].astype(jp.float32), ns.info["register_dist"],
                       ns.info["hind_touchdown"])
                return (ns, k, alive), out

            _, out = jax.lax.scan(body, (state, k, jp.float32(1.0)), None, length=ep_len)
            return out

        run_batch = jax.jit(jax.vmap(run_episode, in_axes=(0, None)))
        rows = []
        for cmd in ecfg.get("commands", [[0.5, 0.0, 0.0]]):
            rng, k = jax.random.split(rng)
            lin, gyro, alive, contact, dist, td = jax.device_get(
                run_batch(jax.random.split(k, n_eps), jp.array(cmd, dtype=jp.float32)))
            row = metrics.tracking_metrics(cmd, lin[..., :2], gyro[..., 2], alive, dt, warmup)
            row.update(metrics.gait_metrics(contact, alive, dt, warmup))
            row.update(metrics.register_metrics(dist, td, alive, dt, tol, warmup))
            rows.append(row)
            print(f"[cat-walk] cmd={cmd}  lin_rmse={row['lin_vel_rmse']:.3f}  falls={row['fall_rate']:.2f}  "
                  f"register_med={row['register_err_median']:.3f} m  hit={row['register_hit_rate']:.2f}  "
                  f"3-foot={row['three_foot_frac']:.2f}  duty={row['duty_factor']:.2f}")
        keys = ["lin_vel_rmse", "fall_rate", "register_err_median", "register_hit_rate",
                "three_foot_frac", "duty_factor", "hind_steps_per_s"]
        results["per_command"] = rows
        results["mean"] = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    else:
        n_eps = args.episodes or ecfg.get("num_episodes", 256)
        ep_len = ecfg.get("episode_length", int(env._config.episode_length))

        def run_episode(k):
            k, k0 = jax.random.split(k)
            state = env.reset(k0)

            def body(carry, _):
                state, k = carry
                k, k1 = jax.random.split(k)
                action, _ = policy(state.obs, k1)
                ns = env.step(state, action)
                i = ns.info
                out = (i["feet_only_touch"], i["touch_up_z"], i["trunk_touch"], i["acc_norm"],
                       -env.get_gravity(ns.data)[2], ns.data.xpos[torso][2], i["landed"].astype(jp.float32))
                return (ns, k), out

            _, out = jax.lax.scan(body, (state, k), None, length=ep_len)
            return state.info["init_tilt"], out

        tilt, (feet_only, up0, trunk, acc, up_z, height, landed) = jax.device_get(
            jax.jit(jax.vmap(run_episode))(jax.random.split(rng, n_eps)))
        res = metrics.landing_metrics(init_tilt=tilt, feet_only_touch=feet_only, touch_up_z=up0[:, -1],
                                      trunk_touch=trunk, acc_norm=acc, up_z=up_z, height=height, landed=landed,
                                      dt=dt, touch_up_z_min=float(env._config.reward_config.land_upright_up_z))
        results["mean"] = res
        for k, v in res.items():
            print(f"[cat-fall] {k:32s} {v:8.3f}")

    out_dir = run_dir / "eval_cat"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"[eval-cat] -> {out_dir / (args.tag + '.json')}")

    if args.video:
        import mediapy

        jit_reset, jit_step, jit_policy = jax.jit(env.reset), jax.jit(env.step), jax.jit(policy)
        n_vid = 3 if kind == "fall" else 1
        cmd = jp.array((ecfg.get("commands") or [[0.5, 0.0, 0.0]])[0], dtype=jp.float32)
        for i in range(n_vid):
            rng, k = jax.random.split(rng)
            state = jit_reset(k)
            if kind == "walk":
                state = state.replace(info={**state.info, "command": cmd})
            traj = [state]
            for _ in range(min(ecfg.get("video_steps", 300), 1000)):
                rng, k = jax.random.split(rng)
                action, _ = jit_policy(state.obs, k)
                state = jit_step(state, action)
                if kind == "walk":
                    state = state.replace(info={**state.info, "command": cmd})
                traj.append(state)
            frames = env.render(traj, camera=ecfg.get("camera", "track"), height=480, width=640)
            path = out_dir / f"{args.tag}_{kind}{i}.mp4"
            mediapy.write_video(str(path), frames, fps=int(round(1.0 / dt)))
            print(f"[eval-cat] video -> {path}")


if __name__ == "__main__":
    main()
