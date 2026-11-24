"""Evaluate a trained policy on fixed velocity commands, optionally render a video.

Metrics (per command, averaged over episodes) are the ones we also log in Gazebo,
so MuJoCo vs Gazebo numbers are directly comparable:
  lin_vel_rmse   RMSE between commanded and measured body-frame (vx, vy)   [m/s]
  yaw_rate_rmse  RMSE between commanded and measured yaw rate              [rad/s]
  fall_rate      fraction of episodes that terminated early
  survival_s     mean time alive                                           [s]

Robustness tests reuse the env config overrides on the eval env only, e.g.
  python scripts/evaluate.py runs/<run> --tag pushes \
      --set env.overrides.pert_config.enable=true
  python scripts/evaluate.py runs/<run> --tag noise2x \
      --set env.overrides.noise_config.level=2.0
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
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint", help="specific params .pkl (default: final or latest)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides for the EVAL env/config")
    ap.add_argument("--tag", default="default", help="name for this eval condition")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--video", action="store_true", help="also render one episode per command")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    train_cfg = deep_merge(cfg, {})
    for key, value in parse_overrides(args.set):
        set_dotted(cfg, key, value)

    import jax
    import jax.numpy as jp
    from brax.io import model

    from quadrucat import agents, envs, metrics

    # Eval env: training config + eval overrides. No domain randomization here
    # (nominal model) unless you explicitly script it.
    env, _ = envs.make_env(cfg)
    if cfg.get("algo", "ppo") == "sac":
        env = envs.StateOnlyObs(env)
    params = model.load_params(str(find_params(run_dir, args.checkpoint)))
    make_policy = agents.make_inference_fn(train_cfg, env)
    policy = make_policy(params[:2] if isinstance(params, (tuple, list)) else params, deterministic=True)

    ecfg = cfg.get("eval", {})
    n_eps = args.episodes or ecfg.get("num_episodes", 8)
    ep_len = ecfg.get("episode_length", 1000)
    warmup = ecfg.get("warmup_steps", 50)
    commands = ecfg.get("commands", [[1.0, 0.0, 0.0]])

    def set_cmd(state, cmd):
        info = dict(state.info)
        info["command"] = cmd
        return state.replace(info=info)

    def run_episode(rng, cmd):
        rng, k = jax.random.split(rng)
        state = set_cmd(env.reset(k), cmd)

        def body(carry, _):
            state, rng, alive = carry
            rng, k = jax.random.split(rng)
            action, _ = policy(state.obs, k)
            nstate = set_cmd(env.step(set_cmd(state, cmd), action), cmd)
            alive = alive * (1.0 - nstate.done)
            linvel = env.get_local_linvel(nstate.data)
            gyro = env.get_gyro(nstate.data)
            return (nstate, rng, alive), (linvel, gyro, alive)

        _, (linvel, gyro, alive) = jax.lax.scan(body, (state, rng, jp.float32(1.0)), None, length=ep_len)
        return linvel, gyro, alive

    run_batch = jax.jit(jax.vmap(run_episode, in_axes=(0, None)))
    dt = float(env.dt)
    results = {"tag": args.tag, "overrides": args.set, "episodes": n_eps, "per_command": []}
    rng = jax.random.PRNGKey(args.seed)
    for cmd in commands:
        rng, k = jax.random.split(rng)
        linvel, gyro, alive = jax.device_get(run_batch(jax.random.split(k, n_eps), jp.array(cmd, dtype=jp.float32)))
        row = metrics.tracking_metrics(cmd, linvel[..., :2], gyro[..., 2], alive, dt, warmup)
        results["per_command"].append(row)
        print("[eval]", metrics.format_row(row))

    results["mean"] = metrics.summarize(results["per_command"])
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.tag}.json").write_text(json.dumps(results, indent=2))
    print(f"[eval] mean: {results['mean']}  -> {out_dir / (args.tag + '.json')}")

    if args.video:
        import mediapy

        jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
        jit_policy = jax.jit(policy)
        for i, cmd in enumerate(commands):
            cmd_arr = jp.array(cmd, dtype=jp.float32)
            rng, k = jax.random.split(rng)
            state = set_cmd(jit_reset(k), cmd_arr)
            traj = [state]
            for _ in range(min(ep_len, ecfg.get("video_steps", 500))):
                rng, k = jax.random.split(rng)
                action, _ = jit_policy(state.obs, k)
                state = set_cmd(jit_step(set_cmd(state, cmd_arr), action), cmd_arr)
                traj.append(state)
                if bool(state.done):
                    break
            frames = env.render(traj, camera=ecfg.get("camera", "track"), height=480, width=640)
            path = out_dir / f"{args.tag}_cmd{i}.mp4"
            mediapy.write_video(str(path), frames, fps=int(round(1.0 / dt)))
            print(f"[eval] video -> {path}")


if __name__ == "__main__":
    main()
