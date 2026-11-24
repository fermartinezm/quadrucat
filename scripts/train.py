"""Train a Go1 velocity-tracking policy with Brax PPO (or SAC) on MuJoCo Playground.

Examples:
  python scripts/train.py --config configs/smoke.yaml
  python scripts/train.py --config configs/base.yaml --seed 1 --notes "baseline, seed 1"
  python scripts/train.py --config configs/base.yaml --name ablate_feet_slip \
      --set env.overrides.reward_config.scales.feet_slip=0.0

Outputs go to runs/<name>_<timestamp>/: config.yaml, env_config.json,
algo_params.json, metrics.jsonl, checkpoints/, params.pkl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import inspect
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quadrucat.config import load_config, save_config  # noqa: E402


def _to_float_dict(metrics) -> dict:
    out = {}
    for k, v in metrics.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", help="experiment name (default: experiment_name in config)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--notes", help="hypothesis / what you changed (saved in config.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.name:
        cfg["experiment_name"] = args.name
    if args.notes:
        cfg["notes"] = args.notes
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("seed", 0)
    cfg.setdefault("algo", "ppo")

    import jax
    from brax.io import model
    from mujoco_playground import wrapper

    from quadrucat import agents, envs

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = pathlib.Path(args.runs_dir) / f"{cfg['experiment_name']}_s{cfg['seed']}_{stamp}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    print(f"[train] run dir: {run_dir}")
    print(f"[train] devices: {jax.devices()}")

    env, env_cfg = envs.make_env(cfg)
    eval_env, _ = envs.make_env(cfg)
    if cfg["algo"] == "sac":  # brax SAC needs flat observations
        env, eval_env = envs.StateOnlyObs(env), envs.StateOnlyObs(eval_env)
    (run_dir / "env_config.json").write_text(env_cfg.to_json_best_effort(indent=2))

    params = agents.algo_params(cfg)
    (run_dir / "algo_params.json").write_text(json.dumps(params, indent=2, default=str))
    factory, _ = agents.network_factory(cfg, params)
    train_kwargs = {k: v for k, v in params.items() if k != "network_factory"}

    metrics_path = run_dir / "metrics.jsonl"
    t0 = time.time()

    def progress_fn(num_steps, metrics):
        row = {"step": int(num_steps), "wall_time": time.time() - t0, **_to_float_dict(metrics)}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[train] step {num_steps:>12,d}  return {row.get('eval/episode_reward', float('nan')):8.3f}  "
              f"ep_len {row.get('eval/avg_episode_length', float('nan')):6.0f}  "
              f"track_lin {row.get('eval/episode_reward/tracking_lin_vel', float('nan')):6.2f}  "
              f"kl {row.get('training/kl_mean', float('nan')):.4f}  "
              f"({row['wall_time']:.0f}s)"
              + (f"  landed_on_feet {row['eval/episode_landed_on_feet']:.2f}" if 'eval/episode_landed_on_feet' in row else "")
              + (f"  register_hits {row['eval/episode_register_hits']:.0f}/{row.get('eval/episode_hind_touchdowns', 0):.0f}"
                 if 'eval/episode_register_hits' in row else ""))

    def policy_params_fn(current_step, make_policy, params):
        del make_policy
        model.save_params(str(run_dir / "checkpoints" / f"params_{int(current_step):012d}.pkl"), params)

    if cfg["algo"] == "ppo":
        from brax.training.agents.ppo import train as algo_train
    else:
        from brax.training.agents.sac import train as algo_train

    kwargs = dict(
        **train_kwargs,
        network_factory=factory,
        randomization_fn=envs.get_randomizer(cfg),
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        seed=cfg["seed"],
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    # Brax versions differ slightly (esp. SAC); drop kwargs this version doesn't accept.
    accepted = inspect.signature(algo_train.train).parameters
    dropped = [k for k in kwargs if k not in accepted]
    if dropped:
        print(f"[train] WARNING: {algo_train.__name__}.train() ignores {dropped}")
    train_fn = functools.partial(algo_train.train, **{k: v for k, v in kwargs.items() if k in accepted})
    _, params_out, _ = train_fn()
    model.save_params(str(run_dir / "params.pkl"), params_out)
    print(f"[train] done in {time.time() - t0:.0f}s -> {run_dir / 'params.pkl'}")


if __name__ == "__main__":
    main()
