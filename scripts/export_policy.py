"""Export a trained PPO actor to a NumPy .npz for the ROS2 node (no JAX at deploy time).

  python scripts/export_policy.py runs/<run>            # -> runs/<run>/policy.npz

The file also stores what the Gazebo side must match (read from the env itself,
so it follows the installed Playground version): default joint pose, action
scale, control period, PD gains, torque limits, joint/actuator order, IMU offset
and observation layout.
After writing, the script checks that NumPy and JAX give the same actions.
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

from quadrucat.config import load_config  # noqa: E402
from quadrucat.observation import OBS_LAYOUT  # noqa: E402

def _get(cfg, key, default=None):
    try:
        return cfg[key]
    except (KeyError, AttributeError):
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out")
    args = ap.parse_args()

    import jax
    import numpy as np
    from brax.io import model

    from quadrucat import agents, envs
    from quadrucat.config import find_params
    from quadrucat.policy_numpy import NumpyPolicy

    run_dir = pathlib.Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    if cfg.get("algo", "ppo") != "ppo":
        raise SystemExit("export currently supports PPO runs only (SAC actor layout differs)")
    out = pathlib.Path(args.out) if args.out else run_dir / "policy.npz"

    env, env_cfg = envs.make_env(cfg)
    params = model.load_params(str(find_params(run_dir, args.checkpoint)))
    normalizer, policy_params = params[0], params[1]
    algo = agents.algo_params(cfg)
    net_kwargs = dict(algo.get("network_factory") or {})
    obs_key = net_kwargs.get("policy_obs_key", "state")

    mean, std = normalizer.mean, normalizer.std
    if isinstance(mean, dict):
        mean, std = mean[obs_key], std[obs_key]
    if not algo.get("normalize_observations", True):
        mean, std = np.zeros_like(mean), np.ones_like(std)

    layers = policy_params["params"]
    names = sorted(layers.keys(), key=lambda n: int(n.split("_")[-1]))
    arrays = {"obs_mean": np.asarray(mean, np.float32), "obs_std": np.asarray(std, np.float32)}
    for i, name in enumerate(names):
        if set(layers[name].keys()) != {"kernel", "bias"}:
            raise SystemExit(f"Unexpected layer '{name}' with keys {list(layers[name])} (layer norm?) - extend exporter")
        arrays[f"W{i}"] = np.asarray(layers[name]["kernel"], np.float32)
        arrays[f"b{i}"] = np.asarray(layers[name]["bias"], np.float32)

    obs_size = arrays["W0"].shape[0]
    layout = list(OBS_LAYOUT)
    scan_meta = None
    if cfg["env"]["name"] == "Go1Terrain" and env_cfg.height_scan.in_actor:
        # terrain policy: the 48 values + the height scan; the deployment rebuilds the same map from
        # `terrain` and computes the scan from the trunk pose (quadrucat.terrain.height_scan_np)
        hs = env_cfg.height_scan
        n_scan = int(hs.x[2]) * int(hs.y[2])
        layout.append(("height_scan", n_scan))
        scan_meta = {"x": list(hs.x), "y": list(hs.y), "offset": float(hs.offset),
                     "terrain": env_cfg.terrain.to_dict(),
                     "heading_command": env_cfg.heading_command.to_dict()}
    layout_size = sum(n for _, n in layout)
    if layout_size != obs_size:
        raise SystemExit(f"[export] OBS_LAYOUT sums to {layout_size} but the policy expects {obs_size}. "
                         "history_len or the obs definition changed: update quadrucat/observation.py first.")

    mj = env.mj_model
    act_joints = [mj.joint(int(mj.actuator_trnid[i, 0])).name for i in range(mj.nu)]
    metadata = {
        "env_name": cfg["env"]["name"],
        "activation": net_kwargs.get("activation", "swish"),
        "action_size": int(env.action_size),
        "obs_size": int(obs_size),
        "obs_layout": layout,
        "height_scan": scan_meta,
        "joint_names": act_joints,
        "default_pose": np.asarray(env._default_pose).tolist(),
        "action_scale": float(_get(env_cfg, "action_scale", 1.0)),
        "ctrl_dt": float(env.dt),
        "sim_dt": float(_get(env_cfg, "sim_dt", float("nan"))),
        "Kp": float(_get(env_cfg, "Kp", float("nan"))),
        "Kd": float(_get(env_cfg, "Kd", float("nan"))),
        "torque_limits": [float(mj.actuator_forcerange[i, 1]) for i in range(mj.nu)],
        "imu_offset": np.asarray(mj.site("imu").pos).tolist(),  # IMU site in trunk frame
        "action_history_quirk": "obs last_action slot holds a_{t-2}; see quadrucat.observation.ActionHistory",
        "run_dir": str(run_dir),
    }
    np.savez(out, metadata=np.array(json.dumps(metadata)), **arrays)
    print(f"[export] wrote {out}")
    print(json.dumps({k: v for k, v in metadata.items() if k != "obs_layout"}, indent=2))

    # --- consistency check: JAX policy vs NumPy policy on real observations ---
    make_policy = agents.make_inference_fn(cfg, env)
    jax_policy = jax.jit(make_policy(params[:2], deterministic=True))
    np_policy = NumpyPolicy(str(out))
    states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 16))
    a_jax = np.asarray(jax_policy(states.obs, jax.random.PRNGKey(1))[0])
    obs_np = np.asarray(states.obs[obs_key] if isinstance(states.obs, dict) else states.obs)
    a_np = np_policy(obs_np)
    err = float(np.abs(a_jax - a_np).max())
    print(f"[export] max |a_jax - a_numpy| = {err:.2e}  ({'OK' if err < 1e-4 else 'MISMATCH - do not deploy'})")


if __name__ == "__main__":
    main()
