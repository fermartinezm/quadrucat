"""Run an exported policy.npz in plain MuJoCo (C engine, CPU) with the SAME code path as
the ROS2/Gazebo node: NumPy policy, quadrucat.observation, our own PD loop.

This is the stepping stone between training (MJX) and Gazebo: if the policy walks here
in `--control effort` mode, the deployment logic is right and any remaining Gazebo
gap comes from the simulator (contacts, solver, actuator model), not from our code.

  python scripts/sim2sim_mujoco.py runs/<run>                    # metrics for eval commands
  python scripts/sim2sim_mujoco.py runs/<run> --control actuator # use MuJoCo's position actuators
  MUJOCO_GL=egl python scripts/sim2sim_mujoco.py runs/<run> --video
  python scripts/sim2sim_mujoco.py runs/<run> --obs-noise 1.0 --push

Writes runs/<run>/eval/sim2sim_<tag>.json with the same metrics as evaluate.py.
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

import numpy as np  # noqa: E402

from quadrucat import metrics  # noqa: E402
from quadrucat.config import load_config  # noqa: E402
from quadrucat.observation import (ActionHistory, build_observation, is_fallen, projected_gravity,  # noqa: E402
                                   quat_to_rotmat, velocity_at_point)
from quadrucat.policy_numpy import NumpyPolicy  # noqa: E402

# Uniform noise amplitudes used in training (noise_config.scales, level=1).
NOISE_SCALES = dict(linvel=0.1, gyro=0.2, gravity=0.05, joint_pos=0.03, joint_vel=1.5)


def load_model(cfg: dict):
    """The exact compiled MjModel used in training (Kp/Kd already applied by the env)."""
    from mujoco_playground import registry

    from quadrucat import cat_envs

    cat_envs.register()  # Go1CatWalk uses the same model/observation as the joystick task
    env_cfg = registry.get_default_config(cfg["env"]["name"])
    env_cfg.impl = "jax"  # only used to build the env object; we step with MuJoCo C
    env = registry.load(cfg["env"]["name"], config=env_cfg)
    return env.mj_model


def run_episode(model, data, policy, meta, cmd, steps, control, rng, obs_noise, push, renderer=None):
    import mujoco

    sim_dt = model.opt.timestep
    n_sub = int(round(meta["ctrl_dt"] / sim_dt))
    default = np.array(meta["default_pose"])
    kp, kd, scale = meta["Kp"], meta["Kd"], meta["action_scale"]
    imu_offset = np.array(model.site("imu").pos)
    torque_limit = model.actuator_forcerange[:, 1].copy()

    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    if control == "effort":
        # Disable MuJoCo's position actuators and passive joint damping (Kd): we apply the PD
        # torque ourselves, exactly like the ROS2 node does.
        model.actuator_gainprm[:, 0] = 0.0
        model.actuator_biasprm[:, 1] = 0.0
        model.actuator_biasprm[:, 2] = 0.0
        model.dof_damping[6:] = 0.0
    mujoco.mj_forward(model, data)

    hist = ActionHistory(len(default))
    q_target = default.copy()
    log_v, log_w, log_alive, frames = [], [], [], []
    alive = 1.0
    next_push = rng.uniform(1.0, 3.0)
    for t in range(steps):
        qpos, qvel = data.qpos, data.qvel
        rot = quat_to_rotmat(qpos[3:7], order="wxyz")
        w_body = qvel[3:6].copy()
        v_imu = velocity_at_point(rot.T @ qvel[0:3], w_body, imu_offset)
        gravity = projected_gravity(qpos[3:7], order="wxyz")
        noise = (lambda k, n: obs_noise * NOISE_SCALES[k] * rng.uniform(-1, 1, n)) if obs_noise > 0 else (lambda k, n: 0.0)
        obs = build_observation(
            local_linvel=v_imu + noise("linvel", 3), gyro=w_body + noise("gyro", 3),
            gravity=gravity + noise("gravity", 3), joint_pos=qpos[7:] + noise("joint_pos", 12),
            joint_vel=qvel[6:] + noise("joint_vel", 12), default_pose=default,
            last_action=hist.for_observation(), command=cmd)
        action = policy(obs)
        hist.push(action)
        q_target = default + scale * action

        if push and t * meta["ctrl_dt"] >= next_push:  # velocity kick like pert_config
            angle = rng.uniform(0, 2 * np.pi)
            data.qvel[0:2] += rng.uniform(0.0, 1.0) * np.array([np.cos(angle), np.sin(angle)])
            next_push += rng.uniform(1.0, 3.0)

        for _ in range(n_sub):
            if control == "effort":
                tau = kp * (q_target - data.qpos[7:]) - kd * data.qvel[6:]
                data.qfrc_applied[6:] = np.clip(tau, -torque_limit, torque_limit)
            else:
                data.ctrl[:] = q_target
            mujoco.mj_step(model, data)

        if is_fallen(projected_gravity(data.qpos[3:7], order="wxyz")):
            alive = 0.0
        log_v.append(velocity_at_point(quat_to_rotmat(data.qpos[3:7], "wxyz").T @ data.qvel[0:3],
                                       data.qvel[3:6], imu_offset)[:2])
        log_w.append(data.qvel[5])
        log_alive.append(alive)
        if renderer is not None:
            renderer.update_scene(data, camera="track")
            frames.append(renderer.render())
        if not alive:
            log_v += [log_v[-1]] * (steps - t - 1)
            log_w += [log_w[-1]] * (steps - t - 1)
            log_alive += [0.0] * (steps - t - 1)
            break
    return np.array(log_v), np.array(log_w), np.array(log_alive), frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--policy", help="policy.npz (default: <run>/policy.npz)")
    ap.add_argument("--control", choices=["effort", "actuator"], default="effort")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, help="control steps per episode (default: eval.episode_length)")
    ap.add_argument("--obs-noise", type=float, default=0.0, help="noise level (1.0 = training noise)")
    ap.add_argument("--push", action="store_true", help="random velocity kicks")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mujoco

    run_dir = pathlib.Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    policy = NumpyPolicy(args.policy or str(run_dir / "policy.npz"))
    meta = policy.metadata
    model = load_model(cfg)
    data = mujoco.MjData(model)
    ecfg = cfg.get("eval", {})
    steps = args.steps or ecfg.get("episode_length", 1000)
    warmup = min(ecfg.get("warmup_steps", 50), steps - 1)
    tag = args.tag or f"sim2sim_{args.control}" + ("_noise" if args.obs_noise else "") + ("_push" if args.push else "")
    rng = np.random.default_rng(args.seed)
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)

    renderer = mujoco.Renderer(model, 480, 640) if args.video else None
    results = {"tag": tag, "control": args.control, "episodes": args.episodes, "per_command": []}
    for i, cmd in enumerate(ecfg.get("commands", [[0.5, 0.0, 0.0]])):
        runs = [run_episode(model, data, policy, meta, np.array(cmd, float), steps, args.control, rng,
                            args.obs_noise, args.push, renderer if (args.video and e == 0) else None)
                for e in range(args.episodes)]
        v, w, alive = (np.stack([r[k] for r in runs]) for k in range(3))
        row = metrics.tracking_metrics(cmd, v, w, alive, meta["ctrl_dt"], warmup)
        results["per_command"].append(row)
        print("[sim2sim]", metrics.format_row(row))
        if args.video and runs[0][3]:
            import mediapy

            path = out_dir / f"{tag}_cmd{i}.mp4"
            mediapy.write_video(str(path), runs[0][3], fps=int(round(1.0 / meta["ctrl_dt"])))
            print(f"[sim2sim] video -> {path}")
    results["mean"] = metrics.summarize(results["per_command"])
    (out_dir / f"{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"[sim2sim] mean: {results['mean']}  -> {out_dir / (tag + '.json')}")


if __name__ == "__main__":
    main()
