"""The deployment observation (quadrucat.observation) must equal the Playground one.

Rebuilds the policy observation from raw state (what Gazebo can give us: trunk
velocity, IMU orientation/gyro, joint states) and compares it with
  1. MuJoCo's own sensors (gyro, velocimeter at the IMU site, site frame) for the same state;
  2. the env's noise-free observation for the joint/action/command part, which also
     checks the action-history quirk handled by ActionHistory.
The env reads its IMU sensors from the last physics substep's forward pass, i.e.
one sim_dt (4 ms) before the final qpos/qvel, so the IMU part is compared via (1).
Skipped when mujoco_playground is not installed.
"""
import numpy as np
import pytest

pytest.importorskip("mujoco_playground")
jax = pytest.importorskip("jax")
mujoco = pytest.importorskip("mujoco")

from quadrucat.observation import ActionHistory, build_observation, projected_gravity, quat_to_rotmat, velocity_at_point  # noqa: E402


def test_observation_matches_env():
    from mujoco_playground import registry

    cfg = registry.get_default_config("Go1JoystickFlatTerrain")
    cfg.impl = "jax"
    cfg.noise_config.level = 0.0
    env = registry.load("Go1JoystickFlatTerrain", config=cfg)
    model = env.mj_model
    mj_data = mujoco.MjData(model)
    imu_offset = np.array(model.site("imu").pos)
    imu_id = model.site("imu").id
    default_pose = np.array(env._default_pose)

    def sensor(name):
        s = model.sensor(name)
        return np.array(mj_data.sensordata[s.adr[0]: s.adr[0] + s.dim[0]])

    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    hist = ActionHistory(env.action_size)
    for i in range(30):
        action = rng.uniform(-0.3, 0.3, size=env.action_size).astype(np.float32)
        state = step(state, action)
        hist.push(action)
        if i % 5:
            continue
        qpos, qvel = np.array(state.data.qpos), np.array(state.data.qvel)
        rot = quat_to_rotmat(qpos[3:7], order="wxyz")
        v_body = rot.T @ qvel[0:3]          # free-joint linear velocity is in world frame
        w_body = qvel[3:6]                  # free-joint angular velocity is in body frame
        obs = build_observation(
            local_linvel=velocity_at_point(v_body, w_body, imu_offset),
            gyro=w_body,
            gravity=projected_gravity(qpos[3:7], order="wxyz"),
            joint_pos=qpos[7:],
            joint_vel=qvel[6:],
            default_pose=default_pose,
            last_action=hist.for_observation(),
            command=np.array(state.info["command"]),
        )
        # (1) IMU part vs MuJoCo sensors evaluated at exactly this state
        mj_data.qpos[:], mj_data.qvel[:] = qpos, qvel
        mujoco.mj_forward(model, mj_data)
        np.testing.assert_allclose(obs[0:3], sensor("local_linvel"), atol=1e-5)
        np.testing.assert_allclose(obs[3:6], sensor("gyro"), atol=1e-5)
        np.testing.assert_allclose(obs[6:9], mj_data.site_xmat[imu_id].reshape(3, 3).T @ [0, 0, -1], atol=1e-6)
        # (2) joints, last action, command vs the env observation
        np.testing.assert_allclose(obs[9:], np.array(state.obs["state"])[9:], atol=1e-5)
