"""Policy observation built from raw robot signals, shared by every deployment target.

Pure NumPy, no JAX/ROS. Used by the ROS2/Gazebo node and by scripts/sim2sim_mujoco.py,
and checked against the Playground env in tests/test_observation_matches_env.py.

The Go1 joystick 'state' observation (playground 0.2.0, history_len=1) is
  [local_linvel(3), gyro(3), gravity(3), q - q_default(12), dq(12), last_action(12), command(3)]
where local_linvel and gyro are measured at the IMU site (body-aligned, offset from the
trunk origin), and gravity is the world -z axis expressed in the body frame.
"""
from __future__ import annotations

import numpy as np

OBS_LAYOUT = [
    ("local_linvel", 3),
    ("gyro", 3),
    ("gravity", 3),
    ("joint_pos_minus_default", 12),
    ("joint_vel", 12),
    ("last_action", 12),
    ("command", 3),
]
OBS_SIZE = sum(n for _, n in OBS_LAYOUT)


def quat_to_rotmat(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Rotation matrix (body -> world) from a unit quaternion.

    order='xyzw' for ROS messages, 'wxyz' for MuJoCo qpos.
    """
    q = np.asarray(quat, dtype=np.float64)
    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError(order)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def projected_gravity(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """World gravity direction (0, 0, -1) expressed in the body frame."""
    return quat_to_rotmat(quat, order).T @ np.array([0.0, 0.0, -1.0])


def velocity_at_point(lin_vel_body: np.ndarray, ang_vel_body: np.ndarray, offset_body: np.ndarray) -> np.ndarray:
    """Linear velocity of a point rigidly attached to the body (all in body frame).

    Gazebo odometry gives the trunk-origin velocity; MuJoCo's velocimeter measures
    at the IMU site, so v_imu = v_trunk + w x r_imu.
    """
    return np.asarray(lin_vel_body) + np.cross(ang_vel_body, offset_body)


def build_observation(
    *,
    local_linvel: np.ndarray,
    gyro: np.ndarray,
    gravity: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    default_pose: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    obs = np.concatenate([
        np.asarray(local_linvel, dtype=np.float64),
        np.asarray(gyro, dtype=np.float64),
        np.asarray(gravity, dtype=np.float64),
        np.asarray(joint_pos, dtype=np.float64) - np.asarray(default_pose, dtype=np.float64),
        np.asarray(joint_vel, dtype=np.float64),
        np.asarray(last_action, dtype=np.float64),
        np.asarray(command, dtype=np.float64),
    ])
    if obs.shape != (OBS_SIZE,):
        raise ValueError(f"observation has shape {obs.shape}, expected ({OBS_SIZE},)")
    return obs


def is_fallen(gravity_body: np.ndarray, threshold: float = 0.0) -> bool:
    """Same criterion as the Playground Go1 termination: body z axis points down.

    Playground terminates when upvector_z < 0, i.e. gravity_body_z > 0.
    """
    return bool(gravity_body[2] > threshold)


class ActionHistory:
    """Tracks which past action goes into the observation.

    Quirk of the Playground Go1 joystick env (0.2.0): step() builds the observation
    BEFORE updating info["last_act"], so the 'last_action' slot seen by the policy when
    choosing a_t is a_{t-2} (the action before the most recent one), not a_{t-1}.
    Deployment must reproduce this or the policy sees out-of-distribution inputs.
    Verified in tests/test_observation_matches_env.py.
    """

    def __init__(self, action_size: int = 12):
        self.latest = np.zeros(action_size)    # a_{t-1}: most recently applied action
        self.previous = np.zeros(action_size)  # a_{t-2}: what the observation uses

    def for_observation(self) -> np.ndarray:
        return self.previous

    def push(self, action: np.ndarray) -> None:
        self.previous = self.latest
        self.latest = np.asarray(action, dtype=np.float64).copy()
