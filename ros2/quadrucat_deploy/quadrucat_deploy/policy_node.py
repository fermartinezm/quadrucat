"""Run an exported quadrucat policy (policy.npz) on the Go1 in Gazebo through ROS2.

Subscribes
  joint_states (sensor_msgs/JointState)   joint positions/velocities
  imu          (sensor_msgs/Imu)          orientation (world->body) + angular velocity (body)
  odom         (nav_msgs/Odometry)        trunk twist, linear part in the BODY frame
  cmd_vel      (geometry_msgs/Twist)      velocity command (ignored when eval_commands is set)
Publishes
  command_topic (std_msgs/Float64MultiArray) for a forward_command_controller:
    command_mode=effort   -> PD torques at pd_rate_hz, tau = Kp (q* - q) - Kd dq, clipped to the
                             training torque limits (reproduces MuJoCo's position actuators;
                             verified in MuJoCo by scripts/sim2sim_mujoco.py --control effort)
    command_mode=position -> joint targets q* (Gazebo's own controller; gains differ, ablation only)

The observation is built by quadrucat.observation exactly as in training (tested against the
Playground env), including the velocity of the IMU point and the action-history quirk.

Evaluation mode: set eval_commands to a flat list [vx, vy, wz, vx, vy, wz, ...] and
eval_segment_s to the duration of each command. The node cycles through them, logs every
control step to log_csv and shuts down at the end; analyse with
scripts/analyze_gazebo_log.py (same metrics as scripts/evaluate.py).

Start-up: the node first ramps the joints from wherever they are to the default pose
(stand_ramp_s), holds it (stand_hold_s) and only then starts the policy, with the same zero
action history as an env reset. Gazebo has no keyframe reset, so without this the policy would
start from whatever pose the robot sank to while the controllers were loading.
"""
from __future__ import annotations

import csv
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

try:
    from ros_gz_interfaces.msg import Contacts
except ImportError:  # contacts are only needed for the gait metrics
    Contacts = None

try:
    from quadrucat.observation import ActionHistory, build_observation, is_fallen, projected_gravity, velocity_at_point
    from quadrucat.policy_numpy import NumpyPolicy
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "quadrucat is not importable. From the repo root run `pip install -e . --no-deps` "
        "in the same Python that ROS2 uses (or export PYTHONPATH=<repo>)."
    ) from e

CSV_HEADER = ["t", "segment", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "gravity_z", "fallen",
              *[f"q{i}" for i in range(12)], *[f"tau{i}" for i in range(12)],
              "px", "py", "pz", "qx", "qy", "qz", "qw", "c_FR", "c_FL", "c_RR", "c_RL", "wall"]
FEET = ("FR", "FL", "RR", "RL")


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("quadrucat_policy")
        p = self.declare_parameter
        policy_path = p("policy_path", "").value
        if not policy_path:
            raise RuntimeError("Set the 'policy_path' parameter to runs/<run>/policy.npz")
        self.policy = NumpyPolicy(policy_path)
        meta = self.policy.metadata

        self.joint_names = list(p("joint_names", list(meta["joint_names"])).value)
        self.default_pose = np.array(meta["default_pose"], dtype=np.float64)
        self.action_scale = float(meta["action_scale"])
        self.ctrl_dt = float(p("ctrl_dt", float(meta["ctrl_dt"])).value)
        self.kp = float(p("kp", float(meta["Kp"])).value)
        self.kd = float(p("kd", float(meta["Kd"])).value)
        self.torque_limits = np.array(meta.get("torque_limits", [23.7, 23.7, 35.55] * 4), dtype=np.float64)
        self.imu_offset = np.array(meta.get("imu_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.mode = p("command_mode", "effort").value
        self.joint_states_topic = p("joint_states_topic", "/joint_states").value
        eval_cmds = list(p("eval_commands", [0.0]).value)
        self.eval_segment_s = float(p("eval_segment_s", 20.0).value)
        self.eval_commands = np.array(eval_cmds, dtype=np.float64).reshape(-1, 3) if len(eval_cmds) >= 3 else None
        log_csv = p("log_csv", "").value
        self.stand_ramp_s = float(p("stand_ramp_s", 1.0).value)
        self.stand_hold_s = float(p("stand_hold_s", 1.0).value)
        self.contact_hold_s = float(p("contact_hold_s", 0.004).value)

        n = len(self.joint_names)
        if n != self.policy.action_size or n != len(self.default_pose):
            raise RuntimeError(f"{n} joint names but the policy has {self.policy.action_size} actions")
        self.get_logger().info(
            f"policy {policy_path}: obs={self.policy.obs_size} act={n} ctrl_dt={self.ctrl_dt} "
            f"Kp={self.kp} Kd={self.kd} mode={self.mode} eval={'on' if self.eval_commands is not None else 'off'}\n"
            f" joints: {self.joint_names}")

        self.q = self.dq = self.quat = self.gyro = self.trunk_linvel = None
        self.cmd = np.zeros(3)
        self.history = ActionHistory(n)
        self.q_target = self.default_pose.copy()
        self._index = None
        self._t_start = None
        self._t_stand = self._q_stand = None
        self._fallen = False
        self.pos = np.full(3, np.nan)
        self.tau = np.zeros(n)
        self._last_contact = np.full(4, -np.inf)

        self.create_subscription(JointState, self.joint_states_topic, self._on_joints, qos_profile_sensor_data)
        self.create_subscription(Imu, p("imu_topic", "/imu").value, self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, p("odom_topic", "/odom").value, self._on_odom, qos_profile_sensor_data)
        if Contacts is not None:
            for i, foot in enumerate(FEET):
                self.create_subscription(Contacts, f"/contact/{foot}", lambda m, i=i: self._on_contact(m, i),
                                         qos_profile_sensor_data)
        self.create_subscription(Twist, p("cmd_vel_topic", "/cmd_vel").value, self._on_cmd, 10)
        self.pub = self.create_publisher(Float64MultiArray, p("command_topic", "/joint_commands/commands").value, 10)

        self.create_timer(self.ctrl_dt, self._control_step)
        if self.mode == "effort":
            self.create_timer(1.0 / float(p("pd_rate_hz", 500.0).value), self._pd_step)
        elif self.mode != "position":
            raise RuntimeError(f"command_mode must be 'effort' or 'position', got {self.mode!r}")

        self._csv = None
        if log_csv:
            self._csv_file = open(log_csv, "w", newline="")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(CSV_HEADER)
            self.get_logger().info(f"logging to {log_csv}")

    # ---------------- callbacks ----------------
    def _on_joints(self, msg: JointState) -> None:
        if self._index is None:
            missing = [j for j in self.joint_names if j not in msg.name]
            if missing:
                self.get_logger().error(f"joints missing from {self.joint_states_topic}: {missing}", once=True)
                return
            self._index = [list(msg.name).index(j) for j in self.joint_names]
        self.q = np.array([msg.position[i] for i in self._index])
        self.dq = np.array([msg.velocity[i] for i in self._index]) if msg.velocity else np.zeros(len(self._index))

    def _on_imu(self, msg: Imu) -> None:
        o, w = msg.orientation, msg.angular_velocity
        self.quat = np.array([o.x, o.y, o.z, o.w])
        self.gyro = np.array([w.x, w.y, w.z])

    def _on_odom(self, msg: Odometry) -> None:
        v = msg.twist.twist.linear
        self.trunk_linvel = np.array([v.x, v.y, v.z])
        x = msg.pose.pose.position
        self.pos = np.array([x.x, x.y, x.z])

    def _on_contact(self, msg, foot: int) -> None:
        if msg.contacts:
            self._last_contact[foot] = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_cmd(self, msg: Twist) -> None:
        if self.eval_commands is None:
            self.cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])

    # ---------------- control ----------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ready(self) -> bool:
        return all(x is not None for x in (self.q, self.quat, self.gyro, self.trunk_linvel))

    def _segment(self) -> int:
        if self.eval_commands is None:
            return 0
        seg = int((self._now() - self._t_start) // self.eval_segment_s)
        if seg >= len(self.eval_commands):
            self.get_logger().info("evaluation finished")
            raise SystemExit
        self.cmd = self.eval_commands[seg]
        return seg

    def _control_step(self) -> None:
        if not self._ready():
            return
        now = self._now()
        if self._t_stand is None:
            self._t_stand, self._q_stand = now, self.q.copy()
        since = now - self._t_stand
        if since < self.stand_ramp_s + self.stand_hold_s:
            a = min(since / max(self.stand_ramp_s, 1e-6), 1.0)
            self.q_target = (1 - a) * self._q_stand + a * self.default_pose
            # logged as segment -1 (negative time) so a bad start can be diagnosed
            gravity = projected_gravity(self.quat, order="xyzw")
            self._log(since - self.stand_ramp_s - self.stand_hold_s, -1, np.zeros(2), gravity, now)
            return
        if self._t_start is None:
            self._t_start = now
        segment = self._segment()

        linvel_imu = velocity_at_point(self.trunk_linvel, self.gyro, self.imu_offset)
        gravity = projected_gravity(self.quat, order="xyzw")
        obs = build_observation(
            local_linvel=linvel_imu, gyro=self.gyro, gravity=gravity,
            joint_pos=self.q, joint_vel=self.dq, default_pose=self.default_pose,
            last_action=self.history.for_observation(), command=self.cmd)
        action = self.policy(obs)
        self.history.push(action)
        self.q_target = self.default_pose + self.action_scale * action
        if self.mode == "position":
            self.pub.publish(Float64MultiArray(data=self.q_target.tolist()))

        self._fallen = self._fallen or is_fallen(gravity)
        self._log(now - self._t_start, segment, linvel_imu, gravity, now)

    def _log(self, t: float, segment: int, linvel_imu, gravity, now: float) -> None:
        if self._csv is None:
            return
        self._csv.writerow([f"{t:.4f}", segment, *self.cmd,
                            linvel_imu[0], linvel_imu[1], self.gyro[2], gravity[2], int(self._fallen),
                            *self.q, *self.tau, *self.pos, *self.quat,
                            *(now - self._last_contact <= self.contact_hold_s).astype(int),
                            f"{time.monotonic():.4f}"])

    def _pd_step(self) -> None:
        if self.q is None:
            return  # nothing to hold yet
        tau = self.kp * (self.q_target - self.q) - self.kd * self.dq
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        self.tau = tau
        self.pub.publish(Float64MultiArray(data=tau.tolist()))

    def destroy_node(self):
        if self._csv is not None:
            self._csv_file.close()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
