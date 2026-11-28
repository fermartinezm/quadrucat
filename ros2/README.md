# ROS2 / Gazebo deployment (`quadrucat_deploy`)

Runs a policy exported with `scripts/export_policy.py` (`policy.npz`, NumPy only, no JAX)
on the Go1 in **Gazebo Harmonic** through `ros2_control`. Target: **Ubuntu 24.04 + ROS 2 Jazzy**.

> Status: runs in Gazebo Harmonic. Results per policy version in `versions/<v>/report.md`
> (MuJoCo vs Gazebo). Main finding: with the physics capped at 1/4 speed (`sim_rate_hz:=250`) the
> cat walk v3 and the base policy transfer with no falls and MuJoCo-level tracking; at full speed
> the Python PD loop over ROS topics lags and some runs fall. So the gap is the control loop timing,
> not the physics.

## Pipeline

```
MJX training ──export──► policy.npz ──► sim2sim_mujoco.py (MuJoCo C, same code path as the node)
                                     └─► policy_node (ROS2) ──► Gazebo ──CSV──► analyze_gazebo_log.py
```

## Setup

```bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-gz-ros2-control ros-jazzy-ros2-controllers \
                 ros-jazzy-robot-state-publisher python3-colcon-common-extensions
source /opt/ros/jazzy/setup.bash
pip install -e /path/to/quadrucat --no-deps --break-system-packages   # quadrucat.observation / policy_numpy
mkdir -p ~/quadrucat_ws/src && ln -s /path/to/quadrucat/ros2/quadrucat_deploy ~/quadrucat_ws/src/
cd ~/quadrucat_ws && colcon build --symlink-install && source install/setup.bash
```

## Install without root (what was used: WSL2 Ubuntu 26.04, no sudo)

```bash
# micromamba + RoboStack Jazzy (Gazebo Harmonic) in ~/quadrucat_gz (~7 GB)
micromamba create -p ~/quadrucat_gz/env -c robostack-jazzy -c conda-forge \
  ros-jazzy-ros-base ros-jazzy-ros-gz ros-jazzy-gz-ros2-control ros-jazzy-ros2-controllers \
  ros-jazzy-robot-state-publisher ros-jazzy-xacro colcon-common-extensions mujoco-python ffmpeg pip pyyaml
# ros2-ros-gz-bridge 1.0.25 (build _31) links a libabseil that the channel does not ship; use 1.0.24:
micromamba install -p ~/quadrucat_gz/env -c robostack-jazzy -c conda-forge "ros2-ros-gz-bridge=1.0.24"
```

## Run

```bash
# Robot only (no policy = no torques, so it will sink to the ground; checks spawning/topics)
ros2 launch quadrucat_deploy gazebo.launch.py

# Policy, interactive
ros2 launch quadrucat_deploy gazebo.launch.py policy_path:=/abs/path/runs/<run>/policy.npz
ros2 topic pub -r 10 /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5}}"

# Policy, scripted evaluation with the same commands as configs/base.yaml (20 s each)
ros2 launch quadrucat_deploy gazebo.launch.py policy_path:=/abs/.../policy.npz headless:=true \
    log_csv:=/tmp/gz_eval.csv \
    eval_commands:="[0.5,0,0, 1.0,0,0, -0.5,0,0, 0,0.4,0, 0,0,1.0, 0.6,0,0.6]"
python scripts/analyze_gazebo_log.py /tmp/gz_eval.csv --run-dir runs/<run> --tag gazebo
python scripts/compare_sims.py collect runs/<run>   # versions/<v>/report.md: MuJoCo vs Gazebo

# Same with the physics capped at 1/4 speed (250 steps/s), so the ROS nodes keep up:
ros2 launch quadrucat_deploy gazebo.launch.py policy_path:=... headless:=true sim_rate_hz:=250 \
    log_csv:=/tmp/gz_slow.csv eval_commands:="[...]"
python scripts/analyze_gazebo_log.py /tmp/gz_slow.csv --run-dir runs/<run> --tag gazebo_slow
```

Several logs of the same eval can be pooled as repetitions:
`analyze_gazebo_log.py rt0.csv rt1.csv rt2.csv --tag gazebo`. The node first ramps the robot to the
default pose (2 s, logged as segment -1); logs where it was not upright when the policy started are
dropped. Gait metrics use foot contacts from the foot height (forward kinematics, < 2.8 cm), because
the bridged contact-sensor messages arrive in bursts.

## Package contents

| File | What |
|---|---|
| `urdf/go1.urdf` | **Generated** by `scripts/mjcf_to_urdf.py` from the training MjModel: same masses, inertias, joint frames/axes/limits, joint friction, feet; plus ros2_control (effort), IMU sensor at the MuJoCo IMU site and odometry. Do not edit by hand. |
| `meshes/` | Visual meshes from MuJoCo Menagerie (BSD-3, license included). |
| `worlds/flat.sdf` | Flat ground, friction 0.6 (as in training), 1 ms physics step. |
| `config/controllers.yaml` | joint_state_broadcaster + effort forward_command_controller (`joint_commands`). |
| `config/bridge.yaml` | `/clock`, `/imu`, `/odom`, `/contact/<foot>` from Gazebo to ROS. |
| `config/policy.yaml` | Policy node parameters. |
| `launch/gazebo.launch.py` | Gazebo + robot + bridge + controllers, then the policy (if `policy_path` is set). |
| `launch/policy.launch.py` | Policy node only (when the simulation is started separately). |
| `quadrucat_deploy/policy_node.py` | Observation → NumPy policy → PD torques; CSV logging; eval mode. |

## What is verified (tests in `tests/`, run without ROS)

* **Robot model**: the URDF loaded back into MuJoCo reproduces every body pose of the training
  model at random joint configurations, and every link's mass and inertia tensor.
* **Observation**: `quadrucat.observation` rebuilt from raw state (trunk twist + IMU + joints)
  equals MuJoCo's IMU sensors and the env observation. This caught a subtle detail: the env's
  "last action" slot is the action from **two** control steps ago (`ActionHistory`).
* **PD control**: in plain MuJoCo, applying our PD torques (`sim2sim_mujoco.py --control effort`)
  gives the same trajectories as MuJoCo's position actuators used in training.

## Known differences MuJoCo → Gazebo (expected sources of the transfer gap)

* Contact model and solver (MuJoCo soft contacts, 1 solver iteration, 4 ms step vs Gazebo's
  physics engine at 1 ms).
* Joint **armature** (rotor inertia, 0.005 kg·m² in MuJoCo) has no URDF/Gazebo equivalent.
* Sensor timing: the node uses the latest message from each topic; MuJoCo's sensors are exact.
* `odom` comes from Gazebo's OdometryPublisher (ground truth, body-frame twist). Check the
  frame on first run: standing still, `vx, vy ≈ 0`; walking forward, `vx > 0`.

## First-run checklist

1. `ros2 launch quadrucat_deploy gazebo.launch.py` and check `ros2 topic hz /imu /odom /joint_states`.
2. `ros2 control list_controllers` shows both controllers active.
3. Start the policy with a zero command: the robot should stand. If it collapses, check joint
   order (`metadata.joint_names`) and the sign/frame of `/imu` orientation and `/odom` twist.
