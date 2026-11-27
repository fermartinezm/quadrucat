"""Gazebo Harmonic + Go1 + ros2_control + (optionally) the quadrucat policy.

  ros2 launch quadrucat_deploy gazebo.launch.py                                  # robot only
  ros2 launch quadrucat_deploy gazebo.launch.py policy_path:=/abs/runs/<run>/policy.npz
  ros2 launch quadrucat_deploy gazebo.launch.py policy_path:=... headless:=true \
      log_csv:=/tmp/gz_eval.csv eval_commands:="[0.5,0.0,0.0,1.0,0.0,0.0]"

Then (interactive mode): ros2 topic pub -r 10 /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5}}"
"""
import os
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, Shutdown)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _setup(context):
    share = get_package_share_directory("quadrucat_deploy")
    with open(os.path.join(share, "urdf", "go1.urdf"), encoding="utf-8") as f:
        urdf = f.read()
    # Absolute paths so Gazebo and gz_ros2_control find meshes/config without resource paths.
    urdf = urdf.replace("package://quadrucat_deploy/", f"file://{share}/")
    urdf = urdf.replace("@CONTROLLERS_YAML@", os.path.join(share, "config", "controllers.yaml"))

    pd = LaunchConfiguration("pd").perform(context).strip()
    if pd not in ("ros", "gazebo"):
        raise RuntimeError(f"pd must be 'ros' or 'gazebo', got {pd!r}")
    if pd == "gazebo":
        urdf = gazebo_pd_urdf(urdf, float(LaunchConfiguration("kp").perform(context)),
                              float(LaunchConfiguration("kd").perform(context)))

    if LaunchConfiguration("video").perform(context).lower() in ("1", "true", "yes"):
        # Chase camera on the trunk (like MuJoCo's 'track' camera, but it also rolls/pitches with the
        # trunk). Gazebo renders it only while its image topic has a subscriber:
        # scripts/record_gazebo_video.py writes the frames to an MP4.
        cam = """
  <link name="video_cam_link"><inertial><mass value="0.001"/>
    <inertia ixx="1e-7" iyy="1e-7" izz="1e-7" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="video_cam_joint" type="fixed"><parent link="trunk"/><child link="video_cam_link"/>
    <origin xyz="0.3 -1.3 0.35" rpy="0 0.2 1.6"/></joint>
  <gazebo reference="video_cam_link"><sensor name="video_cam" type="camera">
    <always_on>1</always_on><update_rate>30</update_rate>
    <camera><horizontal_fov>1.0</horizontal_fov><image><width>640</width><height>480</height></image>
      <clip><near>0.05</near><far>50</far></clip></camera></sensor></gazebo>
"""
        urdf = urdf.replace("</robot>", cam + "</robot>")
    world = LaunchConfiguration("world").perform(context) or os.path.join(share, "worlds", "flat.sdf")
    terrain_type = LaunchConfiguration("terrain_type").perform(context).strip()
    terrain_node, spawn_dz = {}, 0.0
    if terrain_type:   # one tile of the training terrain map (TERRAIN.md)
        world, terrain_node, spawn_dz = terrain_world(
            share, terrain_type, int(LaunchConfiguration("terrain_level").perform(context)),
            LaunchConfiguration("policy_path").perform(context).strip())
    headless = LaunchConfiguration("headless").perform(context).lower() in ("1", "true", "yes")
    # sim_rate_hz caps the physics update rate (gz sim -z). With the 1 ms step, 1000 = real time;
    # lower values give the ROS nodes (policy, PD loop) more wall time per simulated step.
    rate = LaunchConfiguration("sim_rate_hz").perform(context).strip()
    gz_args = f"-r {'-s ' if headless else ''}{'-z ' + rate + ' ' if rate else ''}{world}"

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items(),
    )
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
               parameters=[{"robot_description": urdf, "use_sim_time": True}])
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-topic", "robot_description", "-name", "go1",
                            "-z", str(float(LaunchConfiguration("spawn_z").perform(context) or
                                            ("0.45" if pd == "gazebo" else "0.30")) + spawn_dz)])
    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                  parameters=[{"config_file": os.path.join(share, "config", "bridge.yaml"),
                               "use_sim_time": True}])
    jsb = Node(package="controller_manager", executable="spawner", output="screen",
               arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"])
    cmds = Node(package="controller_manager", executable="spawner", output="screen",
                arguments=["joint_commands", "--controller-manager", "/controller_manager"])

    eval_raw = LaunchConfiguration("eval_commands").perform(context).strip()
    eval_cmds = [float(x) for x in eval_raw.strip("[]").split(",") if x.strip()] or [0.0]
    policy = Node(
        package="quadrucat_deploy", executable="policy_node", name="quadrucat_policy", output="screen",
        on_exit=Shutdown(),   # evaluation finished -> close Gazebo too
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("policy_path"), "' != ''"])),
        parameters=[os.path.join(share, "config", "policy.yaml"), {
            "policy_path": LaunchConfiguration("policy_path"),
            "log_csv": LaunchConfiguration("log_csv"),
            "eval_commands": eval_cmds,
            "eval_segment_s": float(LaunchConfiguration("eval_segment_s").perform(context)),
            "push_enable": LaunchConfiguration("push").perform(context).lower() in ("1", "true", "yes"),
            "push_seed": int(LaunchConfiguration("push_seed").perform(context)),
            # pd:=gazebo: the joint controllers hold the default pose from t=0 (the robot folds its
            # legs while it drops from spawn_z), so no ramp from the current pose
            **({"command_mode": "gz_position", "stand_ramp_s": 0.0, "stand_hold_s": 2.0}
               if pd == "gazebo" else {}),
            **terrain_node,
            "use_sim_time": True,
        }],
    )
    if pd == "gazebo":
        # No ros2_control: Gazebo's JointPositionController runs the PD at every physics step,
        # the policy node only sends joint targets (50 Hz), like the Go1's onboard motor PD.
        js_bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                         arguments=["/world/flat/model/go1/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model"],
                         remappings=[("/world/flat/model/go1/joint_state", "/joint_states")],
                         parameters=[{"use_sim_time": True}])
        return [gz, rsp, bridge, js_bridge, spawn,
                RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[policy]))]
    return [
        gz, rsp, bridge, spawn,
        # Controllers only after the robot exists. The policy node starts before the effort
        # controller so that stand-up torques are already flowing when it activates.
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[policy, cmds])),
    ]


def terrain_world(share: str, terrain_type: str, level: int, policy_path: str):
    """World with one tile of the training terrain (quadrucat.gazebo_terrain), node parameters for
    the height scan / heading hold, and the ground height at the spawn point."""
    import json

    import sys

    import numpy as np
    try:
        from quadrucat import gazebo_terrain, terrain
    except ImportError:   # symlink install: the launch file lives in <repo>/ros2/quadrucat_deploy/launch/
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__))))))
        from quadrucat import gazebo_terrain, terrain

    params = dict(terrain.DEFAULT_PARAMS)
    if policy_path:
        meta = json.loads(str(np.load(policy_path)["metadata"]))
        params.update(((meta.get("height_scan") or {}).get("terrain")) or {})
    world, origin, z0 = gazebo_terrain.write_world(os.path.join(share, "worlds", "flat.sdf"), params,
                                                   terrain_type, level, "/tmp/quadrucat_terrain")
    return str(world), {"terrain_origin": [float(origin[0]), float(origin[1])], "heading_hold": True}, z0


def gazebo_pd_urdf(urdf: str, kp: float, kd: float) -> str:
    """URDF for pd:=gazebo: drop ros2_control, PD inside Gazebo.

    Same law as MuJoCo's position actuators in training, tau = kp (q* - q) - kd dq: the P part is a
    JointPositionController per joint (p_gain = kp, clipped to the motor torque limit) and the D
    part is joint damping = kd (d_gain = 0 avoids the derivative kick when q* jumps).
    """
    urdf = re.sub(r"\s*<ros2_control.*?</ros2_control>", "", urdf, flags=re.S)
    urdf = re.sub(r'\s*<gazebo>\s*<plugin filename="gz_ros2_control-system".*?</gazebo>', "", urdf, flags=re.S)
    urdf = re.sub(r'(<joint name="\w+_joint" type="revolute">.*?<dynamics )damping="0"',
                  rf'\g<1>damping="{kd}"', urdf, flags=re.S)
    plugins = ['<gazebo><plugin filename="gz-sim-joint-state-publisher-system" '
               'name="gz::sim::systems::JointStatePublisher"/></gazebo>']
    for m in re.finditer(r'<joint name="(\w+_joint)" type="revolute">.*?effort="([\d.]+)"', urdf, flags=re.S):
        name, lim = m.group(1), m.group(2)
        leg, part = name.split("_")[:2]
        q0 = DEFAULT_POSE[part] * (-1 if part == "hip" and leg in ("FL", "RL") else 1)
        plugins.append(
            f'<gazebo><plugin filename="gz-sim-joint-position-controller-system" '
            f'name="gz::sim::systems::JointPositionController"><joint_name>{name}</joint_name>'
            f'<p_gain>{kp}</p_gain><i_gain>0</i_gain><d_gain>0</d_gain>'
            f'<cmd_max>{lim}</cmd_max><cmd_min>-{lim}</cmd_min><initial_position>{q0}</initial_position>'
            f'<topic>/model/go1/joint/{name}/cmd_pos</topic></plugin></gazebo>')
    return urdf.replace("</robot>", "\n".join(plugins) + "\n</robot>")


DEFAULT_POSE = {"hip": 0.1, "thigh": 0.9, "calf": -1.8}   # Go1 'home' keyframe (FL/RL hips mirrored)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("policy_path", default_value="", description="absolute path to policy.npz"),
        DeclareLaunchArgument("world", default_value="", description="SDF world (default: worlds/flat.sdf)"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("spawn_z", default_value="",
                              description="spawn height [m] (default 0.30, 0.45 with pd:=gazebo)"),
        DeclareLaunchArgument("sim_rate_hz", default_value="",
                              description="physics steps per wall second (empty = real time)"),
        DeclareLaunchArgument("log_csv", default_value=""),
        DeclareLaunchArgument("eval_commands", default_value="[]",
                              description="flat list vx,vy,wz,... ; empty = follow /cmd_vel"),
        DeclareLaunchArgument("eval_segment_s", default_value="20.0"),
        DeclareLaunchArgument("pd", default_value="ros",
                              description="ros: PD torques from the policy node over ros2_control "
                                          "(500 Hz); gazebo: PD inside Gazebo, node sends targets"),
        DeclareLaunchArgument("kp", default_value="35.0", description="pd:=gazebo only"),
        DeclareLaunchArgument("kd", default_value="0.5", description="pd:=gazebo only"),
        DeclareLaunchArgument("video", default_value="false",
                              description="add a chase camera (record with scripts/record_gazebo_video.py)"),
        DeclareLaunchArgument("push", default_value="false", description="random kicks (pert_config)"),
        DeclareLaunchArgument("terrain_type", default_value="",
                              description="rough|stairs_up|stairs_down|boxes|beam: world = that training tile"),
        DeclareLaunchArgument("terrain_level", default_value="5", description="0 (easy) .. 9 (hard)"),
        DeclareLaunchArgument("push_seed", default_value="0"),
        OpaqueFunction(function=_setup),
    ])
