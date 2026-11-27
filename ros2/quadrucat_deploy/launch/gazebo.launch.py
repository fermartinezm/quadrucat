"""Gazebo Harmonic + Go1 + ros2_control + (optionally) the quadrucat policy.

  ros2 launch quadrucat_deploy gazebo.launch.py                                  # robot only
  ros2 launch quadrucat_deploy gazebo.launch.py policy_path:=/abs/runs/<run>/policy.npz
  ros2 launch quadrucat_deploy gazebo.launch.py policy_path:=... headless:=true \
      log_csv:=/tmp/gz_eval.csv eval_commands:="[0.5,0.0,0.0,1.0,0.0,0.0]"

Then (interactive mode): ros2 topic pub -r 10 /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5}}"
"""
import os

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

    video_dir = LaunchConfiguration("video_dir").perform(context).strip()
    if video_dir:
        # Chase camera on the trunk (like MuJoCo's 'track' camera, but it also rolls/pitches with the
        # trunk); Gazebo writes one PNG per frame to video_dir (needs rendering, use with headless).
        os.makedirs(video_dir, exist_ok=True)
        cam = f"""
  <link name="video_cam_link"><inertial><mass value="0.001"/>
    <inertia ixx="1e-7" iyy="1e-7" izz="1e-7" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="video_cam_joint" type="fixed"><parent link="trunk"/><child link="video_cam_link"/>
    <origin xyz="0.3 -1.3 0.35" rpy="0 0.2 1.6"/></joint>
  <gazebo reference="video_cam_link"><sensor name="video_cam" type="camera">
    <always_on>1</always_on><update_rate>30</update_rate>
    <camera><horizontal_fov>1.0</horizontal_fov><image><width>640</width><height>480</height></image>
      <clip><near>0.05</near><far>50</far></clip>
      <save enabled="true"><path>{video_dir}</path></save></camera></sensor></gazebo>
"""
        urdf = urdf.replace("</robot>", cam + "</robot>")
    world = LaunchConfiguration("world").perform(context) or os.path.join(share, "worlds", "flat.sdf")
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
                            "-z", LaunchConfiguration("spawn_z").perform(context)])
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
            "use_sim_time": True,
        }],
    )
    return [
        gz, rsp, bridge, spawn,
        # Controllers only after the robot exists. The policy node starts before the effort
        # controller so that stand-up torques are already flowing when it activates.
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[policy, cmds])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("policy_path", default_value="", description="absolute path to policy.npz"),
        DeclareLaunchArgument("world", default_value="", description="SDF world (default: worlds/flat.sdf)"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("spawn_z", default_value="0.30"),
        DeclareLaunchArgument("sim_rate_hz", default_value="",
                              description="physics steps per wall second (empty = real time)"),
        DeclareLaunchArgument("log_csv", default_value=""),
        DeclareLaunchArgument("eval_commands", default_value="[]",
                              description="flat list vx,vy,wz,... ; empty = follow /cmd_vel"),
        DeclareLaunchArgument("eval_segment_s", default_value="20.0"),
        DeclareLaunchArgument("video_dir", default_value="",
                              description="save chase-camera frames (PNG) to this folder"),
        DeclareLaunchArgument("push", default_value="false", description="random kicks (pert_config)"),
        DeclareLaunchArgument("push_seed", default_value="0"),
        OpaqueFunction(function=_setup),
    ])
