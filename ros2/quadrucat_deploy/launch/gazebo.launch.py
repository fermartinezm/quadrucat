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
                            RegisterEventHandler)
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

    world = LaunchConfiguration("world").perform(context) or os.path.join(share, "worlds", "flat.sdf")
    headless = LaunchConfiguration("headless").perform(context).lower() in ("1", "true", "yes")
    gz_args = f"-r {'-s ' if headless else ''}{world}"

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
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("policy_path"), "' != ''"])),
        parameters=[os.path.join(share, "config", "policy.yaml"), {
            "policy_path": LaunchConfiguration("policy_path"),
            "log_csv": LaunchConfiguration("log_csv"),
            "eval_commands": eval_cmds,
            "eval_segment_s": float(LaunchConfiguration("eval_segment_s").perform(context)),
            "use_sim_time": True,
        }],
    )
    return [
        gz, rsp, bridge, spawn,
        # Controllers only after the robot exists; the policy once the controllers are up.
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[cmds])),
        RegisterEventHandler(OnProcessExit(target_action=cmds, on_exit=[policy])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("policy_path", default_value="", description="absolute path to policy.npz"),
        DeclareLaunchArgument("world", default_value="", description="SDF world (default: worlds/flat.sdf)"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("spawn_z", default_value="0.30"),
        DeclareLaunchArgument("log_csv", default_value=""),
        DeclareLaunchArgument("eval_commands", default_value="[]",
                              description="flat list vx,vy,wz,... ; empty = follow /cmd_vel"),
        DeclareLaunchArgument("eval_segment_s", default_value="20.0"),
        OpaqueFunction(function=_setup),
    ])
