"""Start the policy node. Gazebo + robot + controllers are launched separately (see ros2/README.md).

  ros2 launch quadrucat_deploy policy.launch.py policy_path:=/abs/path/runs/<run>/policy.npz
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory("quadrucat_deploy"), "config", "policy.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("policy_path", description="absolute path to policy.npz"),
        DeclareLaunchArgument("log_csv", default_value=""),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        Node(
            package="quadrucat_deploy",
            executable="policy_node",
            name="quadrucat_policy",
            output="screen",
            parameters=[
                params,
                {
                    "policy_path": LaunchConfiguration("policy_path"),
                    "log_csv": LaunchConfiguration("log_csv"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                },
            ],
        ),
    ])
