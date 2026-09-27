"""Bring up the simulated wire-harness robot cell.

    ros2 launch harness_bringup cell.launch.py                  # sim + task server + RViz
    ros2 launch harness_bringup cell.launch.py viewer:=true     # also the MuJoCo viewer
    ros2 launch harness_bringup cell.launch.py rviz:=false seed:=7

Then start a routing job:
    ros2 run harness_task route_harness
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    desc = get_package_share_directory("harness_description")
    bringup = get_package_share_directory("harness_bringup")
    xacro_file = os.path.join(desc, "urdf", "harness_cell.urdf.xacro")
    rviz_file = os.path.join(desc, "rviz", "harness_cell.rviz")
    default_cfg = os.path.join(bringup, "config", "cell.yaml")

    args = [
        DeclareLaunchArgument("config", default_value=default_cfg, description="cell YAML"),
        DeclareLaunchArgument("seed", default_value="0", description="layout / wire seed"),
        DeclareLaunchArgument("randomize", default_value="true", description="randomise the layout"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("viewer", default_value="false", description="MuJoCo passive viewer"),
        DeclareLaunchArgument("real_time_factor", default_value="1.0"),
        DeclareLaunchArgument("policy", default_value="expert",
                              description="'expert' or 'package.module:object' for a learned policy"),
        DeclareLaunchArgument("autostart", default_value="false", description="start routing on launch"),
    ]
    cfg = LaunchConfiguration("config")
    robot_description = ParameterValue(Command(["xacro ", xacro_file]), value_type=str)

    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="log",
             parameters=[{"robot_description": robot_description, "use_sim_time": True}]),
        Node(package="harness_sim", executable="mujoco_sim_node", name="mujoco_sim", output="screen",
             parameters=[{
                 "config_file": cfg,
                 "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
                 "randomize": ParameterValue(LaunchConfiguration("randomize"), value_type=bool),
                 "viewer": ParameterValue(LaunchConfiguration("viewer"), value_type=bool),
                 "real_time_factor": ParameterValue(LaunchConfiguration("real_time_factor"), value_type=float),
             }]),
        Node(package="harness_task", executable="routing_task_node", name="routing_task", output="screen",
             parameters=[{
                 "config_file": cfg,
                 "policy": LaunchConfiguration("policy"),
                 "autostart": ParameterValue(LaunchConfiguration("autostart"), value_type=bool),
                 "use_sim_time": True,
             }]),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz_file], output="log",
             parameters=[{"use_sim_time": True}], condition=IfCondition(LaunchConfiguration("rviz"))),
    ]
    return LaunchDescription(args + nodes)
