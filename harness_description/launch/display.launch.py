"""View the robot model on its own: robot_state_publisher + joint sliders + RViz.

    ros2 launch harness_description display.launch.py
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
    share = get_package_share_directory("harness_description")
    xacro_file = os.path.join(share, "urdf", "harness_cell.urdf.xacro")
    rviz_file = os.path.join(share, "rviz", "harness_cell.rviz")
    robot_description = ParameterValue(Command(["xacro ", xacro_file]), value_type=str)
    gui = LaunchConfiguration("gui")
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="joint_state_publisher_gui sliders"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}], output="screen"),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui",
             condition=IfCondition(gui)),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz_file], output="log"),
    ])
