"""Full demo: bring up the cell and start routing automatically.

    ros2 launch harness_bringup demo.launch.py
    ros2 launch harness_bringup demo.launch.py viewer:=true seed:=3
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    cell = os.path.join(get_package_share_directory("harness_bringup"), "launch", "cell.launch.py")
    names = ["seed", "randomize", "rviz", "viewer", "real_time_factor", "policy"]
    defaults = {"seed": "0", "randomize": "true", "rviz": "true", "viewer": "false",
                "real_time_factor": "1.0", "policy": "expert"}
    args = [DeclareLaunchArgument(n, default_value=defaults[n]) for n in names]
    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(cell),
        launch_arguments={**{n: LaunchConfiguration(n) for n in names}, "autostart": "true"}.items())
    return LaunchDescription(args + [include])
