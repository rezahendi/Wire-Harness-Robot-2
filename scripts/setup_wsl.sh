#!/usr/bin/env bash
# One-time setup of the wire-harness robot cell on Ubuntu 24.04 (WSL2) with ROS 2 Jazzy.
#
#   bash scripts/setup_wsl.sh            # installs apt + pip dependencies
#
# Safe to re-run. Needs sudo for apt.
set -euo pipefail

if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "ROS 2 Jazzy not found in /opt/ros/jazzy. Install it first:"
  echo "  https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html"
  exit 1
fi
. /etc/os-release
if [ "${VERSION_ID:-}" != "24.04" ]; then
  echo "warning: tested on Ubuntu 24.04 (found ${PRETTY_NAME:-unknown})"
fi

echo "==> apt packages (ROS tools used by the cell, OpenGL for the MuJoCo viewer / videos)"
sudo apt-get update
sudo apt-get install -y \
  python3-pip python3-numpy python3-yaml python3-pytest \
  python3-colcon-common-extensions python3-rosdep \
  ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-rviz2 ros-jazzy-control-msgs ros-jazzy-tf2-ros-py ros-jazzy-launch-ros \
  libgl1 libegl1 libglfw3 libosmesa6 ffmpeg

echo "==> python packages (user site, next to the system numpy 1.26 that ROS uses)"
# --break-system-packages is required by Ubuntu 24.04 (PEP 668) for pip --user installs.
# "numpy<2" keeps pip from replacing the system numpy that ROS 2 Jazzy is built against.
python3 -m pip install --user --break-system-packages \
  "mujoco>=3.3,<4" "gymnasium>=1.0,<2" "imageio[ffmpeg]>=2.30" "numpy<2"

python3 - <<'PY'
import mujoco, gymnasium, numpy
print(f"mujoco {mujoco.__version__}, gymnasium {gymnasium.__version__}, numpy {numpy.__version__}")
PY

cat <<'MSG'

Setup done. Build the workspace:

  cd ~/harness_ws
  source /opt/ros/jazzy/setup.bash
  colcon build --symlink-install
  source install/setup.bash
  ros2 launch harness_bringup demo.launch.py

MSG
