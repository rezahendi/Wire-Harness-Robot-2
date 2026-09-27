#!/usr/bin/env bash
# Build this checkout from WSL without copying it.
#
#   bash "/mnt/c/Users/<you>/Desktop/Wire harness/wire_harness_robot_1/wire_harness_robot/scripts/wsl_link_workspace.sh"
#
# The source stays where it is (on the Windows side, where it is edited); the colcon
# workspace in ~/harness_ws only holds a symlink to it plus the build output, on the fast
# Linux file system. Built with --symlink-install, so edits to Python files take effect
# without rebuilding; re-run this script after changes to messages, URDF or package lists.
#
# A previous plain copy in ~/harness_ws/src is moved to ~/harness_backups (outside the
# workspace, so colcon does not find its packages a second time).
set -eo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${HARNESS_WS:-$HOME/harness_ws}"
BACKUPS="${HARNESS_BACKUP_DIR:-$HOME/harness_backups}"
LINK="$WS/src/wire_harness_robot"
mkdir -p "$WS/src"

# copies an earlier version of this script left inside the workspace
for old in "$WS"/old_wire_harness_robot_*; do
  if [ -d "$old" ]; then
    mkdir -p "$BACKUPS"
    echo "moving $old to $BACKUPS/"
    mv -n "$old" "$BACKUPS/" || true
    if [ -d "$old" ]; then
      touch "$old/COLCON_IGNORE"          # could not move it: at least hide it from colcon
    fi
  fi
done

if [ -e "$LINK" ] && [ ! -L "$LINK" ]; then
  mkdir -p "$BACKUPS"
  BACKUP="$BACKUPS/wire_harness_robot_$(date +%Y%m%d_%H%M%S)"
  echo "moving the previous copy to $BACKUP"
  mv "$LINK" "$BACKUP"
  echo "removing build output of the previous copy (build/ install/ log/)"
  rm -rf "$WS/build" "$WS/install" "$WS/log"
fi
ln -sfn "$SRC" "$LINK"
echo "workspace $WS/src/wire_harness_robot -> $SRC"

if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "ROS 2 Jazzy not found in /opt/ros/jazzy"; exit 1
fi
source /opt/ros/jazzy/setup.bash
cd "$WS"
colcon build --symlink-install --base-paths src

echo
echo "Built. In every new terminal:  source $WS/install/setup.bash"
