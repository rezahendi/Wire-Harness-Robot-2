#!/usr/bin/env bash
# Set up a Nebius GPU VM (Ubuntu 22.04/24.04 with NVIDIA drivers) for the GR00T work:
#   - system packages (ffmpeg for video, EGL for headless MuJoCo rendering, tmux)
#   - ~/simenv: the simulator + recorder + GR00T client (no ROS needed)
#   - ~/Isaac-GR00T: NVIDIA's GR00T N1.7 code with its own uv environment
#   - ~/harness_env.sh: environment variables for the simulator
#
#   bash ~/Wire-Harness-Robot-2/scripts/setup_groot_vm.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GROOT_DIR="${GROOT_DIR:-$HOME/Isaac-GR00T}"
PY="${PY:-python3}"

echo "== system packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg libegl1 libgl1 libglib2.0-0 tmux git \
    python3-venv python3-dev build-essential curl >/dev/null

echo "== simulator environment (~/simenv)"
$PY -m venv "$HOME/simenv"
"$HOME/simenv/bin/pip" install -q --upgrade pip
"$HOME/simenv/bin/pip" install -q "numpy<2" "mujoco==3.13.0" gymnasium pyyaml imageio imageio-ffmpeg pillow \
    matplotlib pyarrow pandas pyzmq msgpack msgpack-numpy

cat > "$HOME/harness_env.sh" <<EOF
# source this in every shell that runs the simulator
export PYTHONPATH="$REPO_DIR/harness_core:$REPO_DIR/harness_learning:$REPO_DIR/harness_agent\${PYTHONPATH:+:\$PYTHONPATH}"
export MUJOCO_GL=egl
export PATH="\$HOME/simenv/bin:\$PATH"
EOF
# shellcheck disable=SC1091
source "$HOME/harness_env.sh"
python -c "import harness_agent.groot_data, harness_agent.groot_eval; print('simulator environment ok')"
python - <<'EOF'
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos='0 0 1'/><geom size='.1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 64, 64)
r.update_scene(mujoco.MjData(m))
print("headless rendering ok:", r.render().shape)
EOF

echo "== GR00T N1.7 (~/Isaac-GR00T)"
if [ ! -d "$GROOT_DIR" ]; then
    git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T "$GROOT_DIR"
fi
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
cd "$GROOT_DIR"
uv sync --python 3.12
uv run python -c "import torch; print('GR00T environment ok, CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo
echo "Done. Next:"
echo "  cd $GROOT_DIR && uv run huggingface-cli login      # paste your Hugging Face read token"
echo "  source ~/harness_env.sh                               # in every new shell for the simulator"
