#!/usr/bin/env bash
# Mission Control on the GPU VM: GR00T N1.7's policy server and the web app, in one command.
#
#   tmux new -s app          (or tmux a -t app to come back)
#   bash ~/Wire-Harness-Robot-2/scripts/mission_control.sh
#
# Then, on the laptop, forward the app's port and open it in a browser:
#   ssh -L 8000:localhost:8000 reza@<the VM's IP>       and open  http://localhost:8000
#
# Settings: CKPT (~/ckpt/route_v6/checkpoint-20000), PORT (5556, the policy server),
# APP_PORT (8000), RUNS (~/runs/webapp, where builds are kept), NO_GROOT=1 (the app without GR00T).
# Nemotron and the vision check need NEBIUS_API_KEY; when it is not set, the key is read from
# ~/.config/nebius/api_key. Ctrl-c stops the app and the policy server this script started.
set -euo pipefail

CKPT="${CKPT:-$HOME/ckpt/route_v6/checkpoint-20000}"
PORT="${PORT:-5556}"
APP_PORT="${APP_PORT:-8000}"
RUNS="${RUNS:-$HOME/runs/webapp}"
GROOT_DIR="${GROOT_DIR:-$HOME/Isaac-GR00T}"
mkdir -p "$HOME/rounds" "$RUNS"

# shellcheck disable=SC1091
source "$HOME/harness_env.sh"
export PATH="$HOME/.local/bin:$PATH"
say() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ -z "${NEBIUS_API_KEY:-}" ] && [ -f "$HOME/.config/nebius/api_key" ]; then
    NEBIUS_API_KEY="$(tr -d '[:space:]' < "$HOME/.config/nebius/api_key")"
    export NEBIUS_API_KEY
fi
if [ -n "${NEBIUS_API_KEY:-}" ]; then say "Nebius key found: Nemotron and the vision check are on"
else say "no Nebius key (~/.config/nebius/api_key): the app runs with the scripted planner only"; fi

if ! python -c "import fastapi, uvicorn, websockets" >/dev/null 2>&1; then
    say "installing the web server (fastapi, uvicorn) into the simulator environment"
    python -m pip install -q fastapi "uvicorn[standard]"
fi

ping_groot() {
    python -c "
import sys
from harness_agent.groot_client import GrootClient
c = GrootClient('127.0.0.1', $PORT, timeout_ms=3000)
ok = c.ping()
c.close()
sys.exit(0 if ok else 1)" >/dev/null 2>&1
}

STARTED=""
stop_groot() {
    if [ -n "$STARTED" ]; then
        say "stopping the GR00T policy server"
        python -c "from harness_agent.groot_client import GrootClient as C; C('127.0.0.1', $PORT, timeout_ms=3000).call('kill')" \
            >/dev/null 2>&1 || true
    fi
}
trap stop_groot EXIT

GROOT_ARGS=()
if [ "${NO_GROOT:-0}" = "1" ]; then
    say "GR00T off (NO_GROOT=1)"
elif ping_groot; then
    say "a GR00T policy server already answers on port $PORT: using it"
    GROOT_ARGS=(--groot "127.0.0.1:$PORT")
elif [ ! -d "$CKPT" ]; then
    say "no checkpoint $CKPT (ls ~/ckpt): the app runs without GR00T"
else
    say "starting GR00T N1.7's policy server with $CKPT (log: ~/rounds/mission_control.server.log)"
    (cd "$GROOT_DIR" && nohup uv run python gr00t/eval/run_gr00t_server.py --model-path "$CKPT" \
        --embodiment-tag NEW_EMBODIMENT --port "$PORT" > "$HOME/rounds/mission_control.server.log" 2>&1 &)
    STARTED=1
    for _ in $(seq 60); do
        if ping_groot; then break; fi
        sleep 5
    done
    if ping_groot; then
        say "GR00T is ready"
        GROOT_ARGS=(--groot "127.0.0.1:$PORT")
    else
        say "the policy server did not answer within 5 minutes (tail ~/rounds/mission_control.server.log);"
        say "the app starts without GR00T"
    fi
fi

say "Mission Control on port $APP_PORT. On the laptop: ssh -L $APP_PORT:localhost:$APP_PORT $(whoami)@<the VM's IP>"
say "then open http://localhost:$APP_PORT   (Ctrl-c here stops it)"
python -m harness_agent.webapp --port "$APP_PORT" --runs "$RUNS" "${GROOT_ARGS[@]}"
