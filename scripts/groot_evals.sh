#!/usr/bin/env bash
# More closed-loop evaluations of a trained checkpoint: the same weights, run differently (no
# retraining), on the round's test boards, next to the round's own result (~/eval/<name>).
#
#   tmux new -s evals        (or tmux a -t evals to come back)
#   bash ~/Wire-Harness-Robot-2/scripts/groot_evals.sh route_v3
#
# Variants (VARIANTS="ens4 restarts best steps8" by default; each is skipped when its
# summary.md is already there, so the same command carries on after an interruption):
#   ens4      a new action chunk every 4 steps, the overlapping chunks averaged (temporal
#             ensembling, decay 0.1); saves every trial's step-by-step trajectory
#   restarts  8 steps per chunk as in the round, but a stalled try (let go outside the slot, no
#             progress for 12 s) is cleared and the policy starts over, up to 2 times in 60 s
#   steps8    8 denoising steps per chunk instead of 4 (a copy of the checkpoint's config.json
#             next to links to its weights)
#   best      ens4 and restarts together, with videos
#   assist    ens4, and when the policy has held the wire lined up over the slot for 1.5 s without
#             getting it in (or starts to let go there), the expert's force-controlled seating
#             takes over (a hybrid; the summary counts what the policy did alone)
#   assist_best  assist and restarts together, with videos
#   system    the full system: assist, and when the policy's call fails the board is cleared and
#             the expert retries, as the planner does in a build
#   takeovers ens4 + assist on training boards (TAKEOVER_SEEDS, 8000-8199), and where the policy
#             stalls before the slot (no grasp, not lifted, not carried over, dropped) the expert
#             redoes the route; every successful takeover saved as a training episode
#             (~/eval/<name>_takeovers/takeovers: DAgger-style data for the next round)
# Settings: STEPS (12000), CKPT (~/ckpt/<name>/checkpoint-STEPS), EVAL_SEEDS (0-19),
# EVAL_WORKERS (4), FORKS (F1,F2,F3), PORT (5556), ASSIST (1.5 s), TAKEOVER_SEEDS (8000-8199).
# About 30 min per variant on 20 boards. Logged to ~/rounds/<name>_evals.log.
set -euo pipefail

NAME="${1:-route_v3}"
STEPS="${STEPS:-12000}"
CKPT="${CKPT:-$HOME/ckpt/$NAME/checkpoint-$STEPS}"
VARIANTS="${VARIANTS:-ens4 restarts best steps8}"
EVAL_SEEDS="${EVAL_SEEDS:-0-19}"
ASSIST="${ASSIST:-1.5}"
TAKEOVER_SEEDS="${TAKEOVER_SEEDS:-8000-8199}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
FORKS="${FORKS:-F1,F2,F3}"
PORT="${PORT:-5556}"
GROOT_DIR="${GROOT_DIR:-$HOME/Isaac-GR00T}"
mkdir -p "$HOME/rounds" "$HOME/eval"
LOG="$HOME/rounds/${NAME}_evals.log"
exec > >(tee -a "$LOG") 2>&1

# shellcheck disable=SC1091
source "$HOME/harness_env.sh"
export PATH="$HOME/.local/bin:$PATH"
say() { echo; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
trap 'say "stopped (Ctrl-c); run the same command again to carry on"; exit 130' INT

if [ ! -d "$CKPT" ]; then
    echo "no checkpoint $CKPT (ls ~/ckpt/$NAME)"
    exit 1
fi

stop_server() {
    python -c "from harness_agent.groot_client import GrootClient as C; C('127.0.0.1', $PORT, timeout_ms=3000).call('kill')" \
        >/dev/null 2>&1 || true
    sleep 5
}

SERVED=""
serve() {   # serve <checkpoint>: start GR00T's policy server (unless it already serves it) and wait for it
    if [ "$SERVED" = "$1" ]; then return; fi
    stop_server
    say "serving $1 on port $PORT"
    (cd "$GROOT_DIR" && nohup uv run python gr00t/eval/run_gr00t_server.py --model-path "$1" \
        --embodiment-tag NEW_EMBODIMENT --port "$PORT" > "$HOME/rounds/${NAME}_evals.server.log" 2>&1 &)
    python - "$PORT" "$NAME" <<'PY'
import sys
import time

from harness_agent.groot_client import GrootClient

port, name = int(sys.argv[1]), sys.argv[2]
for _ in range(90):
    c = GrootClient("127.0.0.1", port, timeout_ms=3000)
    up = c.ping()
    c.close()
    if up:
        sys.exit(0)
    time.sleep(10)
sys.exit(f"the policy server did not come up; see ~/rounds/{name}_evals.server.log")
PY
    SERVED="$1"
}

denoise_copy() {   # denoise_copy <steps>: the checkpoint with another number of denoising steps
    local dir="${CKPT}-denoise$1"
    if [ ! -f "$dir/config.json" ]; then
        mkdir -p "$dir"
        for f in "$CKPT"/*; do
            b="$(basename "$f")"
            if [ "$b" != config.json ] && [ ! -e "$dir/$b" ]; then ln -s "$f" "$dir/$b"; fi
        done
        python - "$CKPT/config.json" "$dir/config.json" "$1" <<'PY'
import json
import sys

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(src))
found = []


def visit(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "num_inference_timesteps":
                d[k] = n
                found.append(k)
            else:
                visit(v)
    elif isinstance(d, list):
        for v in d:
            visit(v)


visit(cfg)
if not found:
    sys.exit("num_inference_timesteps is not in the checkpoint's config.json")
json.dump(cfg, open(dst, "w"), indent=2)
PY
    fi
    echo "$dir"
}

evaluate() {   # evaluate <out folder> <seeds> <groot_eval options...>
    local out="$1" seeds="$2"
    shift 2
    if [ -f "$out/summary.md" ]; then
        say "evaluation $out: done before"
        return
    fi
    say "evaluation -> $out (seeds $seeds, $*)"
    python -m harness_agent.groot_eval --port "$PORT" --forks "$FORKS" --seeds "$seeds" \
        --workers "$EVAL_WORKERS" --out "$out" "$@"
}

for v in $VARIANTS; do
    out="$HOME/eval/${NAME}_$v"
    case "$v" in
        ens4)     serve "$CKPT"; evaluate "$out" "$EVAL_SEEDS" --execute-horizon 4 --ensemble 0.1 --trajectories ;;
        restarts) serve "$CKPT"; evaluate "$out" "$EVAL_SEEDS" --execute-horizon 8 --restarts 2 --max-seconds 60 ;;
        best)     serve "$CKPT"; evaluate "$out" "$EVAL_SEEDS" --execute-horizon 4 --ensemble 0.1 --restarts 2 \
                      --max-seconds 60 --trajectories --video ;;
        steps8)   dir="$(denoise_copy 8)"; serve "$dir"; evaluate "$out" "$EVAL_SEEDS" --execute-horizon 8 ;;
        assist)   serve "$CKPT"; evaluate "$out" "$EVAL_SEEDS" --execute-horizon 4 --ensemble 0.1 \
                      --seat-assist "$ASSIST" --trajectories ;;
        assist_best) serve "$CKPT"; evaluate "$out" "$EVAL_SEEDS" --execute-horizon 4 --ensemble 0.1 \
                      --seat-assist "$ASSIST" --restarts 2 --max-seconds 60 --trajectories --video ;;
        system)   serve "$CKPT"; evaluate "$out" "$EVAL_SEEDS" --execute-horizon 4 --ensemble 0.1 \
                      --seat-assist "$ASSIST" --fallback --trajectories ;;
        takeovers) serve "$CKPT"; evaluate "$out" "$TAKEOVER_SEEDS" --execute-horizon 4 --ensemble 0.1 \
                      --seat-assist "$ASSIST" --route-assist --max-seconds 60 --record-takeovers ;;
        *)        echo "unknown variant $v (ens4, restarts, steps8, best, assist, assist_best, system, takeovers)"; exit 1 ;;
    esac
done
stop_server

say "results"
for f in "$HOME/eval/$NAME/summary.md" "$HOME/eval/${NAME}"_*/summary.md; do
    if [ -f "$f" ]; then echo "== $f"; sed -n '3,8p' "$f"; grep -E "^(Restarts|Seat assist|Route takeovers|Routed without|Full system)|takeover episodes" "$f" || true; fi
done
say "evaluations finished. Results: ~/eval/${NAME}_*   Log: $LOG"
if [ -z "${QUIET_END:-}" ]; then
    echo "Stop the VM in the Nebius console now (Compute > Virtual machines > Stop)."
fi
