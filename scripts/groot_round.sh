#!/usr/bin/env bash
# One unattended GR00T round on the GPU VM: record recovery demos, package them with the
# earlier ones, fine-tune GR00T N1.7, serve the result and evaluate it in closed loop.
#
#   tmux new -s round        (or tmux a -t round to come back)
#   bash ~/Wire-Harness-Robot-2/scripts/groot_round.sh route_v2
#
# Every step is skipped when its result is already there, so after a preempted VM the same
# command carries on: recording resumes build by build, fine-tuning resumes from its last
# checkpoint, finished evaluations are kept. Everything is logged to ~/rounds/<name>.log.
#
# When it says "round finished", stop the VM in the Nebius console. (Shutting the VM down
# from inside does not stop it: Nebius restarts it and keeps charging.)
#
# Settings (environment variables): BUILDS (240), SEED_START (2000), BASE (earlier set to
# keep, ~/data/harness_route; "" for none), NOISE (1.0), HOLD (1.0), STEPS (12000),
# SAVE_STEPS (3000), WORKERS (vCPUs - 1), EVAL_SEEDS (0-19), EVAL_WORKERS (4), PORT (5556),
# HORIZONS ("8 4": action steps executed per chunk, one evaluation each), BASELINE (a
# checkpoint evaluated on the same boards for comparison; ~/ckpt/route_v1/checkpoint-6000
# when it exists, "" for none).
set -euo pipefail

NAME="${1:-route_v2}"
BUILDS="${BUILDS:-240}"
SEED_START="${SEED_START:-2000}"          # 1000-1329: first set; 0-99: test boards, never trained on
BASE="${BASE-$HOME/data/harness_route}"
NOISE="${NOISE:-1.0}"
HOLD="${HOLD:-1.0}"
STEPS="${STEPS:-12000}"
SAVE_STEPS="${SAVE_STEPS:-3000}"
WORKERS="${WORKERS:-$(( $(nproc) - 1 ))}"
EVAL_SEEDS="${EVAL_SEEDS:-0-19}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
PORT="${PORT:-5556}"
HORIZONS="${HORIZONS:-8 4}"
BASELINE="${BASELINE-$HOME/ckpt/route_v1/checkpoint-6000}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT_DIR="${GROOT_DIR:-$HOME/Isaac-GR00T}"

NEW="$HOME/data/${NAME}_new"
DATA="$HOME/data/$NAME"
CKPT="$HOME/ckpt/$NAME"
EVAL="$HOME/eval/$NAME"
mkdir -p "$HOME/rounds" "$HOME/ckpt" "$HOME/eval"
LOG="$HOME/rounds/$NAME.log"
exec > >(tee -a "$LOG") 2>&1

# shellcheck disable=SC1091
source "$HOME/harness_env.sh"
export PATH="$HOME/.local/bin:$PATH"
say() { echo; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
free_gb() { df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9'; }

say "round $NAME: $BUILDS new builds (seeds from $SEED_START, pushes $NOISE, hold $HOLD s), $STEPS steps"

# 1. new demos -------------------------------------------------------------------------
if [ -f "$NEW/meta/info.json" ]; then
    say "1/5 recording: done before ($NEW)"
else
    say "1/5 recording into $NEW"
    python -m harness_agent.groot_data record --out "$NEW" --seed-start "$SEED_START" --builds "$BUILDS" \
        --workers "$WORKERS" --noise "$NOISE" --hold-after "$HOLD"
fi

# 2. one dataset with the earlier demos --------------------------------------------------
if [ -f "$DATA/meta/info.json" ]; then
    say "2/5 packaging: done before ($DATA)"
else
    say "2/5 packaging $DATA"
    sources=("$NEW")
    if [ -n "$BASE" ]; then sources=("$BASE" "$NEW"); fi
    python -m harness_agent.groot_data merge "${sources[@]}" --out "$DATA"
fi
python -m harness_agent.groot_data check "$DATA" | tail -25

# 3. fine-tune -----------------------------------------------------------------------------
if [ -d "$CKPT/checkpoint-$STEPS" ]; then
    say "3/5 fine-tuning: done before ($CKPT/checkpoint-$STEPS)"
else
    if [ "$(free_gb)" -lt "${MIN_FREE_GB:-80}" ]; then
        say "only $(free_gb) GB free; fine-tuning needs ~80 GB (two 36 GB checkpoints)."
        echo "Free space first, e.g. rm -r ~/ckpt/route_v1/checkpoint-4000, then run this again."
        exit 1
    fi
    resume=()
    if compgen -G "$CKPT/checkpoint-*" > /dev/null; then
        resume=(--resume-from-checkpoint)
        say "3/5 fine-tuning: resuming from $(ls -d "$CKPT"/checkpoint-* | sort -V | tail -1)"
    else
        say "3/5 fine-tuning $STEPS steps -> $CKPT"
    fi
    (cd "$GROOT_DIR" && uv run python gr00t/experiment/launch_finetune.py \
        --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DATA" --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$REPO/groot/harness_config.py" --num-gpus 1 --output-dir "$CKPT" \
        --max-steps "$STEPS" --save-steps "$SAVE_STEPS" --save-total-limit 2 \
        --global-batch-size 32 --dataloader-num-workers 8 "${resume[@]}")
fi

# 4. serve and 5. evaluate -------------------------------------------------------------
stop_server() {
    python -c "from harness_agent.groot_client import GrootClient as C; C('127.0.0.1', $PORT, timeout_ms=3000).call('kill')" \
        >/dev/null 2>&1 || true
    sleep 5
}

serve() {   # serve <checkpoint>: start GR00T's policy server and wait until it answers
    stop_server
    (cd "$GROOT_DIR" && nohup uv run python gr00t/eval/run_gr00t_server.py --model-path "$1" \
        --embodiment-tag NEW_EMBODIMENT --port "$PORT" > "$HOME/rounds/$NAME.server.log" 2>&1 &)
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
sys.exit(f"the policy server did not come up; see ~/rounds/{name}.server.log")
PY
}

evaluate() {   # evaluate <out folder> <action steps executed per chunk>
    if [ -f "$1/summary.md" ]; then
        say "evaluation $1: done before"
        return
    fi
    say "evaluation -> $1 (boards $EVAL_SEEDS, $2 steps per chunk)"
    python -m harness_agent.groot_eval --port "$PORT" --forks F1,F2,F3 --seeds "$EVAL_SEEDS" \
        --workers "$EVAL_WORKERS" --execute-horizon "$2" --out "$1" --video
}

say "4/5 serving $CKPT/checkpoint-$STEPS on port $PORT"
serve "$CKPT/checkpoint-$STEPS"
say "5/5 closed-loop evaluation"
for h in $HORIZONS; do
    if [ "$h" = 8 ]; then evaluate "$EVAL" 8; else evaluate "${EVAL}_h$h" "$h"; fi
done
if [ -n "$BASELINE" ] && [ -d "$BASELINE" ]; then
    base_name="$(basename "$(dirname "$BASELINE")")"
    say "baseline: $BASELINE on the same boards"
    serve "$BASELINE"
    evaluate "$HOME/eval/${base_name}_on_${NAME}_boards" 8
fi
stop_server

say "results"
for f in "$EVAL"/summary.md "$EVAL"_h*/summary.md "$HOME/eval/"*"_on_${NAME}_boards/summary.md"; do
    if [ -f "$f" ]; then echo "== $f"; sed -n '1,9p' "$f"; fi
done

say "round finished. Results: ~/eval/$NAME*/summary.md   Log: $LOG"
echo "Stop the VM in the Nebius console now (Compute > Virtual machines > Stop)."
