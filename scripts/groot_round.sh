#!/usr/bin/env bash
# One unattended GR00T round on the GPU VM: record demos (with recovery pushes), fine-tune
# GR00T N1.7, check how well it fits demos it saw and demos it did not see (open loop), serve
# it and evaluate it in closed loop next to the previous model.
#
#   tmux new -s round        (or tmux a -t round to come back)
#   bash ~/Wire-Harness-Robot-2/scripts/groot_round.sh route_v3
#
# Every step is skipped when its result is already there, so after a preempted VM the same
# command carries on: recording resumes build by build, fine-tuning resumes from its last
# checkpoint, finished evaluations are kept. Ctrl-c stops the round; during recording it first
# packages the demos recorded so far, and the next run trains on those. Everything is logged
# to ~/rounds/<name>.log.
#
# When it says "round finished", stop the VM in the Nebius console. (Shutting the VM down
# from inside does not stop it: Nebius restarts it and keeps charging.)
#
# Settings (environment variables): BUILDS (400), SEED_START (3000), BASE (an earlier set
# with the same state layout to add; "" by default), DATASET (train on this packaged set
# instead of recording one, e.g. ~/data/route_v3: steps 1-2 are skipped and its _val set is
# reused), NOISE (1.0), HOLD (1.0), STEPS (12000),
# SAVE_STEPS (3000), EXTRA_TRAIN_ARGS (e.g. "--tune-visual"), WORKERS (vCPUs - 1),
# EVAL_SEEDS (0-19), EVAL_WORKERS (4), PORT (5556), HORIZONS ("8": action steps executed per
# chunk, one evaluation each), BASELINE (a checkpoint evaluated on the same boards for
# comparison; ~/ckpt/route_v2/checkpoint-12000 when it exists, "" for none), OPEN_LOOP (1).
set -euo pipefail

NAME="${1:-route_v3}"
BUILDS="${BUILDS:-400}"
SEED_START="${SEED_START:-3000}"          # 1000s: first set, 2000s: second; 0-99: test boards, never trained on
BASE="${BASE:-}"
DATASET="${DATASET:-}"
NOISE="${NOISE:-1.0}"
HOLD="${HOLD:-1.0}"
STEPS="${STEPS:-12000}"
SAVE_STEPS="${SAVE_STEPS:-3000}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
WORKERS="${WORKERS:-$(( $(nproc) - 1 ))}"
EVAL_SEEDS="${EVAL_SEEDS:-0-19}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
PORT="${PORT:-5556}"
HORIZONS="${HORIZONS:-8}"
BASELINE="${BASELINE-$HOME/ckpt/route_v2/checkpoint-12000}"
OPEN_LOOP="${OPEN_LOOP:-1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT_DIR="${GROOT_DIR:-$HOME/Isaac-GR00T}"

NEW="$HOME/data/${NAME}_new"
DATA="${DATASET:-$HOME/data/$NAME}"
VAL="$HOME/data/${NAME}_val"
if [ -n "$DATASET" ] && [ ! -f "$VAL/meta/info.json" ] && [ -f "${DATASET%/}_val/meta/info.json" ]; then
    VAL="${DATASET%/}_val"                      # the expert's demos from the test boards, already recorded
fi
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
trap 'say "stopped (Ctrl-c); run the same command again to carry on"; exit 130' INT

if [ -n "$DATASET" ]; then
    say "round $NAME: training on $DATASET, $STEPS steps ${EXTRA_TRAIN_ARGS}"
else
    say "round $NAME: $BUILDS builds (seeds from $SEED_START, pushes $NOISE, hold $HOLD s), $STEPS steps"
fi

# 1. demos -----------------------------------------------------------------------------
if [ -n "$DATASET" ]; then
    if [ ! -f "$DATASET/meta/info.json" ]; then echo "no packaged set at $DATASET"; exit 1; fi
    say "1/6 recording: none, training on $DATASET"
elif [ -f "$NEW/meta/info.json" ]; then
    say "1/6 recording: done before ($NEW)"
else
    say "1/6 recording into $NEW"
    python -m harness_agent.groot_data record --out "$NEW" --seed-start "$SEED_START" --builds "$BUILDS" \
        --workers "$WORKERS" --noise "$NOISE" --hold-after "$HOLD"
fi

# 2. the training set ------------------------------------------------------------------
if [ -f "$DATA/meta/info.json" ]; then
    say "2/6 packaging: done before ($DATA)"
else
    say "2/6 packaging $DATA"
    sources=("$NEW")
    if [ -n "$BASE" ]; then sources=("$BASE" "$NEW"); fi
    python -m harness_agent.groot_data merge "${sources[@]}" --out "$DATA"
fi
python -m harness_agent.groot_data check "$DATA" | tail -25

# 3. fine-tune -------------------------------------------------------------------------
if [ -d "$CKPT/checkpoint-$STEPS" ]; then
    say "3/6 fine-tuning: done before ($CKPT/checkpoint-$STEPS)"
else
    if [ "$(free_gb)" -lt "${MIN_FREE_GB:-75}" ]; then
        say "only $(free_gb) GB free; fine-tuning needs ~75 GB (three 22 GB checkpoints while it saves)."
        echo "Delete checkpoints you no longer need (ls ~/ckpt), then run this again."
        exit 1
    fi
    resume=()
    if compgen -G "$CKPT/checkpoint-*" > /dev/null; then
        resume=(--resume-from-checkpoint)
        say "3/6 fine-tuning: resuming from $(ls -d "$CKPT"/checkpoint-* | sort -V | tail -1)"
    else
        say "3/6 fine-tuning $STEPS steps -> $CKPT"
    fi
    # shellcheck disable=SC2086
    (cd "$GROOT_DIR" && uv run python gr00t/experiment/launch_finetune.py \
        --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DATA" --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$REPO/groot/harness_config.py" --num-gpus 1 --output-dir "$CKPT" \
        --max-steps "$STEPS" --save-steps "$SAVE_STEPS" --save-total-limit 2 \
        --global-batch-size 32 --dataloader-num-workers 8 $EXTRA_TRAIN_ARGS "${resume[@]}")
fi

# serving helpers ----------------------------------------------------------------------
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

open_loop() {   # open_loop <dataset> <label> <episode ids...>: predicted vs recorded actions
    local data="$1" label="$2"
    shift 2
    if [ -f "$EVAL/open_loop_$label.log" ]; then
        return
    fi
    say "open-loop check on $label episodes $*"
    (cd "$GROOT_DIR" && uv run python gr00t/eval/open_loop_eval.py --dataset-path "$data" \
        --embodiment-tag new_embodiment --port "$PORT" --traj-ids "$@" --steps 400 --execution-horizon 8 \
        --save-plot-path "$EVAL/open_loop_$label.png") > "$EVAL/open_loop_$label.tmp" 2>&1 \
        && mv "$EVAL/open_loop_$label.tmp" "$EVAL/open_loop_$label.log" \
        || say "open-loop check on $label failed (see $EVAL/open_loop_$label.tmp); carrying on"
    grep -E "MSE for trajectory|Average M" "$EVAL/open_loop_$label.log" 2>/dev/null || true
}

# 4. fit: demos it trained on, demos from the test boards it never saw ----------------
mkdir -p "$EVAL"
say "4/6 serving $CKPT/checkpoint-$STEPS on port $PORT"
serve "$CKPT/checkpoint-$STEPS"
if [ "$OPEN_LOOP" = 1 ]; then
    if [ ! -f "$VAL/meta/info.json" ]; then
        say "recording the expert on 10 test boards (open-loop reference) -> $VAL"
        python -m harness_agent.groot_data record --out "$VAL" --seeds "${VAL_SEEDS:-0-9}" --workers "$WORKERS" --popped 0
    fi
    open_loop "$DATA" train 0 200 400 600 800 1000
    open_loop "$VAL" unseen 0 5 10 15 20 25
fi

# 5. closed loop -------------------------------------------------------------------------
say "5/6 closed-loop evaluation"
for h in $HORIZONS; do
    if [ "$h" = 8 ]; then evaluate "$EVAL" 8; else evaluate "${EVAL}_h$h" "$h"; fi
done

# 6. the previous model on the same boards -------------------------------------------------
if [ -n "$BASELINE" ] && [ -d "$BASELINE" ]; then
    base_name="$(basename "$(dirname "$BASELINE")")"
    say "6/6 baseline: $BASELINE on the same boards"
    serve "$BASELINE"
    evaluate "$HOME/eval/${base_name}_on_${NAME}_boards" 8
fi
stop_server

say "results"
for f in "$EVAL"/summary.md "$EVAL"_h*/summary.md "$HOME/eval/"*"_on_${NAME}_boards/summary.md"; do
    if [ -f "$f" ]; then echo "== $f"; cat "$f"; fi
done
for f in "$EVAL"/open_loop_*.log; do
    if [ -f "$f" ]; then echo "== $f"; grep -E "Average M" "$f" || true; fi
done

say "round finished. Results: ~/eval/$NAME*   Log: $LOG"
echo "Stop the VM in the Nebius console now (Compute > Virtual machines > Stop)."
