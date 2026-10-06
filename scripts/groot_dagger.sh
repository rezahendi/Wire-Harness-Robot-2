#!/usr/bin/env bash
# One DAgger-style round, unattended, on the GPU VM:
#   1. the current model (PREV) on the 40 test boards with the seat assist (the hybrid), alone
#      and with restarts;
#   2. the same on training boards, recording every takeover (the expert seating the wire from
#      the policy's own stuck states) as training episodes;
#   3. the next round (NEXT, scripts/groot_round.sh): new demos (30% of them seat recoveries
#      from deliberately bad descents), PREV's training set + the takeovers + the new demos,
#      fine-tuning, and its evaluations, alone and with the seat assist.
#
#   tmux new -s night        (or tmux a -t night to come back)
#   bash ~/Wire-Harness-Robot-2/scripts/groot_dagger.sh route_v5 route_v6
#
# Every step is skipped when its result is there, so after a preempted VM the same command
# carries on. Settings: PREV_DATA (PREV's training set, ~/data/PREV), PREV_STEPS (PREV's last
# checkpoint),
# EVAL_SEEDS (0-39), EVAL_WORKERS (6), TAKEOVER_SEEDS (8000-8199), ASSIST (1.5), BUILDS (400),
# SEED_START (6000), SEAT_RECOVERIES (0.3), STEPS (20000), SAVE_STEPS (5000), MIN_START_GB (90).
set -euo pipefail

PREV="${1:?usage: groot_dagger.sh PREV NEXT, e.g. route_v5 route_v6}"
NEXT="${2:?usage: groot_dagger.sh PREV NEXT, e.g. route_v5 route_v6}"
PREV_DATA="${PREV_DATA:-$HOME/data/$PREV}"
PREV_STEPS="${PREV_STEPS:-$(ls -d "$HOME/ckpt/$PREV"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' \
    | grep -E '^[0-9]+$' | sort -n | tail -1)}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAKEOVERS="$HOME/eval/${PREV}_takeovers/takeovers"
export EVAL_SEEDS="${EVAL_SEEDS:-0-39}" EVAL_WORKERS="${EVAL_WORKERS:-6}"
export TAKEOVER_SEEDS="${TAKEOVER_SEEDS:-8000-8199}" ASSIST="${ASSIST:-1.5}"
say() { echo; echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
free_gb() { df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9'; }

if [ -z "$PREV_STEPS" ]; then
    echo "no checkpoint of $PREV in ~/ckpt/$PREV"
    exit 1
fi
if [ ! -f "$PREV_DATA/meta/info.json" ]; then
    echo "no training set at $PREV_DATA (ls ~/data); run again with PREV_DATA=<the set $PREV was trained on>"
    exit 1
fi
if [ ! -d "$HOME/ckpt/$NEXT/checkpoint-${STEPS:-20000}" ] && [ ! -d "$HOME/ckpt/$NEXT" ] \
        && [ "$(free_gb)" -lt "${MIN_START_GB:-90}" ]; then
    echo "only $(free_gb) GB free; the night needs ~${MIN_START_GB:-90} GB (new demos, the merged set, three"
    echo "22 GB checkpoints while training saves). Delete checkpoints you no longer need (du -sh ~/ckpt/*/*)."
    exit 1
fi

say "DAgger round: $PREV (checkpoint-$PREV_STEPS) with the seat assist, takeovers on boards $TAKEOVER_SEEDS, then $NEXT"
QUIET_END=1 STEPS="$PREV_STEPS" VARIANTS="assist assist_best takeovers" bash "$REPO/scripts/groot_evals.sh" "$PREV"

base="$PREV_DATA"
if compgen -G "$TAKEOVERS/staging/*/meta.json" > /dev/null; then     # (gone once a merge packaged them)
    base="$base $TAKEOVERS"
    say "takeover episodes: $(ls -d "$TAKEOVERS"/staging/*/ 2>/dev/null | wc -l) in $TAKEOVERS"
else
    say "no takeover episodes in $TAKEOVERS; training without them"
fi
BASE="$base" BUILDS="${BUILDS:-400}" SEED_START="${SEED_START:-6000}" SEAT_RECOVERIES="${SEAT_RECOVERIES:-0.3}" \
    STEPS="${STEPS:-20000}" SAVE_STEPS="${SAVE_STEPS:-5000}" BASELINE="" VARIANTS="ens4 assist" \
    bash "$REPO/scripts/groot_round.sh" "$NEXT"
