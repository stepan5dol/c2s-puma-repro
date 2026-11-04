#!/bin/bash
#SBATCH --job-name=c2s-notify
#SBATCH --account=vkalin
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/xdisk/vkalin/dolzhenko/c2s/logs/%x_%j.out

# Send the result of a grid analysis job to Telegram: a short message (job states, STOPPED, NEXT) and the full
# report as a text file (the decider's output: violins, eval_loss, reverse task, McNemar, energy guard).
#   bash grid/notify_tg.sh --test                       login node: check the bot, find and save your chat id, send "test"
#   sbatch --dependency=afterany:<analysis job> grid/notify_tg.sh <analysis job>
# afterany: it also reports an analysis that failed or was cancelled because a cell failed.
# The bot token is read from ~/c2s/c2s.env (TG_TOKEN=...), next to HF_TOKEN; TG_CHAT is added there by --test.
# The token never appears on a command line (curl reads the URL from a file descriptor) or in the job log.
set -euo pipefail
D=${C2S_D:-/xdisk/vkalin/dolzhenko/c2s}
GRID=$D/grid
ENV_FILE=${C2S_ENV:-$HOME/c2s/c2s.env}
TG_API=${TG_API:-https://api.telegram.org}

[ -s "$ENV_FILE" ] || { echo "no $ENV_FILE" >&2; exit 1; }
TG_TOKEN=$(sed -n 's/^TG_TOKEN=//p' "$ENV_FILE" | tail -n 1)
TG_CHAT=$(sed -n 's/^TG_CHAT=//p' "$ENV_FILE" | tail -n 1)
[ -n "$TG_TOKEN" ] || { echo "no TG_TOKEN= line in $ENV_FILE" >&2; exit 1; }

api() {  # api <method> [curl args...]; prints the JSON reply, fails unless "ok":true
    local m=$1; shift
    local out
    out=$(curl -sS --max-time 60 --retry 3 -K <(printf 'url = "%s/bot%s/%s"\n' "$TG_API" "$TG_TOKEN" "$m") "$@") \
        || { echo "Telegram $m: no connection" >&2; return 1; }
    grep -q '"ok":true' <<< "$out" || { echo "Telegram $m: $(grep -o '"description":"[^"]*"' <<< "$out" || echo "$out" | cut -c1-200)" >&2; return 1; }
    printf '%s\n' "$out"
}

if [ "${1:-}" = --test ]; then
    if [ -z "$TG_CHAT" ]; then
        UPD=$(api getUpdates)
        TG_CHAT=$(grep -oE '"chat":\{"id":-?[0-9]+' <<< "$UPD" | tail -n 1 | grep -oE -- '-?[0-9]+$' || true)
        [ -n "$TG_CHAT" ] || { echo "the bot has no messages: write it anything in Telegram, then run --test again" >&2; exit 1; }
        echo "TG_CHAT=$TG_CHAT" >> "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        echo "chat id $TG_CHAT saved to $ENV_FILE"
    fi
    api sendMessage --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text=C2S grid: test from $(hostname -s)" > /dev/null
    echo "sent: check Telegram"
    exit 0
fi

AJ=${1:?usage: notify_tg.sh <analysis job id> | --test}
[[ $AJ =~ ^[0-9]+$ ]] || { echo "bad job id $AJ" >&2; exit 1; }
[ -n "$TG_CHAT" ] || { echo "no TG_CHAT in $ENV_FILE: run bash grid/notify_tg.sh --test first" >&2; exit 1; }
mkdir -p "$GRID/reports"
REP=$GRID/reports/analysis_$AJ.txt
MSG=$(mktemp "${TMPDIR:-/tmp}/c2s_notify.XXXXXX")
trap 'rm -f "$MSG"' EXIT

STATE=$(sacct -n -X -P -j "$AJ" -o State,Elapsed 2>/dev/null | head -n 1 | tr '|' ' ')
# the part of chain.log written by this analysis job, and in it the decider's output
PART=$(awk -v j="analysis job $AJ " 'index($0, j) {on = 1} on' "$GRID/chain.log" 2>/dev/null || true)
DEC=$(awk '/ decider$/ {buf = ""; on = 1; next} on {buf = buf $0 "\n"} END {printf "%s", buf}' <<< "$PART")
STOP=$(cat "$GRID/STOPPED" 2>/dev/null || true)
[ -n "$STOP" ] && STOP="$STOP (file written $( (stat -c %y "$GRID/STOPPED" 2>/dev/null || stat -f %Sm "$GRID/STOPPED") | cut -c1-19))"
{
    echo "C2S grid, analysis job $AJ: ${STATE:-no sacct record}"
    echo "report written $(date '+%F %T')"
    echo
    echo "== cells (cells.tsv) and their jobs"
    while IFS=$'\t' read -r name run tj ij; do
        s=$( [ "$tj" != 0 ] && sacct -n -X -P -j "$tj" -o State,Elapsed 2>/dev/null | head -n 1 | tr '|' ' ' || true)
        echo "$name  job ${tj}  ${s:--}"
    done < "$GRID/cells.tsv"
    echo
    echo "== STOPPED"
    echo "${STOP:-(none: the chain goes on)}"
    echo
    if [ -n "$DEC" ]; then
        echo "== decision (decider output)"
        printf '%s\n' "$DEC"
    else
        echo "== the decider did not run; this job's lines in chain.log"
        printf '%s\n' "${PART:-(none)}"
        echo
        echo "== end of the Slurm log of the analysis job"
        tail -n 40 "$D/logs/c2s-analyze_$AJ.out" 2>/dev/null || echo "(no log)"
    fi
} > "$REP"

{
    echo "C2S grid: analysis $AJ ${STATE:-?}"
    [ -n "$STOP" ] && echo "STOPPED: $STOP"
    grep -E '^  LR set by' <<< "$DEC" || true
    # the last stage in full (B if it ran, else A): violins, eval_loss, reverse task, McNemar, energy; then NEXT
    if grep -q '^STAGE B' <<< "$DEC"; then sed -n '/^STAGE B/,$p' <<< "$DEC"; else sed -n '/^STAGE A/,$p' <<< "$DEC"; fi \
        | grep -vE '^(STOP|TRAIN):' || true
    echo
    echo "full report: file below; on Puma $REP"
} | cut -c1-600 | head -c 3800 > "$MSG"

api sendMessage --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text@$MSG" > /dev/null
api sendDocument -F "chat_id=$TG_CHAT" -F "document=@$REP" > /dev/null
echo "sent to Telegram: message and $REP"
