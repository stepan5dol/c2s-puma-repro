#!/bin/bash
# Telegram for the old grid re-run that submit.sh already queued: one message per run when its analysis job ends
# (notify_run.slurm) and the summary at the end (summary.slurm). Replaces the pending summary job of submit.sh.
# Touches no running or queued analysis/inference job. Run once on the Puma login node:
#   bash /xdisk/vkalin/dolzhenko/c2s/repo/grid/reanalysis_old/notify_setup.sh
set -euo pipefail
D=/xdisk/vkalin/dolzhenko/c2s
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$D/reanalysis_old
[ -s "$OUT/jobs.tsv" ] || { echo "no $OUT/jobs.tsv: run submit.sh first" >&2; exit 1; }
for f in tg.sh notify_run.slurm summary.slurm summarize.py; do [ -s "$HERE/$f" ] || { echo "missing $HERE/$f" >&2; exit 1; }; done

old=$(squeue -u "$USER" -h -n re-summary -t PENDING -o %i | tr '\n' ' ')
if [ -n "${old// /}" ]; then scancel $old; echo "cancelled the pending summary job(s) of submit.sh: $old"; fi

in_queue() { squeue -h -j "$1" 2>/dev/null | grep -q .; }
DEPS=""
while IFS=$'\t' read -r name run gj aj test azi; do
    dep=""
    if in_queue "$aj"; then dep="--dependency=afterany:$aj"; DEPS+="${DEPS:+:}$aj"; fi
    nj=$(sbatch --parsable $dep --job-name="tg-$name" --export=ALL,NAME="$name",REPO="$D/repo" "$HERE/notify_run.slurm")
    echo "$name: message after analysis job $aj -> job $nj"
    DEPS+="${DEPS:+:}$nj"
done < "$OUT/jobs.tsv"
sj=$(sbatch --parsable --dependency=afterany:$DEPS --job-name=re-summary --export=ALL,REPO="$D/repo" "$HERE/summary.slurm")
echo "summary job $sj (after all runs and their messages) -> $OUT/report.txt, message.txt and Telegram"
