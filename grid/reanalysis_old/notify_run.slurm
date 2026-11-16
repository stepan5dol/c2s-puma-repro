#!/bin/bash
#SBATCH --job-name=tg-run
#SBATCH --account=vkalin
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=/xdisk/vkalin/dolzhenko/c2s/logs/%x_%j.out

# Telegram message for one run of the old grid re-run, after its analysis job ended (any state): how many runs are
# done, how long its AZI and analysis jobs took, its own table, violins shared -> own table (summarize.py --run).
# Submitted by notify_setup.sh. env: NAME REPO
set -euo pipefail
D=/xdisk/vkalin/dolzhenko/c2s
REPO=${REPO:-$D/repo}
OUT=$D/reanalysis_old
SIF=$D/images/analysis-sif-$(sha256sum "$REPO/apptainer/c2s-analysis.def" | cut -c1-12).sif
source "$REPO/grid/reanalysis_old/tg.sh"
MSG=$(mktemp "${TMPDIR:-/tmp}/c2s_run.XXXXXX")
trap 'rm -f "$MSG"' EXIT
while IFS=$'\t' read -r name run gj aj test azi; do
    g=-; [ "$gj" != 0 ] && g=$(sacct -n -X -P -j "$gj" -o State,Elapsed 2>/dev/null | head -n 1)
    a=$(sacct -n -X -P -j "$aj" -o State,Elapsed 2>/dev/null | head -n 1)
    printf '%s\t%s\t%s\n' "$name" "${g:--}" "${a:--}"
done < "$OUT/jobs.tsv" \
  | apptainer exec --cleanenv --no-home --writable-tmpfs --bind "$D:$D" --bind "$REPO:/repo:ro" \
        --env PYTHONPATH=/repo/grid/lib,HOME=/tmp,C2S_WORK=/tmp,C2S_ROOT=/tmp \
        "$SIF" python /repo/grid/reanalysis_old/summarize.py --run "${NAME:?}" --jobinfo - > "$MSG" \
  || echo "C2S old grid re-run: $NAME, message failed; log $D/logs/tg-${NAME}_$SLURM_JOB_ID.out" > "$MSG"
cat "$MSG"
tg_send "$MSG"
