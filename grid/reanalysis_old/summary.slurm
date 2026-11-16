#!/bin/bash
#SBATCH --job-name=re-summary
#SBATCH --account=vkalin
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/xdisk/vkalin/dolzhenko/c2s/logs/%x_%j.out

# Summary of the re-run of the old grid (summarize.py) -> /xdisk/vkalin/dolzhenko/c2s/reanalysis_old/report.txt (full,
# with the per-stratum matrix) and message.txt (short), then Telegram (tg.sh): message.txt as one monospace block and
# report.txt as a file. If Telegram fails, both stay on disk.
# Started after all analysis jobs (afterany: also when some failed; they show as FAILED / MISSING).
set -euo pipefail
D=/xdisk/vkalin/dolzhenko/c2s
REPO=${REPO:-$D/repo}
OUT=$D/reanalysis_old
SIF=$D/images/analysis-sif-$(sha256sum "$REPO/apptainer/c2s-analysis.def" | cut -c1-12).sif
source "$REPO/grid/reanalysis_old/tg.sh"
rc=0
while IFS=$'\t' read -r name run gj aj test azi; do
    g=-; [ "$gj" != 0 ] && g=$(sacct -n -X -P -j "$gj" -o State,Elapsed 2>/dev/null | head -n 1)
    a=$(sacct -n -X -P -j "$aj" -o State,Elapsed 2>/dev/null | head -n 1)
    printf '%s\t%s\t%s\n' "$name" "${g:--}" "${a:--}"
done < "$OUT/jobs.tsv" \
  | apptainer exec --cleanenv --no-home --writable-tmpfs --bind "$D:$D" --bind "$REPO:/repo:ro" \
        --env PYTHONPATH=/repo/grid/lib,HOME=/tmp,C2S_WORK=/tmp,C2S_ROOT=/tmp \
        "$SIF" python /repo/grid/reanalysis_old/summarize.py --jobinfo - || rc=$?
if [ "$rc" -ne 0 ] || [ ! -s "$OUT/message.txt" ]; then
    M=$(mktemp "${TMPDIR:-/tmp}/c2s_sum.XXXXXX")
    echo "C2S old grid re-run: summary FAILED (exit $rc), job $SLURM_JOB_ID; log $D/logs/re-summary_$SLURM_JOB_ID.out" > "$M"
    tg_send "$M"
    rm -f "$M"
else
    tg_send "$OUT/message.txt" "$OUT/report.txt"
fi
exit $rc
