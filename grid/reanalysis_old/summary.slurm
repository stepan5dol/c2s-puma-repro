#!/bin/bash
#SBATCH --job-name=re-summary
#SBATCH --account=vkalin
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/xdisk/vkalin/dolzhenko/c2s/logs/%x_%j.out

# Summary of the re-run of the old grid (grid/reanalysis_old/summarize.py) -> /xdisk/vkalin/dolzhenko/c2s/reanalysis_old/report.txt
# Started by submit.sh after all analysis jobs (afterany: also when some failed; they show as MISSING).
set -euo pipefail
D=/xdisk/vkalin/dolzhenko/c2s
REPO=${REPO:-$D/repo}
SIF=$D/images/analysis-sif-$(sha256sum "$REPO/apptainer/c2s-analysis.def" | cut -c1-12).sif
apptainer exec --cleanenv --no-home --writable-tmpfs --bind "$D:$D" --bind "$REPO:/repo:ro" \
    --env PYTHONPATH=/repo/grid/lib,HOME=/tmp,C2S_WORK=/tmp,C2S_ROOT=/tmp \
    "$SIF" python /repo/grid/reanalysis_old/summarize.py
