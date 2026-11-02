#!/bin/bash
# Start the whole grid with one command on the Puma login node (UA policy: you submit, not an agent):
#   bash /xdisk/vkalin/dolzhenko/c2s/repo/grid/start.sh
# It checks everything first (analysis image published on GitHub, data in data_src), then
#   the starting cell lr1e-4_ep2_s42 = run 2026-10-28_1152: its fp32 inference of the test set, unless already there;
#   stage A: trainings 1e-5 and 3e-5 (2 epochs, seed 42) with their inferences, and the analysis job after all of them.
# From there the chain runs by itself (grid/README.md).
set -euo pipefail
D=/xdisk/vkalin/dolzhenko/c2s
GRID=$D/grid
REPO=$(cd "$(dirname "$0")/.." && pwd)
START=$D/results/2026-10-28_1152_puma-a100mig-repro-ep2-fp32

[ -e "$GRID/cells.tsv" ] && { echo "$GRID/cells.tsv exists: the grid was already started" >&2; exit 1; }
[ -d "$START" ] || { echo "no starting run $START" >&2; exit 1; }
bash "$REPO/grid/submit_cells.sh" --check

AFTER=""
if [ -s "$START/inference_float32/test_t10.jsonl" ]; then
    echo "starting cell: fp32 inference is already there"
elif [ -e "$START/inference_float32" ]; then
    echo "$START/inference_float32 exists but has no test_t10.jsonl: look at it before starting" >&2; exit 1
else
    AFTER=$(RUN=$START SETS=test_t10 sbatch --parsable --job-name=i-lr1e-4_ep2_s42 --time=04:00:00 "$REPO/hpc/infer_only.slurm")
    echo "starting cell: fp32 inference job $AFTER"
fi
mkdir -p "$GRID"
printf 'lr1e-4_ep2_s42\t%s\t0\t%s\n' "$START" "${AFTER:-0}" | tee "$GRID/cells.tsv"
AFTER=$AFTER bash "$REPO/grid/submit_cells.sh" 1e-5:2:42 3e-5:2:42
