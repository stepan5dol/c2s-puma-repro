#!/bin/bash
# Queue grid cells on Puma. For every cell: one training, then (afterok) one fp32 inference of the test set at
# T 1.0; for all cells of the call together: one analysis job (grid/analyze.slurm) after all their inferences.
# The analysis job decides what to train next and calls this script again for those cells, so after the first
# call the grid runs by itself until the rule says stop, a job fails, or the cap (MAX_CELLS) is reached.
# Run the first call yourself on the login node (UA policy: an agent must not submit jobs).
#
# usage:  bash grid/submit_cells.sh LR:EPOCHS:SEED [LR:EPOCHS:SEED ...]       e.g.  1e-5:2:42 3e-5:2:42
set -euo pipefail
D=/xdisk/vkalin/dolzhenko/c2s
GRID=$D/grid
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
mkdir -p "$D/logs" "$GRID"
[ -s "$D/images/c2s-analysis.sif" ] || { echo "no $D/images/c2s-analysis.sif: build it first (grid/build_analysis_image.sh)" >&2; exit 1; }
[ -s "$D/images/c2s-train.sif" ] || { echo "no $D/images/c2s-train.sif" >&2; exit 1; }
mkdir -p "$GRID/reference" && cp -n "$REPO/grid/reference/identity_scores.json" "$GRID/reference/"
INF_JOBS=""
for cell in "$@"; do
    IFS=: read -r LR EP SEED <<< "$cell"
    [[ $LR =~ ^[0-9]+(\.[0-9]+)?e-[0-9]+$ && $EP =~ ^[0-9]+$ && $SEED =~ ^[0-9]+$ ]] \
        || { echo "bad cell '$cell' (want LR:EPOCHS:SEED, e.g. 3e-5:2:42)" >&2; exit 1; }
    NAME="lr${LR}_ep${EP}_s${SEED}"
    grep -q "^$NAME	" "$GRID/cells.tsv" 2>/dev/null && { echo "cell $NAME is already in the grid" >&2; exit 1; }
    STAMP=$(date +%Y-%m-%d_%H%M%S)
    RUN=$D/results/${STAMP}_puma-a100mig-repro-ep${EP}-fp32-lr${LR}-s${SEED}
    # time of the training: 2.5 h per epoch (measured, fp32 on the A100 slice) x 1.6 + 1 h for setup and saving
    HOURS=$(awk -v e="$EP" 'BEGIN{h = e * 2.5 * 1.6 + 1; printf "%d", (h == int(h)) ? h : int(h) + 1}')
    TRAIN=$(MODEL=pythia1b EPOCHS=$EP LR=$LR SEED=$SEED PRECISION=fp32 INFER=0 NO_EARLY_STOP=1 CKPTS=none RUN_STAMP=$STAMP \
        sbatch --parsable --job-name="g-$NAME" --time="$HOURS:00:00" "$REPO/hpc/train.slurm")
    INFER=$(RUN=$RUN SETS=test_t10 sbatch --parsable --job-name="i-$NAME" --time=04:00:00 \
        --dependency=afterok:$TRAIN --kill-on-invalid-dep=yes "$REPO/hpc/infer_only.slurm")
    printf '%s\t%s\t%s\t%s\n' "$NAME" "$RUN" "$TRAIN" "$INFER" | tee -a "$GRID/cells.tsv"
    INF_JOBS+="${INF_JOBS:+:}$INFER"
done
# the analysis waits for all inferences of this call; if one of them fails, it is cancelled and the mail says so
sbatch --parsable --job-name=c2s-analyze --dependency=afterok:$INF_JOBS --kill-on-invalid-dep=yes \
    --export=ALL,REPO="$REPO" "$REPO/grid/analyze.slurm" | tee -a "$GRID/chain.log"
