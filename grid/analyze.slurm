#!/bin/bash
#SBATCH --job-name=c2s-analyze
#SBATCH --account=vkalin
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/xdisk/vkalin/dolzhenko/c2s/logs/%x_%j.out

# Grid analysis step: the manuscript violins for the finished cells, then the decider (grid/lib/decide.py),
# which writes its decision and the cells to train next. It submits those cells itself (sbatch); the rule is
# fixed in decide.py, the chain is limited to MAX_CELLS trainings in total and stops on any error.
# Started by grid/submit_cells.sh (afterok on the inference jobs of the cells it queued).
#
# memory: the counts matrix of the 5 human conditions is ~10k cells x 36.6k genes float32 = 1.5 GB, plus the atlas
# and the model matrices; 8 CPUs x 5 GB = 40 GB.
set -euo pipefail

D=/xdisk/vkalin/dolzhenko/c2s
GRID=$D/grid                      # state, decisions, per-run outputs
# the analysis image: built on GitHub from apptainer/c2s-analysis.def, its release tag is the hash of that recipe
SIF_TAG=analysis-sif-$(sha256sum "${REPO:-$D/repo}/apptainer/c2s-analysis.def" | cut -c1-12)
SIF=$D/images/$SIF_TAG.sif
SIF_URL=https://github.com/stepan5dol/c2s-puma-repro/releases/download/$SIF_TAG
REPO=${REPO:-$D/repo}             # a checkout of this repository (grid/lib, grid/tree, hpc)
MAX_CELLS=${MAX_CELLS:-9}
WORK=/tmp/c2s_analyze_$SLURM_JOB_ID
mkdir -p "$GRID"/{runs,decisions,logs} "$WORK"
trap 'rm -rf "$WORK"' EXIT
note() { echo "$(date '+%F %T') $*" | tee -a "$GRID/chain.log"; }

note "analysis job $SLURM_JOB_ID on $(hostname)"
fetch_image() {
    mkdir -p "$D/images"
    curl -fsSL --retry 3 "$SIF_URL/c2s-analysis.sif.sha256" -o "$SIF.sha256" || return 1
    curl -fL --retry 3 -sS "$SIF_URL/c2s-analysis.sif" -o "$SIF.part" || return 1
    [ "$(sha256sum < "$SIF.part" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$SIF.sha256")" ] || { note "sha256 of the download differs"; return 1; }
    mv "$SIF.part" "$SIF"
}
if [ ! -s "$SIF" ]; then
    note "downloading the analysis image $SIF_TAG"
    fetch_image || { note "FAIL: could not download $SIF_URL/c2s-analysis.sif (did the build-analysis-sif workflow finish?)"; exit 1; }
fi
# the data: deploy.sh put them flat in data_src; the code wants the counts under BPD-PH/
mkdir -p "$WORK/data/BPD-PH"
for f in GSE275938_compiled_counts.csv GSE275938_cell_metadata.csv; do ln -s "$D/data_src/$f" "$WORK/data/BPD-PH/$f"; done
for f in rat.ho.azi.integrated.h5ad he_lung_atlas.h5ad 2022FetalLungIntCounts.h5ad cell_types_config.json; do ln -s "$D/data_src/$f" "$WORK/data/$f"; done
for f in "$WORK"/data/BPD-PH/* "$WORK"/data/*.h5ad "$WORK"/data/*.json; do
    [ -s "$f" ] || { note "FAIL: missing data file $(readlink "$f") (run deploy.sh)"; exit 1; }
done
ARUN() { apptainer exec --cleanenv --no-home --writable-tmpfs \
    --bind "$WORK:/work" --bind "$D:$D" --bind "$REPO:/repo:ro" \
    --env C2S_ROOT=/work/root,C2S_WORK=$GRID,C2S_RESULTS=$GRID/reference,PYTHONPATH=/repo/grid/lib,PYTHONFAULTHANDLER=1,HOME=/tmp \
    "$SIF" "$@"; }

# working root: the manuscript code with its path prefix rewritten, data linked from data_src
ARUN python /repo/grid/lib/prepare_root.py /repo/grid/tree /work/data /work/root >> "$GRID/chain.log" 2>&1

# the cells: grid/cells.tsv has one line per submitted cell: name  run_folder  train_job  infer_job
[ -s "$GRID/cells.tsv" ] || { note "FAIL: no $GRID/cells.tsv"; exit 1; }
ARUN python - "$GRID" <<'PY'
import json, sys
from pathlib import Path
g = Path(sys.argv[1]); runs = {}
for line in open(g / "cells.tsv"):
    name, run, tj, ij = line.rstrip("\n").split("\t")[:4]
    r = Path(run)
    inf = r / "inference_float32" / "test_t10.jsonl"
    ts = r / "trainer_state.json"
    if not ts.exists():
        cks = sorted(r.glob("sft_pythia1b_short_full/checkpoint-*/trainer_state.json"), key=lambda q: int(q.parent.name.split("-")[1]))
        ts = cks[-1] if cks else ts
    if inf.exists() and ts.exists() and (g / "runs" / name / "violins" / "scores.json").exists() is False:
        runs[name] = str(inf)
json.dump(runs, open(g / "todo_runs.json", "w"))
print(len(runs), "cell(s) need the violins:", list(runs))
PY
if [ "$(ARUN python -c "import json;print(len(json.load(open('$GRID/todo_runs.json'))))")" -gt 0 ]; then
    note "violins for the new cells (one pass over the counts CSV)"
    ARUN python -u /repo/grid/lib/violins_only.py "$GRID/todo_runs.json" >> "$GRID/chain.log" 2>&1
fi

note "decider"
rm -f "$GRID/next_cells.txt"
ARUN python -u /repo/grid/lib/chain_step.py --grid "$GRID" --repo /repo --max-cells "$MAX_CELLS" 2>&1 | tee -a "$GRID/chain.log"
# the rule's next cells, if any: submit them (outside the container, from this job; the rule and the cap are fixed in code)
if [ -s "$GRID/next_cells.txt" ]; then
    note "submitting: $(cat "$GRID/next_cells.txt")"
    bash "$REPO/grid/submit_cells.sh" $(cat "$GRID/next_cells.txt")
else
    note "nothing to submit; see $GRID/decisions and $GRID/STOPPED"
fi
