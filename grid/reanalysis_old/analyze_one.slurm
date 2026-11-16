#!/bin/bash
#SBATCH --job-name=re-old
#SBATCH --account=vkalin
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/xdisk/vkalin/dolzhenko/c2s/logs/%x_%j.out

# One old grid run (or the manuscript model as a check): its human rescued table and the violins S2-S6, by the
# manuscript's code unchanged (grid/reanalysis_old/tree = grid/tree at deployed-20261104-1346 + the two table scripts).
# In a fresh working root the run's answers are put where the manuscript code reads its own
# (retrain_v2/results/inference_azi_results.jsonl and test_inference_results_t10.jsonl), so the scripts run as they
# are, as scripts, as in the manuscript:
#   1. human_rescue_pipeline_genomewide.py   this run's AZI answers -> reports/human_rescue_genomewide.csv
#   2. filter_to_rat_schema.py               -> reports/rat_schema/human_*_rescue_predicted_azi.xlsx (12 files)
#   3. record_violins.py: build_disease_arms_trajectory.main() (3 human arms) and build_all_panel_variants.main()
#      (2 rat arms), unchanged; it only records the per-cell values they draw (scores.json, read by grid/lib/decide.py)
# The manuscript's own table is removed from the working root first: a missing table of this run stops the job
# instead of the manuscript's table being read in its place.
# env (set by submit.sh): NAME  TEST (test answers)  AZI (AZI answers)  REPO
# output: /xdisk/vkalin/dolzhenko/c2s/reanalysis_old/$NAME/violins/  (must not exist)
set -euo pipefail
D=/xdisk/vkalin/dolzhenko/c2s
REPO=${REPO:-$D/repo}
B=$REPO/grid/reanalysis_old
OUT=$D/reanalysis_old/${NAME:?}/violins
H=retrain_v2/analysis/real_counts_suite_ortho/human_panel_variants
SIF=$D/images/analysis-sif-$(sha256sum "$REPO/apptainer/c2s-analysis.def" | cut -c1-12).sif
WORK=/tmp/c2s_re_$SLURM_JOB_ID

[ -s "$SIF" ] || { echo "no analysis image $SIF" >&2; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists: not overwriting" >&2; exit 1; }
for f in "${TEST:?}" "${AZI:?}"; do [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }; done
echo "$(date '+%F %T') $NAME  test $TEST  AZI $AZI  on $(hostname)"

mkdir -p "$WORK/data/BPD-PH"
trap 'rm -rf "$WORK"' EXIT
for f in GSE275938_compiled_counts.csv GSE275938_cell_metadata.csv; do ln -s "$D/data_src/$f" "$WORK/data/BPD-PH/$f"; done
for f in rat.ho.azi.integrated.h5ad he_lung_atlas.h5ad 2022FetalLungIntCounts.h5ad cell_types_config.json; do ln -s "$D/data_src/$f" "$WORK/data/$f"; done
for f in "$WORK"/data/BPD-PH/* "$WORK"/data/*.h5ad "$WORK"/data/*.json; do
    [ -s "$f" ] || { echo "missing data file $(readlink "$f")" >&2; exit 1; }
done
ARUN() { apptainer exec --cleanenv --no-home --writable-tmpfs \
    --bind "$WORK:/work" --bind "$D:$D" --bind "$REPO:/repo:ro" \
    --env C2S_ROOT=/work/root,PYTHONFAULTHANDLER=1,HOME=/tmp \
    "$SIF" "$@"; }

ARUN python /repo/grid/lib/prepare_root.py /repo/grid/reanalysis_old/tree /work/data /work/root
R=$WORK/root
mkdir -p "$R/retrain_v2/results"
ln -s "$AZI" "$R/retrain_v2/results/inference_azi_results.jsonl"
ln -s "$TEST" "$R/retrain_v2/results/test_inference_results_t10.jsonl"
rm "$R/$H/reports/rat_schema/"human_*_rescue_predicted_azi.xlsx
[ -z "$(ls "$R/$H/reports/rat_schema/" 2>/dev/null)" ] || { echo "rat_schema not empty after removal" >&2; exit 1; }

echo "$(date '+%F %T') 1-2. rescued table: human_rescue_pipeline_genomewide.py, filter_to_rat_schema.py"
ARUN bash -c "cd /work/root/$H && python human_rescue_pipeline_genomewide.py && python filter_to_rat_schema.py"
n=$(ls "$R/$H/reports/rat_schema/"human_*_rescue_predicted_azi.xlsx | wc -l)
[ "$n" -eq 12 ] || { echo "the rescued table has $n of 12 workbooks (condition x cell type): stopping" >&2; exit 1; }

echo "$(date '+%F %T') 3. violins S2-S6"
ARUN python /repo/grid/reanalysis_old/record_violins.py /work/root /work/out

mkdir -p "$(dirname "$OUT")"
cp -r "$WORK/out" "$OUT"
sha256sum "$TEST" "$AZI" "$B/MANIFEST.json" > "$OUT/inputs.sha256"
echo "$(date '+%F %T') done -> $OUT"
