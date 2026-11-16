#!/bin/bash
# Re-run of the 8 old grid runs (2026-11-02..04, /xdisk/.../c2s/grid_old_20261105) with each run's own human rescued
# table, as the manuscript builds it. Run on the Puma login node (UA policy: you submit, not an agent):
#   bash /xdisk/vkalin/dolzhenko/c2s/repo/grid/reanalysis_old/submit.sh
# Per old run:
#   - no AZI answers yet: one GPU job, its end-of-run model answers the 631 AZI prompts, with infer_only_20261104.slurm
#     unchanged (the old grid's own inference script: inputs from Hugging Face with their sha256, vLLM float32);
#   - then one CPU job (analyze_one.slurm): its rescued table and the violins S2-S6 by the manuscript's code.
# Plus one CPU job for the manuscript model as a check (its outputs against the manuscript's files) and one summary job
# after all of them (summary.slurm -> /xdisk/vkalin/dolzhenko/c2s/reanalysis_old/report.txt).
# Everything goes to /xdisk/vkalin/dolzhenko/c2s/reanalysis_old/; if that folder exists the script stops.
set -euo pipefail
D=/xdisk/vkalin/dolzhenko/c2s
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$D/reanalysis_old
CELLS=$D/grid_old_20261105/cells.tsv

[ -s "$CELLS" ] || { echo "no $CELLS" >&2; exit 1; }
[ -s "$HERE/MANIFEST.json" ] || { echo "no $HERE/MANIFEST.json: run build_bundle.py on the Mac, then deploy.sh" >&2; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists: look at it, move it away to re-run" >&2; exit 1; }
for f in gold/test_t10.jsonl gold/inference_azi_results.jsonl infer_only_20261104.slurm analyze_one.slurm \
         record_violins.py summary.slurm summarize.py \
         tree/retrain_v2/analysis/real_counts_suite_ortho/human_panel_variants/human_rescue_pipeline_genomewide.py \
         tree/retrain_v2/analysis/real_counts_suite_ortho/human_panel_variants/filter_to_rat_schema.py; do
    [ -s "$HERE/$f" ] || { echo "missing $HERE/$f" >&2; exit 1; }
done
# check every run before submitting anything. The runs come only from the old grid's own list (grid_old_20261105/cells.tsv);
# its answers must be in the old format: 1566 test answers (the new test set has 4198) and the input cell in the prompt
while IFS=$'\t' read -r name run tj ij; do
    [ -f "$run/sft_pythia1b_short_full/config.json" ] || { echo "$name: no model in $run/sft_pythia1b_short_full" >&2; exit 1; }
    t=$run/inference_float32/test_t10.jsonl
    [ -s "$t" ] || { echo "$name: no $t" >&2; exit 1; }
    n=$(wc -l < "$t"); [ "$n" -eq 1566 ] || { echo "$name: $n test answers in $t, the old test set has 1566" >&2; exit 1; }
    grep -q "Unexposed:" "$t" || { echo "$name: $t has no input cell in its prompts: not the old format" >&2; exit 1; }
    a=$run/inference_float32/inference_azi_results.jsonl
    if [ -s "$a" ]; then
        n=$(wc -l < "$a"); [ "$n" -eq 631 ] || { echo "$name: $n AZI answers in $a, expected 631" >&2; exit 1; }
        grep -q "Unexposed:" "$a" || { echo "$name: $a has no input cell in its prompts: not the old format" >&2; exit 1; }
    fi
done < "$CELLS"

mkdir -p "$OUT" "$D/logs"
cp "$CELLS" "$OUT/cells_old.tsv"
analysis() {  # analysis <name> <test answers> <AZI answers> [GPU job to wait for]
    sbatch --parsable ${4:+--dependency=afterok:$4 --kill-on-invalid-dep=yes} --job-name="re-$1" \
        --export=ALL,NAME="$1",TEST="$2",AZI="$3",REPO="$D/repo" "$HERE/analyze_one.slurm"
}
ALL=""
while IFS=$'\t' read -r name run tj ij; do
    mkdir "$OUT/$name"
    test=$run/inference_float32/test_t10.jsonl
    azi=$run/inference_float32/inference_azi_results.jsonl
    gj=""
    if [ -s "$azi" ]; then
        echo "$name: AZI answers already there ($azi)"
    else
        gj=$(RUN="$OUT/$name" MODEL_PATH="$run/sft_pythia1b_short_full" SETS=inference_azi DTYPE=float32 \
             sbatch --parsable --job-name="azi-$name" --time=04:00:00 "$HERE/infer_only_20261104.slurm")
        azi=$OUT/$name/inference_float32/inference_azi_results.jsonl
        echo "$name: AZI inference job $gj"
    fi
    aj=$(analysis "$name" "$test" "$azi" "$gj")
    echo "$name: analysis job $aj"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$run" "${gj:-0}" "$aj" "$test" "$azi" >> "$OUT/jobs.tsv"
    ALL+="${ALL:+:}$aj"
done < "$CELLS"
mkdir "$OUT/manuscript"
aj=$(analysis manuscript "$HERE/gold/test_t10.jsonl" "$HERE/gold/inference_azi_results.jsonl")
echo "manuscript model (check): analysis job $aj"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' manuscript "-" 0 "$aj" "$HERE/gold/test_t10.jsonl" "$HERE/gold/inference_azi_results.jsonl" >> "$OUT/jobs.tsv"
ALL+=":$aj"
sj=$(sbatch --parsable --dependency=afterany:$ALL --job-name=re-summary --export=ALL,REPO="$D/repo" "$HERE/summary.slurm")
echo "summary job $sj -> $OUT/report.txt"
echo "jobs: $OUT/jobs.tsv   watch: squeue -u \$USER"
