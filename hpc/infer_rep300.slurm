#!/bin/bash
#SBATCH --job-name=c2s-rep300
#SBATCH --account=vkalin
#SBATCH --partition=gpu_standard
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                 # 4 x 5 GB = 20 GB RAM; 1B model on disk in fp32 is 4 GB
#SBATCH --time=08:00:00                   # 1200 generations; fp32 AZI of lr1e-4 ep4 ran 631 in 1.2 h -> <= ~2.3 h in bf16
#SBATCH --array=0-5
#SBATCH --output=/xdisk/vkalin/dolzhenko/c2s/logs/%x_%A_%a.out

# Repeated sampling for the human disease states: N generations per prompt (inference_rep300/build_rep300.py), one array
# task per (state, arm), on the final model of an existing training run. Copy of hpc/infer_only.slurm: same image, same
# patched run_azi.py (vLLM dtype, forward prompt recognised by its new first words, forward limit 7936), same dtype checks;
# changed: vLLM in bfloat16 (author's decision 2026-11-18: inference in bf16; fp32 weights downcast, as ALLOW_DOWNCAST=1 in
# c2s-puma-std; basis results/fp32_2026-10-30/fp32_vs_bf16.log), inputs from the repository (inference_rep300/, checked against its sha256.txt), output folder
# $RUN/inference_bfloat16_rep300/, the model checked against final_model_sha256 of record.md. Both arms go through run_azi.py
# (run_test.py encodes prompts the same way: bos + prompt, no special tokens).
# usage: RUN=/xdisk/vkalin/dolzhenko/c2s/results/<run folder> sbatch infer_rep300.slurm

set -euo pipefail

RUN=${RUN:?set RUN to the training run folder}
DTYPE=bfloat16
TASKS=(Acute26_noazi Acute26_azi BPD7mo_noazi BPD7mo_azi BPDPH7mo_noazi BPDPH7mo_azi)  # order of build_rep300.py
NAME=${TASKS[$SLURM_ARRAY_TASK_ID]}
DATA=/xdisk/vkalin/dolzhenko/c2s
REPO=$DATA/repo
IN=$REPO/inference_rep300
SIF=$DATA/images/c2s-train.sif
MODEL_PATH=$RUN/sft_pythia1b_short_full
OUT=$RUN/inference_${DTYPE}_rep300
RES=$OUT/${NAME}_results.jsonl
WORK=/tmp/c2s_rep300_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

[ -f "$MODEL_PATH/config.json" ] || { echo "no model in $MODEL_PATH" >&2; exit 1; }
mkdir -p "$OUT"
[ ! -e "$RES" ] || { echo "$RES exists: not overwriting" >&2; exit 1; }
mkdir -p "$WORK"/{in,code,hf,triton}
trap 'rm -rf "$WORK"' EXIT

# model = the end-of-run model the training job recorded
want=$(sed -n 's/^final_model_sha256: //p' "$RUN/record.md")
have=$(cat "$MODEL_PATH"/model*.safetensors | sha256sum | cut -d' ' -f1)
[ -n "$want" ] && [ "$want" = "$have" ] || { echo "model sha256 $have, record.md says '$want'" >&2; exit 1; }
{ echo "model: $MODEL_PATH"; echo "final_model_sha256: $have"; grep -E "status|num_epochs|best_checkpoint" "$RUN/record.md" || true
  echo "vLLM dtype: $DTYPE (fp32 weights downcast)"; cat "$REPO/REVISION"; } > "$OUT/${NAME}.model.txt"

# input of this task
grep -E "^[0-9a-f]{64}  $NAME.jsonl\$" "$IN/sha256.txt" > "$OUT/${NAME}.inputs_sha256.txt"
cp "$IN/$NAME.jsonl" "$WORK/in/"
(cd "$WORK/in" && sha256sum -c "$OUT/${NAME}.inputs_sha256.txt")

# run_azi.py of the image, patched as in infer_only.slurm
FWD_OLD="\"Determine the single cell's expression changes\" in prompt\[:64\]"
FWD_NEW='prompt.startswith("Create a ranked list of genes in decreasing order of expression")'
apptainer exec "$SIF" cat /opt/c2s/run_azi.py > "$WORK/code/run_azi.py"
sed -i "s/dtype=\"bfloat16\"/dtype=\"$DTYPE\"/" "$WORK/code/run_azi.py"
sed -i "s/$FWD_OLD/$FWD_NEW/; s/^MAX_NEW_TOKENS_FWD = 4096\$/MAX_NEW_TOKENS_FWD = 7936/" "$WORK/code/run_azi.py"
[ "$(grep -c "dtype=\"$DTYPE\"" "$WORK/code/run_azi.py")" -eq 1 ]
[ "$(grep -cF "$FWD_NEW" "$WORK/code/run_azi.py")" -eq 1 ] || { echo "run_azi.py: forward prompts not recognised" >&2; exit 1; }
[ "$(grep -c '^MAX_NEW_TOKENS_FWD = 7936$' "$WORK/code/run_azi.py")" -eq 1 ] || { echo "run_azi.py: forward limit not set" >&2; exit 1; }
apptainer exec "$SIF" cat /opt/c2s/run_azi.py | diff - "$WORK/code/run_azi.py" > "$OUT/${NAME}.run_azi.diff" || true
cp "$0" "$OUT/${NAME}.infer_rep300.slurm"

echo "$(date '+%F %T') $NAME" >&2
apptainer run --app infer --nv --cleanenv --no-home --writable-tmpfs \
    --bind "$MODEL_PATH:/model:ro" --bind "$WORK/in:/in:ro" --bind "$WORK/code:/code:ro" \
    --bind "$OUT:/out" --bind "$WORK/hf:/hf" --bind "$WORK/triton:/triton" \
    --env HF_HOME=/hf,TRITON_CACHE_DIR=/triton,VLLM_CACHE_ROOT=/triton/vllm,HOME=/tmp,PYTHONFAULTHANDLER=1,VLLM_NO_USAGE_STATS=1,MODEL_DIR=/model,TEMPERATURE=1.0 \
    "$SIF" /code/run_azi.py "/in/$NAME.jsonl" "/out/${NAME}_results.jsonl" > "$OUT/$NAME.log" 2>&1

nin=$(wc -l < "$WORK/in/$NAME.jsonl"); nout=$(wc -l < "$RES")
[ "$nin" -eq "$nout" ] || { echo "$NAME: $nout records for $nin prompts" >&2; exit 1; }
[ "$(grep -c '"direction": "fwd"' "$RES")" -eq "$nin" ] || { echo "$NAME: not every prompt ran as forward" >&2; exit 1; }
grep -q "'dtype': '$DTYPE'" "$OUT/$NAME.log" || { echo "$NAME: vLLM did not run in $DTYPE" >&2; exit 1; }
if grep -q "Upcasting" "$OUT/$NAME.log"; then echo "$NAME: weights on disk are not fp32" >&2; exit 1; fi
echo "$(date '+%F %T') done -> $RES ($(grep -c '"finish_reason": "length"' "$RES" || true) cut at the limit)" >&2
