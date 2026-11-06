#!/bin/bash
#SBATCH --job-name=c2s-infer
#SBATCH --account=vkalin
#SBATCH --partition=gpu_standard
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                 # 4 x 5 GB = 20 GB RAM; 1B model in fp32 is 4 GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/xdisk/vkalin/dolzhenko/c2s/logs/%x_%j.out

# Inference only, on the final model of an existing training run (train.slurm of c2s-puma-repro),
# with the image's own run_test.py / run_azi.py; the only change is the vLLM dtype.
# usage: RUN=/xdisk/vkalin/dolzhenko/c2s/results/<run folder> [SETS="test_t10 ..."] [DTYPE=float32] sbatch infer_only.slurm
# MODEL_PATH: the model folder (default $RUN/$OUTNAME); train.slurm with INFER_FP32=1 runs this script at the end of
#       training, inside its own job, on the end-of-run model in its work folder.
# ALLOW_UPCAST=1: a bf16-trained run (e.g. a reference) inferred in fp32 for a same-dtype comparison.
# SETS (default: the full A100 set; the grid asks for SETS=test_t10 only): test_t08 test_t10 test_t12 (test set at T 0.8/1.0/1.2),
#       inference_azi inference_azi_concise
#       inference_azi_ratmatch (T 1.0). Outputs: $RUN/inference_$DTYPE/, same names as train.slurm.

set -euo pipefail

RUN=${RUN:?set RUN to the training run folder}
SETS=${SETS:-test_t10 test_t08 test_t12 inference_azi inference_azi_concise inference_azi_ratmatch}
DTYPE=${DTYPE:-float32}
OUTNAME=sft_pythia1b_short_full
DATA=/xdisk/vkalin/dolzhenko/c2s
SIF=$DATA/images/c2s-train.sif
SRC=https://huggingface.co/dolzhenkosv/c2s-pythia-1b-azithro-transfer/resolve/040f129aafc483852e54a7ccf078f7cc1a2286f5
AZI_SRC=https://huggingface.co/dolzhenkosv/c2s-gemma2-2b-azithro-transfer/resolve/be9e317b2791e2518308dfdc211342815ced72de
OUT=$RUN/inference_$DTYPE
MODEL_PATH=${MODEL_PATH:-$RUN/$OUTNAME}
WORK=/tmp/c2s_infer_$SLURM_JOB_ID

[ -f "$MODEL_PATH/config.json" ] || { echo "no model in $MODEL_PATH" >&2; exit 1; }
mkdir "$OUT"  # fails if it exists: never overwrite earlier outputs
mkdir -p "$WORK"/{in/dataset,in/inference,code,hf,triton}
trap 'rm -rf "$WORK"' EXIT
HF_AUTH=$WORK/hf_auth_header
(umask 077 && sed -n 's/^HF_TOKEN=/Authorization: Bearer /p' "$HOME/c2s/c2s.env" > "$HF_AUTH")

# inputs: the same files and checksums as the training run recorded
# data_src/dataset or data_src/inference on xdisk = the local dataset is meant: without its checksum list stop here
# instead of inferring on the Hugging Face inputs (2026-11-05: the list was missing, the grid ran on the old test set)
if [ -d "$DATA/data_src/dataset" ] || [ -d "$DATA/data_src/inference" ]; then
    [ -s "$DATA/data_src/inputs_sha256.txt" ] \
        || { echo "$DATA/data_src/dataset exists but $DATA/data_src/inputs_sha256.txt does not: not falling back to Hugging Face (run push.sh)" >&2; exit 1; }
fi
if [ -s "$DATA/data_src/inputs_sha256.txt" ]; then echo "data_src $DATA/data_src"; else echo "huggingface $SRC $AZI_SRC"; fi > "$OUT/inputs_source.txt"
for s in $SETS; do
    case $s in
        test_t*|valid_t*) f=dataset/${s%_t*}_segments.jsonl; url=$SRC/$f ;;
        inference_azi*) f=inference/$s.jsonl; url=$AZI_SRC/$f ;;
        *) echo "unknown set $s" >&2; exit 1 ;;
    esac
    if [ -s "$DATA/data_src/inputs_sha256.txt" ]; then
        [ -s "$WORK/in/$f" ] || cp "$DATA/data_src/$f" "$WORK/in/$f"
        list=$DATA/data_src/inputs_sha256.txt
    else
        [ -s "$WORK/in/$f" ] || curl -fsSL -H @"$HF_AUTH" "$url" -o "$WORK/in/$f"
        # checksums from the sources, as train.slurm does: a run made with INFER=0 has no test line in its own file
        case $s in
            inference_azi*) list=$WORK/azi_sha256.txt; [ -s "$list" ] || curl -fsSL -H @"$HF_AUTH" "$AZI_SRC/inference/sha256.txt" -o "$list" ;;
            *) list=$WORK/src_sha256.txt; [ -s "$list" ] || curl -fsSL -H @"$HF_AUTH" "$SRC/env_snapshot/inputs_sha256.txt" -o "$list" ;;
        esac
    fi
    n=$(grep -c -E "^[0-9a-f]{64}  $f\$" "$list")
    [ "$n" -ge 1 ] || { echo "no checksum for $f in $list" >&2; exit 1; }
    grep -E "^[0-9a-f]{64}  $f\$" "$list" | sort -u
done | tee "$OUT/inputs_sha256.txt" | (cd "$WORK/in" && sha256sum -c)

# the image's inference scripts, dtype line changed and checked
for p in run_test run_azi; do
    apptainer exec "$SIF" cat /opt/c2s/$p.py > "$WORK/code/$p.py"
    sed -i "s/dtype=\"bfloat16\"/dtype=\"$DTYPE\"/" "$WORK/code/$p.py"
    [ "$(grep -c "dtype=\"$DTYPE\"" "$WORK/code/$p.py")" -eq 1 ]
    apptainer exec "$SIF" cat /opt/c2s/$p.py | diff - "$WORK/code/$p.py" > "$OUT/$p.diff" || true
done
cp "$0" "$OUT/infer_only.slurm"
if [ "$MODEL_PATH" = "$RUN/$OUTNAME" ]; then
    grep -E "status|num_epochs|best_checkpoint|final_model_sha256" "$RUN/record.md" > "$OUT/model.txt" || true
else
    echo "model: $MODEL_PATH, end of run, inside training job ${SLURM_JOB_ID:-}" > "$OUT/model.txt"
fi

run_infer() {
    apptainer run --app infer --nv --cleanenv --no-home --writable-tmpfs \
        --bind "$MODEL_PATH:/model:ro" --bind "$WORK/in:/in:ro" --bind "$WORK/code:/code:ro" \
        --bind "$OUT:/out" --bind "$WORK/hf:/hf" --bind "$WORK/triton:/triton" \
        --env HF_HOME=/hf,TRITON_CACHE_DIR=/triton,VLLM_CACHE_ROOT=/triton/vllm,HOME=/tmp,PYTHONFAULTHANDLER=1,VLLM_NO_USAGE_STATS=1,MODEL_DIR=/model,"$1" \
        "$SIF" "${@:2}"
}

for s in $SETS; do
    echo "$(date '+%F %T') $s" >&2
    case $s in
        test_t*|valid_t*)
            t=$(echo "${s##*_t}" | sed 's/^\(.\)/\1./')  # t08 -> 0.8, t10 -> 1.0
            run_infer "TEST_PATH=/in/dataset/${s%_t*}_segments.jsonl,OUT_PATH=/out/$s.jsonl,TEMPERATURE=$t" \
                /code/run_test.py > "$OUT/$s.log" 2>&1
            # one answer per line of the test set (a short file passed the old non-empty check)
            nin=$(wc -l < "$WORK/in/dataset/${s%_t*}_segments.jsonl"); nout=$(wc -l < "$OUT/$s.jsonl")
            [ "$nin" -eq "$nout" ] || { echo "$s: $nout records for $nin lines of the test set" >&2; exit 1; } ;;
        inference_azi*)
            run_infer "TEMPERATURE=1.0" /code/run_azi.py "/in/inference/$s.jsonl" "/out/${s}_results.jsonl" \
                > "$OUT/$s.log" 2>&1 ;;
    esac
    grep -q "'dtype': '$DTYPE'" "$OUT/$s.log" || { echo "$s: vLLM did not run in $DTYPE" >&2; exit 1; }
    if grep -q "Downcasting" "$OUT/$s.log"; then echo "$s: weights were downcast" >&2; exit 1; fi
    if [ "${ALLOW_UPCAST:-0}" != 1 ] && grep -q "Upcasting" "$OUT/$s.log"; then
        echo "$s: weights on disk are not fp32 (ALLOW_UPCAST=1 for a bf16-trained run)" >&2; exit 1
    fi
done
echo "$(date '+%F %T') done -> $OUT" >&2
