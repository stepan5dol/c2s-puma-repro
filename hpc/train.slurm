#!/bin/bash
#SBATCH --job-name=c2s-sft
#SBATCH --account=vkalin
#SBATCH --partition=gpu_standard
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=%x_%j.out
#SBATCH --signal=B:TERM@900

# Same training as dolzhenkosv/c2s-pythia-1b-azithro-transfer@040f129,
# unmodified sft_pythia1b.yaml, on a Puma A100 MIG slice.
# Needs ~/.cache/huggingface/token (read access to the private image repo).

set -euo pipefail

STAMP=$(date +%Y-%m-%d_%H%M)
SLUG=puma-a100mig-repro
BASE=$HOME/c2s
RUN=$BASE/results/${STAMP}_${SLUG}
REV=040f129aafc483852e54a7ccf078f7cc1a2286f5
SRC=https://huggingface.co/dolzhenkosv/c2s-pythia-1b-azithro-transfer/resolve/$REV
SIF_URL=https://huggingface.co/dolzhenkosv/c2s-puma-sif/resolve/main
SIF=$BASE/images/c2s-train.sif
WORK=/tmp/c2s_$SLURM_JOB_ID
T0=$(date +%s)

if [ -e "$RUN" ]; then echo "run dir exists: $RUN" >&2; exit 1; fi
mkdir -p "$RUN" "$BASE/images"
ln -sfn "$RUN" "$BASE/results/последний"
mkdir -p "$WORK"/{mainwork/dataset,last_run_prepared,hf,triton}
HF_TOKEN=$(cat ~/.cache/huggingface/token)
STATUS=failed

finish() {
    set +e
    # Keep everything except intermediate checkpoints; the best one is kept.
    OUT=$WORK/mainwork/sft_pythia1b_short_full
    BEST=
    BASE_REV=$(cat "$WORK"/hf/hub/models--vandijklab--C2S-Scale-Pythia-1b-pt/refs/main 2>/dev/null)
    if [ -d "$OUT" ]; then
        BEST=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('best_model_checkpoint') or '')" \
            "$OUT/trainer_state.json" 2>/dev/null || true)
        rsync -a --exclude 'checkpoint-*' "$OUT/" "$RUN/sft_pythia1b_short_full/"
        if [ -n "$BEST" ]; then
            rsync -a "$OUT/$(basename "$BEST")" "$RUN/sft_pythia1b_short_full/"
        fi
    fi
    {
        echo "# ${STAMP}_${SLUG}"
        echo
        echo "status: $STATUS"
        echo "slurm_job: $SLURM_JOB_ID on $(hostname)"
        echo "duration_s: $(( $(date +%s) - T0 ))"
        echo "source_revision: $REV"
        echo "image_sha256: $(cut -d' ' -f1 "$RUN/image.sha256" 2>/dev/null)"
        echo "config_sha256: $(sha256sum "$WORK/mainwork/sft_pythia1b.yaml" 2>/dev/null | cut -d' ' -f1)"
        echo "base_model_revision: ${BASE_REV:-}"
        echo "best_checkpoint: ${BEST}"
        echo "slurm_log: $SLURM_SUBMIT_DIR/c2s-sft_$SLURM_JOB_ID.out"
        echo
        echo '## image build_info'
        cat "$RUN/build_info.txt" 2>/dev/null
        echo
        echo '## inputs sha256'
        cat "$RUN/inputs_sha256.txt" 2>/dev/null
        echo
        echo '## author notes'
    } > "$RUN/record.md"
    rm -rf "$WORK"
}
trap finish EXIT
# Sent 15 min before the time limit (--signal), so finish() has time to copy.
PID=
trap 'STATUS=timeout; [ -n "$PID" ] && kill "$PID" 2>/dev/null; exit 143' TERM

# Image: download once, re-download when the published hash changes.
curl -fsSL -H "Authorization: Bearer $HF_TOKEN" "$SIF_URL/c2s-train.sif.sha256" -o "$RUN/image.sha256"
if ! (cd "$BASE/images" && sha256sum -c --quiet "$RUN/image.sha256" 2>/dev/null); then
    curl -fL -H "Authorization: Bearer $HF_TOKEN" "$SIF_URL/c2s-train.sif" -o "$SIF.part"
    mv "$SIF.part" "$SIF"
    (cd "$BASE/images" && sha256sum -c "$RUN/image.sha256")
fi

# Original config and data from the pinned revision, checked against the run's own hashes.
curl -fsSL "$SRC/env_snapshot/inputs_sha256.txt" -o "$RUN/inputs_sha256_pod.txt"
curl -fsSL "$SRC/sft_pythia1b.yaml" -o "$WORK/mainwork/sft_pythia1b.yaml"
for f in train_segments.jsonl valid_segments.jsonl; do
    curl -fsSL "$SRC/dataset/$f" -o "$WORK/mainwork/dataset/$f"
done
(cd "$WORK/mainwork" && grep -E ' (sft_pythia1b.yaml|dataset/(train|valid)_segments.jsonl)$' \
    "$RUN/inputs_sha256_pod.txt" | tee "$RUN/inputs_sha256.txt" | sha256sum -c)
cp "$WORK/mainwork/sft_pythia1b.yaml" "$RUN/"

nvidia-smi > "$RUN/nvidia-smi.txt"

run() {
    apptainer exec --nv --cleanenv --no-home --writable-tmpfs \
        --bind "$WORK/mainwork:/workspace/axolotl/mainwork" \
        --bind "$WORK/last_run_prepared:/workspace/axolotl/last_run_prepared" \
        --bind "$WORK/hf:/hf" --bind "$WORK/triton:/triton" \
        --env HF_HOME=/hf,TRITON_CACHE_DIR=/triton,HOME=/tmp \
        --pwd /workspace/axolotl "$SIF" "$@"
}

run cat /opt/c2s/build_info.txt > "$RUN/build_info.txt"
run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" \
    | tee "$RUN/torch_check.txt"

# Logs go straight to the run dir so they can be followed live from a login node.
# Background + wait: bash only handles the time-limit TERM between commands.
run axolotl preprocess mainwork/sft_pythia1b.yaml > "$RUN/preprocess.log" 2>&1 &
PID=$!; wait $PID
run axolotl train mainwork/sft_pythia1b.yaml > "$RUN/train.log" 2>&1 &
PID=$!; wait $PID
STATUS=completed
