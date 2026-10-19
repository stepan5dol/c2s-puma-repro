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

set -euo pipefail

MODEL=${MODEL:-pythia1b}
INFER=${INFER:-0}
INFER_TEMPS=${INFER_TEMPS:-0.8 1.0 1.2}
case $MODEL in
    pythia1b)
        SLUG=puma-a100mig-repro
        HF_SRC=dolzhenkosv/c2s-pythia-1b-azithro-transfer
        REV=040f129aafc483852e54a7ccf078f7cc1a2286f5
        INPUTS=env_snapshot/inputs_sha256.txt
        CFG=sft_pythia1b.yaml
        OUTNAME=sft_pythia1b_short_full
        BASE_ID=vandijklab--C2S-Scale-Pythia-1b-pt
        ;;
    gemma2b)
        SLUG=gemma2b-puma-a100mig
        HF_SRC=dolzhenkosv/c2s-gemma2-2b-azithro-transfer
        REV=e28b53e18cdb6236aff5804a46654e4b83abaa60
        INPUTS=inputs_sha256.txt
        CFG=sft_gemma2_2b.yaml
        OUTNAME=sft_gemma2_2b_short_full
        BASE_ID=vandijklab--C2S-Scale-Gemma-2-2B
        ;;
    *)
        echo "unknown MODEL=$MODEL (pythia1b or gemma2b)" >&2
        exit 1
        ;;
esac

STAMP=$(date +%Y-%m-%d_%H%M)
BASE=$HOME/c2s
RUN=$BASE/results/${STAMP}_${SLUG}
SRC=https://huggingface.co/$HF_SRC/resolve/$REV
SIF_URL=https://huggingface.co/dolzhenkosv/c2s-puma-sif/resolve/main
SIF=$BASE/images/c2s-train.sif
WORK=/tmp/c2s_$SLURM_JOB_ID
T0=$(date +%s)

if [ -e "$RUN" ]; then echo "run dir exists: $RUN" >&2; exit 1; fi
mkdir -p "$RUN" "$BASE/images"
ln -sfn "$RUN" "$BASE/results/последний"
mkdir -p "$WORK"/{mainwork/dataset,last_run_prepared,hf,triton}
ENV_FILE=$BASE/c2s.env
grep -q '^HF_TOKEN=hf_' "$ENV_FILE" || { echo "no HF_TOKEN in $ENV_FILE" >&2; exit 1; }
chmod 600 "$ENV_FILE"
HF_AUTH=$WORK/hf_auth_header
(umask 077 && sed -n 's/^HF_TOKEN=/Authorization: Bearer /p' "$ENV_FILE" > "$HF_AUTH")
STATUS=failed
INFER_STATUS=off
MB=
UPID=
TGPID=

TG_TOKEN=$(sed -n 's/^TG_TOKEN=//p' "$ENV_FILE")
TG_CHAT=$(sed -n 's/^TG_CHAT=//p' "$ENV_FILE")
tg() {
    [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ] || return 0
    curl -fsS -m 30 "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
        --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text=$1" > /dev/null || true
}
cp "$BASE/tg_monitor.py" "$WORK/tg_monitor.py" 2>/dev/null || : > "$WORK/tg_monitor.py"

cat > "$WORK/hf_sync.py" <<'EOF'
import os, re, sys
from huggingface_hub import HfApi

mode, run_name, outname = sys.argv[1], sys.argv[2], sys.argv[3]
repo, prefix = "dolzhenkosv/c2s-puma-runs", f"runs/{run_name}"
out, run_dir = f"/workspace/axolotl/mainwork/{outname}", "/run_out"
done_file = f"{run_dir}/hf_uploaded.txt"
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)

done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()
local = sorted((d for d in os.listdir(out) if re.fullmatch(r"checkpoint-\d+", d))
               if os.path.isdir(out) else [], key=lambda d: int(d.split("-")[1]))
ready = [d for d in (local[:-1] if mode == "sync" else local)
         if os.path.exists(f"{out}/{d}/trainer_state.json")]

for d in ready:
    if d not in done:
        api.upload_folder(folder_path=f"{out}/{d}", path_in_repo=f"{prefix}/{d}", repo_id=repo,
                          ignore_patterns=["optimizer.pt"], commit_message=f"{run_name}: {d}")
        done.add(d)
        open(done_file, "a").write(d + "\n")
        print("uploaded", d, flush=True)
if mode == "sync":
    for d in sorted(done - set(local)):
        api.delete_folder(path_in_repo=f"{prefix}/{d}", repo_id=repo,
                          commit_message=f"{run_name}: rotate out {d}")
        done.discard(d)
        open(done_file, "w").write("".join(x + "\n" for x in sorted(done)))
        print("rotated out", d, flush=True)
else:
    api.upload_folder(folder_path=run_dir, path_in_repo=f"{prefix}/result", repo_id=repo,
                      commit_message=f"{run_name}: result")
    print("uploaded result", flush=True)
EOF

uploader() {
    while sleep 300; do
        run_hf python /hf_sync.py sync "${STAMP}_${SLUG}" "$OUTNAME" >> "$RUN/hf_sync.log" 2>&1 || true
    done
}

finish() {
    set +e
    if [ -n "$UPID" ]; then
        pkill -P "$UPID" 2>/dev/null; kill "$UPID" 2>/dev/null; wait "$UPID" 2>/dev/null
    fi
    if [ -n "$TGPID" ]; then
        kill "$TGPID" 2>/dev/null; wait "$TGPID" 2>/dev/null
    fi
    OUT=$WORK/mainwork/$OUTNAME
    BEST=
    BASE_REV=$(cat "$WORK/hf/hub/models--$BASE_ID/refs/main" 2>/dev/null)
    if [ -d "$OUT" ]; then
        BEST=$(run python /opt/c2s/finalize_best.py "mainwork/$OUTNAME" \
            2>> "$RUN/finalize_best.log")
        if [ -z "$BEST" ] && [ "$STATUS" = completed ]; then
            STATUS=failed
            echo "finalize_best.py failed" >> "$RUN/finalize_best.log"
        fi
        rsync -a --exclude 'checkpoint-*' "$OUT/" "$RUN/$OUTNAME/"
        if [ -n "$BEST" ]; then
            EXCL=(--exclude optimizer.pt)
            if [ -e "$OUT/model.safetensors" ] || [ -e "$OUT/model.safetensors.index.json" ]; then
                EXCL+=(--exclude 'model*.safetensors*')
            fi
            rsync -a "${EXCL[@]}" "$OUT/$BEST" "$RUN/$OUTNAME/"
        fi
    fi
    {
        echo "# ${STAMP}_${SLUG}"
        echo
        echo "status: $STATUS"
        echo "model: $MODEL"
        echo "slurm_job: $SLURM_JOB_ID on $(hostname)"
        echo "duration_s: $(( $(date +%s) - T0 ))"
        echo "source: $HF_SRC@$REV"
        echo "image_sha256: $(cut -d' ' -f1 "$RUN/image.sha256" 2>/dev/null)"
        echo "config_sha256: $(grep " $CFG\$" "$RUN/inputs_sha256.txt" 2>/dev/null | cut -d' ' -f1)"
        echo "micro_batch_size: ${MB}"
        echo "base_model_revision: ${BASE_REV:-}"
        echo "best_checkpoint: ${BEST}"
        echo "final_model_sha256: $(cat "$RUN/$OUTNAME"/model*.safetensors 2>/dev/null | sha256sum | cut -d' ' -f1)"
        echo "inference: $INFER_STATUS ($INFER_TEMPS)"
        echo "tg_monitor_sha256: $(sha256sum "$WORK/tg_monitor.py" 2>/dev/null | cut -d' ' -f1)"
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
    run_hf python /hf_sync.py final "${STAMP}_${SLUG}" "$OUTNAME" >> "$RUN/hf_sync.log" 2>&1 \
        || echo "final HF upload failed, see hf_sync.log" >> "$RUN/record.md"
    MSG="$([ "$STATUS" = completed ] && echo '✅' || echo '❌') ${STAMP}_${SLUG}: $STATUS"
    MSG+=", $(( ($(date +%s) - T0) / 60 )) min, job $SLURM_JOB_ID"
    [ -n "$BEST" ] && MSG+=$'\n'"best: $BEST"
    [ "$INFER_STATUS" != off ] && MSG+=$'\n'"inference: $INFER_STATUS"
    if [ "$STATUS" != completed ]; then
        LOG=$RUN/train.log
        [ -s "$LOG" ] || LOG=$RUN/preprocess.log
        [ -s "$LOG" ] || LOG=$SLURM_SUBMIT_DIR/c2s-sft_$SLURM_JOB_ID.out
        MSG+=$'\n\n'"$(tail -c 3000 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -n 15)"
    fi
    tg "$MSG"
    rm -rf "$WORK"
}
trap finish EXIT
PID=
trap '[ "$STATUS" = completed ] || STATUS=timeout; [ -n "$PID" ] && kill "$PID" 2>/dev/null; exit 143' TERM

curl -fsSL -H @"$HF_AUTH" "$SIF_URL/c2s-train.sif.sha256" -o "$RUN/image.sha256"
if ! (cd "$BASE/images" && sha256sum -c --quiet "$RUN/image.sha256" 2>/dev/null); then
    curl -fL -H @"$HF_AUTH" "$SIF_URL/c2s-train.sif" -o "$SIF.part"
    mv "$SIF.part" "$SIF"
    (cd "$BASE/images" && sha256sum -c "$RUN/image.sha256")
fi

FILES="$CFG dataset/train_segments.jsonl dataset/valid_segments.jsonl"
[ "$INFER" = 1 ] && FILES+=" dataset/test_segments.jsonl"
curl -fsSL -H @"$HF_AUTH" "$SRC/$INPUTS" -o "$RUN/inputs_sha256_source.txt"
for f in $FILES; do
    curl -fsSL -H @"$HF_AUTH" "$SRC/$f" -o "$WORK/mainwork/$f"
    grep -E "^[0-9a-f]{64}  $f\$" "$RUN/inputs_sha256_source.txt" >> "$RUN/inputs_sha256.txt"
done
[ "$(wc -l < "$RUN/inputs_sha256.txt")" -eq "$(echo $FILES | wc -w)" ]
(cd "$WORK/mainwork" && sha256sum -c "$RUN/inputs_sha256.txt")
cp "$WORK/mainwork/$CFG" "$RUN/"

nvidia-smi > "$RUN/nvidia-smi.txt"

run() {
    apptainer exec --nv --cleanenv --no-home --writable-tmpfs \
        --bind "$WORK/mainwork:/workspace/axolotl/mainwork" \
        --bind "$WORK/last_run_prepared:/workspace/axolotl/last_run_prepared" \
        --bind "$WORK/hf:/hf" --bind "$WORK/triton:/triton" \
        --env HF_HOME=/hf,TRITON_CACHE_DIR=/triton,HOME=/tmp \
        --pwd /workspace/axolotl "$SIF" "$@"
}

run_hf() {
    apptainer exec --cleanenv --no-home --writable-tmpfs \
        --bind "$WORK/mainwork:/workspace/axolotl/mainwork" \
        --bind "$WORK/hf_sync.py:/hf_sync.py" --bind "$WORK/tg_monitor.py:/tg_monitor.py" \
        --bind "$RUN:/run_out" \
        --env-file "$ENV_FILE" --env HF_HOME=/tmp/hf_upload,HOME=/tmp \
        "$SIF" "$@"
}

run_infer() {
    apptainer run --app infer --nv --cleanenv --no-home --writable-tmpfs \
        --bind "$WORK/mainwork:/workspace/axolotl/mainwork" \
        --bind "$WORK/hf:/hf" --bind "$WORK/triton:/triton" --bind "$RUN:/run_out" \
        --env HF_HOME=/hf,TRITON_CACHE_DIR=/triton,VLLM_CACHE_ROOT=/triton/vllm,HOME=/tmp,"$1" \
        "$SIF" "${@:2}"
}

run cat /opt/c2s/build_info.txt > "$RUN/build_info.txt"
run test -f /opt/c2s/finalize_best.py
[ "$INFER" = 1 ] && run_infer TEMPERATURE=1 -c 'import vllm'
run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" \
    | tee "$RUN/torch_check.txt"
tg "▶️ ${STAMP}_${SLUG}: job $SLURM_JOB_ID on $(hostname)
$(cat "$RUN/torch_check.txt")"

run axolotl preprocess "mainwork/$CFG" > "$RUN/preprocess.log" 2>&1 &
PID=$!; wait $PID
uploader &
UPID=$!
if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ] && [ -s "$WORK/tg_monitor.py" ]; then
    run_hf python /tg_monitor.py "/workspace/axolotl/mainwork/$OUTNAME" "c2s Puma MIG $MODEL" \
        > "$RUN/tg_monitor.log" 2>&1 &
    TGPID=$!
fi

MB=$(sed -n 's/^micro_batch_size: *\([0-9]*\).*/\1/p' "$WORK/mainwork/$CFG")
EFFECTIVE=$(( MB * $(sed -n 's/^gradient_accumulation_steps: *\([0-9]*\).*/\1/p' "$WORK/mainwork/$CFG") ))
while :; do
    run axolotl train "mainwork/$CFG" > "$RUN/train.log" 2>&1 &
    PID=$!
    if wait $PID; then
        STATUS=completed
        break
    fi
    if [ "$MB" -le 1 ] || ! grep -q -E 'OutOfMemoryError|CUDA out of memory' "$RUN/train.log" \
        || compgen -G "$WORK/mainwork/$OUTNAME/checkpoint-*" > /dev/null; then
        exit 1
    fi
    mv "$RUN/train.log" "$RUN/train_oom_micro$MB.log"
    rm -rf "$WORK/mainwork/$OUTNAME"
    tg "⚠️ ${STAMP}_${SLUG}: OOM at micro_batch_size $MB, retrying with $(( MB / 2 ))"
    MB=$(( MB / 2 ))
    sed -i -e "s/^micro_batch_size: .*/micro_batch_size: $MB/" \
        -e "s/^gradient_accumulation_steps: .*/gradient_accumulation_steps: $(( EFFECTIVE / MB ))/" \
        "$WORK/mainwork/$CFG"
    cp "$WORK/mainwork/$CFG" "$RUN/${CFG%.yaml}_micro$MB.yaml"
done

if [ "$INFER" = 1 ]; then
    INFER_STATUS=failed
    run python /opt/c2s/finalize_best.py "mainwork/$OUTNAME" > /dev/null 2>> "$RUN/finalize_best.log"
    mkdir -p "$RUN/inference"
    for t in $INFER_TEMPS; do
        tag=t$(echo "$t" | tr -d .)
        run_infer "MODEL_DIR=/workspace/axolotl/mainwork/$OUTNAME,TEST_PATH=/workspace/axolotl/mainwork/dataset/test_segments.jsonl,OUT_PATH=/run_out/inference/test_$tag.jsonl,TEMPERATURE=$t" \
            /opt/c2s/run_test.py > "$RUN/inference/test_$tag.log" 2>&1 &
        PID=$!; wait $PID
    done
    INFER_STATUS=completed
fi
