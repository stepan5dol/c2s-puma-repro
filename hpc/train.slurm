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
# Needs ~/c2s/c2s.env with HF_TOKEN=<write token> (checkpoint uploads only).
# Optional: TG_TOKEN and TG_CHAT there plus ~/c2s/tg_monitor.py for Telegram updates.

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
ENV_FILE=$BASE/c2s.env
grep -q '^HF_TOKEN=hf_' "$ENV_FILE" || { echo "no HF_TOKEN in $ENV_FILE" >&2; exit 1; }
chmod 600 "$ENV_FILE"
STATUS=failed
UPID=
TGPID=

TG_TOKEN=$(sed -n 's/^TG_TOKEN=//p' "$ENV_FILE")
TG_CHAT=$(sed -n 's/^TG_CHAT=//p' "$ENV_FILE")
tg() {
    [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ] || return 0
    curl -fsS -m 30 "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
        --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text=$1" > /dev/null || true
}
# run_hf binds it either way; empty means no per-step monitor.
cp "$BASE/tg_monitor.py" "$WORK/tg_monitor.py" 2>/dev/null || : > "$WORK/tg_monitor.py"

# Mirrors checkpoints to a public HF repo while training runs, so a dead node
# loses at most one save interval. "sync" uploads every checkpoint except the
# newest (it may still be written) and drops the ones the Trainer rotated away
# (save_total_limit); "final" uploads the rest and the run dir.
cat > "$WORK/hf_sync.py" <<'EOF'
import os, re, sys
from huggingface_hub import HfApi

mode, run_name = sys.argv[1], sys.argv[2]
repo, prefix = "dolzhenkosv/c2s-puma-runs", f"runs/{run_name}"
out, run_dir = "/workspace/axolotl/mainwork/sft_pythia1b_short_full", "/run_out"
done_file = f"{run_dir}/hf_uploaded.txt"
api = HfApi()
api.create_repo(repo, private=False, exist_ok=True)

done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()
local = sorted((d for d in os.listdir(out) if re.fullmatch(r"checkpoint-\d+", d))
               if os.path.isdir(out) else [], key=lambda d: int(d.split("-")[1]))
ready = [d for d in (local[:-1] if mode == "sync" else local)
         if os.path.exists(f"{out}/{d}/trainer_state.json")]

for d in ready:
    if d not in done:
        api.upload_folder(folder_path=f"{out}/{d}", path_in_repo=f"{prefix}/{d}", repo_id=repo,
                          commit_message=f"{run_name}: {d}")
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
        run_hf python /hf_sync.py sync "${STAMP}_${SLUG}" >> "$RUN/hf_sync.log" 2>&1 || true
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
    run_hf python /hf_sync.py final "${STAMP}_${SLUG}" >> "$RUN/hf_sync.log" 2>&1 \
        || echo "final HF upload failed, see hf_sync.log" >> "$RUN/record.md"
    MSG="$([ "$STATUS" = completed ] && echo '✅' || echo '❌') ${STAMP}_${SLUG}: $STATUS"
    MSG+=", $(( ($(date +%s) - T0) / 60 )) min, job $SLURM_JOB_ID"
    [ -n "$BEST" ] && MSG+=$'\n'"best: $(basename "$BEST")"
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
# Sent 15 min before the time limit (--signal), so finish() has time to copy.
PID=
trap 'STATUS=timeout; [ -n "$PID" ] && kill "$PID" 2>/dev/null; exit 143' TERM

# Image: download once, re-download when the published hash changes.
curl -fsSL "$SIF_URL/c2s-train.sif.sha256" -o "$RUN/image.sha256"
if ! (cd "$BASE/images" && sha256sum -c --quiet "$RUN/image.sha256" 2>/dev/null); then
    curl -fL "$SIF_URL/c2s-train.sif" -o "$SIF.part"
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

# Same container, plus the token and the run dir; only the uploader gets the token.
run_hf() {
    apptainer exec --cleanenv --no-home --writable-tmpfs \
        --bind "$WORK/mainwork:/workspace/axolotl/mainwork" \
        --bind "$WORK/hf_sync.py:/hf_sync.py" --bind "$WORK/tg_monitor.py:/tg_monitor.py" \
        --bind "$RUN:/run_out" \
        --env-file "$ENV_FILE" --env HF_HOME=/tmp/hf_upload,HOME=/tmp \
        "$SIF" "$@"
}

run cat /opt/c2s/build_info.txt > "$RUN/build_info.txt"
run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" \
    | tee "$RUN/torch_check.txt"
tg "▶️ ${STAMP}_${SLUG}: job $SLURM_JOB_ID on $(hostname)
$(cat "$RUN/torch_check.txt")"

# Logs go straight to the run dir so they can be followed live from a login node.
# Background + wait: bash only handles the time-limit TERM between commands.
run axolotl preprocess mainwork/sft_pythia1b.yaml > "$RUN/preprocess.log" 2>&1 &
PID=$!; wait $PID
uploader &
UPID=$!
if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ] && [ -s "$WORK/tg_monitor.py" ]; then
    run_hf python /tg_monitor.py /workspace/axolotl/mainwork/sft_pythia1b_short_full "c2s Puma MIG" \
        > "$RUN/tg_monitor.log" 2>&1 &
    TGPID=$!
fi
run axolotl train mainwork/sft_pythia1b.yaml > "$RUN/train.log" 2>&1 &
PID=$!; wait $PID
STATUS=completed
