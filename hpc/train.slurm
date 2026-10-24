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
#SBATCH --output=/xdisk/vkalin/dolzhenko/c2s/logs/%x_%j.out
#SBATCH --signal=B:TERM@900

set -euo pipefail

MODEL=${MODEL:-pythia1b}
INFER=${INFER:-0}
INFER_TEMPS=${INFER_TEMPS:-1.0 0.8 1.2}
AZI_TEMP=${AZI_TEMP:-1.0}
AZI_SETS="inference_azi inference_azi_concise inference_azi_ratmatch"
AZI_SRC=https://huggingface.co/dolzhenkosv/c2s-gemma2-2b-azithro-transfer/resolve/be9e317b2791e2518308dfdc211342815ced72de
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
EPOCHS=${EPOCHS:-}
if [ -n "$EPOCHS" ]; then
    [[ $EPOCHS =~ ^[0-9]+$ ]] || { echo "EPOCHS must be an integer" >&2; exit 1; }
    SLUG+=-ep$EPOCHS
fi
PRECISION=${PRECISION:-bf16}
case $PRECISION in
    bf16) PYENV= ;;
    fp32) SLUG+=-fp32; PYENV=,PYTHONPATH=/workspace/axolotl/mainwork ;;
    *) echo "PRECISION must be bf16 or fp32" >&2; exit 1 ;;
esac

STAMP=$(date +%Y-%m-%d_%H%M)
BASE=$HOME/c2s
DATA=/xdisk/vkalin/dolzhenko/c2s
RUN=$DATA/results/${STAMP}_${SLUG}
SRC=https://huggingface.co/$HF_SRC/resolve/$REV
SIF_URL=https://huggingface.co/dolzhenkosv/c2s-puma-sif/resolve/main
SIF=$DATA/images/c2s-train.sif
WORK=/tmp/c2s_$SLURM_JOB_ID
T0=$(date +%s)

if [ -e "$RUN" ]; then echo "run dir exists: $RUN" >&2; exit 1; fi
mkdir -p "$RUN" "$DATA/images"
ln -sfn "$RUN" "$DATA/results/последний"
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
MONPID=
INFPID=
STEP=setup
STEP_T0=$(date +%s)
RC=0
DIAG=$RUN/diag
mkdir -p "$DIAG"
CG=/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup)
MEM_CUR=$CG/memory.current
MEM_PEAK=$CG/memory.peak
MEM_EV=$CG/memory.events
if [ ! -r "$MEM_CUR" ]; then
    CG=/sys/fs/cgroup/memory$(sed -n 's/^[0-9]*:memory://p' /proc/self/cgroup)
    MEM_CUR=$CG/memory.usage_in_bytes
    MEM_PEAK=$CG/memory.max_usage_in_bytes
    MEM_EV=$CG/memory.oom_control
fi
oom_kills() { awk '$1 == "oom_kill" {print $2}' "$MEM_EV" 2>/dev/null; }
gib() { awk -v b="$(cat "$1" 2>/dev/null)" 'BEGIN {if (b ~ /^[0-9]+$/) printf "%.2f", b / 2^30}'; }
step() {
    local now=$(date +%s)
    local took=$(( (now - STEP_T0) / 60 ))
    echo "$(date '+%F %T') $1 (after $STEP, $took min)" | tee -a "$DIAG/steps.log"
    case $1 in
        train_*|finalize_best|infer_*) tg "⏩ ${STAMP}_${SLUG}: $1 (after $STEP, $took min)" ;;
    esac
    STEP=$1
    STEP_T0=$now
    echo "$STEP" > "$DIAG/step"
}
infer_progress() {
    set +e
    while sleep 1800; do
        local s=$(cat "$DIAG/step" 2>/dev/null)
        local p=$(tr '\r' '\n' < "$RUN/inference/${s#infer_}.log" 2>/dev/null \
            | grep -oE 'Processed prompts: +[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+ \[[^]]*\]' | tail -1 | sed -E 's/\|[^|]*\|/ /')
        [ -n "$p" ] && tg "⏳ ${STAMP}_${SLUG}: $s, $p"
    done
}
monitor() {
    printf 'time\tmem_gib\tpeak_gib\toom_kill\tprocs\ttmp_free_gib\tstep\n'
    while :; do
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%T)" "$(gib "$MEM_CUR")" "$(gib "$MEM_PEAK")" \
            "$(oom_kills)" "$(wc -l < "$CG/cgroup.procs" 2>/dev/null)" \
            "$(df -Pk /tmp | awk 'NR == 2 {printf "%.1f", $4 / 2^20}')" "$(cat "$DIAG/step" 2>/dev/null)"
        sleep 5
    done
}
{
    echo "start: $(date '+%F %T %Z')"
    echo "node: $(hostname)"
    echo "slurm: job=$SLURM_JOB_ID cpus_per_task=${SLURM_CPUS_PER_TASK:-} mem_per_cpu=${SLURM_MEM_PER_CPU:-} mem_per_node=${SLURM_MEM_PER_NODE:-}"
    echo "nproc: $(nproc) online_cpus: $(getconf _NPROCESSORS_ONLN)"
    echo "cgroup: $CG"
    for d in "$CG" "${CG%/*}" "${CG%/*/*}"; do
        for f in memory.max memory.limit_in_bytes; do
            if [ -r "$d/$f" ]; then echo "$d/$f: $(cat "$d/$f")"; fi
        done
    done
    df -h /tmp "$HOME"
    ulimit -a
} > "$DIAG/start.txt" 2>&1
monitor > "$DIAG/resources.tsv" 2>&1 &
MONPID=$!

TG_TOKEN=$(sed -n 's/^TG_TOKEN=//p' "$ENV_FILE")
TG_CHAT=$(sed -n 's/^TG_CHAT=//p' "$ENV_FILE")
tg() {
    [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ] || return 0
    printf 'url = "https://api.telegram.org/bot%s/sendMessage"\n' "$TG_TOKEN" | curl -fsS -m 30 -K - \
        --data-urlencode "chat_id=$TG_CHAT" --data-urlencode "text=$1" > /dev/null 2>&1 || true
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
    RC=$?
    set +e
    echo "$(date '+%F %T') exit $RC in step $STEP, stopping background jobs" | tee -a "$DIAG/steps.log"
    if [ -n "$UPID" ]; then
        pkill -P "$UPID" 2>/dev/null; kill "$UPID" 2>/dev/null; wait "$UPID" 2>/dev/null
    fi
    if [ -n "$TGPID" ]; then
        kill "$TGPID" 2>/dev/null; wait "$TGPID" 2>/dev/null
    fi
    if [ -n "$INFPID" ]; then
        kill "$INFPID" 2>/dev/null; wait "$INFPID" 2>/dev/null
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
    if [ -n "$MONPID" ]; then
        kill "$MONPID" 2>/dev/null; wait "$MONPID" 2>/dev/null
    fi
    {
        echo "end: $(date '+%F %T %Z')"
        echo "exit_code: $RC"
        echo "step: $STEP"
        echo "memory_peak_gib: $(gib "$MEM_PEAK")"
        cat "$MEM_EV"
        df -h /tmp "$HOME"
    } > "$DIAG/end.txt" 2>&1
    {
        echo "# ${STAMP}_${SLUG}"
        echo
        echo "status: $STATUS"
        echo "exit_code: $RC$([ "$RC" -gt 128 ] && echo " (signal $((RC - 128)))")"
        echo "last_step: $STEP"
        echo "memory_peak_gib: $(gib "$MEM_PEAK")"
        echo "oom_kill: $(oom_kills)"
        echo "model: $MODEL"
        echo "slurm_job: $SLURM_JOB_ID on $(hostname)"
        echo "duration_s: $(( $(date +%s) - T0 ))"
        echo "source: $HF_SRC@$REV"
        echo "image_sha256: $(cut -d' ' -f1 "$RUN/image.sha256" 2>/dev/null)"
        echo "config_sha256: $(grep " $CFG\$" "$RUN/inputs_sha256.txt" 2>/dev/null | cut -d' ' -f1)"
        echo "num_epochs: $(sed -n 's/^num_epochs: *\([0-9]*\).*/\1/p' "$WORK/mainwork/$CFG" 2>/dev/null)"
        echo "micro_batch_size: ${MB}"
        echo "precision: $PRECISION ($(grep -o -m1 '"torch_dtype": "[^"]*"' "$RUN/train.log" 2>/dev/null))"
        echo "base_model_revision: ${BASE_REV:-}"
        echo "best_checkpoint: ${BEST}"
        echo "final_model_sha256: $(cat "$RUN/$OUTNAME"/model*.safetensors 2>/dev/null | sha256sum | cut -d' ' -f1)"
        echo "inference: $INFER_STATUS (test at $INFER_TEMPS, azithromycin at $AZI_TEMP)"
        echo "tg_monitor_sha256: $(sha256sum "$WORK/tg_monitor.py" 2>/dev/null | cut -d' ' -f1)"
        echo "slurm_log: $DATA/logs/c2s-sft_$SLURM_JOB_ID.out"
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
        MSG+=$'\n'"step $STEP, exit $RC, memory peak $(gib "$MEM_PEAK") GiB, oom_kill $(oom_kills)"
        LOG=$RUN/train.log
        [ -s "$LOG" ] || LOG=$RUN/preprocess.log
        [ -s "$LOG" ] || LOG=$DATA/logs/c2s-sft_$SLURM_JOB_ID.out
        MSG+=$'\n\n'"$(tail -c 3000 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -n 15)"
    fi
    tg "$MSG"
    rm -rf "$WORK"
}
trap finish EXIT
PID=
trap '[ "$STATUS" = completed ] || STATUS=timeout; [ -n "$PID" ] && kill "$PID" 2>/dev/null; exit 143' TERM

step download_image
curl -fsSL -H @"$HF_AUTH" "$SIF_URL/c2s-train.sif.sha256" -o "$RUN/image.sha256"
if ! (cd "$DATA/images" && sha256sum -c --quiet "$RUN/image.sha256" 2>/dev/null); then
    curl -fL -H @"$HF_AUTH" "$SIF_URL/c2s-train.sif" -o "$SIF.part"
    mv "$SIF.part" "$SIF"
    (cd "$DATA/images" && sha256sum -c "$RUN/image.sha256")
fi

step download_inputs
FILES="$CFG dataset/train_segments.jsonl dataset/valid_segments.jsonl"
[ "$INFER" = 1 ] && FILES+=" dataset/test_segments.jsonl"
curl -fsSL -H @"$HF_AUTH" "$SRC/$INPUTS" -o "$RUN/inputs_sha256_source.txt"
for f in $FILES; do
    curl -fsSL -H @"$HF_AUTH" "$SRC/$f" -o "$WORK/mainwork/$f"
    grep -E "^[0-9a-f]{64}  $f\$" "$RUN/inputs_sha256_source.txt" >> "$RUN/inputs_sha256.txt"
done
[ "$(wc -l < "$RUN/inputs_sha256.txt")" -eq "$(echo $FILES | wc -w)" ]
if [ "$INFER" = 1 ]; then
    mkdir -p "$WORK/mainwork/inference"
    curl -fsSL -H @"$HF_AUTH" "$AZI_SRC/inference/sha256.txt" -o "$RUN/inference_inputs_source.txt"
    for n in $AZI_SETS; do
        curl -fsSL -H @"$HF_AUTH" "$AZI_SRC/inference/$n.jsonl" -o "$WORK/mainwork/inference/$n.jsonl"
        grep -E "^[0-9a-f]{64}  inference/$n.jsonl\$" "$RUN/inference_inputs_source.txt" >> "$RUN/inputs_sha256.txt"
    done
fi
(cd "$WORK/mainwork" && sha256sum -c "$RUN/inputs_sha256.txt")
cp "$WORK/mainwork/$CFG" "$RUN/"
if [ -n "$EPOCHS" ]; then
    sed -i "s/^num_epochs: .*/num_epochs: $EPOCHS/" "$WORK/mainwork/$CFG"
    [ "$(grep -c "^num_epochs: $EPOCHS\$" "$WORK/mainwork/$CFG")" -eq 1 ]
    cp "$WORK/mainwork/$CFG" "$RUN/${CFG%.yaml}_ep$EPOCHS.yaml"
fi
if [ "$PRECISION" = fp32 ]; then
    cat > "$WORK/mainwork/c2s_fp32.py" <<'PY'
"""Master weights in fp32, compute in bf16 autocast.

axolotl 1855055 sets torch_dtype = bfloat16 whenever bf16 is on (utils/config resolve_dtype),
so the model is loaded and its weights are updated in bf16. register() runs in load_cfg before
normalize_config(), which calls resolve_dtype by module-global name: the wrapper keeps cfg.bf16
(Trainer autocast) and loads the model in float32.
"""
import torch

import axolotl.utils.config as axolotl_config
from axolotl.integrations.base import BasePlugin

_resolve_dtype = axolotl_config.resolve_dtype


def _resolve_dtype_fp32_master(cfg):
    _resolve_dtype(cfg)
    if cfg.bf16:
        cfg.torch_dtype = torch.float32


class Fp32MasterWeightsPlugin(BasePlugin):
    def register(self, cfg):
        axolotl_config.resolve_dtype = _resolve_dtype_fp32_master
PY
    if grep -q '^plugins:' "$WORK/mainwork/$CFG"; then
        sed -i '/^plugins:/a\  - c2s_fp32.Fp32MasterWeightsPlugin' "$WORK/mainwork/$CFG"
    else
        printf '\nplugins:\n  - c2s_fp32.Fp32MasterWeightsPlugin\n' >> "$WORK/mainwork/$CFG"
    fi
    [ "$(grep -c '^  - c2s_fp32.Fp32MasterWeightsPlugin$' "$WORK/mainwork/$CFG")" -eq 1 ]
    cp "$WORK/mainwork/$CFG" "$RUN/${CFG%.yaml}_fp32.yaml"
    cp "$WORK/mainwork/c2s_fp32.py" "$RUN/c2s_fp32.py"
fi

nvidia-smi > "$RUN/nvidia-smi.txt"

run() {
    apptainer exec --nv --cleanenv --no-home --writable-tmpfs \
        --bind "$WORK/mainwork:/workspace/axolotl/mainwork" \
        --bind "$WORK/last_run_prepared:/workspace/axolotl/last_run_prepared" \
        --bind "$WORK/hf:/hf" --bind "$WORK/triton:/triton" \
        --env HF_HOME=/hf,TRITON_CACHE_DIR=/triton,HOME=/tmp,PYTHONFAULTHANDLER=1$PYENV \
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
        --env HF_HOME=/hf,TRITON_CACHE_DIR=/triton,VLLM_CACHE_ROOT=/triton/vllm,HOME=/tmp,PYTHONFAULTHANDLER=1,VLLM_NO_USAGE_STATS=1,"$1" \
        "$SIF" "${@:2}"
}

step container_checks
run cat /opt/c2s/build_info.txt > "$RUN/build_info.txt"
run test -f /opt/c2s/finalize_best.py
[ "$INFER" = 1 ] && run_infer TEMPERATURE=1 -c 'import vllm'
run python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))" \
    | tee "$RUN/torch_check.txt"
tg "▶️ ${STAMP}_${SLUG}: job $SLURM_JOB_ID on $(hostname)
$(cat "$RUN/torch_check.txt")"

step preprocess
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
    step "train_micro$MB"
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
    step finalize_best
    INFER_BEST=$(run python /opt/c2s/finalize_best.py "mainwork/$OUTNAME" 2>> "$RUN/finalize_best.log")
    tg "🏁 ${STAMP}_${SLUG}: training done, best $INFER_BEST, inference: test at $INFER_TEMPS, azithromycin at $AZI_TEMP"
    mkdir -p "$RUN/inference"
    infer_progress &
    INFPID=$!
    for t in $INFER_TEMPS; do
        tag=t$(echo "$t" | tr -d .)
        step "infer_test_$tag"
        run_infer "MODEL_DIR=/workspace/axolotl/mainwork/$OUTNAME,TEST_PATH=/workspace/axolotl/mainwork/dataset/test_segments.jsonl,OUT_PATH=/run_out/inference/test_$tag.jsonl,TEMPERATURE=$t" \
            /opt/c2s/run_test.py > "$RUN/inference/test_$tag.log" 2>&1 &
        PID=$!; wait $PID
    done
    for n in $AZI_SETS; do
        step "infer_$n"
        run_infer "MODEL_DIR=/workspace/axolotl/mainwork/$OUTNAME,TEMPERATURE=$AZI_TEMP" \
            /opt/c2s/run_azi.py "/workspace/axolotl/mainwork/inference/$n.jsonl" "/run_out/inference/${n}_results.jsonl" \
            > "$RUN/inference/$n.log" 2>&1 &
        PID=$!; wait $PID
    done
    INFER_STATUS=completed
fi
