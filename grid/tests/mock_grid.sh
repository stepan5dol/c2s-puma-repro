#!/bin/bash
# Mock run of the grid start on a Linux machine with sudo (GitHub Actions): grid/start.sh -> submit_cells.sh ->
# the queued jobs run in dependency order with stubs for sbatch, apptainer, curl and nvidia-smi. Checks that one job per
# cell trains and then writes the fp32 inference, that the analysis waits for exactly those jobs, and that a failed
# inference fails the job (no silent pass). Nothing real is trained; the stubs are in grid/tests/stubs.
#   bash grid/tests/mock_grid.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/../.." && pwd)
D=/xdisk/vkalin/dolzhenko/c2s
START=$D/results/2026-10-28_1152_puma-a100mig-repro-ep2-fp32
fail() { echo "MOCK FAIL: $*" >&2; exit 1; }

[ -e /xdisk ] && fail "/xdisk exists: run this only on a throwaway machine"
sudo mkdir -p "$D" && sudo chown -R "$USER" /xdisk
mkdir -p "$D/images" "$D/data_src" "$START/inference_float32" "$HOME/c2s"
cp -r "$SRC" "$D/repo"
for f in GSE275938_compiled_counts.csv GSE275938_cell_metadata.csv rat.ho.azi.integrated.h5ad he_lung_atlas.h5ad \
         2022FetalLungIntCounts.h5ad cell_types_config.json; do echo stub > "$D/data_src/$f"; done
echo '{"stub": 1}' > "$START/inference_float32/test_t10.jsonl"
printf 'HF_TOKEN=hf_mock\n' > "$HOME/c2s/c2s.env"
printf 'stub c2s-train.sif\n' > "$D/images/c2s-train.sif"
export PATH=$HERE/stubs:$PATH MOCK_QUEUE=$(mktemp -d)

echo "== start.sh"
bash "$D/repo/grid/start.sh"
echo "== queued jobs"
python3 - "$MOCK_QUEUE" <<'PY'
import json, os, sys
q = sys.argv[1]
jobs = [json.load(open(f"{q}/{f}")) for f in sorted(os.listdir(q)) if f.endswith(".json")]
for j in jobs:
    print(j["id"], j["opts"].get("job-name"), j["opts"].get("time", "-"), j["opts"].get("dependency", "-"))
names = [j["opts"]["job-name"] for j in jobs]
assert names == ["g-lr1e-5_ep2_s42", "g-lr3e-5_ep2_s42", "c2s-analyze"], names
train = [j for j in jobs if j["opts"]["job-name"].startswith("g-")]
for j in train:
    e = j["env"]
    assert (e["INFER_FP32"], e["INFER"], e["NO_EARLY_STOP"], e["PRECISION"], e["CKPTS"]) == ("1", "0", "1", "fp32", "none"), e
    assert e["REPO"] == "/xdisk/vkalin/dolzhenko/c2s/repo", e["REPO"]
    assert j["opts"]["time"] == "12:00:00", j["opts"]["time"]           # 2 epochs: 2 x 2.5 x 1.6 + 1 + 3
assert jobs[2]["opts"]["dependency"] == f"afterok:{train[0]['id']}:{train[1]['id']}", jobs[2]["opts"]["dependency"]
PY
echo "== cells.tsv"; cat "$D/grid/cells.tsv"
[ "$(wc -l < "$D/grid/cells.tsv")" -eq 3 ] || fail "cells.tsv should have 3 lines"

echo "== run the jobs"
python3 "$HERE/run_queue.py" "$MOCK_QUEUE"
for id in 1001 1002; do
    rc=$(python3 -c "import json;print(json.load(open('$MOCK_QUEUE/states.json'))['$id'])")
    [ "$rc" = 0 ] || { tail -n 40 "$MOCK_QUEUE/$id.out"; fail "job $id exit $rc"; }
done
while IFS=$'\t' read -r name run tj ij; do
    [ "$name" = lr1e-4_ep2_s42 ] && continue
    echo "-- $name: $run"
    [ "$tj" = "$ij" ] || fail "$name: train and infer job differ in cells.tsv"
    j=$run/inference_float32/test_t10.jsonl
    [ -s "$j" ] || fail "$name: no $j"
    python3 - "$j" "$run" <<'PY'
import json, sys
r = json.loads(open(sys.argv[1]).readline())
assert r["dtype"] == "float32", r
assert r["T"] == "1.0", r
assert r["model_host_path"].startswith("/tmp/c2s_"), r     # the end-of-run model in the job's work folder
PY
    grep -q "^status: completed$" "$run/record.md" || fail "$name: record.md status"
    grep -q "^inference: completed (fp32" "$run/record.md" || fail "$name: record.md inference"
    grep -q "end of run, inside training job $tj" "$run/inference_float32/model.txt" || fail "$name: model.txt"
    [ -s "$run/sft_pythia1b_short_full/model.safetensors" ] || fail "$name: model not saved to the run folder"
    [ -s "$run/trainer_state.json" ] || fail "$name: no trainer_state.json"
    grep -q 'dtype="float32"' "$run/inference_float32/run_test.diff" || fail "$name: dtype not patched"
    echo "   ok: fp32 inference of the end-of-run model, model and trainer_state saved"
done < "$D/grid/cells.tsv"

echo "== a failing inference must fail the job"
J=$(ls "$MOCK_QUEUE"/1001.json)
python3 - "$J" "$MOCK_QUEUE" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
j["id"] = 2001; j["env"]["RUN_STAMP"] = "mock-bad"; j["env"]["MOCK_FORCE_DTYPE"] = "bfloat16"
import os; q2 = sys.argv[2] + "/bad"; os.makedirs(q2)
json.dump(j, open(f"{q2}/2001.json", "w"))
PY
python3 "$HERE/run_queue.py" "$MOCK_QUEUE/bad"
rc=$(python3 -c "import json;print(json.load(open('$MOCK_QUEUE/bad/states.json'))['2001'])")
[ "$rc" != 0 ] || fail "job with a bf16 inference exited 0"
BAD=$(ls -d "$D"/results/mock-bad_*)
grep -q "^inference: failed" "$BAD/record.md" || fail "record.md of the failed job does not say inference: failed"
grep -q "did not run in float32" "$BAD/infer_fp32.log" || fail "infer_fp32.log does not say why"
[ -s "$BAD/sft_pythia1b_short_full/model.safetensors" ] || fail "the trained model must be kept when inference fails"
echo "   ok: exit $rc, record.md says inference failed, the model is kept"

echo "== data_src: the local dataset is used, never silently replaced by the Hugging Face inputs"
DS=$D/data_src
python3 - "$DS" "$HERE/stubs" <<'PY'
import os, sys
sys.path.insert(0, sys.argv[2])
import mockdata as M
for rel in M.KNOWN:
    p = os.path.join(sys.argv[1], rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(M.content(rel))
PY
rerun() {  # rerun <id> <stamp>: the first training job of the queue again, under a new id and stamp; prints its exit code
    python3 - "$MOCK_QUEUE" "$1" "$2" <<'PY'
import json, os, sys
q, i, stamp = sys.argv[1:]
j = json.load(open(f"{q}/1001.json"))
j["id"] = int(i); j["env"]["RUN_STAMP"] = stamp
os.makedirs(f"{q}/q{i}")
json.dump(j, open(f"{q}/q{i}/{i}.json", "w"))
PY
    python3 "$HERE/run_queue.py" "$MOCK_QUEUE/q$1" > /dev/null
    python3 -c "import json;print(json.load(open('$MOCK_QUEUE/q$1/states.json'))['$1'])"
}
sums() { (cd "$DS" && sha256sum sft_pythia1b.yaml dataset/*.jsonl inference/*.jsonl > inputs_sha256.txt); }

rc=$(rerun 3001 mock-ds-nolist)
[ "$rc" != 0 ] || fail "data_src/dataset without inputs_sha256.txt: the job exited 0"
grep -q "not falling back to Hugging Face" "$MOCK_QUEUE/q3001/3001.out" || fail "the job does not say why it stopped"
echo "   ok: data_src/dataset without its checksum list stops the job (exit $rc)"

sums
rc=$(rerun 3002 mock-ds-ok)
[ "$rc" = 0 ] || { tail -n 40 "$MOCK_QUEUE/q3002/3002.out"; fail "data_src job exit $rc"; }
OK=$(ls -d "$D"/results/mock-ds-ok_*)
grep -q "^data_src " "$OK/inputs_source.txt" || fail "training did not take data_src: $(cat "$OK/inputs_source.txt")"
grep -q "^data_src " "$OK/inference_float32/inputs_source.txt" || fail "inference did not take data_src"
grep -q "  dataset/test_segments.jsonl$" "$OK/inference_float32/inputs_sha256.txt" || fail "inference does not record its test set"
echo "   ok: training and inference take data_src and record it"

printf '{"stub": "second line"}\n' >> "$DS/dataset/test_segments.jsonl"; sums
rc=$(rerun 3003 mock-ds-short)
[ "$rc" != 0 ] || fail "1 answer for a 2-line test set: the job exited 0"
SH=$(ls -d "$D"/results/mock-ds-short_*)
grep -q "test_t10: 1 records for 2 lines of the test set" "$SH/infer_fp32.log" || fail "infer_fp32.log does not say why"
echo "   ok: fewer answers than test lines fail the job (exit $rc)"
echo "MOCK OK"
