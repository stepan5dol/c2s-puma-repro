#!/bin/bash
# grid/FORCE_LR end to end through chain_step.py, on a fake grid shaped like the real one on 2026-11-04:
# stage A at 2 epochs (1e-5, 3e-5, 1e-4 = the starting run, 3e-4), stage B at 3e-4 (1, 4), and 1e-4 at 1 and 4 epochs.
# The measures are scripted (stage A picks 3e-4, stage B picks 4 epochs), so this checks the plumbing only:
# the file is read, stage B runs over the 1e-4 epochs, the cap stops the chain and says what the rule wanted,
# a bad file stops the chain, and without the file the rule still asks for 3e-4 at 8 epochs.
#   bash grid/tests/force_lr.sh          (needs python3 with numpy, scipy, scikit-learn)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LIB=$(cd "$HERE/../lib" && pwd)
PY=${PYTHON:-python3}
T=$(mktemp -d "${TMPDIR:-/tmp}/force_lr.XXXXXX")
trap 'rm -rf "$T"' EXIT
fail() { echo "FORCE_LR TEST FAIL: $*" >&2; exit 1; }

# chain_step.py next to a wrapper decide.py: the real decide.py with scripted measures (no model outputs needed)
mkdir "$T/bin"
cp "$LIB/chain_step.py" "$T/bin/"
cat > "$T/bin/decide.py" <<'PYW'
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("decide_real", os.environ["REAL_DECIDE"])
real = importlib.util.module_from_spec(spec)
sys.modules["decide_real"] = real
spec.loader.exec_module(real)

def pick(names, runs, thr, log, tiebreak):
    stage_a = len({n.split("_")[0] for n in names}) > 1
    want = "lr3e-4_ep2_s42" if stage_a else next(n for n in names if "_ep4_" in n)
    for n in names:
        log(f"  {n:22s} scripted")
    return want, "scripted"

real.load_run = lambda d: dict(loss=0.0)
real.pick_best = pick
real.violin_mismatch = lambda scores: 22
next_step, lr_str, violin_mismatch = real.next_step, real.lr_str, real.violin_mismatch
PYW

make_grid() {
    G=$1
    rm -rf "$G"; mkdir -p "$G/reference" "$G/results"
    echo '{}' > "$G/reference/identity_scores.json"
    : > "$G/cells.tsv"
    for c in lr1e-4_ep2_s42 lr1e-5_ep2_s42 lr3e-5_ep2_s42 lr3e-4_ep2_s42 lr3e-4_ep1_s42 lr3e-4_ep4_s42 lr1e-4_ep1_s42 lr1e-4_ep4_s42; do
        r=$G/results/$c
        mkdir -p "$r/inference_float32" "$G/runs/$c/violins"
        echo '{}' > "$r/inference_float32/test_t10.jsonl"
        if [ $c = lr1e-4_ep2_s42 ]; then   # the starting run keeps trainer_state.json only in its checkpoint folders
            mkdir -p "$r/sft_pythia1b_short_full/checkpoint-264"
            echo '{"log_history": []}' > "$r/sft_pythia1b_short_full/checkpoint-264/trainer_state.json"
        else
            echo '{"log_history": []}' > "$r/trainer_state.json"
        fi
        echo '{}' > "$G/runs/$c/violins/scores.json"
        printf '%s\t%s\t1\t1\n' "$c" "$r" >> "$G/cells.tsv"
    done
}
# no -I: it drops the user site-packages, where pip puts numpy/scipy on the GitHub runner
run() { REAL_DECIDE=$LIB/decide.py "$PY" "$T/bin/chain_step.py" --grid "$1" --repo /repo --max-cells "$2"; }

echo "== FORCE_LR 1e-4, MAX_CELLS 0 (the real call)"
make_grid "$T/g1"; echo 1e-4 > "$T/g1/FORCE_LR"
run "$T/g1" 0 > "$T/g1.out" || { cat "$T/g1.out"; fail "chain_step exited non-zero"; }
cat "$T/g1.out"
grep -q "LR set by the author (grid/FORCE_LR): 1e-4; the rule chose 3e-4" "$T/g1.out" || fail "the log does not say the LR was forced"
grep -q "^STAGE B  epochs at LR 0.0001" "$T/g1.out" || fail "stage B did not run at 1e-4"
for c in lr1e-4_ep1_s42 lr1e-4_ep2_s42 lr1e-4_ep4_s42; do
    sed -n '/^STAGE B/,$p' "$T/g1.out" | grep -q "  $c " || fail "stage B did not compare $c"
done
sed -n '/^STAGE B/,$p' "$T/g1.out" | grep -q "lr3e-4_ep" && fail "stage B compared 3e-4 runs"
grep -q "the cap of 0 trainings would be exceeded" "$T/g1/STOPPED" || fail "STOPPED does not name the cap"
grep -q "try 8" "$T/g1/STOPPED" || fail "STOPPED does not say what the rule wanted"
[ -e "$T/g1/next_cells.txt" ] && fail "next_cells.txt written although the cap stops the chain"
"$PY" - "$T/g1/decisions/decision_00.json" <<'PYC'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["state"]["force_lr"] == 1e-4, d["state"]["force_lr"]
assert d["step"] == {"action": "train", "cells": [{"lr": 1e-4, "ep": 8, "seed": 42}],
                     "why": "stage B+: the winner 4 epochs is at the upper edge; try 8"}, d["step"]
PYC
echo "   ok: stage B over 1e-4 epochs 1/2/4, decision written, cap stops, nothing to submit"

echo "== FORCE_LR 1e-4, MAX_CELLS 9: the rule's next cell is handed to analyze.slurm"
make_grid "$T/g2"; printf ' 1e-4 \n' > "$T/g2/FORCE_LR"
run "$T/g2" 9 > "$T/g2.out" || { cat "$T/g2.out"; fail "chain_step exited non-zero"; }
[ "$(cat "$T/g2/next_cells.txt")" = "1e-4:8:42" ] || fail "next_cells.txt: $(cat "$T/g2/next_cells.txt")"
[ -e "$T/g2/STOPPED" ] && fail "STOPPED written although under the cap"
echo "   ok: next_cells.txt = 1e-4:8:42"

echo "== no FORCE_LR: unchanged rule"
make_grid "$T/g3"
run "$T/g3" 9 > "$T/g3.out" || { cat "$T/g3.out"; fail "chain_step exited non-zero"; }
[ "$(cat "$T/g3/next_cells.txt")" = "3e-4:8:42" ] || fail "next_cells.txt: $(cat "$T/g3/next_cells.txt")"
grep -q "FORCE_LR" "$T/g3.out" && fail "FORCE_LR mentioned without the file"
echo "   ok: next_cells.txt = 3e-4:8:42"

echo "== bad FORCE_LR stops the chain"
for bad in "0.0001" "1e-4 3e-4" ""; do
    make_grid "$T/g4"; printf '%s\n' "$bad" > "$T/g4/FORCE_LR"
    if run "$T/g4" 9 > "$T/g4.out" 2>&1; then fail "FORCE_LR '$bad' accepted"; fi
    grep -q "grid/FORCE_LR must hold one LR" "$T/g4/STOPPED" || fail "STOPPED for '$bad': $(cat "$T/g4/STOPPED")"
    ls "$T/g4/decisions"/decision_*.json >/dev/null 2>&1 && fail "decision written for FORCE_LR '$bad'"
    echo "   ok: '$bad' -> $(cat "$T/g4/STOPPED")"
done

echo "== FORCE_LR without a 2-epoch run"
make_grid "$T/g5"; echo 3e-6 > "$T/g5/FORCE_LR"
run "$T/g5" 9 > "$T/g5.out" || true
grep -q "FORCE_LR 3e-6 has no 2-epoch run" "$T/g5/STOPPED" || fail "STOPPED: $(cat "$T/g5/STOPPED")"
echo "   ok: $(cat "$T/g5/STOPPED")"
echo "FORCE_LR OK"
