#!/bin/bash
# Manuscript violins (Figure 5A/B, 5-S2 to 5-S6) and human rescued genes of newgrid_lr1e-4_ep4_s42:
# prepare_root.py, then violins_only.py of c2s-puma-std/grid/lib on the run spec from
# results/2026-11-20_c2s-puma-std_unified_all11/runs.json.
# usage: bash run_violins.sh <out folder> <azi jsonl> [model jsonl ...]
#   azi jsonl     the AZI arm (rescued genes)
#   model jsonl   generated cells of the model arm of the human violins (default: the spec's test file)
set -euo pipefail
OUT=$(cd "$1" && pwd)
AZI=$2
shift 2
AZ=/Users/stepandolzhenko/Documents/AzithroGemma
LIB=$AZ/retrain_v2/c2s-puma-std/grid/lib
TREE=$AZ/retrain_v2/c2s-puma-std/grid/tree
W=$OUT/work/violins
[ ! -e "$W" ] || { echo "$W exists" >&2; exit 1; }
mkdir -p "$W/reference"
python3 - "$AZ/retrain_v2/results/2026-11-20_c2s-puma-std_unified_all11/runs.json" "$W/runs.json" "$AZI" "$@" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))["newgrid_lr1e-4_ep4_s42"]
spec["azi"] = sys.argv[3]
if sys.argv[4:]:
    spec["model"] = sys.argv[4:]
json.dump({"newgrid_lr1e-4_ep4_s42": spec}, open(sys.argv[2], "w"), indent=1)
print(spec)
PY
TD=$(mktemp -d "${TMPDIR:-/tmp}/c2s_ms_violins_XXXX")
mkdir -p "$TD/data/BPD-PH"
for f in GSE275938_compiled_counts.h5ad GSE275938_compiled_counts.csv GSE275938_cell_metadata.csv; do
    [ -e "$AZ/BPD-PH/$f" ] && ln -s "$AZ/BPD-PH/$f" "$TD/data/BPD-PH/$f"
done
for f in rat.ho.azi.integrated.h5ad he_lung_atlas.h5ad 2022FetalLungIntCounts.h5ad; do ln -s "$AZ/$f" "$TD/data/$f"; done
ln -s "$AZ/retrain_v2/scripts/cell_types_config.json" "$TD/data/cell_types_config.json"
python3 "$LIB/prepare_root.py" "$TREE" "$TD/data" "$TD/root"
C2S_ROOT=$TD/root C2S_WORK=$W C2S_RESULTS=$W/reference PYTHONPATH=$LIB \
    python3 -u "$LIB/violins_only.py" "$W/runs.json" 2>&1 | tee "$OUT/work/violins.log"
rm -rf "$TD"
