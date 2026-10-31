"""[scan-only copy of results/fp32_2026-10-30/scripts/sweep_standard_once.py: the helpers and the one read of
the counts CSV, without the rescue table and panel C]
One process for the standard human-AZI analysis on the checkpoint sweep.

Variants: ms264 = the manuscript's own run (results/inference_azi_results.jsonl, checkpoint 264,
pod sampler) and ckpt264/297/330/396 = the local MLX sweep (same per-cell seed across
checkpoints). The manuscript's variant scripts are imported unchanged:
  ablation_rescue_pipeline.py      -> reports/ablation_rescue_genomewide[_summary]_<v>.csv
  ablation_export_for_limma.py     -> reports/ablation_limma_input_bpd7mo_<v>/
  ablation_limma_sets.py           -> autophagy_program/reports/ablation_bpd7mo_limma_sets_<v>.csv
Only plumbing is patched: the calibration fit is the cached one, load_human_bpd_raw scans the
3.2 GB counts CSV once for all five conditions and serves subsets from memory (same rows, same
order), and the ms264 variant name is pointed at the original file.

"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
ROOT = Path(os.environ["C2S_ROOT"])  # tree root: holds retrain_v2/, pipeline/, train-after-grpo-analysis/
R = ROOT / "retrain_v2"
SUITE = R / "analysis/real_counts_suite_ortho"
HV = SUITE / "human_panel_variants"
SCR = Path(os.environ["C2S_WORK"])
VARIANTS = []  # scan-only: no rescue table, no panel C
MS_FILE = R / "results/inference_azi_results.jsonl"

for p in (HV, SUITE, R / "analysis"):
    sys.path.insert(0, str(p))

import rank_expr_model as rem  # noqa: E402

z = np.load(R / "analysis/reports/rank_expr_model.npz", allow_pickle=True)
_names = list(z["gene_names"])
HUMAN_MODEL = rem.RankExprModel(float(z["slope"]), float(z["intercept"]), float(z["r2"]),
                                {g: i for i, g in enumerate(_names)}, _names)
rem.fit_from_csv = lambda *a, **k: HUMAN_MODEL

sys.argv = ["x"] + VARIANTS
import ablation_rescue_pipeline as ARP  # noqa: E402
import ablation_export_for_limma as AEL  # noqa: E402
import ablation_limma_sets as ALS  # noqa: E402
import common_human as CH  # noqa: E402
import common_real as CR  # noqa: E402

RC = sys.modules["real_counts"]
L = sys.modules["load_data"]

# ---- one scan of the counts CSV, subsets served from memory ----
ALL_CONDS = ["Term0d", "Term20d", "Acute26", "BPD7mo", "BPDPH7mo"]
_scan = {}


def _scan_once():
    if not _scan:
        print("[scan] one chunked read of the counts CSV for", ALL_CONDS, flush=True)
        meta = pd.read_csv(L.C.BPD_META_PATH, index_col="id")
        meta["condition"] = meta["dataset"].map(L.C.DATASET_TO_CONDITION)
        meta = meta[meta["condition"].isin(ALL_CONDS) & meta["celltype"].isin(L.C.TARGET_CELL_TYPES)]
        needed = set(meta.index)
        chunks_X, chunks_bc, genes = [], [], None
        for chunk in pd.read_csv(L.C.BPD_COUNTS_PATH, index_col="id", chunksize=5000):
            if genes is None:
                genes = L.remap_gene_names(chunk.columns.tolist())
            rows_here = [bc for bc in chunk.index if bc in needed]
            if rows_here:
                chunks_X.append(chunk.loc[rows_here].values.astype(np.float32))
                chunks_bc.extend(rows_here)
        _scan.update(X=np.vstack(chunks_X), genes=genes, bc=np.array(chunks_bc),
                     ct=meta.loc[chunks_bc, "celltype"].values, cond=meta.loc[chunks_bc, "condition"].values)
        print(f"[scan] {_scan['X'].shape[0]} cells x {_scan['X'].shape[1]} genes", flush=True)
    return _scan


def load_human_bpd_raw(conditions):
    s = _scan_once()
    m = np.isin(s["cond"], list(conditions))
    return s["X"][m], s["genes"], s["ct"][m], s["cond"][m]


L.load_human_bpd_raw = load_human_bpd_raw

_orig_load_azi = CH.load_azi_counterfactual


def load_azi(path=CH.AZI_RESULTS_PATH):
    if Path(path).name == "inference_azi_results_ms264.jsonl":
        path = MS_FILE
    return _orig_load_azi(path)


CH.load_azi_counterfactual = load_azi
ARP.AZI_JSONL_OF["ms264"] = MS_FILE

