#!/usr/bin/env python3
"""Fit the human rank -> log1p(CPM10k) calibration for K = 2000 once and save it next to the old file.

reports/rank_expr_model.npz (5.11, r2 0.8361) is the K = 800 fit; scan_only.py and build_human_h5ads.py loaded it for
K = 2000 runs, so model ranks 801-2000 were placed on a line fitted to ranks 1-800. This is the same call that
results/2026-11-19_ali_he22_corrected_analysis/run_corrected_analysis.py makes (rank_expr_model.fit_from_csv on
GSE275938_compiled_counts.csv, k = 2000, n_cells = 2000, seed = 0; r2 0.8665 there), written to
reports/rank_expr_model_k2000.npz so that nothing reads the counts CSV for the calibration again.

usage: python3 fit_calibration_k2000.py
"""
import sys
from pathlib import Path

import numpy as np

TREE = Path(__file__).resolve().parents[1] / "tree/retrain_v2/analysis"
sys.path.insert(0, str(TREE))
import rank_expr_model as rem  # noqa: E402

CSV = Path("/Users/stepandolzhenko/Documents/AzithroGemma/BPD-PH/GSE275938_compiled_counts.csv")
K, N_CELLS, SEED = 2000, 2000, 0
OUT = TREE / "reports/rank_expr_model_k2000.npz"

if OUT.exists():
    sys.exit(f"{OUT} exists: not overwriting")
m = rem.fit_from_csv(CSV, k=K, n_cells=N_CELLS, seed=SEED)
np.savez(OUT, slope=m.slope, intercept=m.intercept, r2=m.r2, gene_names=np.array(m.gene_names, dtype=object),
         k=K, n_cells=N_CELLS, seed=SEED, source=str(CSV))
print(f"k={K} slope={m.slope:.6f} intercept={m.intercept:.6f} r2={m.r2:.4f} genes={len(m.gene_names)} -> {OUT}")
