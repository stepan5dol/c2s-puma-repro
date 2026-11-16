#!/usr/bin/env python3
"""
build_disease_arms_trajectory.py — the three disease arms that compare a
diseased state against a healthy one, which is the contrast the rescued-gene
panel itself is defined on (p_val_adj_HO is term-born -> disease):

    fetal lung GW22       -> acute preterm lung injury GW26   (paired)
    term-born control     -> BPD, 7 months postnatal          (random baseline)
    term-born control     -> BPD-PH, 7 months postnatal        (random baseline)

The SFT data instead pairs BPD and BPD-PH off the acute-injury cell
(common_human.py:18-20), i.e. disease against an earlier disease stage. Those
arms are deliberately NOT built: a delta measured from an already-diseased
baseline is not comparable to a panel defined against health.

Term0d/Term20d cells are absent from the pairing registry entirely, so no
paired term-born -> disease example exists; the term-born baseline is drawn at
random within cell type (seeded), which is recorded per arm as
pairing="random_within_ct".

He22 source cells live in the atlas h5ad, not compiled_counts.csv, so they
are loaded separately and reindexed onto the model's gene axis.

Model side is unchanged in all three: pred is the model's prediction for that
test example, differenced against the same baseline cell as the real arm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import human_updown_percell as DIS
import panel_sets_human as PSH

import common_human as CH
import rank_expr_model as rem
import load_data as L
import real_counts as RC

PIPELINE_SHORT = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/pipeline_short")
sys.path.insert(0, str(PIPELINE_SHORT))
import stage1_sft_bidir as S1

PAIRS_CSV = HERE / "recovered_human_test_pairs.csv"
TEST_INFERENCE = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/test_inference_results_t10.jsonl")
SEED = 0

GATES = ("full_list",)
SOLO_CT = ("gCap",)

CT_MAP = {"general capillary endothelial cell (endothelial)": "gCap",
          "aerocyte capillary endothelial cell (endothelial)": "aCap",
          "pericyte (mural)": "Pericyte",
          "vascular endothelial cell (endothelial)": "VEC",
          "pulmonary venous endothelial cell (endothelial)": "VEC"}

ARMS = [
    ("Acute26", "He22", "human_He22_Acute26", "atlas"),
    ("BPD7mo", "Term", "human_Acute26_BPD7mo", "term"),
    ("BPDPH7mo", "Term", "human_Acute26_BPDPH7mo", "term"),
]

LABELS = {
    ("He22", "Acute26"): (
        "fetal lung, normal development, 22 weeks of gestation",
        "acute preterm lung injury, 26 weeks of gestation",
        "fetal", "acute injury", "fetal_gw22", "ali_gw26"),
    ("Term", "BPD7mo"): (
        "term-born control",
        "bronchopulmonary dysplasia, 7 months postnatal",
        "Term-born", "BPD 7mo", "term_born", "bpd_7mo"),
    ("Term", "BPDPH7mo"): (
        "term-born control",
        "bronchopulmonary dysplasia with pulmonary hypertension, 7 months postnatal",
        "Term-born", "BPD-PH 7mo", "term_born", "bpd_ph_7mo"),
}


def load_bpd_by_barcode(barcodes, model):
    print("  [bpd] chunked read of compiled_counts.csv (once)...")
    chunks_X, chunks_bc, genes = [], [], None
    for chunk in pd.read_csv(L.C.BPD_COUNTS_PATH, index_col="id", chunksize=5000):
        if genes is None:
            genes = L.remap_gene_names(chunk.columns.tolist())
        rows = [bc for bc in chunk.index if bc in barcodes]
        if rows:
            chunks_X.append(chunk.loc[rows].values.astype(np.float64))
            chunks_bc.extend(rows)
    X = np.vstack(chunks_X)
    lib = X.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
    Xn = RC._reindex_to_model_genes(np.log1p(X / lib * 1e4).astype(np.float32),
                                    genes, model.gene_names)
    print(f"  [bpd] {Xn.shape[0]} cells")
    return {bc: Xn[i] for i, bc in enumerate(chunks_bc)}


def load_atlas_by_barcode(barcodes, model):
    """He22 source cells. The clean atlas carries the barcodes the pairing used;
    counts come from the raw atlas, matched the same way stage1 matches them
    (exact, else drop the trailing sample suffix)."""
    print("  [atlas] reading clean atlas for barcode space...")
    clean = anndata.read_h5ad(S1.ATLAS_CLEAN_PATH)
    clean_obs = clean.obs_names.tolist()
    del clean
    print("  [atlas] reading raw atlas counts...")
    raw = anndata.read_h5ad(S1.ATLAS_RAW_PATH)
    genes = L.remap_gene_names(raw.var_names.tolist())
    raw_obs = raw.obs_names.tolist()
    raw_set = set(raw_obs)
    raw_short = {}
    for rbc in raw_obs:
        parts = rbc.split("-")
        raw_short["-".join(parts[:2]) if len(parts) >= 2 else rbc] = rbc

    want = {}
    for cbc in clean_obs:
        if cbc not in barcodes:
            continue
        if cbc in raw_set:
            want[cbc] = cbc
        else:
            parts = cbc.split("-")
            key = "-".join(parts[:2]) if len(parts) >= 2 else cbc
            if key in raw_short:
                want[cbc] = raw_short[key]
    print(f"  [atlas] {len(want)}/{len(barcodes)} requested barcodes matched")

    pos = {bc: i for i, bc in enumerate(raw_obs)}
    keys = list(want)
    rows = [pos[want[k]] for k in keys]
    X = raw.X[rows]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    del raw
    lib = X.sum(axis=1, keepdims=True); lib[lib == 0] = 1.0
    Xn = RC._reindex_to_model_genes(np.log1p(X / lib * 1e4).astype(np.float32),
                                    genes, model.gene_names)
    return {k: Xn[i] for i, k in enumerate(keys)}


def main():
    print("[calibration] fitting rank->expr model (once)...")
    model = rem.fit_from_csv(CH.COUNTS_CSV)
    print(f"  r2={model.r2:.4f}")

    pairs = pd.read_csv(PAIRS_CSV)
    pairs = pairs[pairs["verified"]].copy()
    pairs["ct"] = pairs["cell_type"].map(CT_MAP)

    bpd_needed, atlas_needed, idx_needed = set(), set(), set()
    for cond, src, pt, kind in ARMS:
        sub = pairs[pairs["pair_type"] == pt]
        bpd_needed |= set(sub["tgt_barcode"])
        idx_needed |= set(sub["idx"])
        if kind == "atlas":
            atlas_needed |= set(sub["src_barcode"])
    print(f"[plan] bpd={len(bpd_needed)} atlas={len(atlas_needed)} preds={len(idx_needed)}")

    real_by_bc = load_bpd_by_barcode(bpd_needed, model)
    if atlas_needed:
        real_by_bc.update(load_atlas_by_barcode(atlas_needed, model))

    print("[term] real Term0d+Term20d matrices (random-match pool)...")
    term_mats = RC.real_term_matrices(model)
    print("  " + ", ".join(f"{ct}:{X.shape[0]}" for ct, X in term_mats.items()))

    pred_by_idx = {}
    with open(TEST_INFERENCE) as f:
        for line in f:
            r = json.loads(line)
            if r["idx"] in idx_needed:
                pred_by_idx[r["idx"]] = r["pred"].strip().split()
    print(f"[pred] {len(pred_by_idx)} predictions")

    rng = np.random.default_rng(SEED)
    for cond, src_label, pt, kind in ARMS:
        df = pairs[(pairs["pair_type"] == pt) & pairs["ct"].notna()].copy()
        if kind == "term":
            keys = []
            for i, (_, r) in enumerate(df.iterrows()):
                X = term_mats.get(r["ct"])
                if X is None or X.shape[0] == 0:
                    keys.append(None)
                    continue
                k = f"__term__{cond}__{i}"
                real_by_bc[k] = X[rng.integers(X.shape[0])]
                keys.append(k)
            df["src_barcode"] = keys
            df = df[df["src_barcode"].notna()]

        DIS.CONDITION, DIS.SRC_LABEL, DIS.PAIR_TYPE = cond, src_label, pt
        DIS.PANEL_CONDITION = cond
        (DIS.SRC_TITLE, DIS.TGT_TITLE, DIS.SRC_SHORT,
         DIS.TGT_SHORT, DIS.SRC_SLUG, DIS.TGT_SLUG) = LABELS[(src_label, cond)]
        DIS.SRC_FIG, DIS.TGT_FIG = {("He22", "Acute26"): ("gestational week 24", "acute injury, week 26")}.get(
            (src_label, cond), (None, None))
        DIS.PAIRING = "paired" if kind == "atlas" else "random_within_ct"
        print(f"\n{'=' * 78}\n=== {DIS.SRC_TITLE} -> {DIS.TGT_TITLE}  (n={len(df)})\n{'=' * 78}")
        print(df.groupby("ct").size().to_string())
        for gate in GATES:
            print(f"--- gate={gate}, all cell types")
            DIS.ONLY_CT = None
            DIS.run(gate, df, real_by_bc, pred_by_idx, model, CT_MAP)
            for ct in SOLO_CT:
                print(f"--- gate={gate}, {ct} alone")
                DIS.ONLY_CT = ct
                DIS.run(gate, df, real_by_bc, pred_by_idx, model, CT_MAP)
            DIS.ONLY_CT = None


if __name__ == "__main__":
    main()
