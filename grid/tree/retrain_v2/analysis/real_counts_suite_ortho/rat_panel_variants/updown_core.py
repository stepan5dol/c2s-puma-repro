#!/usr/bin/env python3
"""
updown_core.py — shared machinery for the rat_panel_variants figures.

Same per-cell delta construction as
rat_top800_both_arms/rescue_delta_*_top800_both_arms.py (BOTH real sides
top-800-truncated, model arm = pred - topk(real source cell)), with two
changes, both of them the point of this folder:

  1. Rescued_HO_down and Rescued_HO_up are drawn as SEPARATE violins instead
     of only "down" being plotted. Background is now everything outside
     down UNION up -- in the older scripts the "up" genes sat inside the
     background set, which blunts the background baseline.
  2. The panel gate is a parameter (panel_sets.GATES), so the p_val_adj_AZI
     selection can be switched off.

Six violins per cell type: REAL[bg, down, up] | MODEL[bg, down, up].
Brackets are bg-vs-down and bg-vs-up within each arm -- the two arms are
never tested against each other, same as in the original scripts.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ABS_DIR = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/analysis")
sys.path.insert(0, str(ABS_DIR))
import common as C
import plot_style as PS
import rank_expr_model as rem

sys.path.insert(0, str(Path(__file__).parent.parent))
import common_real as CR

sys.path.insert(0, str(Path(__file__).parent))
import panel_sets as PSets

HERE = Path(__file__).parent
CSV_PATH = HERE.parent / "rat" / "recovered_rat_test_pairs.csv"
EXAMPLE_REG_PATH = Path(os.environ.get(
    "EXAMPLE_REG_PATH",
    "/Users/stepandolzhenko/Documents/AzithroGemma/retrain_prompt/pipeline/sft_example_registry.csv",
))
TEST_INFERENCE_PATH = Path(os.environ.get("TEST_INFERENCE_PATH",
    "/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/test_inference_results_t10.jsonl"))

CT_RAW_TO_CT = {"general capillary endothelial cell (endothelial)": "gCap",
                "aerocyte capillary endothelial cell (endothelial)": "aCap",
                "pericyte (mural)": "Pericyte",
                "pulmonary venous endothelial cell (endothelial)": "VEC",
                "vascular endothelial cell (endothelial)": "VEC",
                "gCAP": "gCap",
                "aCAP": "aCap",
                "Peri": "Pericyte",
                "VEC": "VEC"}

SUBSETS = ("bg", "down", "up")
X_POS = {("real", "bg"): 0.0, ("real", "down"): 1.0, ("real", "up"): 2.0,
         ("model", "bg"): 3.4, ("model", "down"): 4.4, ("model", "up"): 5.4}
BG_MIN_GENES = 3
MIN_PANEL_GENES = 3
ROW_H = 3.3  # inches per row of two panels (PS.FIG_W wide)

ARMS = {
    "azi": {"pair_type": "rat_HO_AZI", "cond": "AZI",
            "delta_label": "+azithromycin − hyperoxia",
            "title": "treatment arm, hyperoxia vs hyperoxia + azithromycin",
            "slug": "hyperoxia_to_azithromycin",
            "short": "hyperoxia vs +azithromycin"},
    "ho": {"pair_type": "rat_RA_HO", "cond": "HO",
           "delta_label": "hyperoxia − room air",
           "title": "injury arm, room air vs hyperoxia",
           "slug": "room_air_to_hyperoxia",
           "short": "room air vs hyperoxia"},
}

CT_TITLE = {"gCap": "General capillary endothelial cell",
            "aCap": "Aerocyte capillary endothelial cell",
            "Pericyte": "Pericyte",
            "VEC": "Vascular endothelial cell",
            "POOLED": "All cell types pooled"}
PANEL_TITLE = {"down": "hyperoxia-suppressed genes",
               "up": "hyperoxia-induced genes"}
GATE_SLUG = {"full_list": "all_rescued", "gated": "dge_filtered"}
ONLY_CT = None


def load_real_counts_by_barcode(barcodes: list[str], gene_names: list[str]) -> dict[str, np.ndarray]:
    """log1p(CPM10k), EXACT same recipe as rank_expr_model._fit_from_matrix."""
    print("  [real] reading rat.ho.azi.integrated.h5ad raw counts...")
    adata = anndata.read_h5ad(C.H5AD)
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    assert list(adata.var_names) == gene_names, "gene order mismatch vs calibration model"
    bc_idx = {bc: i for i, bc in enumerate(adata.obs_names)}
    needed = [bc for bc in barcodes if bc in bc_idx]
    missing = set(barcodes) - set(needed)
    if missing:
        print(f"  WARN: {len(missing)} barcodes not found in h5ad: {list(missing)[:5]}")
    rows = [bc_idx[bc] for bc in needed]
    Xs = X[rows]
    if sp.issparse(Xs):
        Xs = Xs.toarray()
    Xs = np.asarray(Xs, dtype=np.float64)
    lib = Xs.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    Xn = np.log1p(Xs / lib * 1e4).astype(np.float32)
    del adata
    return {bc: Xn[i] for i, bc in enumerate(needed)}


def per_cell_means(delta_X, mask_X, bool_by_subset):
    """-> {subset: array of per-cell mean delta, NaN where too few genes}"""
    n = delta_X.shape[0]
    out = {s: np.full(n, np.nan) for s in bool_by_subset}
    for i in range(n):
        m = mask_X[i]
        for s, gene_bool in bool_by_subset.items():
            sel = m & gene_bool
            need = BG_MIN_GENES if s == "bg" else 1
            if sel.sum() >= need:
                out[s][i] = delta_X[i, sel].mean()
    return out


def draw_panel(ax, scores, title, delta_label, expected, show_legend=False):
    keys = [(arm, s) for arm in ("real", "model") for s in SUBSETS]
    data = [scores[k] for k in keys]
    all_data = np.concatenate(data)
    parts = ax.violinplot(data, positions=[X_POS[k] for k in keys], widths=0.8,
                          showmedians=True, showextrema=False)
    for body, (arm, s) in zip(parts["bodies"], keys):
        color = PS.HO_COLOR if s == "bg" else (PS.GT_COLOR if arm == "real" else PS.PRED_COLOR)
        body.set_facecolor(color); body.set_edgecolor("#333333")
        body.set_alpha(0.85); body.set_linewidth(1.1)
        if s == "up":
            body.set_hatch("///")
    parts["cmedians"].set_color("#222222"); parts["cmedians"].set_linewidth(1.3)
    ax.axhline(0, color="#999999", lw=1, ls="--", zorder=0)

    yr = all_data.max() - all_data.min()
    step = max(yr * 0.09, 1e-6)
    y0 = all_data.max() + step * 0.6

    stats = {}
    for arm in ("real", "model"):
        for j, s in enumerate(("down", "up")):
            p = mannwhitneyu(scores[(arm, "bg")], scores[(arm, s)], alternative="two-sided").pvalue
            PS.sig_bracket(ax, X_POS[(arm, "bg")], X_POS[(arm, s)], y0 + step * 2.2 * j, step * 0.5, p)
            stats[f"p_{arm}_{s}"] = p
        for s in SUBSETS:
            stats[f"median_{arm}_{s}"] = float(np.median(scores[(arm, s)]))

    ax.axvline((X_POS[("real", "up")] + X_POS[("model", "bg")]) / 2, color="#cccccc", lw=1, ls=":")
    ax.set_ylim(top=y0 + step * 5.0)
    ax.set_xticks([X_POS[k] for k in keys])
    ax.set_xticklabels(["background", "suppressed", "induced"] * 2,
                       fontsize=PS.FS_TICK, rotation=45, ha="right")
    for x, arm in ((1.0, "datasets"), (4.4, "model")):
        ax.text(x, ax.get_ylim()[1], arm, ha="center", va="bottom", fontsize=PS.FS_ANNOT,
                color="#666666")
    n_real = len(scores[("real", "down")])
    ax.set_title(f"{title}\nn = {n_real} cells", fontsize=PS.FS_TITLE, pad=16)
    ax.set_ylabel(f"mean Δ per cell\n({delta_label})", fontsize=PS.FS_LABEL)
    return stats


def legend_handles_labels(expected):
    """Placed on the FIGURE, under the axes -- inside the axes it sat on top of
    the left-hand violin. Same construction as the human core, so the two panels
    of the main figure carry identical legends."""
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=PS.HO_COLOR, edgecolor="#333333", alpha=0.85),
         plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", edgecolor="#333333", alpha=0.85),
         plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", edgecolor="#333333", alpha=0.85, hatch="///")]
    return h, ["background", PANEL_TITLE["down"], PANEL_TITLE["up"]]


def load_pairs_df(arm_key: str) -> pd.DataFrame:
    arm = ARMS[arm_key]
    if EXAMPLE_REG_PATH.exists():
        reg = pd.read_csv(EXAMPLE_REG_PATH)
        reg = reg[(reg["split"] == "test") & (reg["direction"] == "fwd") & (reg["source"] == "rat")].copy()
        reg["ct"] = reg["cell_type"].map(CT_RAW_TO_CT)
        assert reg["ct"].notna().all(), f"unmapped rat cell types: {reg.loc[reg['ct'].isna(), 'cell_type'].unique()}"
        src_cond = "RA" if arm_key == "ho" else "HO"
        tgt_cond = arm["cond"]
        rng = np.random.default_rng(C.SEED)
        rows = []
        for ct in C.CT_ORDER:
            tgt_sub = reg[(reg["condition"] == tgt_cond) & (reg["ct"] == ct)]
            src_pool = reg[(reg["condition"] == src_cond) & (reg["ct"] == ct)]["barcode"].tolist()
            if tgt_sub.empty or not src_pool:
                continue
            replace = len(src_pool) < len(tgt_sub)
            src_bcs = rng.choice(src_pool, size=len(tgt_sub), replace=replace)
            for (_, r), sbc in zip(tgt_sub.iterrows(), src_bcs):
                rows.append({
                    "idx": int(r["line_idx"]),
                    "pair_type": arm["pair_type"],
                    "cell_type": r["cell_type"],
                    "ct": ct,
                    "src_barcode": str(sbc),
                    "tgt_barcode": str(r["barcode"]),
                    "verified": True,
                })
        return pd.DataFrame(rows)

    df = pd.read_csv(CSV_PATH)
    df = df[df["pair_type"] == arm["pair_type"]].copy()
    assert df["verified"].all(), "unverified rows present -- do not trust unverified barcodes"
    df["ct"] = df["cell_type"].map(CT_RAW_TO_CT)
    assert df["ct"].notna().all()
    return df


def run(arm_key: str, gate: str):
    arm = ARMS[arm_key]
    expected = {"down": PSets.EXPECTED[("Rescued_HO_down", arm["cond"])],
                "up": PSets.EXPECTED[("Rescued_HO_up", arm["cond"])]}

    model = rem.fit(C.H5AD)
    print(f"[calibration] slope={model.slope:.5f} intercept={model.intercept:.4f} r2={model.r2:.4f}")
    panels_by_ct = PSets.load_panels(gate)

    df = load_pairs_df(arm_key)

    all_bc = list(pd.unique(df[["src_barcode", "tgt_barcode"]].values.ravel()))
    real_by_bc = load_real_counts_by_barcode(all_bc, model.gene_names)

    idx_set = set(df["idx"])
    pred_by_idx = {}
    with open(TEST_INFERENCE_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["idx"] in idx_set:
                pred_by_idx[r["idx"]] = r["pred"].strip().split()

    violins = {}
    gene_counts = {}
    for ct in (C.CT_ORDER if ONLY_CT is None else [ONLY_CT]):
        sub = df[df["ct"] == ct]
        down = panels_by_ct[ct]["Rescued_HO_down"]
        up = panels_by_ct[ct]["Rescued_HO_up"]
        if len(sub) < C.MIN_N or len(down) < MIN_PANEL_GENES or len(up) < MIN_PANEL_GENES:
            print(f"{ct}: skipped (n_cells={len(sub)}, down={len(down)}, up={len(up)})")
            continue
        down_bool = np.array([g in down for g in model.gene_names])
        up_bool = np.array([g in up for g in model.gene_names])
        bool_by_subset = {"bg": ~(down_bool | up_bool), "down": down_bool, "up": up_bool}
        gene_counts[ct] = {"n_genes_down": int(down_bool.sum()), "n_genes_up": int(up_bool.sum())}

        real_src_X = np.stack([real_by_bc[bc] for bc in sub["src_barcode"]])
        real_tgt_X = np.stack([real_by_bc[bc] for bc in sub["tgt_barcode"]])
        pred_X = rem.reconstruct_batch([pred_by_idx[i] for i in sub["idx"]], model, C.K)
        real_src_topk = CR.truncate_topk(real_src_X, C.K)
        real_tgt_topk = CR.truncate_topk(real_tgt_X, C.K)

        d_real, m_real = C.delta_and_mask(real_src_topk, real_tgt_topk)
        d_model, m_model = C.delta_and_mask(real_src_topk, pred_X)
        means_real = per_cell_means(d_real, m_real, bool_by_subset)
        means_model = per_cell_means(d_model, m_model, bool_by_subset)

        violins[ct] = {}
        for s in SUBSETS:
            violins[ct][("real", s)] = means_real[s][~np.isnan(means_real[s])]
            violins[ct][("model", s)] = means_model[s][~np.isnan(means_model[s])]

    if not violins:
        print("  WARN: no cell type passed the gates -- nothing to plot")
        return
    if ONLY_CT is None:
        violins["POOLED"] = {k: np.concatenate([violins[ct][k] for ct in violins]) for k in X_POS}
        gene_counts["POOLED"] = {"n_genes_down": -1, "n_genes_up": -1}

    PS.apply()
    fig, axes, spare = PS.two_per_row(len(violins), ROW_H)
    report_rows = []
    for i, (ax, (name, scores)) in enumerate(zip(axes, violins.items())):
        stats = draw_panel(ax, scores, CT_TITLE.get(name, name), arm["delta_label"],
                           expected, show_legend=False)
        stats.update({"cell_type": name, "arm": arm_key, "gate": gate,
                      "expected_down": expected["down"], "expected_up": expected["up"],
                      **gene_counts[name]})
        report_rows.append(stats)

    fig.suptitle(f"Rat, {arm['short']}", fontsize=PS.FS_SUPTITLE)
    h, lab = legend_handles_labels(expected)
    PS.put_legend(fig, spare, h, lab)

    stem = (f"rat_delta_{arm['slug']}_{GATE_SLUG[gate]}"
            + ("" if ONLY_CT is None else f"_{ONLY_CT.lower()}"))
    fig_path = HERE / "figures" / f"{stem}.png"
    PS.save(fig, fig_path)
    plt.close(fig)
    print(f"-> {fig_path}")

    cols = ["cell_type", "arm", "gate", "n_genes_down", "n_genes_up",
            "expected_down", "expected_up",
            "p_real_down", "p_real_up", "p_model_down", "p_model_up",
            "median_real_bg", "median_real_down", "median_real_up",
            "median_model_bg", "median_model_down", "median_model_up"]
    rep = pd.DataFrame(report_rows)[cols]
    rep_path = HERE / "reports" / f"report_{stem}.csv"
    rep.to_csv(rep_path, index=False)
    print(rep.to_string(index=False))
    print(f"-> {rep_path}")
