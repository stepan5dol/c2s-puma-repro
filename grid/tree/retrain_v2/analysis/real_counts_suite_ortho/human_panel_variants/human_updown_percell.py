#!/usr/bin/env python3
"""
human_updown_percell.py — the human figure built exactly the way the rat one
is: per-CELL paired deltas against TRUE raw counts, using the barcodes
recovered by recover_human_test_barcodes.py.

Mirrors rat_panel_variants/updown_core.py line for line in construction:

  delta_real  = topk(real_tgt cell)  - topk(real_src cell)   -- fully real, paired
  delta_model = pred(reconstructed)  - topk(real_src cell)   -- model vs its OWN input cell

src/tgt are the actual cells behind each test example (human_Acute26_BPD7mo
arm: src = real Acute26 cell, tgt = real BPD7mo cell), not population means.
That is the whole point of the barcode recovery -- until now every human
figure in this suite compared population averages because the pairing was
lost when test.jsonl was written.

PAIRING says whether that holds for the arm being drawn. It is "paired" above,
but build_disease_arms_trajectory.py also drives this module for Term->disease
arms, where no pairing exists (Term cells are absent from the pairing registry)
and the src cell is drawn at random within cell type -- there PAIRING is
"random_within_ct". Both the output name and a report column carry it: the two
drivers build the same CONDITION from different baselines, and before this they
silently overwrote each other's figures.

Six violins per cell type: REAL[bg, down, up] | MODEL[bg, down, up],
background = outside down UNION up, two panel gates (panel_sets_human).

Normalization matches rank_expr_model._fit_from_matrix: log1p of CPM10k with
the library size taken over the FULL transcriptome, then top-800 truncation
of both real sides (pred is <=800-nonzero by construction).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ABS_DIR = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/analysis")
PMA_DIR = ABS_DIR / "pathway_module_analysis"
sys.path.insert(0, str(ABS_DIR))
import common as C
import common_human as CH
import plot_style as PS
import rank_expr_model as rem

sys.path.insert(0, str(Path(__file__).parent.parent))
import common_real as CR

sys.path.insert(0, str(PMA_DIR))
_saved_common = sys.modules.pop("common", None)
import load_data as L
import real_counts as RC
if _saved_common is not None:
    sys.modules["common"] = _saved_common

sys.path.insert(0, str(Path(__file__).parent))
import panel_sets_human as PSH

HERE = Path(__file__).parent
PAIRS_CSV = HERE / "recovered_human_test_pairs.csv"
TEST_INFERENCE_PATH = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/test_inference_results_t10.jsonl")
PAIR_TYPE = "human_Acute26_BPD7mo"
CONDITION = "BPD7mo"
SRC_LABEL = "Acute26"
PAIRING = "paired"

PANEL_CONDITION = "BPD7mo"
SRC_TITLE = "acute preterm lung injury, GW26"
TGT_TITLE = "bronchopulmonary dysplasia, 7 months postnatal"
SRC_SHORT = "ALI GW26"
TGT_SHORT = "BPD 7mo"
SRC_FIG = TGT_FIG = None
SRC_SLUG = "ali_gw26"
TGT_SLUG = "bpd_7mo"
GATE_SLUG = {"full_list": "all_rescued", "gated": "dge_filtered"}
ROW_H = 3.3  # inches per row of two panels (PS.FIG_W wide)
PANEL_TITLE = {"down": "disease-suppressed genes", "up": "disease-induced genes"}
CT_TITLE = {"gCap": "General capillary endothelial cell",
            "aCap": "Aerocyte capillary endothelial cell",
            "Pericyte": "Pericyte",
            "VEC": "Pulmonary venous endothelial cell",
            "POOLED": "All cell types pooled"}
ONLY_CT = None

SUBSETS = ("bg", "down", "up")
X_POS = {("real", "bg"): 0.0, ("real", "down"): 1.0, ("real", "up"): 2.0,
         ("model", "bg"): 3.4, ("model", "down"): 4.4, ("model", "up"): 5.4}
BG_MIN_GENES = 3
MIN_PANEL_GENES = 3


def load_real_counts_by_barcode(barcodes: set[str], model) -> dict[str, np.ndarray]:
    """log1p(CPM10k) per cell, model gene axis, for the given barcodes only.
    Chunked read of the same compiled_counts.csv load_data.load_human_bpd_raw
    uses -- that function does not return barcodes, so the row identity is
    kept here instead of re-deriving it."""
    print("  [real] chunked read of compiled_counts.csv...")
    chunks_X, chunks_bc, genes = [], [], None
    for chunk in pd.read_csv(L.C.BPD_COUNTS_PATH, index_col="id", chunksize=5000):
        if genes is None:
            genes = L.remap_gene_names(chunk.columns.tolist())
        rows_here = [bc for bc in chunk.index if bc in barcodes]
        if rows_here:
            chunks_X.append(chunk.loc[rows_here].values.astype(np.float64))
            chunks_bc.extend(rows_here)
    X = np.vstack(chunks_X)
    missing = barcodes - set(chunks_bc)
    if missing:
        print(f"  WARN: {len(missing)} barcodes absent from counts: {list(missing)[:5]}")

    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    Xn = np.log1p(X / lib * 1e4).astype(np.float32)
    Xn = RC._reindex_to_model_genes(Xn, genes, model.gene_names)
    print(f"  [real] {Xn.shape[0]} cells x {Xn.shape[1]} genes on the model axis")
    return {bc: Xn[i] for i, bc in enumerate(chunks_bc)}


def per_cell_means(delta_X, mask_X, bool_by_subset):
    n = delta_X.shape[0]
    out = {s: np.full(n, np.nan) for s in bool_by_subset}
    for i in range(n):
        m = mask_X[i]
        for s, gene_bool in bool_by_subset.items():
            sel = m & gene_bool
            if sel.sum() >= (BG_MIN_GENES if s == "bg" else 1):
                out[s][i] = delta_X[i, sel].mean()
    return out


def draw_panel(ax, scores, title, show_legend=False):
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
    ax.set_ylim(top=y0 + step * 5.4)
    ax.set_xticks([X_POS[k] for k in keys])
    ax.set_xticklabels(["background", "suppressed", "induced"] * 2,
                       fontsize=PS.FS_TICK, rotation=45, ha="right")
    for x, lbl in ((1.0, "datasets"), (4.4, "model")):
        ax.text(x, ax.get_ylim()[1], lbl, ha="center", va="bottom", fontsize=PS.FS_ANNOT,
                color="#666666")
    ax.set_title(f"{title}\nn = {len(scores[('real','down')])} cells",
                 fontsize=PS.FS_TITLE, pad=16)
    ax.set_ylabel(f"mean Δ per cell\n({TGT_SHORT} − {SRC_SHORT})", fontsize=PS.FS_LABEL)
    return stats


def legend_handles_labels():
    """Built once and placed on the FIGURE, under the axes. Inside the axes it
    sat on top of the left-hand violin; below the axes it cannot collide, and
    when these panels are assembled into one main figure the per-panel legends
    are dropped in favour of a single one."""
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=PS.HO_COLOR, edgecolor="#333333", alpha=0.85),
         plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", edgecolor="#333333", alpha=0.85),
         plt.Rectangle((0, 0), 1, 1, facecolor="#bbbbbb", edgecolor="#333333", alpha=0.85, hatch="///")]
    return h, ["background", PANEL_TITLE["down"], PANEL_TITLE["up"]]


def run(gate, df, real_by_bc, pred_by_idx, model, ct_map):
    panels_by_ct = PSH.load_panels(gate, condition=PANEL_CONDITION)
    violins, report_rows = {}, []

    for ct in (PSH.CT_ORDER if ONLY_CT is None else [ONLY_CT]):
        sub = df[df["ct"] == ct]
        down = panels_by_ct[ct]["Rescued_HO_down"]
        up = panels_by_ct[ct]["Rescued_HO_up"]
        if len(sub) < CH.MIN_N or len(down) < MIN_PANEL_GENES or len(up) < MIN_PANEL_GENES:
            print(f"  {ct}: skipped (n_cells={len(sub)}, down={len(down)}, up={len(up)})")
            continue
        down_bool = np.array([g in down for g in model.gene_names])
        up_bool = np.array([g in up for g in model.gene_names])
        by_subset = {"bg": ~(down_bool | up_bool), "down": down_bool, "up": up_bool}

        real_src = np.stack([real_by_bc[bc] for bc in sub["src_barcode"]])
        real_tgt = np.stack([real_by_bc[bc] for bc in sub["tgt_barcode"]])
        pred_X = rem.reconstruct_batch([pred_by_idx[i] for i in sub["idx"]], model, CR.K)
        src_t8 = CR.truncate_topk(real_src, CR.K)
        tgt_t8 = CR.truncate_topk(real_tgt, CR.K)

        d_real, m_real = C.delta_and_mask(src_t8, tgt_t8)
        d_model, m_model = C.delta_and_mask(src_t8, pred_X)
        mr = per_cell_means(d_real, m_real, by_subset)
        mm = per_cell_means(d_model, m_model, by_subset)

        violins[ct] = {}
        for s in SUBSETS:
            violins[ct][("real", s)] = mr[s][~np.isnan(mr[s])]
            violins[ct][("model", s)] = mm[s][~np.isnan(mm[s])]

    if not violins:
        print(f"  WARN: no cell type usable for gate={gate} -- skipping plot")
        return
    if ONLY_CT is None:
        violins["POOLED"] = {k: np.concatenate([violins[ct][k] for ct in violins]) for k in X_POS}

    PS.apply()
    fig, axes, spare = PS.two_per_row(len(violins), ROW_H)
    for i, (ax, (name, scores)) in enumerate(zip(axes, violins.items())):
        st = draw_panel(ax, scores, CT_TITLE.get(name, name), show_legend=False)
        st.update({"cell_type": name, "gate": gate,
                   "source": SRC_TITLE, "target": TGT_TITLE,
                   "src_label": SRC_LABEL, "pairing": PAIRING})
        report_rows.append(st)

    fig.suptitle(f"Human, {SRC_FIG or SRC_SHORT} vs {TGT_FIG or TGT_SHORT}", fontsize=PS.FS_SUPTITLE)
    h, lab = legend_handles_labels()
    PS.put_legend(fig, spare, h, lab)

    stem = (f"human_delta_{SRC_SLUG}_to_{TGT_SLUG}_{GATE_SLUG[gate]}"
            + ("" if ONLY_CT is None else f"_{ONLY_CT.lower()}"))
    fig_path = HERE / "figures" / f"{stem}.png"
    PS.save(fig, fig_path)
    plt.close(fig)
    print(f"-> {fig_path}")

    cols = ["cell_type", "source", "target", "src_label", "pairing", "gate",
            "p_real_down", "p_real_up", "p_model_down", "p_model_up",
            "median_real_bg", "median_real_down", "median_real_up",
            "median_model_bg", "median_model_down", "median_model_up"]
    rep = pd.DataFrame(report_rows)[cols]
    rep_path = HERE / "reports" / f"report_{stem}.csv"
    rep.to_csv(rep_path, index=False)
    print(rep.to_string(index=False))
    print(f"-> {rep_path}")


def main():
    df = pd.read_csv(PAIRS_CSV)
    df = df[(df["pair_type"] == PAIR_TYPE) & df["verified"]].copy()
    print(f"[pairs] {len(df)} verified {PAIR_TYPE} pairs")
    print("[pairs] cell_type values:", sorted(df["cell_type"].unique()))

    ct_map = {"general capillary endothelial cell (endothelial)": "gCap",
              "aerocyte capillary endothelial cell (endothelial)": "aCap",
              "pericyte (mural)": "Pericyte",
              "vascular endothelial cell (endothelial)": "VEC",
              "pulmonary venous endothelial cell (endothelial)": "VEC",
              "gCap": "gCap",
              "aCap": "aCap",
              "Pericyte": "Pericyte",
              "Pulmonary venous EC": "VEC"}
    df["ct"] = df["cell_type"].map(ct_map)
    unmapped = sorted(df.loc[df["ct"].isna(), "cell_type"].unique())
    assert not unmapped, f"unmapped cell types: {unmapped}"
    print(df.groupby("ct").size().to_string())

    print("[calibration] fitting rank->expr model...")
    model = rem.fit_from_csv(CH.COUNTS_CSV)
    print(f"  r2={model.r2:.4f}")

    bcs = set(df["src_barcode"]) | set(df["tgt_barcode"])
    real_by_bc = load_real_counts_by_barcode(bcs, model)

    idx_set = set(df["idx"])
    pred_by_idx = {}
    with open(TEST_INFERENCE_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["idx"] in idx_set:
                pred_by_idx[r["idx"]] = r["pred"].strip().split()
    print(f"[pred] {len(pred_by_idx)} predictions loaded")

    for gate in PSH.GATES:
        print(f"\n{'=' * 78}\n=== gate={gate}\n{'=' * 78}")
        run(gate, df, real_by_bc, pred_by_idx, model, ct_map)


if __name__ == "__main__":
    main()
