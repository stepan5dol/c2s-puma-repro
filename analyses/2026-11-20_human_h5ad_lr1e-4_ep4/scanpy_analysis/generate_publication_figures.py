#!/usr/bin/env python3
"""
Reproducible publication-grade figure suite for the standard scRNA-seq analysis
of human_Acute26, human_BPD7mo, and human_BPDPH7mo (.h5ad real + model cells).

Follows the `dataviz` skill standards (Ocean Dusk palette, 300 DPI PNG + vector PDF,
despined top/right axes, clean typography, statistical boxes with Spearman rho / Pearson r / p-values)
and uses the processed outputs from `run_standard_scrna_analysis.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

BASE_DIR = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/2026-11-20_human_h5ad_newgrid_lr1e-4_ep4_s42")
ANA_DIR = BASE_DIR / "scanpy_analysis"
H5AD_DIR = ANA_DIR / "processed_h5ad"
CONTRAST_DIR = ANA_DIR / "contrasts"
FIG_DIR = ANA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Dataviz Skill Global Styling & Ocean Dusk Palette
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.labelsize": 10.5,
    "axes.labelweight": "medium",
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.edgecolor": "#666666",
    "axes.linewidth": 0.8,
    "legend.fontsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE_ARM = {
    "CTRL": "#2A9D8F",         # Teal-green (healthy control)
    "DISEASE": "#E76F51",      # Warm coral (real pathology)
    "MODEL_NOAZI": "#F4A261",  # Sandy orange (reconstructed disease, no drug)
    "MODEL_AZI": "#264653",    # Deep indigo (reconstructed + Azithromycin)
}

PALETTE_CT = {
    "gCap": "#264653",      # Deep indigo
    "aCap": "#2A9D8F",      # Teal-green
    "VEC": "#F4A261",       # Sandy orange
    "Pericyte": "#E76F51",  # Warm coral
}

# Ocean Dusk extended distinct cluster palette (colorblind-safe, non-neon)
CLUSTER_COLORS = [
    "#264653", "#2A9D8F", "#E76F51", "#F4A261", "#E9C46A",
    "#457B9D", "#8D99AE", "#6D597A", "#B56576", "#52796F",
    "#355070", "#E56B6F", "#84A59D", "#BC6C25",
]

COHORTS = ["Acute26", "BPD7mo", "BPDPH7mo"]
COHORT_LABELS = {
    "Acute26": "Acute26 (GW24 He22 Ctrl vs Acute Preterm Injury)",
    "BPD7mo": "BPD 7mo (Term Ctrl vs Established BPD)",
    "BPDPH7mo": "BPD+PH 7mo (Term Ctrl vs BPD + Pulmonary Hypertension)",
}
CTS = ["gCap", "aCap", "Pericyte", "VEC"]
ARMS = ["CTRL", "DISEASE", "MODEL_NOAZI", "MODEL_AZI"]


def style_axis(ax: plt.Axes, grid_axis: str | None = "both") -> None:
    """Apply dataviz skill despining and subtle grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.15, color="#cccccc", zorder=0)
    else:
        ax.grid(False)


def save_fig(fig: plt.Figure, stem: str) -> None:
    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path.name} & {pdf_path.name}", flush=True)


def fit_line_with_bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 250, seed: int = 42):
    """Fit OLS trend line with 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(np.min(x), np.max(x), 100)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x_grid + intercept

    n = len(x)
    boot_preds = np.empty((n_boot, len(x_grid)), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.std(x[idx]) < 1e-8:
            boot_preds[b] = y_hat
        else:
            s_b, i_b = np.polyfit(x[idx], y[idx], 1)
            boot_preds[b] = s_b * x_grid + i_b
    ci_low = np.percentile(boot_preds, 2.5, axis=0)
    ci_high = np.percentile(boot_preds, 97.5, axis=0)
    return x_grid, y_hat, ci_low, ci_high, slope


# =============================================================================
# FIGURE 1: Multi-Cohort UMAP Atlas & Co-Clustering (3 cohorts x 3 views)
# =============================================================================
def build_figure1_umap_atlas(adatas: dict[str, ad.AnnData], summary: dict) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 12.5))
    fig.subplots_adjust(hspace=0.30, wspace=0.22)

    for row_idx, cohort in enumerate(COHORTS):
        adata = adatas[cohort]
        umap = adata.obsm["X_umap"]
        obs = adata.obs

        # --- Column 0: Cell Type ---
        ax0 = axes[row_idx, 0]
        style_axis(ax0, grid_axis=None)
        for ct in CTS:
            mask = (obs["cell_type"] == ct).values
            ax0.scatter(
                umap[mask, 0], umap[mask, 1],
                c=PALETTE_CT[ct], s=7, alpha=0.65, edgecolors="none", rasterized=True, label=ct
            )
            if mask.sum() > 10:
                cx, cy = np.median(umap[mask, 0]), np.median(umap[mask, 1])
                txt = ax0.text(
                    cx, cy, ct, fontsize=9.0, fontweight="bold", color="#1D3557",
                    ha="center", va="center", zorder=10
                )
                txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

        ax0.set_title(f"{cohort}: Vascular Cell Types (n={adata.n_obs:,})", pad=6)
        ax0.set_xlabel("UMAP 1")
        ax0.set_ylabel(f"{COHORT_LABELS[cohort].split(' (')[0]}\nUMAP 2")
        ax0.set_xticks([])
        ax0.set_yticks([])

        # --- Column 1: Experimental Arm (Real background, Model foreground) ---
        ax1 = axes[row_idx, 1]
        style_axis(ax1, grid_axis=None)
        for arm in ["CTRL", "DISEASE"]:
            mask = (obs["orig.ident"] == arm).values
            ax1.scatter(
                umap[mask, 0], umap[mask, 1],
                c=PALETTE_ARM[arm], s=6, alpha=0.42, edgecolors="none", rasterized=True,
                label=f"{arm} (n={mask.sum():,})", zorder=2
            )
        for arm in ["MODEL_NOAZI", "MODEL_AZI"]:
            mask = (obs["orig.ident"] == arm).values
            ax1.scatter(
                umap[mask, 0], umap[mask, 1],
                c=PALETTE_ARM[arm], s=18, alpha=0.88, edgecolors="white", linewidths=0.35,
                rasterized=True, label=f"{arm} (n={mask.sum():,})", zorder=5
            )
        ax1.set_title(f"{cohort}: Real vs Reconstructed Model Arms", pad=6)
        ax1.set_xlabel("UMAP 1")
        ax1.set_ylabel("UMAP 2")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.legend(loc="lower left", frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC",
                   framealpha=0.92, markerscale=1.4, borderpad=0.35, handletextpad=0.3)

        # --- Column 2: Leiden 0.5 Clusters ---
        ax2 = axes[row_idx, 2]
        style_axis(ax2, grid_axis=None)
        clusters = sorted(obs["leiden_0.5"].unique(), key=lambda x: int(x))
        for c_id in clusters:
            c_idx = int(c_id) % len(CLUSTER_COLORS)
            mask = (obs["leiden_0.5"] == c_id).values
            ax2.scatter(
                umap[mask, 0], umap[mask, 1],
                c=CLUSTER_COLORS[c_idx], s=7, alpha=0.65, edgecolors="none", rasterized=True
            )
            if mask.sum() > 15:
                cx, cy = np.median(umap[mask, 0]), np.median(umap[mask, 1])
                txt = ax2.text(
                    cx, cy, f"C{c_id}", fontsize=8.5, fontweight="bold", color="#111111",
                    ha="center", va="center", zorder=10
                )
                txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

        c_stats = summary[cohort]["step4_clustering"]["resolutions"]["0.5"]
        badge = (
            f"Leiden res=0.5 ({c_stats['n_clusters']} clusters)\n"
            f"NMI(cell_type) = {c_stats['NMI_with_cell_type']:.3f}\n"
            f"ARI(cell_type) = {c_stats['ARI_with_cell_type']:.3f}"
        )
        ax2.text(
            0.03, 0.96, badge, transform=ax2.transAxes, fontsize=8.2, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.92)
        )
        ax2.set_title(f"{cohort}: Leiden Clustering (res=0.5)", pad=6)
        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")
        ax2.set_xticks([])
        ax2.set_yticks([])

    # Top shared legend for Cell Types
    ct_handles = [
        Line2D([0], [0], marker="o", color="w", label=ct, markerfacecolor=PALETTE_CT[ct], markersize=8)
        for ct in CTS
    ]
    fig.legend(
        handles=ct_handles, loc="upper center", ncol=4, frameon=True,
        facecolor="#F8F9FA", edgecolor="#CCCCCC", bbox_to_anchor=(0.22, 0.935),
        title="Vascular Cell Types", title_fontsize=9.0
    )
    fig.suptitle(
        "Figure 1. Standard Scanpy UMAP Atlas: Cell-Type Identity & Co-Clustering of Real and Reconstructed Model Cells",
        fontsize=13.0, fontweight="bold", y=0.975
    )
    save_fig(fig, "fig1_umap_atlas_coclustering")


# =============================================================================
# FIGURE 2: Pathology Fidelity Scatter Grid (3 cohorts x 4 cell types)
# =============================================================================
def build_figure2_pathology_fidelity() -> None:
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 11.2))
    fig.subplots_adjust(hspace=0.36, wspace=0.26)

    HIGHLIGHT_GENES = {
        "gCap": ["FCN3", "CD36", "IL7R", "EPAS1", "BTNL9", "COL1A1", "GDF15"],
        "aCap": ["EDNRB", "HPGD", "SOSTDC1", "S100A4", "EMCN", "GDF15"],
        "Pericyte": ["PDGFRB", "COL1A2", "COL1A1", "NOTCH3", "ACTA2", "TIMP1"],
        "VEC": ["ACKR1", "CLU", "CPE", "VWF", "SELE", "PTGDS"],
    }

    for r_idx, cohort in enumerate(COHORTS):
        df = pd.read_csv(CONTRAST_DIR / f"{cohort}_within_ct_contrasts.csv")
        for c_idx, ct in enumerate(CTS):
            ax = axes[r_idx, c_idx]
            style_axis(ax, grid_axis="both")
            sub = df[df["cell_type"] == ct].dropna(subset=["score_dis_vs_ctrl", "score_noazi_vs_ctrl"]).copy()

            x = sub["score_dis_vs_ctrl"].values
            y = sub["score_noazi_vs_ctrl"].values

            # Clip extreme z-scores at +/-35 for clean visual display while computing stats on exact values
            rho, p_rho = spearmanr(x, y)
            r_p, _ = pearsonr(x, y)
            x_plot = np.clip(x, -35, 35)
            y_plot = np.clip(y, -25, 25)

            ax.axhline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)
            ax.axvline(0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

            ax.scatter(
                x_plot, y_plot, c=PALETTE_CT[ct], s=9, alpha=0.45,
                edgecolors="white", linewidths=0.2, rasterized=True, zorder=2
            )

            xg, yh, cil, cih, slope = fit_line_with_bootstrap_ci(x_plot, y_plot, n_boot=200)
            ax.fill_between(xg, cil, cih, color="#1D3557", alpha=0.18, zorder=3)
            ax.plot(xg, yh, color="#1D3557", linewidth=1.6, zorder=4)

            # Highlight top outlier / canonical genes without overlapping
            sub["abs_prod"] = np.abs(x_plot * y_plot)
            top_genes = sub.sort_values("abs_prod", ascending=False).head(3)["gene"].tolist()
            for g in HIGHLIGHT_GENES[ct]:
                if g in sub["gene"].values and g not in top_genes and len(top_genes) < 5:
                    top_genes.append(g)

            for g in top_genes[:4]:
                row_g = sub[sub["gene"] == g].iloc[0]
                gx = float(np.clip(row_g["score_dis_vs_ctrl"], -35, 35))
                gy = float(np.clip(row_g["score_noazi_vs_ctrl"], -25, 25))
                ax.scatter([gx], [gy], c="#E76F51", s=24, edgecolors="#1D3557", linewidths=0.7, zorder=6)
                txt = ax.text(
                    gx + 0.8, gy + 0.5, g, fontsize=7.5, fontweight="bold",
                    color="#1D3557", zorder=7
                )
                txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

            p_str = "p < 1e-100" if p_rho < 1e-100 else f"p = {p_rho:.1e}"
            stat_box = (
                f"Spearman ρ = {rho:.3f}\n"
                f"Pearson r = {r_p:.3f}\n"
                f"{p_str} (n={len(sub):,})"
            )
            ax.text(
                0.04, 0.95, stat_box, transform=ax.transAxes, fontsize=8.0,
                va="top", ha="left", zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.92)
            )

            ax.set_title(f"{cohort} — {ct}", pad=5)
            if r_idx == 2:
                ax.set_xlabel("Real Pathology Shift: z(DISEASE vs CTRL)")
            else:
                ax.set_xlabel("")
            if c_idx == 0:
                ax.set_ylabel(f"{cohort}\nModel Shift: z(NOAZI vs CTRL)")
            else:
                ax.set_ylabel("")

    fig.suptitle(
        "Figure 2. Pathology Fidelity Across Vascular Cell Types: Real Disease Shift vs Reconstructed MODEL_NOAZI Shift",
        fontsize=13.0, fontweight="bold", y=0.98
    )
    save_fig(fig, "fig2_pathology_fidelity_grid")


# =============================================================================
# FIGURE 3: Polygenic Reversal Under Azithromycin & 40D PCA Axis Shift
# =============================================================================
def build_figure3_azi_reversal(adatas: dict[str, ad.AnnData], summary: dict) -> None:
    fig = plt.figure(figsize=(15.5, 10.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.28)

    # --- Top row (Panels A1-A4): Polygenic reversal scatter plots on representative cohort x cell type ---
    exemplars = [
        ("Acute26", "VEC", "A1. Acute26 — VEC"),
        ("Acute26", "Pericyte", "A2. Acute26 — Pericyte"),
        ("BPD7mo", "gCap", "A3. BPD7mo — gCap"),
        ("BPDPH7mo", "aCap", "A4. BPDPH7mo — aCap"),
    ]
    for idx, (cohort, ct, title) in enumerate(exemplars):
        ax = fig.add_subplot(gs[0, idx])
        style_axis(ax, grid_axis="both")
        df = pd.read_csv(CONTRAST_DIR / f"{cohort}_within_ct_contrasts.csv")
        sub = df[df["cell_type"] == ct].copy()
        mod_degs = sub[(sub["padj_noazi_vs_ctrl"] < 0.05) & (sub["score_noazi_vs_ctrl"].abs() >= 3.0)].copy()

        x = mod_degs["score_noazi_vs_ctrl"].values
        y = mod_degs["score_azi_vs_noazi"].values
        rho, p_val = spearmanr(x, y)
        r_p, _ = pearsonr(x, y)
        rev_mask = np.sign(x) != np.sign(y)
        rev_pct = float(rev_mask.mean() * 100.0)

        ax.axhline(0, color="#666666", linestyle="--", linewidth=0.8, zorder=1)
        ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8, zorder=1)

        # Shade reversal quadrants (Q2: x<0, y>0 and Q4: x>0, y<0)
        xlim = (float(np.min(x) * 1.12), float(np.max(x) * 1.12))
        ylim = (float(np.min(y) * 1.18), float(np.max(y) * 1.25))
        ax.axvspan(xlim[0], 0, ymin=0.5, ymax=1.0, color="#2A9D8F", alpha=0.06, zorder=0)
        ax.axvspan(0, xlim[1], ymin=0.0, ymax=0.5, color="#2A9D8F", alpha=0.06, zorder=0)

        ax.scatter(
            x[~rev_mask], y[~rev_mask], c="#8D99AE", s=11, alpha=0.45,
            edgecolors="none", label="Non-reversed", rasterized=True, zorder=2
        )
        ax.scatter(
            x[rev_mask], y[rev_mask], c="#2A9D8F", s=14, alpha=0.70,
            edgecolors="white", linewidths=0.25, label=f"Reversed ({rev_pct:.1f}%)",
            rasterized=True, zorder=3
        )

        xg, yh, cil, cih, _ = fit_line_with_bootstrap_ci(x, y, n_boot=200)
        ax.fill_between(xg, cil, cih, color="#E76F51", alpha=0.20, zorder=4)
        ax.plot(xg, yh, color="#E76F51", linewidth=1.8, zorder=5)

        # Annotate top 4 reversed genes with largest |y|
        rev_df = mod_degs[rev_mask].copy()
        rev_df["abs_y"] = rev_df["score_azi_vs_noazi"].abs()
        for _, r_g in rev_df.sort_values("abs_y", ascending=False).head(4).iterrows():
            gx, gy = float(r_g["score_noazi_vs_ctrl"]), float(r_g["score_azi_vs_noazi"])
            ax.scatter([gx], [gy], c="#E76F51", s=26, edgecolors="#1D3557", linewidths=0.7, zorder=6)
            txt = ax.text(gx, gy + 0.08 * (ylim[1] - ylim[0]) * np.sign(gy if gy != 0 else 1),
                          str(r_g["gene"]), fontsize=7.5, fontweight="bold", color="#1D3557",
                          ha="center", zorder=7)
            txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

        p_str = f"p = {p_val:.1e}" if p_val >= 1e-99 else "p < 1e-99"
        box_txt = (
            f"Reversal = {rev_pct:.1f}% ({rev_mask.sum()}/{len(x)})\n"
            f"Spearman ρ = {rho:.3f} ({p_str})\n"
            f"Pearson r = {r_p:.3f}"
        )
        ax.text(
            0.04, 0.96, box_txt, transform=ax.transAxes, fontsize=7.8, va="top", ha="left", zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.93)
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title, pad=6)
        ax.set_xlabel("Pathology Shift: z(MODEL_NOAZI vs CTRL)")
        if idx == 0:
            ax.set_ylabel("Azithromycin Shift:\nz(MODEL_AZI vs MODEL_NOAZI)")

    # --- Bottom Left (Panel B, spanning 2 cols): Summary of Reversal % across all 12 Cohort x Cell Type pairs ---
    ax_b = fig.add_subplot(gs[1, :2])
    style_axis(ax_b, grid_axis="y")
    x_pos = np.arange(len(CTS))
    width = 0.24
    cohort_colors = {"Acute26": "#264653", "BPD7mo": "#2A9D8F", "BPDPH7mo": "#E76F51"}

    ax_b.axhline(50.0, color="#666666", linestyle="--", linewidth=1.0, label="50% random baseline", zorder=1)
    for i_c, cohort in enumerate(COHORTS):
        vals = []
        rhos = []
        for ct in CTS:
            st = summary[cohort]["step5_markers_and_contrasts"]["within_ct_contrasts"][ct]
            vals.append(st["reversal_rate_on_model_path_degs_pct"])
            rhos.append(st["reversal_spearman_rho_on_model_path_degs"])
        offset = (i_c - 1) * width
        bars = ax_b.bar(
            x_pos + offset, vals, width=width, color=cohort_colors[cohort],
            edgecolor="white", linewidth=0.7, alpha=0.90, label=cohort, zorder=3
        )
        for bar, v, rho_v in zip(bars, vals, rhos):
            ax_b.text(
                bar.get_x() + bar.get_width() / 2.0, v + 1.2,
                f"{v:.1f}%\n(ρ={rho_v:.2f})",
                ha="center", va="bottom", fontsize=7.2, color="#222222", fontweight="medium"
            )

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(CTS, fontweight="bold")
    ax_b.set_ylim(0, 112)
    ax_b.set_ylabel("Pathology DEGs Shifted Toward CTRL (%)")
    ax_b.set_title("B. Polygenic Pathology Reversal Rate Under Azithromycin Across All 12 Cohort × Cell-Type Pairs", pad=8)
    ax_b.legend(loc="upper left", ncol=4, frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC")

    # --- Bottom Right (Panel C, spanning 2 cols): 40D PCA Projection Shift along DISEASE (0) -> CTRL (1) Axis ---
    ax_c = fig.add_subplot(gs[1, 2:])
    style_axis(ax_c, grid_axis="x")

    pairs_to_show = [
        ("BPDPH7mo", "gCap"),
        ("BPDPH7mo", "aCap"),
        ("BPDPH7mo", "VEC"),
        ("BPD7mo", "gCap"),
        ("BPD7mo", "Pericyte"),
        ("BPD7mo", "VEC"),
        ("Acute26", "aCap"),
        ("Acute26", "VEC"),
    ]
    y_ticks = np.arange(len(pairs_to_show))[::-1]
    labels_c = []

    for y_pos, (cohort, ct) in zip(y_ticks, pairs_to_show):
        st = summary[cohort]["step3_dimred"]["pca_disease_to_ctrl_shift_by_ct"][ct]
        p_noazi = st["axis_pos_noazi_0dis_1ctrl"]
        p_azi = st["axis_pos_azi_0dis_1ctrl"]
        pval = st["shift_mannwhitney_p"]
        delta = st["delta_shift_toward_ctrl"]

        # Connecting arrow/segment from MODEL_NOAZI to MODEL_AZI
        ax_c.plot([p_noazi, p_azi], [y_pos, y_pos], color="#1D3557", linewidth=2.0, zorder=2)
        ax_c.scatter([p_noazi], [y_pos], color=PALETTE_ARM["MODEL_NOAZI"], s=70, edgecolors="#1D3557",
                     linewidths=0.8, zorder=4, label="MODEL_NOAZI" if y_pos == y_ticks[0] else "")
        ax_c.scatter([p_azi], [y_pos], color=PALETTE_ARM["MODEL_AZI"], s=70, marker="D", edgecolors="white",
                     linewidths=0.8, zorder=5, label="MODEL_AZI" if y_pos == y_ticks[0] else "")

        sig_tag = f"p={pval:.3f}*" if pval < 0.05 else (f"p={pval:.3f}" if pval < 0.15 else f"Δ=+{delta:.3f}")
        ax_c.text(
            max(p_noazi, p_azi) + 0.015, y_pos,
            f"+{delta:.3f} ({sig_tag})",
            va="center", ha="left", fontsize=8.0,
            fontweight="bold" if pval < 0.05 else "normal",
            color="#264653" if pval < 0.05 else "#444444"
        )
        labels_c.append(f"{cohort} — {ct}")

    ax_c.axvline(0.0, color=PALETTE_ARM["DISEASE"], linestyle="--", linewidth=1.1, label="Real DISEASE centroid (0.0)")
    ax_c.set_yticks(y_ticks)
    ax_c.set_yticklabels(labels_c, fontweight="medium")
    ax_c.set_xlim(-0.04, 0.62)
    ax_c.set_xlabel("Projection on 40D PCA Axis: Real DISEASE (0.0) → Healthy CTRL (1.0)")
    ax_c.set_title("C. Latent 40D PCA Shift of MODEL_AZI vs MODEL_NOAZI Toward Healthy CTRL", pad=8)
    ax_c.legend(loc="lower right", frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC")

    fig.suptitle(
        "Figure 3. Therapeutic Transcriptomic Shift of Azithromycin (MODEL_AZI vs MODEL_NOAZI)",
        fontsize=13.0, fontweight="bold", y=0.98
    )
    save_fig(fig, "fig3_azithromycin_reversal_and_pca_shift")


# =============================================================================
# FIGURE 4: Canonical Vascular Marker DotPlot by Cell Type x Arm (1 x 3 cohorts)
# =============================================================================
def build_figure4_canonical_dotplot(adatas: dict[str, ad.AnnData]) -> None:
    marker_panel = [
        # gCap
        "CD36", "FCN3", "BTNL9", "IL7R",
        # aCap
        "EDNRB", "HPGD", "SOSTDC1", "S100A4",
        # VEC
        "ACKR1", "CLU", "CPE", "MMRN1",
        # Pericyte
        "PDGFRB", "NOTCH3", "COL1A2", "COX4I2",
    ]
    # Custom Ocean Dusk colormap: light neutral -> teal -> deep indigo
    cmap_dusk = mpl.colors.LinearSegmentedColormap.from_list(
        "ocean_dusk_seq", ["#F4F6F7", "#A8DADC", "#2A9D8F", "#1D3557", "#264653"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.8), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    row_groups = [(ct, arm) for ct in CTS for arm in ARMS]
    row_labels = [f"{ct} | {arm}" for ct, arm in row_groups]

    for idx, cohort in enumerate(COHORTS):
        ax = axes[idx]
        style_axis(ax, grid_axis="both")
        adata = adatas[cohort]
        raw_X = adata.raw[:, marker_panel].X
        if sp.issparse(raw_X):
            raw_X = raw_X.toarray()

        n_rows = len(row_groups)
        n_cols = len(marker_panel)
        mean_mat = np.zeros((n_rows, n_cols), dtype=np.float32)
        frac_mat = np.zeros((n_rows, n_cols), dtype=np.float32)

        for r_i, (ct, arm) in enumerate(row_groups):
            mask = ((adata.obs["cell_type"] == ct) & (adata.obs["orig.ident"] == arm)).values
            if mask.sum() > 0:
                sub_x = raw_X[mask]
                mean_mat[r_i] = sub_x.mean(axis=0)
                frac_mat[r_i] = (sub_x > 0).mean(axis=0) * 100.0

        # Normalize mean expression per gene (0 to 1 across rows) for clean visual contrast
        col_max = np.maximum(mean_mat.max(axis=0, keepdims=True), 1e-5)
        norm_mean = mean_mat / col_max

        xx, yy = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
        sizes = np.clip(frac_mat * 1.65, 4.0, 165.0)
        sc_plot = ax.scatter(
            xx.ravel(), yy.ravel(),
            s=sizes.ravel(), c=norm_mean.ravel(),
            cmap=cmap_dusk, vmin=0.0, vmax=1.0,
            edgecolors="#444444", linewidths=0.4, zorder=3
        )

        # Horizontal dividers between cell types and vertical dividers between marker blocks
        for sep in [3.5, 7.5, 11.5]:
            ax.axhline(sep, color="#888888", linestyle="-", linewidth=0.8, alpha=0.6, zorder=2)
            ax.axvline(sep, color="#888888", linestyle="--", linewidth=0.7, alpha=0.5, zorder=2)

        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(marker_panel, rotation=45, ha="right", fontweight="medium")
        ax.set_xlim(-0.7, n_cols - 0.3)
        ax.set_ylim(n_rows - 0.4, -0.8)
        ax.set_title(f"{cohort} (n={adata.n_obs:,})", pad=10)

        # Block headers for marker lineage
        for b_idx, b_name in enumerate(["gCap markers", "aCap markers", "VEC markers", "Pericyte markers"]):
            ax.text(b_idx * 4 + 1.5, -0.95, b_name, ha="center", va="bottom",
                    fontsize=8.2, fontweight="bold", color=PALETTE_CT[CTS[b_idx]])

        if idx == 0:
            ax.set_yticks(np.arange(n_rows))
            ax.set_yticklabels(row_labels, fontsize=8.5)
            for tick_i, (ct, arm) in enumerate(row_groups):
                ax.get_yticklabels()[tick_i].set_color(PALETTE_CT[ct])

    # Colorbar & Dot Size Legend
    cbar_ax = fig.add_axes([0.92, 0.25, 0.012, 0.45])
    cbar = fig.colorbar(sc_plot, cax=cbar_ax)
    cbar.set_label("Relative Mean Expression\nlog1p(CPM10k) [0–1 scaled]", fontsize=9.0)

    size_handles = [
        plt.scatter([], [], s=pct * 1.65, c="#2A9D8F", edgecolors="#444444", linewidths=0.5, label=f"{pct}%")
        for pct in [25, 50, 75, 100]
    ]
    fig.legend(
        handles=size_handles, title="Cells Expressing (%)", loc="upper left",
        bbox_to_anchor=(0.915, 0.86), frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC"
    )

    fig.suptitle(
        "Figure 4. Canonical Vascular Lineage Marker Expression Across Real (CTRL, DISEASE) and Model (NOAZI, AZI) Arms",
        fontsize=13.0, fontweight="bold", y=0.98
    )
    save_fig(fig, "fig4_canonical_markers_dotplot")


# =============================================================================
# FIGURE 5: QC Metrics & Functional Pathway Module Profiles (2 x 3 panels)
# =============================================================================
def build_figure5_qc_and_modules(adatas: dict[str, ad.AnnData], summary: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8))
    fig.subplots_adjust(hspace=0.38, wspace=0.25)

    # --- Top row: Library complexity (n_genes_nonzero_full) & MT% across arms ---
    for idx, cohort in enumerate(COHORTS):
        ax = axes[0, idx]
        style_axis(ax, grid_axis="y")
        obs = adatas[cohort].obs

        data_genes = [obs.loc[obs["orig.ident"] == arm, "n_genes_nonzero_full"].values for arm in ARMS]
        bp = ax.boxplot(
            data_genes, positions=np.arange(len(ARMS)), widths=0.52,
            patch_artist=True, showfliers=False, medianprops=dict(color="#111111", linewidth=1.6),
            boxprops=dict(linewidth=0.8, edgecolor="#333333"),
            whiskerprops=dict(linewidth=0.8, color="#555555"),
            capprops=dict(linewidth=0.8, color="#555555"),
        )
        for patch, arm in zip(bp["boxes"], ARMS):
            patch.set_facecolor(PALETTE_ARM[arm])
            patch.set_alpha(0.82)

        # Annotate medians and MT% on top of each box
        for i_a, arm in enumerate(ARMS):
            med_g = float(np.median(data_genes[i_a]))
            med_mt = summary[cohort]["step1_qc"]["by_arm"][arm]["pct_mt_top2000_median"]
            ax.text(
                i_a, np.percentile(data_genes[i_a], 75) + 120,
                f"{med_g:.0f} genes\n(MT {med_mt:.1f}%)",
                ha="center", va="bottom", fontsize=7.6, fontweight="medium", color="#222222"
            )

        ax.set_xticks(np.arange(len(ARMS)))
        ax.set_xticklabels(ARMS, rotation=15, ha="right", fontweight="bold")
        ax.set_ylim(0, 4400)
        ax.set_title(f"A{idx+1}. {cohort}: Library Complexity & MT%", pad=6)
        if idx == 0:
            ax.set_ylabel("Detected Genes per Cell\n(n_genes_nonzero_full)")

    # --- Bottom row: Functional Gene Signature Scores (mean +/- 95% CI) ---
    mod_names = ["ECM_Fibrosis", "Inflammation_IFN", "Angiogenesis_Vascular", "Hypoxia_Stress"]
    mod_labels = ["ECM / Fibrosis", "Inflammation / IFN", "Angiogenesis", "Hypoxia / Stress"]
    x_pos = np.arange(len(mod_names))
    width = 0.19

    for idx, cohort in enumerate(COHORTS):
        ax = axes[1, idx]
        style_axis(ax, grid_axis="y")
        obs = adatas[cohort].obs
        ax.axhline(0.0, color="#888888", linestyle=":", linewidth=0.8, zorder=1)

        for i_a, arm in enumerate(ARMS):
            sub_obs = obs[obs["orig.ident"] == arm]
            means = [float(sub_obs[f"module_{m}"].mean()) for m in mod_names]
            sems = [1.96 * float(sub_obs[f"module_{m}"].sem()) for m in mod_names]
            offset = (i_a - 1.5) * width
            ax.bar(
                x_pos + offset, means, width=width, yerr=sems,
                color=PALETTE_ARM[arm], edgecolor="white", linewidth=0.6,
                alpha=0.88, capsize=2.0, error_kw=dict(lw=0.8, ecolor="#222222"),
                label=arm, zorder=3
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(mod_labels, rotation=15, ha="right", fontweight="medium")
        ax.set_title(f"B{idx+1}. {cohort}: Functional Pathway Modules", pad=6)
        if idx == 0:
            ax.set_ylabel("Scanpy Module Score\n(sc.tl.score_genes, mean ± 95% CI)")
        if idx == 2:
            ax.legend(loc="upper right", frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC")

    fig.suptitle(
        "Figure 5. Quality Control Concordance (Detected Genes & MT%) and Functional Pathway Module Scores Across Arms",
        fontsize=13.0, fontweight="bold", y=0.98
    )
    save_fig(fig, "fig5_qc_and_functional_modules")


def main() -> None:
    with open(ANA_DIR / "scanpy_standard_analysis_summary.json") as f:
        summary = json.load(f)

    adatas = {}
    for cohort in COHORTS:
        p = H5AD_DIR / f"human_{cohort}_scanpy_processed.h5ad"
        print(f"Loading processed AnnData: {p.name}...", flush=True)
        adatas[cohort] = ad.read_h5ad(p)

    build_figure1_umap_atlas(adatas, summary)
    build_figure2_pathology_fidelity()
    build_figure3_azi_reversal(adatas, summary)
    build_figure4_canonical_dotplot(adatas)
    build_figure5_qc_and_modules(adatas, summary)
    print("ALL 5 PUBLICATION FIGURES GENERATED SUCCESSFULLY.", flush=True)


if __name__ == "__main__":
    main()
