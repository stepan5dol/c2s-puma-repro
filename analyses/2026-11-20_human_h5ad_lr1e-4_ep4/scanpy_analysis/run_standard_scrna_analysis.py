#!/usr/bin/env python3
"""
Standard scRNA-seq analysis across all 3 human cohorts (Acute26, BPD7mo, BPDPH7mo)
following the K-Dense `scanpy` (v1.12.4) and `anndata` (v0.13.4) skill contracts.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Import skill helpers directly from the installed scanpy skill
SKILL_SCRIPTS = Path("/Users/stepandolzhenko/.agents/skills/scanpy/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from _common import configure_scanpy, compute_pca, build_neighbors, info

BASE_DIR = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/2026-11-20_human_h5ad_newgrid_lr1e-4_ep4_s42")
OUT_DIR = BASE_DIR / "scanpy_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "markers").mkdir(exist_ok=True)
(OUT_DIR / "contrasts").mkdir(exist_ok=True)
(OUT_DIR / "processed_h5ad").mkdir(exist_ok=True)

COHORTS = {
    "Acute26": BASE_DIR / "human_Acute26_real_and_model.h5ad",
    "BPD7mo": BASE_DIR / "human_BPD7mo_real_and_model.h5ad",
    "BPDPH7mo": BASE_DIR / "human_BPDPH7mo_real_and_model.h5ad",
}

CANONICAL_MARKERS = {
    "gCap": ["CA4", "FCN3", "IL7R", "CD36", "BTNL9", "GPIHBP1", "AFF3"],
    "aCap": ["EDNRB", "HPGD", "SOSTDC1", "APLN", "EMCN", "TBX2", "S100A4"],
    "VEC": ["ACKR1", "VWF", "CPE", "POSTN", "MMRN1", "CLU", "PTGDS"],
    "Pericyte": ["PDGFRB", "CSPG4", "RGS5", "NOTCH3", "ACTA2", "COL1A2", "TAGLN"],
}

GENE_SIGNATURES = {
    "ECM_Fibrosis": ["COL1A1", "COL1A2", "COL3A1", "FN1", "POSTN", "SPARC", "TIMP1", "ACTA2", "TAGLN", "CTGF", "TGFBI"],
    "Inflammation_IFN": ["IFITM3", "IFITM2", "ISG15", "IFI6", "IFI27", "STAT1", "HLA-A", "HLA-B", "HLA-C", "B2M", "CXCL8", "IL1B", "NFKBIA"],
    "Angiogenesis_Vascular": ["KDR", "FLT1", "TEK", "TIE1", "PECAM1", "CDH5", "ENG", "NOS3", "EPAS1", "CA4", "EDNRB"],
    "Hypoxia_Stress": ["HIF1A", "VEGFA", "LDHA", "PGK1", "ENO1", "BNIP3", "DDIT4", "HSPA1A", "HSP90AA1", "JUN", "FOS"],
}


def analyze_cohort(cohort: str, path: Path) -> dict:
    print(f"\n{'='*78}\nANALYZING COHORT: {cohort} ({path.name})\n{'='*78}", flush=True)
    adata = sc.read_h5ad(path)
    summary: dict = {"cohort": cohort, "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars)}

    # -------------------------------------------------------------------------
    # STEP 0: Representation & Inference Contract Check (anndata + scanpy skill)
    # -------------------------------------------------------------------------
    x_data = adata.X.data if sp.issparse(adata.X) else np.asarray(adata.X).ravel()
    cpm_layer = adata.layers["cpm10k_top2000"]
    full_layer = adata.layers["log1p_cpm10k_full"]
    cpm_sums = np.asarray(cpm_layer.sum(axis=1)).ravel()

    summary["step0_contract"] = {
        "X_dtype": str(adata.X.dtype),
        "X_min": float(x_data.min()) if len(x_data) else 0.0,
        "X_max": float(x_data.max()) if len(x_data) else 0.0,
        "X_nnz": int(adata.X.nnz),
        "X_density_pct": float(100.0 * adata.X.nnz / (adata.n_obs * adata.n_vars)),
        "cpm10k_top2000_sum_median": float(np.median(cpm_sums)),
        "cpm10k_top2000_sum_iqr": [float(np.percentile(cpm_sums, 25)), float(np.percentile(cpm_sums, 75))],
        "layers": [k for k in adata.layers.keys() if isinstance(k, str)],
        "cell_counts_by_arm": adata.obs["orig.ident"].value_counts().to_dict(),
        "cell_counts_by_ct_and_arm": pd.crosstab(adata.obs["cell_type"], adata.obs["orig.ident"]).to_dict(),
        "datasets": pd.crosstab(adata.obs["dataset"], adata.obs["orig.ident"]).to_dict(),
    }

    # -------------------------------------------------------------------------
    # STEP 1: Quality Control (QC) on cpm10k_top2000 and log1p_cpm10k_full
    # -------------------------------------------------------------------------
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]", regex=True)

    # Compute QC metrics on linear CPM10k layer to get true percentage of MT/Ribo expression
    sc.pp.calculate_qc_metrics(
        adata,
        expr_type="cpm10k",
        var_type="genes",
        layer="cpm10k_top2000",
        qc_vars=["mt", "ribo", "hb"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # Also compute full-layer MT% for real cells
    mt_mask = adata.var["mt"].values
    ribo_mask = adata.var["ribo"].values
    full_cpm = full_layer.copy()
    full_cpm.data = np.expm1(full_cpm.data).astype(np.float32)
    full_tot = np.asarray(full_cpm.sum(axis=1)).ravel()
    full_tot[full_tot == 0] = 1.0
    adata.obs["pct_cpm_mt_full"] = (np.asarray(full_cpm[:, mt_mask].sum(axis=1)).ravel() / full_tot * 100.0).astype(np.float32)
    adata.obs["pct_cpm_ribo_full"] = (np.asarray(full_cpm[:, ribo_mask].sum(axis=1)).ravel() / full_tot * 100.0).astype(np.float32)

    qc_by_arm = {}
    for arm, df_arm in adata.obs.groupby("orig.ident", observed=True):
        qc_by_arm[str(arm)] = {
            "n_cells": int(len(df_arm)),
            "n_genes_top2000_median": float(df_arm["n_genes_nonzero"].median()),
            "n_genes_top2000_min": int(df_arm["n_genes_nonzero"].min()),
            "n_genes_top2000_max": int(df_arm["n_genes_nonzero"].max()),
            "n_genes_full_median": float(df_arm["n_genes_nonzero_full"].median()),
            "n_genes_full_iqr": [
                float(np.percentile(df_arm["n_genes_nonzero_full"], 25)),
                float(np.percentile(df_arm["n_genes_nonzero_full"], 75)),
            ],
            "cells_lt_200_genes": int((df_arm["n_genes_nonzero_full"] < 200).sum()),
            "pct_mt_top2000_median": float(df_arm["pct_cpm10k_mt"].median()),
            "pct_mt_top2000_p95": float(np.percentile(df_arm["pct_cpm10k_mt"], 95)),
            "pct_mt_gt_10": int((df_arm["pct_cpm10k_mt"] > 10.0).sum()),
            "pct_ribo_top2000_median": float(df_arm["pct_cpm10k_ribo"].median()),
        }
    n_active_genes = int((np.asarray((adata.X > 0).sum(axis=0)).ravel() >= 3).sum())
    summary["step1_qc"] = {
        "n_mt_genes_in_var": int(adata.var["mt"].sum()),
        "n_ribo_genes_in_var": int(adata.var["ribo"].sum()),
        "n_genes_detected_in_ge_3_cells": n_active_genes,
        "by_arm": qc_by_arm,
    }

    # -------------------------------------------------------------------------
    # STEP 2: Normalization Snapshot (.raw) & Highly Variable Genes (HVG)
    # -------------------------------------------------------------------------
    # As required by scanpy skill: preserve full-gene log-normalized snapshot in .raw
    adata.uns["log1p"] = {"base": None}
    adata.raw = adata.copy()

    # Select 2000 HVGs on log1p-normalized X (flavor='seurat')
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    hvg_genes = list(adata.var_names[adata.var["highly_variable"]])

    # Also check batch-aware HVG across real vs model
    adata_tmp = adata.copy()
    sc.pp.highly_variable_genes(adata_tmp, n_top_genes=2000, flavor="seurat", batch_key="source")
    hvg_batch_genes = set(adata_tmp.var_names[adata_tmp.var["highly_variable"]])
    del adata_tmp

    summary["step2_hvg"] = {
        "n_hvg": len(hvg_genes),
        "overlap_with_source_batch_hvg": len(set(hvg_genes) & hvg_batch_genes),
        "top15_hvg_by_dispersion_norm": list(
            adata.var[adata.var["highly_variable"]]
            .sort_values("dispersions_norm", ascending=False)
            .head(15)
            .index
        ),
        "canonical_markers_in_hvg": {
            ct: [g for g in genes if g in hvg_genes]
            for ct, genes in CANONICAL_MARKERS.items()
        },
    }

    # -------------------------------------------------------------------------
    # STEP 3: Dimensionality Reduction (PCA -> Neighbors -> UMAP)
    # -------------------------------------------------------------------------
    # Compute PCA on HVGs (using scanpy skill parameters: n_comps=50, svd_solver='arpack', n_pcs=40, n_neighbors=15)
    sc.pp.pca(adata, n_comps=50, svd_solver="arpack", mask_var="highly_variable", random_state=0)
    var_ratio = adata.uns["pca"]["variance_ratio"]
    build_neighbors(sc, adata, use_rep="X_pca", n_pcs=40, n_neighbors=15)
    sc.tl.umap(adata, random_state=0)

    # Quantitative Latent Space Shift Analysis in 40D PCA space per cell type:
    # Project cells onto the CTRL -> DISEASE axis and measure where MODEL_NOAZI and MODEL_AZI land!
    pca_coords = adata.obsm["X_pca"][:, :40]
    pca_shift_by_ct = {}
    for ct in ["gCap", "aCap", "Pericyte", "VEC"]:
        m_ct = (adata.obs["cell_type"] == ct).values
        m_ctrl = m_ct & (adata.obs["orig.ident"] == "CTRL").values
        m_dis = m_ct & (adata.obs["orig.ident"] == "DISEASE").values
        m_noazi = m_ct & (adata.obs["orig.ident"] == "MODEL_NOAZI").values
        m_azi = m_ct & (adata.obs["orig.ident"] == "MODEL_AZI").values

        if m_ctrl.sum() < 2 or m_dis.sum() < 2 or m_noazi.sum() < 2 or m_azi.sum() < 2:
            continue

        c_ctrl = pca_coords[m_ctrl].mean(axis=0)
        c_dis = pca_coords[m_dis].mean(axis=0)
        c_noazi = pca_coords[m_noazi].mean(axis=0)
        c_azi = pca_coords[m_azi].mean(axis=0)

        # Axis from DISEASE (0.0) to CTRL (1.0)
        axis_vec = c_ctrl - c_dis
        axis_norm_sq = float(np.dot(axis_vec, axis_vec))
        proj_noazi = float(np.dot(c_noazi - c_dis, axis_vec) / axis_norm_sq) if axis_norm_sq > 0 else 0.0
        proj_azi = float(np.dot(c_azi - c_dis, axis_vec) / axis_norm_sq) if axis_norm_sq > 0 else 0.0

        # Per-cell projections for Mann-Whitney test between MODEL_AZI and MODEL_NOAZI along DISEASE->CTRL axis
        cell_proj_noazi = np.dot(pca_coords[m_noazi] - c_dis, axis_vec) / axis_norm_sq
        cell_proj_azi = np.dot(pca_coords[m_azi] - c_dis, axis_vec) / axis_norm_sq
        _, pval_shift = mannwhitneyu(cell_proj_azi, cell_proj_noazi, alternative="two-sided")

        pca_shift_by_ct[ct] = {
            "dist_ctrl_to_disease": float(np.linalg.norm(c_ctrl - c_dis)),
            "dist_noazi_to_disease": float(np.linalg.norm(c_noazi - c_dis)),
            "dist_noazi_to_ctrl": float(np.linalg.norm(c_noazi - c_ctrl)),
            "dist_azi_to_ctrl": float(np.linalg.norm(c_azi - c_ctrl)),
            "dist_azi_to_noazi": float(np.linalg.norm(c_azi - c_noazi)),
            "axis_pos_noazi_0dis_1ctrl": proj_noazi,
            "axis_pos_azi_0dis_1ctrl": proj_azi,
            "delta_shift_toward_ctrl": proj_azi - proj_noazi,
            "shift_mannwhitney_p": float(pval_shift),
        }

    summary["step3_dimred"] = {
        "pca_var_explained_top10_pct": float(var_ratio[:10].sum() * 100.0),
        "pca_var_explained_top40_pct": float(var_ratio[:40].sum() * 100.0),
        "top5_pc_var_pct": [round(float(v * 100.0), 2) for v in var_ratio[:5]],
        "pca_disease_to_ctrl_shift_by_ct": pca_shift_by_ct,
    }

    # -------------------------------------------------------------------------
    # STEP 4: Clustering (Leiden at multiple resolutions: 0.3, 0.5, 0.8, 1.0)
    # -------------------------------------------------------------------------
    resolutions = [0.3, 0.5, 0.8, 1.0]
    clustering_stats = {}
    for res in resolutions:
        key = f"leiden_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key, flavor="igraph", n_iterations=2, directed=False, random_state=0)
        ari_ct = adjusted_rand_score(adata.obs["cell_type"], adata.obs[key])
        nmi_ct = normalized_mutual_info_score(adata.obs["cell_type"], adata.obs[key])
        ari_arm = adjusted_rand_score(adata.obs["orig.ident"], adata.obs[key])
        nmi_arm = normalized_mutual_info_score(adata.obs["orig.ident"], adata.obs[key])
        clustering_stats[str(res)] = {
            "n_clusters": int(adata.obs[key].nunique()),
            "ARI_with_cell_type": round(float(ari_ct), 4),
            "NMI_with_cell_type": round(float(nmi_ct), 4),
            "ARI_with_arm": round(float(ari_arm), 4),
            "NMI_with_arm": round(float(nmi_arm), 4),
        }
    adata.obs["leiden"] = adata.obs["leiden_0.5"]
    ct_leiden_crosstab = pd.crosstab(adata.obs["leiden_0.5"], adata.obs["cell_type"])
    arm_leiden_crosstab = pd.crosstab(adata.obs["leiden_0.5"], adata.obs["orig.ident"])
    summary["step4_clustering"] = {
        "resolutions": clustering_stats,
        "leiden_0.5_vs_cell_type": ct_leiden_crosstab.to_dict(),
        "leiden_0.5_vs_arm": arm_leiden_crosstab.to_dict(),
    }

    # -------------------------------------------------------------------------
    # STEP 5: Marker Gene Identification & Condition/Treatment Contrasts
    # -------------------------------------------------------------------------
    # 5a. Exploratory markers per Leiden cluster (0.5) and per canonical cell_type
    sc.tl.rank_genes_groups(
        adata, groupby="leiden_0.5", method="wilcoxon", use_raw=True,
        corr_method="benjamini-hochberg", tie_correct=True, key_added="rank_genes_leiden_0.5"
    )
    sc.tl.rank_genes_groups(
        adata, groupby="cell_type", method="wilcoxon", use_raw=True,
        corr_method="benjamini-hochberg", tie_correct=True, key_added="rank_genes_cell_type"
    )

    top_markers_by_ct = {}
    all_ct_dfs = []
    for ct in ["gCap", "aCap", "Pericyte", "VEC"]:
        df_ct = sc.get.rank_genes_groups_df(adata, group=ct, key="rank_genes_cell_type")
        df_ct.insert(0, "cell_type", ct)
        all_ct_dfs.append(df_ct.head(50))
        top_markers_by_ct[ct] = df_ct.head(10)[["names", "scores", "logfoldchanges", "pvals_adj"]].to_dict(orient="records")
    pd.concat(all_ct_dfs, ignore_index=True).to_csv(OUT_DIR / "markers" / f"{cohort}_celltype_top50_markers.csv", index=False)

    # 5b. Within-cell-type condition & treatment contrasts:
    #   1) DISEASE vs CTRL (real pathology signature)
    #   2) MODEL_NOAZI vs CTRL (model pathology fidelity)
    #   3) MODEL_AZI vs MODEL_NOAZI (azithromycin treatment shift)
    #   4) Reversal analysis: among significant DISEASE vs CTRL DEGs, how does MODEL_AZI vs MODEL_NOAZI shift them?
    contrasts_by_ct = {}
    contrast_rows = []
    for ct in ["gCap", "aCap", "Pericyte", "VEC"]:
        sub = adata[adata.obs["cell_type"] == ct].copy()
        arm_counts = sub.obs["orig.ident"].value_counts().to_dict()
        if min(arm_counts.get("CTRL", 0), arm_counts.get("DISEASE", 0), arm_counts.get("MODEL_NOAZI", 0), arm_counts.get("MODEL_AZI", 0)) < 5:
            continue

        # 1. DISEASE vs CTRL
        sc.tl.rank_genes_groups(
            sub, groupby="orig.ident", groups=["DISEASE"], reference="CTRL",
            method="wilcoxon", use_raw=True, corr_method="benjamini-hochberg", tie_correct=True,
            key_added="de_dis_vs_ctrl"
        )
        df_dis = sc.get.rank_genes_groups_df(sub, group="DISEASE", key="de_dis_vs_ctrl").set_index("names")

        # 2. MODEL_NOAZI vs CTRL
        sc.tl.rank_genes_groups(
            sub, groupby="orig.ident", groups=["MODEL_NOAZI"], reference="CTRL",
            method="wilcoxon", use_raw=True, corr_method="benjamini-hochberg", tie_correct=True,
            key_added="de_noazi_vs_ctrl"
        )
        df_noazi = sc.get.rank_genes_groups_df(sub, group="MODEL_NOAZI", key="de_noazi_vs_ctrl").set_index("names")

        # 3. MODEL_AZI vs MODEL_NOAZI
        sc.tl.rank_genes_groups(
            sub, groupby="orig.ident", groups=["MODEL_AZI"], reference="MODEL_NOAZI",
            method="wilcoxon", use_raw=True, corr_method="benjamini-hochberg", tie_correct=True,
            key_added="de_azi_vs_noazi"
        )
        df_azi = sc.get.rank_genes_groups_df(sub, group="MODEL_AZI", key="de_azi_vs_noazi").set_index("names")

        # Combine on common genes
        joined = pd.DataFrame({
            "score_dis_vs_ctrl": df_dis["scores"],
            "lfc_dis_vs_ctrl": df_dis["logfoldchanges"],
            "padj_dis_vs_ctrl": df_dis["pvals_adj"],
            "score_noazi_vs_ctrl": df_noazi.reindex(df_dis.index)["scores"],
            "lfc_noazi_vs_ctrl": df_noazi.reindex(df_dis.index)["logfoldchanges"],
            "padj_noazi_vs_ctrl": df_noazi.reindex(df_dis.index)["pvals_adj"],
            "score_azi_vs_noazi": df_azi.reindex(df_dis.index)["scores"],
            "lfc_azi_vs_noazi": df_azi.reindex(df_dis.index)["logfoldchanges"],
            "padj_azi_vs_noazi": df_azi.reindex(df_dis.index)["pvals_adj"],
        }).dropna()

        # Filter to genes expressed in at least 5% of cells in this cell type
        expr_frac = np.asarray((sub.X > 0).mean(axis=0)).ravel()
        active_genes = sub.var_names[expr_frac >= 0.05]
        joined_active = joined.loc[joined.index.intersection(active_genes)].copy()

        # Fidelity correlation: DISEASE vs CTRL score vs MODEL_NOAZI vs CTRL score
        rho_fidelity, p_fidelity = spearmanr(joined_active["score_dis_vs_ctrl"], joined_active["score_noazi_vs_ctrl"])

        # Pathology DEGs (|score_dis_vs_ctrl| >= 3.0 & padj < 0.05)
        path_degs = joined_active[(joined_active["padj_dis_vs_ctrl"] < 0.05) & (joined_active["score_dis_vs_ctrl"].abs() >= 3.0)]
        if len(path_degs) >= 10:
            rho_rev, p_rev = spearmanr(path_degs["score_dis_vs_ctrl"], path_degs["score_azi_vs_noazi"])
            reversal_rate = float((np.sign(path_degs["score_dis_vs_ctrl"]) != np.sign(path_degs["score_azi_vs_noazi"])).mean() * 100.0)
        else:
            rho_rev, p_rev, reversal_rate = 0.0, 1.0, 0.0

        # Also check model-defined pathology DEGs (|score_noazi_vs_ctrl| >= 3.0 & padj < 0.05)
        mod_degs = joined_active[(joined_active["padj_noazi_vs_ctrl"] < 0.05) & (joined_active["score_noazi_vs_ctrl"].abs() >= 3.0)]
        if len(mod_degs) >= 10:
            rho_rev_mod, p_rev_mod = spearmanr(mod_degs["score_noazi_vs_ctrl"], mod_degs["score_azi_vs_noazi"])
            reversal_rate_mod = float((np.sign(mod_degs["score_noazi_vs_ctrl"]) != np.sign(mod_degs["score_azi_vs_noazi"])).mean() * 100.0)
        else:
            rho_rev_mod, p_rev_mod, reversal_rate_mod = 0.0, 1.0, 0.0

        sig_azi = joined_active[joined_active["padj_azi_vs_noazi"] < 0.05].sort_values("score_azi_vs_noazi", ascending=False)
        top_azi_up = sig_azi[sig_azi["score_azi_vs_noazi"] > 0].head(10).index.tolist()
        top_azi_down = sig_azi[sig_azi["score_azi_vs_noazi"] < 0].tail(10).index.tolist()[::-1]

        # If fewer than 5 pass padj < 0.05 (e.g. small n in BPD7mo), also record top by Wilcoxon z-score
        top_azi_up_by_score = joined_active.sort_values("score_azi_vs_noazi", ascending=False).head(8).index.tolist()
        top_azi_down_by_score = joined_active.sort_values("score_azi_vs_noazi", ascending=True).head(8).index.tolist()

        joined_active.insert(0, "cohort", cohort)
        joined_active.insert(1, "cell_type", ct)
        contrast_rows.append(joined_active.reset_index().rename(columns={"index": "gene"}))

        contrasts_by_ct[ct] = {
            "n_active_genes_ge_5pct": int(len(joined_active)),
            "n_deg_dis_vs_ctrl_padj05": int((joined_active["padj_dis_vs_ctrl"] < 0.05).sum()),
            "n_deg_noazi_vs_ctrl_padj05": int((joined_active["padj_noazi_vs_ctrl"] < 0.05).sum()),
            "n_deg_azi_vs_noazi_padj05": int((joined_active["padj_azi_vs_noazi"] < 0.05).sum()),
            "fidelity_spearman_rho_dis_vs_noazi": round(float(rho_fidelity), 4),
            "fidelity_spearman_p": float(p_fidelity),
            "real_path_degs_n": int(len(path_degs)),
            "reversal_spearman_rho_on_real_path_degs": round(float(rho_rev), 4),
            "reversal_rate_on_real_path_degs_pct": round(float(reversal_rate), 2),
            "model_path_degs_n": int(len(mod_degs)),
            "reversal_spearman_rho_on_model_path_degs": round(float(rho_rev_mod), 4),
            "reversal_rate_on_model_path_degs_pct": round(float(reversal_rate_mod), 2),
            "top_azi_up_padj05": top_azi_up,
            "top_azi_down_padj05": top_azi_down,
            "top_azi_up_by_zscore": top_azi_up_by_score,
            "top_azi_down_by_zscore": top_azi_down_by_score,
        }

    if contrast_rows:
        pd.concat(contrast_rows, ignore_index=True).to_csv(
            OUT_DIR / "contrasts" / f"{cohort}_within_ct_contrasts.csv", index=False
        )
    summary["step5_markers_and_contrasts"] = {
        "top_celltype_markers": top_markers_by_ct,
        "within_ct_contrasts": contrasts_by_ct,
    }

    # -------------------------------------------------------------------------
    # STEP 6: Cell Type Annotation Validation & Gene Signature Scoring
    # -------------------------------------------------------------------------
    # Score canonical cell-type signatures and biological pathway modules
    for ct_name, genes in CANONICAL_MARKERS.items():
        valid = [g for g in genes if g in adata.raw.var_names]
        sc.tl.score_genes(adata, valid, score_name=f"sig_{ct_name}", use_raw=True, random_state=0)

    for sig_name, genes in GENE_SIGNATURES.items():
        valid = [g for g in genes if g in adata.raw.var_names]
        sc.tl.score_genes(adata, valid, score_name=f"module_{sig_name}", use_raw=True, random_state=0)

    # Summarize canonical signature specificity across cell types
    ct_sig_matrix = {}
    for ct in ["gCap", "aCap", "Pericyte", "VEC"]:
        sub_obs = adata.obs[adata.obs["cell_type"] == ct]
        ct_sig_matrix[ct] = {
            f"sig_{target}": round(float(sub_obs[f"sig_{target}"].mean()), 4)
            for target in ["gCap", "aCap", "Pericyte", "VEC"]
        }

    # Summarize functional module scores across arms (CTRL, DISEASE, MODEL_NOAZI, MODEL_AZI)
    module_by_arm = {}
    for sig_name in GENE_SIGNATURES:
        col = f"module_{sig_name}"
        module_by_arm[sig_name] = {
            str(arm): round(float(adata.obs.loc[adata.obs["orig.ident"] == arm, col].mean()), 4)
            for arm in ["CTRL", "DISEASE", "MODEL_NOAZI", "MODEL_AZI"]
        }
        # Mann-Whitney test for MODEL_AZI vs MODEL_NOAZI
        v_azi = adata.obs.loc[adata.obs["orig.ident"] == "MODEL_AZI", col].values
        v_noazi = adata.obs.loc[adata.obs["orig.ident"] == "MODEL_NOAZI", col].values
        _, p_mod = mannwhitneyu(v_azi, v_noazi, alternative="two-sided")
        module_by_arm[sig_name]["p_azi_vs_noazi"] = float(p_mod)

    summary["step6_annotation_and_scoring"] = {
        "canonical_signature_means_by_cell_type": ct_sig_matrix,
        "functional_module_means_by_arm": module_by_arm,
    }

    # -------------------------------------------------------------------------
    # STEP 7: Save Processed AnnData & Metadata
    # -------------------------------------------------------------------------
    out_h5ad = OUT_DIR / "processed_h5ad" / f"human_{cohort}_scanpy_processed.h5ad"
    adata.write_h5ad(out_h5ad, compression="gzip")
    adata.obs.to_csv(OUT_DIR / f"{cohort}_obs_metadata.csv")
    summary["step7_saved"] = {
        "processed_h5ad": str(out_h5ad),
        "obs_csv": str(OUT_DIR / f"{cohort}_obs_metadata.csv"),
    }
    print(f"Finished {cohort} -> saved {out_h5ad.name}", flush=True)
    return summary


def main():
    sc.settings.verbosity = 1
    all_summaries = {}
    for cohort, path in COHORTS.items():
        all_summaries[cohort] = analyze_cohort(cohort, path)

    summary_json = OUT_DIR / "scanpy_standard_analysis_summary.json"
    with open(summary_json, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nALL COHORTS COMPLETE. Summary written to {summary_json}", flush=True)


if __name__ == "__main__":
    main()
