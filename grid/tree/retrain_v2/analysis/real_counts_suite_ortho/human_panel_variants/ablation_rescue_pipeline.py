#!/usr/bin/env python3
"""
human_rescue_pipeline_genomewide.py — genome-wide human rescue table, the
missing piece for building human panels the way the rat ones are built.

WHY THIS EXISTS
pathway_module_analysis/human_rescue_gene_pipeline.py already ports rat's
scripts/rescue_gene_pipeline.py faithfully -- it computes BOTH arms and BOTH
p-values (Mann-Whitney + Bonferroni, same as rat's wilcoxon_padj), plus
recovery_ratio / recovery_type with rat's constants. What it does NOT do is
run over all genes: it is a per-gene lookup tool (`main(genes)`, default
FOXF1), because it reads the real side through load_data.load_human_bpd_raw,
whose gene axis (remap_gene_names-canonicalized) differs from the AZI-pred
matrix's axis (model.gene_to_idx, raw CSV column names), and reconciling the
two for the full panel was out of scope there.

That blocker does not exist here: pathway_module_analysis/real_counts.py's
loaders take `model` and already reindex the real matrices onto the model's
gene axis (_reindex_to_model_genes), which is the convention the whole
real_counts_suite uses. So all three matrices share one axis and the sweep is
just vectorized arithmetic.

ARMS
  control  real Term0d+Term20d pooled    (real counts)
  disease  real BPD7mo                   (real counts)
  azi      model AZI-counterfactual pred (reconstructed; <=800 nonzero by
                                          construction, no real human patient
                                          ever received AZI)

GENE BUDGET -- the one place this deliberately departs from rat
rat's pipeline has all three arms real and runs on the FULL transcriptome;
top-800 never appears there. Here the AZI arm cannot leave the K=800 budget,
and recovery_ratio = |log2FC_azi| / |log2FC_disease| divides one arm by the
other -- so a full-transcriptome denominator under a top-800 numerator would
make the ratio, and the recovery_ratio<0.9 gate on it, meaningless. Hence:

  log2FC and p  ->  computed with ALL sides top-800-truncated (both contrasts)
  min.pct       ->  computed on the FULL untruncated real matrices

The min.pct split is intentional. After truncation "detected" means "made it
into this cell's own top 800", not "expressed", so MIN_PCT on truncated data
is a different and much harsher filter. min.pct is an eligibility filter, not
a measurement, so it stays on full data (as in rat); the measurements live in
the matched budget.

The disease arm is ALSO reported untruncated (`*_disease_full` columns) --
that is the direct analogue of rat's avg_log2FC_HO / p_val_adj_HO and shows
what the truncation costs. It is not what feeds recovery_ratio.

CAVEAT carried over from human_rescue_gene_pipeline.py's docstring (lines
38-47): GSE275938 has 1 Acute26 / 2 BPD7mo / 2 BPDPH7mo / 1 Term0d / 1
Term20d patient. Per-cell Mann-Whitney across these groups is
pseudoreplicated -- condition confounds with patient. These p-values exist so
the human and rat tables are formula-identical, not because they carry the
usual inferential meaning.

Only T=1.0 / inference_azi_results.jsonl (see real_counts_suite/README.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ABS_DIR = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/analysis")
PMA_DIR = ABS_DIR / "pathway_module_analysis"
sys.path.insert(0, str(ABS_DIR))
import common_human as CH
import rank_expr_model as rem

sys.path.insert(0, str(Path(__file__).parent.parent))
import common_real as CR

sys.path.insert(0, str(PMA_DIR))
_saved_common = sys.modules.pop("common", None)
import real_counts as RC
if _saved_common is not None:
    sys.modules["common"] = _saved_common

HERE = Path(__file__).parent
VARIANTS = sys.argv[1:] or ["azi"]
RESULTS_DIR = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results")
AZI_JSONL_OF = {v: RESULTS_DIR / f"inference_azi_results_{v}.jsonl" for v in VARIANTS}

MIN_PCT = 0.1
LOGFC_THRESHOLD = 0.25
RECOVERY_RATIO_THRESHOLD = 0.9

CONDITIONS = {"Acute26": "Acute", "BPD7mo": "BPD", "BPDPH7mo": "BPD-PH"}
MIN_AZI_CELLS = 3
_DISEASE_CACHE: dict | None = None


def write_rat_schema_workbooks(out: pd.DataFrame, rat_schema_dir: Path | None = None) -> list[Path]:
    """Write human_{cond}_{ct}_rescue_predicted_azi.xlsx files in rat_schema format."""
    target_dir = rat_schema_dir or (HERE / "reports" / "rat_schema")
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cond_csv in CONDITIONS:
        for ct in CH.CT_ORDER:
            if len(out):
                sub = out[(out["condition"] == cond_csv) & (out["cell_type"] == ct) & out["rescued"]].copy()
            else:
                sub = pd.DataFrame()
            if len(sub):
                wb_df = pd.DataFrame({
                    "gene": sub["gene"].values,
                    "avg_log2FC_HO": sub["avg_log2FC_disease_top800"].values,
                    "p_val_adj_HO": sub["p_val_adj_disease_top800"].values,
                    "avg_log2FC_AZI": sub["avg_log2FC_azi_top800"].values,
                    "p_val_adj_AZI": sub["p_val_adj_azi_top800"].values,
                    "recovery_ratio": sub["recovery_ratio"].values,
                    "recovery_type": sub["recovery_type"].values,
                })
            else:
                wb_df = pd.DataFrame(columns=[
                    "gene", "avg_log2FC_HO", "p_val_adj_HO",
                    "avg_log2FC_AZI", "p_val_adj_AZI",
                    "recovery_ratio", "recovery_type",
                ])
            xlsx_path = target_dir / f"human_{cond_csv}_{ct}_rescue_predicted_azi.xlsx"
            wb_df.to_excel(xlsx_path, index=False)
            written.append(xlsx_path)
    return written


def seurat_log2fc(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """log2(mean(expm1(x1))+1) - log2(mean(expm1(x2))+1), per gene (axis 0).
    Same formula as scripts/rescue_gene_pipeline.py::seurat_log2fc."""
    return np.log2(np.expm1(x1).mean(axis=0) + 1) - np.log2(np.expm1(x2).mean(axis=0) + 1)


def mwu_padj(x1: np.ndarray, x2: np.ndarray, n_genes_total: int) -> np.ndarray:
    """Two-sided Mann-Whitney U per gene, Bonferroni-adjusted by total gene
    count -- rat's wilcoxon_padj, vectorized over the gene axis.

    Genes constant in both groups give an undefined U statistic (all ties);
    scipy returns NaN there. Those carry no evidence of a difference, so they
    are set to p=1 rather than propagated as NaN into the gate."""
    p = stats.mannwhitneyu(x1, x2, alternative="two-sided", axis=0).pvalue
    p = np.where(np.isnan(p), 1.0, p)
    return np.minimum(p * n_genes_total, 1.0)


def _get_disease_cache():
    global _DISEASE_CACHE
    if _DISEASE_CACHE is not None and _DISEASE_CACHE.get("k") == CR.K:
        return _DISEASE_CACHE
    print("[calibration] fitting rank->expr model on GSE275938 compiled counts (all human conditions)...")
    model = rem.fit_from_csv(CH.COUNTS_CSV)
    n_genes_total = len(model.gene_names)
    print(f"  r2={model.r2:.4f}  n_genes={n_genes_total}\n")

    print("[data] real Term0d+Term20d control (real counts, model gene axis)...")
    term_mats = RC.real_term_matrices(model)
    term_top800 = {ct: CR.truncate_topk(X) for ct, X in term_mats.items()}

    by_cond = {}
    for cond_csv in CONDITIONS:
        print(f"\n[data] real {cond_csv} disease counts...")
        disease_mats = RC.real_disease_matrices(cond_csv, model)
        by_ct = {}
        for ct in CH.CT_ORDER:
            if ct not in disease_mats or ct not in term_mats:
                continue
            ctrl_full = term_mats[ct]
            dis_full = disease_mats[ct]
            ctrl_t8 = term_top800[ct]
            dis_t8 = CR.truncate_topk(dis_full)
            by_ct[ct] = (
                ctrl_full,
                dis_full,
                ctrl_t8,
                seurat_log2fc(dis_t8, ctrl_t8),
                mwu_padj(dis_t8, ctrl_t8, n_genes_total),
                seurat_log2fc(dis_full, ctrl_full),
                mwu_padj(dis_full, ctrl_full, n_genes_total),
            )
        by_cond[cond_csv] = by_ct

    _DISEASE_CACHE = {
        "k": CR.K,
        "model": model,
        "n_genes_total": n_genes_total,
        "by_cond": by_cond,
    }
    return _DISEASE_CACHE


def main():
    cache = _get_disease_cache()
    model = cache["model"]
    n_genes_total = cache["n_genes_total"]

    azi_groups_of = {}
    for v in VARIANTS:
        print(f"[data] model AZI-counterfactual, variant {v}...")
        azi_groups_of[v] = CH.load_azi_counterfactual(AZI_JSONL_OF[v])

    rows_of = {v: [] for v in VARIANTS}
    for cond_csv, cond_model in CONDITIONS.items():
        by_ct = cache["by_cond"].get(cond_csv, {})
        for ct in CH.CT_ORDER:
            if ct not in by_ct:
                continue
            (ctrl_full, dis_full, ctrl_t8,
             lfc_dis_t8_shared, padj_dis_t8_shared,
             lfc_dis_full_shared, padj_dis_full_shared) = by_ct[ct]

            for variant in VARIANTS:
                arecs = azi_groups_of[variant].get((ct, cond_model), [])
                if len(arecs) < MIN_AZI_CELLS:
                    print(f"  {cond_csv}/{ct}/{variant}: skipped, only {len(arecs)} AZI pred cells")
                    continue
                azi_X = rem.reconstruct_batch([r["pred"] for r in arecs], model, CR.K)
                _run_one(rows_of[variant], model, n_genes_total, cond_csv, ct, ctrl_full, dis_full,
                         ctrl_t8, azi_X, lfc_dis_t8_shared, padj_dis_t8_shared,
                         lfc_dis_full_shared, padj_dis_full_shared, variant)

    for variant in VARIANTS:
        out = pd.concat(rows_of[variant], ignore_index=True) if rows_of[variant] else pd.DataFrame()
        out_path = HERE / "reports" / f"ablation_rescue_genomewide_{variant}.csv"
        out.to_csv(out_path, index=False)
        print(f"\n-> {out_path}  ({len(out)} rows)")
        if len(out):
            print(f"=== rescued genes per condition/cell type, variant {variant} "
                  f"(rat's gate: both candidates + recovery_ratio<0.9) ===")
            summary = (out[out["rescued"]]
                       .groupby(["condition", "cell_type", "recovery_type"])
                       .size().rename("n_genes").reset_index())
            print(summary.to_string(index=False))
        else:
            summary = pd.DataFrame(columns=["condition", "cell_type", "recovery_type", "n_genes"])
        summary_path = HERE / "reports" / f"ablation_rescue_genomewide_summary_{variant}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"-> {summary_path}")
        write_rat_schema_workbooks(out)


def _run_one(rows, model, n_genes_total, cond_csv, ct, ctrl_full, dis_full, ctrl_t8, azi_X,
             lfc_dis_t8, padj_dis_t8, lfc_dis_full, padj_dis_full, variant):
            pct_ctrl = (ctrl_full > 0).mean(axis=0)
            pct_dis = (dis_full > 0).mean(axis=0)
            pct_azi = (azi_X > 0).mean(axis=0)
            pass_min_pct_dis = (pct_ctrl >= MIN_PCT) | (pct_dis >= MIN_PCT)
            pass_min_pct_azi = (pct_ctrl >= MIN_PCT) | (pct_azi >= MIN_PCT)

            lfc_azi_t8 = seurat_log2fc(azi_X, ctrl_t8)
            padj_azi_t8 = mwu_padj(azi_X, ctrl_t8, n_genes_total)

            with np.errstate(divide="ignore", invalid="ignore"):
                recovery_ratio = np.abs(lfc_azi_t8) / np.abs(lfc_dis_t8)
            recovery_ratio[~np.isfinite(recovery_ratio)] = np.nan

            dis_candidate = pass_min_pct_dis & (np.abs(lfc_dis_t8) > LOGFC_THRESHOLD)
            azi_candidate = pass_min_pct_azi & (np.abs(lfc_azi_t8) > LOGFC_THRESHOLD)
            rescued = dis_candidate & azi_candidate & (recovery_ratio < RECOVERY_RATIO_THRESHOLD)
            recovery_type = np.where(lfc_dis_t8 > 0, "Rescued_HO_up", "Rescued_HO_down")

            rows.append(pd.DataFrame({
                "gene": model.gene_names,
                "cell_type": ct,
                "condition": cond_csv,
                "n_control_cells": ctrl_full.shape[0],
                "n_disease_cells": dis_full.shape[0],
                "n_azi_pred_cells": azi_X.shape[0],
                "pct_control_full": pct_ctrl.round(4),
                "pct_disease_full": pct_dis.round(4),
                "pct_azi_pred": pct_azi.round(4),
                "passes_min_pct_disease": pass_min_pct_dis,
                "passes_min_pct_azi": pass_min_pct_azi,
                "avg_log2FC_disease_top800": lfc_dis_t8.round(4),
                "p_val_adj_disease_top800": padj_dis_t8,
                "avg_log2FC_azi_top800": lfc_azi_t8.round(4),
                "p_val_adj_azi_top800": padj_azi_t8,
                "avg_log2FC_disease_full": lfc_dis_full.round(4),
                "p_val_adj_disease_full": padj_dis_full,
                "disease_candidate": dis_candidate,
                "azi_candidate": azi_candidate,
                "recovery_ratio": recovery_ratio.round(4),
                "recovery_type": recovery_type,
                "rescued": rescued,
            }))
            print(f"  {cond_csv}/{ct}/{variant}: n_ctrl={ctrl_full.shape[0]} n_dis={dis_full.shape[0]} "
                  f"n_azi={azi_X.shape[0]}  min.pct-eligible dis/azi="
                  f"{int(pass_min_pct_dis.sum())}/{int(pass_min_pct_azi.sum())}  "
                  f"rescued={int(rescued.sum())}")


if __name__ == "__main__":
    main()
