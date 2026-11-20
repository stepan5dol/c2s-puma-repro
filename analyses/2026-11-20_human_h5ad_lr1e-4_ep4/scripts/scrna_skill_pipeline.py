#!/usr/bin/env python3
"""Standard scRNA-seq analysis of the human real + model h5ad files, by the skills
scrnaseq-bioinformatics (scanpy pipeline, Wilcoxon DE, volcano/dotplot recipes) and
single-cell-rna-qc (qc_core.calculate_qc_metrics / detect_outliers_mad, scverse MAD thresholds).

Per cohort (Acute26, BPD7mo, BPDPH7mo; groups CTRL, DISEASE, MODEL_NOAZI, MODEL_AZI):
  1. QC (single-cell-rna-qc): metrics from layers['log1p_cpm10k_full'] -> CPM (no raw counts in the files;
     CPM gives the same n_genes and MT/ribo/HB percentages; total_counts is 1e4 for every cell, so its MAD
     filter does not apply). Real cells (droplets): outliers by MAD within each sample (obs 'dataset' = donor,
     the scverse per-sample practice), n_genes_by_counts 5 MADs, pct_counts_mt 3 MADs + hard 8 % (skill
     defaults), removed. Model cells are generated cell sentences, not droplets: the dying-cell / empty-droplet
     filters do not apply to them and removing them would change the model's output distribution, so they are
     kept; their metrics are plotted. Counts per sample in qc_summary.csv.
  2. Preprocess (scrnaseq skill): X is already log1p(CPM10k) (normalize_total 1e4 + log1p), top-2000 matched
     between real and model; .raw = X; genes in >= 20 cells (single-cell-rna-qc min_cells); HVG 2000, scale 10,
     PCA 50, neighbors 15, UMAP, Leiden 0.5. No batch integration: it would remove the model-vs-real difference
     this analysis looks at.
  3. Cell-type markers: Wilcoxon on real CTRL cells by cell_type, top 5 per type -> dotplot over all groups.
  4. DE per cell type, scanpy rank_genes_groups(method='wilcoxon', tie_correct=True), Benjamini-Hochberg
     (scanpy default), genes detected in >= 10 % of cells of either group (Seurat min.pct of the skill);
     DEG = padj < 0.05 and |log2FC| >= 0.25 (skill logfc threshold). Contrasts:
        disease_vs_ctrl   DISEASE vs CTRL          the real disease signature
        noazi_vs_ctrl     MODEL_NOAZI vs CTRL      does the model reproduce it
        noazi_vs_disease  MODEL_NOAZI vs DISEASE   where the model differs from real disease
        azi_vs_noazi      MODEL_AZI vs MODEL_NOAZI azithromycin in the prompt
  5. Summaries: DEG counts; real disease DEGs reproduced by the model (same / opposite sign);
     azi_vs_noazi DEGs against the disease direction; pan-cell genes = DEG with one sign in all 4 cell types.

usage: python3 scrna_skill_pipeline.py --h5ad-dir <folder with human_<cohort>_real_and_model.h5ad> --out-root <folder>
Output: <out-root>/scrna_skill_<YYYY-MM-DD_HHMM>/ (fails if it exists).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

QC_SKILL = Path.home() / ".claude/plugins/synced/891c489b-3e8a-4908-acef-e82ba8472cde_ade76d41-81f9-4da4-b866-763738ced717/bio-research~g2/skills/single-cell-rna-qc/scripts"
sys.path.insert(0, str(QC_SKILL))
from qc_core import calculate_qc_metrics, detect_outliers_mad  # noqa: E402

COHORTS = ["Acute26", "BPD7mo", "BPDPH7mo"]
CTS = ["gCap", "aCap", "Pericyte", "VEC"]
GROUPS = ["CTRL", "DISEASE", "MODEL_NOAZI", "MODEL_AZI"]
CONTRASTS = {
    "disease_vs_ctrl": ("DISEASE", "CTRL"),
    "noazi_vs_ctrl": ("MODEL_NOAZI", "CTRL"),
    "noazi_vs_disease": ("MODEL_NOAZI", "DISEASE"),
    "azi_vs_noazi": ("MODEL_AZI", "MODEL_NOAZI"),
}
MAD_GENES, MAD_MT, MT_HARD = 5, 3, 8.0      # single-cell-rna-qc defaults
MIN_CELLS_GENE = 20                          # single-cell-rna-qc default
MIN_PCT, PADJ, LFC = 0.10, 0.05, 0.25       # scrnaseq skill (Seurat FindAllMarkers defaults)
MIN_GROUP = 3

log = logging.getLogger("scrna_skill")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def volcano_plot(ax, df, title):
    """scrnaseq skill recipe (section 10), drawn into a given axis."""
    d = df.dropna(subset=["logfoldchanges", "pvals_adj"])
    y = -np.log10(d["pvals_adj"].clip(lower=1e-300))
    up = (d["logfoldchanges"] >= LFC) & (d["pvals_adj"] < PADJ)
    dn = (d["logfoldchanges"] <= -LFC) & (d["pvals_adj"] < PADJ)
    ax.scatter(d.loc[~up & ~dn, "logfoldchanges"], y[~up & ~dn], c="grey", alpha=0.4, s=4, label="NS")
    ax.scatter(d.loc[up, "logfoldchanges"], y[up], c="red", alpha=0.7, s=6, label=f"Up ({int(up.sum())})")
    ax.scatter(d.loc[dn, "logfoldchanges"], y[dn], c="blue", alpha=0.7, s=6, label=f"Down ({int(dn.sum())})")
    ax.axhline(-np.log10(PADJ), ls="--", c="black", lw=0.6)
    for x in (LFC, -LFC):
        ax.axvline(x, ls="--", c="black", lw=0.6)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("log2 fold change", fontsize=7)
    ax.set_ylabel("-log10 adj. p", fontsize=7)
    ax.legend(fontsize=6, loc="upper left")


class CohortPipeline:
    def __init__(self, cohort: str, h5ad: Path, out: Path, seed: int, n_top_genes: int, n_pcs: int,
                 n_neighbors: int, resolution: float):
        self.cohort, self.h5ad, self.out, self.seed = cohort, h5ad, out, seed
        self.n_top_genes, self.n_pcs, self.n_neighbors, self.resolution = n_top_genes, n_pcs, n_neighbors, resolution
        self.fig = out / "figures"
        self.fig.mkdir(parents=True, exist_ok=True)
        (out / "de").mkdir(exist_ok=True)
        self.adata: ad.AnnData | None = None
        self.de: dict[tuple[str, str], pd.DataFrame] = {}
        self.summary: dict = {"cohort": cohort, "h5ad": str(h5ad)}
        np.random.seed(seed)
        sc.settings.seed = seed
        sc.settings.figdir = str(self.fig)

    def load(self):
        self.adata = ad.read_h5ad(self.h5ad)
        self.adata.var_names_make_unique()
        log.info("%s: loaded %d cells x %d genes", self.cohort, *self.adata.shape)
        self.summary["n_cells_loaded"] = int(self.adata.n_obs)
        return self

    def quality_control(self):
        a = self.adata
        full = a.layers["log1p_cpm10k_full"].tocsr().copy()
        full.data = np.expm1(full.data)
        q = ad.AnnData(X=full, obs=a.obs[["orig.ident", "cell_type"]].copy(), var=pd.DataFrame(index=a.var_names))
        calculate_qc_metrics(q, mt_pattern="MT-", ribo_pattern="RPL,RPS", inplace=True)
        for k in ("n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"):
            a.obs[k] = q.obs[k].values
        q.obs["dataset"] = a.obs["dataset"].values
        q.obs["source"] = a.obs["source"].values
        flag = np.zeros(a.n_obs, dtype=bool)
        rows = []
        for (src, g, ds), idx in q.obs.groupby(["source", "orig.ident", "dataset"], observed=True).indices.items():
            sub = q[idx]
            if src == "real":
                f_g = np.asarray(detect_outliers_mad(sub, "n_genes_by_counts", MAD_GENES, verbose=False))
                f_mt = np.asarray(detect_outliers_mad(sub, "pct_counts_mt", MAD_MT, verbose=False)) | (sub.obs["pct_counts_mt"].values > MT_HARD)
            else:
                f_g = f_mt = np.zeros(len(idx), dtype=bool)
            flag[idx] = f_g | f_mt
            rows.append({"cohort": self.cohort, "source": src, "group": g, "sample": ds, "n_cells": len(idx),
                         "out_n_genes": int(f_g.sum()), "out_mt": int(f_mt.sum()), "removed": int((f_g | f_mt).sum()),
                         "kept": int(len(idx) - (f_g | f_mt).sum()),
                         "median_n_genes": float(np.median(sub.obs["n_genes_by_counts"])),
                         "median_pct_mt": float(np.median(sub.obs["pct_counts_mt"]))})
        a.obs["qc_outlier"] = flag
        sc.pl.violin(a, ["n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"], groupby="orig.ident",
                     jitter=0.4, rotation=45, show=False, save=f"_{self.cohort}_qc_before.png")
        self.adata = a[~flag].copy()
        log.info("%s: QC removed %d of %d cells", self.cohort, int(flag.sum()), len(flag))
        self.summary["qc_removed"] = int(flag.sum())
        return pd.DataFrame(rows)

    def preprocess(self):
        a = self.adata
        sc.pp.filter_genes(a, min_cells=MIN_CELLS_GENE)
        a.raw = a                                        # log1p CPM10k, top-2000 matched: used for DE
        sc.pp.highly_variable_genes(a, n_top_genes=self.n_top_genes, flavor="seurat")
        self.adata = a[:, a.var.highly_variable].copy()
        sc.pp.scale(self.adata, max_value=10)
        log.info("%s: %d genes after min_cells, %d HVG", self.cohort, a.n_vars, self.adata.n_vars)
        return self

    def dim_reduction_and_clusters(self):
        a = self.adata
        sc.tl.pca(a, n_comps=self.n_pcs, svd_solver="arpack", random_state=self.seed)
        sc.pp.neighbors(a, n_neighbors=self.n_neighbors, n_pcs=self.n_pcs, random_state=self.seed)
        sc.tl.umap(a, random_state=self.seed)
        sc.tl.leiden(a, resolution=self.resolution, random_state=self.seed, flavor="igraph", n_iterations=2,
                     directed=False)
        sc.pl.umap(a, color=["orig.ident", "cell_type", "leiden", "dataset"], ncols=2, wspace=0.5, show=False,
                   save=f"_{self.cohort}.png")
        ct_tab = pd.crosstab([a.obs["orig.ident"], a.obs["cell_type"]], a.obs["leiden"])
        ct_tab.to_csv(self.out / f"{self.cohort}_leiden_by_group_celltype.csv")
        return self

    def markers(self):
        a = self.adata
        real = a[a.obs["orig.ident"] == "CTRL"].copy()
        sc.tl.rank_genes_groups(real, groupby="cell_type", method="wilcoxon", use_raw=True, tie_correct=True)
        mk = sc.get.rank_genes_groups_df(real, group=None)
        mk.to_csv(self.out / f"{self.cohort}_ctrl_celltype_markers.csv", index=False)
        top = {ct: mk[mk["group"] == ct].nlargest(5, "scores")["names"].tolist() for ct in CTS}
        a.obs["group_celltype"] = (a.obs["cell_type"].astype(str) + " | " + a.obs["orig.ident"].astype(str)).astype("category")
        order = [f"{ct} | {g}" for ct in CTS for g in GROUPS if f"{ct} | {g}" in set(a.obs["group_celltype"])]
        a.obs["group_celltype"] = a.obs["group_celltype"].cat.reorder_categories(order)
        sc.pl.dotplot(a, top, groupby="group_celltype", use_raw=True, standard_scale="var", show=False,
                      save=f"_{self.cohort}_ctrl_markers.png")
        self.summary["ctrl_markers_top5"] = top
        return self

    def differential_expression(self):
        a = self.adata
        rows = []
        for ct in CTS:
            for name, (ga, gb) in CONTRASTS.items():
                m = (a.obs["cell_type"] == ct) & a.obs["orig.ident"].isin([ga, gb])
                obs = a.obs.loc[m, ["orig.ident"]].copy()
                na, nb = int((obs["orig.ident"] == ga).sum()), int((obs["orig.ident"] == gb).sum())
                rec = {"cohort": self.cohort, "cell_type": ct, "contrast": name, "n_A": na, "n_B": nb}
                if min(na, nb) < MIN_GROUP:
                    rows.append({**rec, "skipped": f"group < {MIN_GROUP} cells"})
                    continue
                X = a.raw[m.values].X
                X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
                isA = (obs["orig.ident"] == ga).values
                pctA = np.asarray((X[isA] > 0).mean(0)).ravel()
                pctB = np.asarray((X[~isA] > 0).mean(0)).ravel()
                keep = np.maximum(pctA, pctB) >= MIN_PCT
                sub = ad.AnnData(X=X[:, keep], obs=obs, var=pd.DataFrame(index=a.raw.var_names[keep]))
                sub.obs["orig.ident"] = sub.obs["orig.ident"].astype(str).astype("category")
                sc.tl.rank_genes_groups(sub, groupby="orig.ident", groups=[ga], reference=gb, method="wilcoxon",
                                        use_raw=False, tie_correct=True, corr_method="benjamini-hochberg")
                df = sc.get.rank_genes_groups_df(sub, group=ga)
                df["pct_A"] = pd.Series(pctA[keep], index=sub.var_names).reindex(df["names"]).values
                df["pct_B"] = pd.Series(pctB[keep], index=sub.var_names).reindex(df["names"]).values
                df["deg"] = np.where((df["pvals_adj"] < PADJ) & (df["logfoldchanges"] >= LFC), "up",
                                     np.where((df["pvals_adj"] < PADJ) & (df["logfoldchanges"] <= -LFC), "down", ""))
                df.insert(0, "cell_type", ct)
                df.insert(0, "contrast", name)
                df.to_csv(self.out / "de" / f"{self.cohort}_{ct}_{name}.csv", index=False)
                self.de[(ct, name)] = df.set_index("names")
                rows.append({**rec, "n_tested": int(keep.sum()), "n_up": int((df["deg"] == "up").sum()),
                             "n_down": int((df["deg"] == "down").sum()), "min_padj": float(df["pvals_adj"].min())})
        for name in CONTRASTS:
            fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
            for ax, ct in zip(axs, CTS):
                if (ct, name) in self.de:
                    volcano_plot(ax, self.de[(ct, name)].reset_index(), f"{self.cohort} {ct}: {name}")
                else:
                    ax.set_axis_off()
                    ax.set_title(f"{ct}: skipped (< {MIN_GROUP} cells)", fontsize=8)
            fig.tight_layout()
            fig.savefig(self.fig / f"volcano_{self.cohort}_{name}.png", dpi=130)
            plt.close(fig)
        return pd.DataFrame(rows)

    def cross_contrast_summary(self):
        rows, pan = [], []
        for ct in CTS:
            dis = self.de.get((ct, "disease_vs_ctrl"))
            if dis is None:
                continue
            dis_deg = dis[dis["deg"] != ""]
            sign_dis = np.sign(dis_deg["logfoldchanges"])
            for name in ("noazi_vs_ctrl", "azi_vs_noazi"):
                other = self.de.get((ct, name))
                if other is None:
                    continue
                o = other.reindex(dis_deg.index)
                is_deg = o["deg"].fillna("") != ""
                same = is_deg & (np.sign(o["logfoldchanges"]) == sign_dis)
                opp = is_deg & (np.sign(o["logfoldchanges"]) == -sign_dis)
                rows.append({"cohort": self.cohort, "cell_type": ct, "contrast": name,
                             "real_disease_degs": int(len(dis_deg)), "also_deg_same_sign": int(same.sum()),
                             "also_deg_opposite_sign": int(opp.sum()), "not_tested": int(o["deg"].isna().sum())})
        for name in CONTRASTS:
            tabs = [self.de[(ct, name)] for ct in CTS if (ct, name) in self.de]
            if len(tabs) < 4:
                continue
            common = set.intersection(*[set(t.index[t["deg"] != ""]) for t in tabs])
            for g in sorted(common):
                s = [np.sign(t.loc[g, "logfoldchanges"]) for t in tabs]
                if len(set(s)) == 1:
                    pan.append({"cohort": self.cohort, "contrast": name, "gene": g, "direction": "up" if s[0] > 0 else "down",
                                **{f"lfc_{ct}": float(t.loc[g, "logfoldchanges"]) for ct, t in zip(CTS, tabs)},
                                **{f"padj_{ct}": float(t.loc[g, "pvals_adj"]) for ct, t in zip(CTS, tabs)}})
        return pd.DataFrame(rows), pd.DataFrame(pan)

    def save(self):
        self.adata.write_h5ad(self.out / f"{self.cohort}_processed.h5ad", compression="gzip")
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-top-genes", type=int, required=True)
    ap.add_argument("--n-pcs", type=int, required=True)
    ap.add_argument("--n-neighbors", type=int, required=True)
    ap.add_argument("--resolution", type=float, required=True)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out = args.out_root / f"scrna_skill_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "scripts").mkdir()
    shutil.copy2(__file__, out / "scripts" / Path(__file__).name)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()])
    t0 = time.time()
    record = {"stamp": stamp, "args": {k: str(v) for k, v in vars(args).items()},
              "thresholds": {"mad_genes": MAD_GENES, "mad_mt": MAD_MT, "mt_hard": MT_HARD, "min_cells_gene": MIN_CELLS_GENE,
                             "min_pct": MIN_PCT, "padj": PADJ, "lfc": LFC, "min_group": MIN_GROUP},
              "versions": {p: version(p) for p in ("scanpy", "anndata", "numpy", "scipy", "pandas", "igraph", "leidenalg")},
              "inputs": {}, "cohorts": {}}
    qc_all, deg_all, cross_all, pan_all = [], [], [], []
    for cohort in COHORTS:
        h5 = args.h5ad_dir / f"human_{cohort}_real_and_model.h5ad"
        record["inputs"][h5.name] = sha256(h5)
        p = CohortPipeline(cohort, h5, out, args.seed, args.n_top_genes, args.n_pcs, args.n_neighbors, args.resolution)
        p.load()
        qc_all.append(p.quality_control())
        p.preprocess().dim_reduction_and_clusters().markers()
        deg_all.append(p.differential_expression())
        cross, pan = p.cross_contrast_summary()
        cross_all.append(cross)
        pan_all.append(pan)
        p.save()
        record["cohorts"][cohort] = p.summary
    pd.concat(qc_all).to_csv(out / "qc_summary.csv", index=False)
    pd.concat(deg_all).to_csv(out / "deg_counts.csv", index=False)
    pd.concat(cross_all).to_csv(out / "disease_degs_in_model.csv", index=False)
    pd.concat(pan_all).to_csv(out / "pan_cell_degs.csv", index=False)
    record["duration_s"] = round(time.time() - t0, 1)
    record["status"] = "ok"
    json.dump(record, open(out / "run.json", "w"), indent=2, ensure_ascii=False)
    log.info("done -> %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
