"""Raw-data loaders for the rat and human sides. No subsampling, no caching
tricks -- every function reads straight from the source files listed in
common.py. Human loaders return raw (unnormalized) counts; caller normalizes."""
from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import common as C

sys.path.insert(0, str(C.BASE))
from pipeline_short.common import remap_gene_names


def load_rat():
    """Returns (adata) with raw counts normalized+log1p'd, ribo/mito/hgb genes
    stripped, var_names upper-cased to human symbols for pathway matching."""
    adata = sc.read_h5ad(C.RAT_PATH)
    keep = [g for g in adata.var_names if not C.STRIP_PATTERN_RAT.match(g)]
    adata = adata[:, keep].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.var_names = [g.upper() for g in adata.var_names]
    adata.var_names_make_unique()
    return adata


def load_human_he22_raw():
    """He22 control: cell-type labels from ATLAS_CLEAN_PATH, raw counts
    barcode-matched from ATLAS_RAW_PATH (ATLAS_CLEAN_PATH's X is already
    log-normalized, has no raw counts)."""
    a = ad.read_h5ad(C.ATLAS_CLEAN_PATH, backed="r")
    obs = a.obs[["new_celltype", "stage"]].copy()
    del a
    he22_obs = obs[obs["stage"].astype(str) == "22.0"].copy()
    he22_obs["cell_type"] = he22_obs["new_celltype"].map(lambda x: C.harmonize_atlas_celltype(str(x)))
    he22_obs = he22_obs[he22_obs["cell_type"].isin(C.TARGET_CELL_TYPES)]

    a_raw = ad.read_h5ad(C.ATLAS_RAW_PATH)
    genes = remap_gene_names(a_raw.var_names.tolist())
    raw_obs_names = a_raw.obs_names.tolist()
    raw_bc_set = set(raw_obs_names)
    raw_short = {}
    for rbc in raw_obs_names:
        parts = rbc.split("-")
        raw_short["-".join(parts[:2]) if len(parts) >= 2 else rbc] = rbc
    clean_to_raw = {}
    for cbc in he22_obs.index:
        if cbc in raw_bc_set:
            clean_to_raw[cbc] = cbc
        else:
            parts = cbc.split("-")
            key = "-".join(parts[:2]) if len(parts) >= 2 else cbc
            if key in raw_short:
                clean_to_raw[cbc] = raw_short[key]
    matched = [bc for bc in he22_obs.index if bc in clean_to_raw]
    raw_idx = {bc: i for i, bc in enumerate(raw_obs_names)}
    rows = [raw_idx[clean_to_raw[bc]] for bc in matched]
    X = a_raw.X[rows]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    del a_raw

    ct = he22_obs.loc[matched, "cell_type"].values
    return X, genes, ct


def load_human_bpd_raw(conditions: list[str]):
    """BPD-PH cohort (Acute26 / BPD7mo / BPDPH7mo / Term0d / Term20d), raw
    counts from the compiled_counts.csv, chunked read."""
    meta = pd.read_csv(C.BPD_META_PATH, index_col="id")
    meta["condition"] = meta["dataset"].map(C.DATASET_TO_CONDITION)
    meta = meta[meta["condition"].isin(conditions) & meta["celltype"].isin(C.TARGET_CELL_TYPES)]
    needed = set(meta.index)

    chunks_X, chunks_bc, genes = [], [], None
    for chunk in pd.read_csv(C.BPD_COUNTS_PATH, index_col="id", chunksize=5000):
        if genes is None:
            genes = remap_gene_names(chunk.columns.tolist())
        rows_here = [bc for bc in chunk.index if bc in needed]
        if rows_here:
            chunks_X.append(chunk.loc[rows_here].values.astype(np.float32))
            chunks_bc.extend(rows_here)
    X = np.vstack(chunks_X)
    ct = meta.loc[chunks_bc, "celltype"].values
    cond = meta.loc[chunks_bc, "condition"].values
    return X, genes, ct, cond


def intersect_and_strip(genes_a, X_a, genes_b, X_b):
    """Common genes between two gene lists, then strip ribo/mito/hgb (human
    naming convention). Returns (common_genes_stripped, X_a_sub, X_b_sub)."""
    common = [g for g in genes_a if g in set(genes_b)]
    idx_a = {g: i for i, g in enumerate(genes_a)}
    idx_b = {g: i for i, g in enumerate(genes_b)}
    X_a_c = X_a[:, [idx_a[g] for g in common]]
    X_b_c = X_b[:, [idx_b[g] for g in common]]
    keep_idx = [i for i, g in enumerate(common) if not C.STRIP_PATTERN_HUMAN.match(g)]
    keep_genes = [common[i] for i in keep_idx]
    return keep_genes, X_a_c[:, keep_idx], X_b_c[:, keep_idx]
