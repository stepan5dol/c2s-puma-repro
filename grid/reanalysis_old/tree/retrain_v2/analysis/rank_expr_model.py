"""
rank_expr_model.py — calibrated rank -> absolute expression reconstruction.

Cell sentences carry gene ORDER only (rank), not values. To compare gt/pred
cell sentences in expression space rather than rank-distance space, we need
a rank -> expression map.

C2S-Scale (NIHPP 2025.04.14.648850v4, Supp. Fig. 8) fits this as LINEAR IN
LOG-RANK, not raw rank:

    log1p_cpm10k(gene) ~= intercept + slope * log(rank + 1)

Verified empirically on rat.ho.azi.integrated.h5ad (counts layer, K=800,
n=2000 cells, seed=0):
    raw-rank fit:  R^2 = 0.639
    log-rank fit:  R^2 = 0.836   <- matches paper's 0.82-0.87 range

So log-rank is used here, not raw rank.

The model is fit ONCE on real counts from the ground-truth h5ad, then
applied IDENTICALLY to both gt and pred gene-rank-lists coming out of
inference. Both sides pass through the same reconstruction loss, so
gt-vs-pred comparisons in reconstructed-expression space are fair (the
comparison never touches real counts for pred, which has none).

Caveat carried through by design: this is one global linear curve, not
per-cell true counts. Always report r2 alongside any downstream metric
built on top of this reconstruction.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import anndata


class RankExprModel(NamedTuple):
    slope: float
    intercept: float
    r2: float
    gene_to_idx: dict[str, int]
    gene_names: list[str]

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)


def _fit_from_matrix(X: np.ndarray, gene_names: list[str], k: int, n_cells: int, seed: int) -> RankExprModel:
    """Shared calibration core: raw counts [n_cells, n_genes] -> RankExprModel.
    Used by both fit() (h5ad source) and fit_from_csv() (CSV source, e.g. the
    human GSE275938 compiled-counts table) so the log-rank -> log1p(CPM10k)
    math is defined exactly once."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_cells, len(X)), replace=False)
    X = X[idx]
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    X_norm = np.log1p(X / lib * 1e4)

    log_ranks_parts, exprs_parts = [], []
    for row in X_norm:
        order = np.argsort(-row)[:k]
        e = row[order]
        mask = e > 0
        r = np.arange(k)[mask]
        log_ranks_parts.append(np.log(r + 1.0))
        exprs_parts.append(e[mask])
    log_ranks = np.concatenate(log_ranks_parts)
    exprs = np.concatenate(exprs_parts)

    A = np.column_stack([np.ones_like(log_ranks), log_ranks])
    (intercept, slope), *_ = np.linalg.lstsq(A, exprs, rcond=None)
    y_pred = intercept + slope * log_ranks
    ss_res = float(((exprs - y_pred) ** 2).sum())
    ss_tot = float(((exprs - exprs.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    return RankExprModel(float(slope), float(intercept), float(r2), gene_to_idx, gene_names)


def fit(
    h5ad_path: str | Path,
    k: int = 800,
    n_cells: int = 2000,
    seed: int = 0,
    layer: str = "counts",
) -> RankExprModel:
    """Fit log-rank -> log1p(CPM10k) linear model on real counts (h5ad source)."""
    adata = anndata.read_h5ad(h5ad_path)
    X = adata.layers[layer] if layer in adata.layers else adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    gene_names = list(adata.var_names)
    return _fit_from_matrix(X, gene_names, k, n_cells, seed)


def fit_from_csv(
    counts_csv_path: str | Path,
    k: int = 800,
    n_cells: int = 2000,
    seed: int = 0,
    id_col: str = "id",
) -> RankExprModel:
    """Fit log-rank -> log1p(CPM10k) linear model on real counts stored as a
    wide cells x genes CSV (e.g. BPD-PH/GSE275938_compiled_counts.csv -- the
    human Acute/BPD/BPD-PH raw-counts source per pipeline/common.py's
    BPD_COUNTS_PATH). Reads a random n_cells-row subsample directly (the file
    is tens of thousands of rows / 3+ GB, too large to load whole), via
    skiprows on a pre-drawn random row-index set -- same sampling semantics
    as fit()'s rng.choice over an in-memory matrix."""
    import pandas as pd

    with open(counts_csv_path) as f:
        total_rows = sum(1 for _ in f) - 1

    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(total_rows, size=min(n_cells, total_rows), replace=False).tolist())
    skip = lambda i: i != 0 and (i - 1) not in chosen

    df = pd.read_csv(counts_csv_path, skiprows=skip, index_col=id_col)
    X = df.to_numpy(dtype=np.float64)
    gene_names = df.columns.tolist()
    return _fit_from_matrix(X, gene_names, k, n_cells, seed)


def reconstruct(genes: list[str], model: RankExprModel, k: int = 800,
                 floor_rank: int | None = None) -> np.ndarray:
    """Rank-ordered gene-name list -> dense [n_genes] log1p(CPM10k) vector.

    Genes absent from the list default to 0 ("not measured"). Pass
    floor_rank (e.g. k+1) to instead give absent genes the model's calibrated
    value AT that rank -- i.e. "at most this low", not "exactly zero". Only
    meaningful for cell sentences with no real-count alternative (model
    output); real measured cells should skip reconstruction entirely and use
    their actual counts instead (see pathway_module_analysis/real_counts.py)."""
    vec = np.zeros(model.n_genes, dtype=np.float32)
    if floor_rank is not None:
        floor_val = model.intercept + model.slope * np.log(floor_rank + 1.0)
        if floor_val > 0:
            vec[:] = floor_val
    for rank, g in enumerate(genes[:k]):
        j = model.gene_to_idx.get(g)
        if j is None:
            continue
        val = model.intercept + model.slope * np.log(rank + 1.0)
        if val > 0:
            vec[j] = val
    return vec


def reconstruct_batch(gene_lists: list[list[str]], model: RankExprModel, k: int = 800,
                       floor_rank: int | None = None) -> np.ndarray:
    """[n_cells] gene-name lists -> [n_cells, n_genes] dense reconstructed matrix."""
    out = np.zeros((len(gene_lists), model.n_genes), dtype=np.float32)
    for i, genes in enumerate(gene_lists):
        out[i] = reconstruct(genes, model, k, floor_rank=floor_rank)
    return out


if __name__ == "__main__":
    H5AD = "/Users/stepandolzhenko/Documents/AzithroGemma/rat.ho.azi.integrated.h5ad"
    m = fit(H5AD)
    print(f"slope={m.slope:.5f} intercept={m.intercept:.4f} r2={m.r2:.4f} n_genes={m.n_genes}")
