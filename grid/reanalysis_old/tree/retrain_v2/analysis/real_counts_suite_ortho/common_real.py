"""
common_real.py — shared primitives for real_counts_suite: every test here
compares REAL raw counts of real cells (top-800-truncated to match the
model's forced K=800 budget) against the model's RECONSTRUCTED top-800
output (rank_expr_model, since pred never has real counts to begin with).

This is deliberately a small, separate module from abs_analysis/common.py --
that module's tests reconstruct BOTH sides from rank lists (gt AND pred),
which is a fundamentally different (weaker, rank-only) comparison. See
real_counts_suite/README.md for the criterion this suite holds to.

Only T=1.0. Only these two files are source of truth for model answers:
  /Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/test_inference_results_t10.jsonl  (rat+human non-AZI)
  /Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/inference_azi_results.jsonl    (human AZI, default variant)
"""
from __future__ import annotations

import numpy as np

K = 800


def truncate_topk(X: np.ndarray, k: int = K) -> np.ndarray:
    """Zero everything except each row's own top-k values -- same K=800 budget
    pred is stuck with by cell-sentence format, applied to real values too.
    Without this, real (unbounded ~18-20k genes) vs pred (K=800-capped) is
    apples-to-oranges: most background genes are real-nonzero/pred-structural-
    zero purely from the vocabulary-size mismatch, not from model behavior."""
    Xt = np.zeros_like(X)
    if X.shape[1] <= k:
        return X.copy()
    part = np.argpartition(-X, k, axis=1)[:, :k]
    rows = np.arange(X.shape[0])[:, None]
    Xt[rows, part] = X[rows, part]
    return Xt


def expression_matched_null(anchor_vec: np.ndarray, gene_idx: np.ndarray, indicator: np.ndarray,
                             n_bins: int = 20, n_perm: int = 1000, seed: int = 0):
    """Shared expression-matched permutation null: bin ALL genes by anchor_vec
    (e.g. real reference mean expression) into n_bins quantile bins; resample
    a panel-sized set of genes from the SAME bins n_perm times; compare the
    observed indicator rate (e.g. 'falls'/'rises' boolean) on the real panel
    against this null distribution.

    Returns (observed_rate, null_median, fold, empirical_p)."""
    rng = np.random.default_rng(seed)
    bin_edges = np.quantile(anchor_vec, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-6
    gene_bins = np.digitize(anchor_vec, bin_edges[1:-1])
    bin_to_genes = {b: np.where(gene_bins == b)[0] for b in range(n_bins)}

    observed_bins = gene_bins[gene_idx]
    observed_rate = float(indicator[gene_idx].mean())
    null_rates = np.empty(n_perm)
    for p in range(n_perm):
        sampled = np.array([rng.choice(bin_to_genes[b]) for b in observed_bins])
        null_rates[p] = indicator[sampled].mean()
    null_median = float(np.median(null_rates))
    fold = observed_rate / null_median if null_median > 0 else float("nan")
    empirical_p = float((null_rates >= observed_rate).mean())
    return observed_rate, null_median, fold, empirical_p
