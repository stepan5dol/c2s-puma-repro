"""
Shared paths, constants, and helper functions for the cross-species
(rat HO/RA vs human disease/control) pathway-module comparison.

Design decisions carried over from the conversation that produced this:
  - Effect sizes (Cohen's d), not p-values: both rat (n=1 animal/condition)
    and the human BPD-PH cohort (n=1-2 samples/condition) lack biological
    replication, so per-cell tests (Wilcoxon etc.) are pseudoreplicated --
    FDR/p-values from them don't mean what they normally mean. See
    memory/rat_rescue_gene_pipeline.md and the conversation for the full
    argument.
  - AUCell/ULM (via decoupler) instead of raw log2FC + GSEA: rank-based
    scoring is far less sensitive to sequencing-depth differences between
    groups, which is a real, confirmed confound here (rat disease has HIGHER
    depth than control; human disease has LOWER depth than control -- see
    diagnose_depth.py).
  - Ribosomal/mitochondrial/hemoglobin genes stripped from the scoring
    matrix (not just excluded from gene sets) before scoring, since they
    dominate the composition and are highly depth-sensitive.
  - Pathways collapsed into modules by Jaccard gene-overlap (>=0.5) BEFORE
    scoring, not after, to avoid ~100 collinear numbers.
"""
from pathlib import Path
import re

import numpy as np

BASE = Path("/Users/stepandolzhenko/Documents/AzithroGemma")
RAT_PATH = BASE / "rat.ho.azi.integrated.h5ad"
ATLAS_CLEAN_PATH = BASE / "he_lung_atlas.h5ad"
ATLAS_RAW_PATH = BASE / "2022FetalLungIntCounts.h5ad"
BPD_META_PATH = BASE / "BPD-PH/GSE275938_cell_metadata.csv"
BPD_COUNTS_PATH = BASE / "BPD-PH/GSE275938_compiled_counts.csv"

HERE = Path(__file__).parent
OUT_DIR = HERE / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

TARGET_CELL_TYPES = {"gCap", "aCap", "Pulmonary venous EC", "Pericyte"}
DISEASE_CONDITIONS = ["Acute26", "BPD7mo", "BPDPH7mo"]

DATASET_TO_CONDITION = {
    "Acute preterm injury 1": "Acute26",
    "BPD 1": "BPD7mo", "BPD 2": "BPD7mo",
    "BPD+PH 1": "BPDPH7mo", "BPD+PH 2": "BPDPH7mo",
    "Term infant 1": "Term0d", "Term infant 2": "Term20d",
}

CT_MATCH_RAT_TO_HUMAN = {"Peri": "Pericyte", "VEC": "Pulmonary venous EC", "aCAP": "aCap", "gCAP": "gCap"}

STRIP_PATTERN_RAT = re.compile(r"^(Rpl|Rps|Mrpl|Mrps|Mt-|Hba|Hbb)", re.IGNORECASE)
STRIP_PATTERN_HUMAN = re.compile(r"^(RPL|RPS|MRPL|MRPS|MT-|HBA|HBB)", re.IGNORECASE)


def harmonize_atlas_celltype(raw: str) -> str:
    m = {
        "Aerocyte": "aCap", "Early cap": "gCap", "Mid cap": "gCap", "Late cap": "gCap",
        "Venous endo": "Pulmonary venous EC",
    }
    return m.get(raw, raw)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """x - y, pooled-SD standardized. Positive = higher in x."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / pooled_std) if pooled_std > 0 else 0.0


def downsample_counts(X: np.ndarray, target_depth: int, rng: np.random.Generator) -> np.ndarray:
    """Multinomial downsample each row (cell) to target_depth total counts.
    Rows already at or below target_depth are left unchanged."""
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        row = X[i]
        total = row.sum()
        if total <= target_depth:
            out[i] = row
            continue
        p = (row / total).astype(np.float64)
        p = p / p.sum()
        out[i] = rng.multinomial(target_depth, p)
    return out
