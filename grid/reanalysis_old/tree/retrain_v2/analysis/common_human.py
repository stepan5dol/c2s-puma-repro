"""
common_human.py — human-side counterpart to common.py.

Data sources (per pipeline/common.py, the pipeline that actually built these
datasets -- reused here directly instead of re-derived):
  - Cell records: the SAME inference JSONLs as the rat analysis (common.RUNS).
    They already contain "Homo sapiens" fwd-direction records (453 of them);
    the rat scripts just filter those out via species.startswith("Rattus").
  - Calibration counts: BPD-PH/GSE275938_compiled_counts.csv (pipeline.common.
    BPD_COUNTS_PATH) -- the raw-counts source pipeline/stage0_split.py's
    _load_bpd() uses for the Acute26/BPD7mo/BPDPH7mo conditions. This is NOT
    he_lung_atlas.h5ad / 2022FetalLungIntCounts.h5ad (those back the He15-22
    healthy-atlas timepoints, a different transition not covered here).

No AZI (treatment) arm exists for human -- these are three disease-trajectory
transitions, each measured against its OWN baseline (embedded in the record's
"Unexposed:" field, exactly like common.py's rat convention):
  Acute   : GW22 healthy fetal lung          -> GW26 acute preterm injury
  BPD     : GW26 acute preterm injury        -> 7mo bronchopulmonary dysplasia
  BPD-PH  : GW26 acute preterm injury        -> 7mo BPD with pulmonary hypertension
BPD and BPD-PH are independent chronic outcomes from the same acute baseline,
not a progression from one to the other.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("/Users/stepandolzhenko/Documents/AzithroGemma")
sys.path.insert(0, str(BASE))
from pipeline.common import BPD_COUNTS_PATH

from common import (
    K, MIN_N, SEED, RUNS,
    dedupe_first, reconstruct_group, filtered_gene_domain, pairwise_dist, cdist,
    fit_pca_pooled, fit_pca_shared, centroid_distance, centroid_perm_test,
    centroid_distance_bootstrap_ci, delta_and_mask, per_cell_rescue_auc,
    forest_plot, knn_mixing_lisi, unpaired_perm_test, bootstrap_ci_two_sample,
    bootstrap_ci, auc_mannwhitney, swap_perm_test, permdisp,
)

COUNTS_CSV = BPD_COUNTS_PATH

RESCUED_HUMAN_PATH = BASE / "train-after-grpo-analysis" / "rescued_human_genes.txt"


def load_rescue_genes() -> set[str]:
    return set(RESCUED_HUMAN_PATH.read_text().splitlines()) - {""}

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures_human"
FIG_DIR.mkdir(exist_ok=True)

CT_ORDER = ["gCap", "aCap", "Pericyte", "VEC"]
CT_MAP = {
    "general capillary endothelial cell":  "gCap",
    "aerocyte capillary endothelial cell": "aCap",
    "pericyte":                            "Pericyte",
    "pulmonary venous endothelial cell":   "VEC",
}
COND_ORDER = ["Acute", "BPD", "BPD-PH"]

_PROMPT_RE = {
    "species":      re.compile(r"Species:\s*([^\n]+)"),
    "cell_type":    re.compile(r"Cell type:\s*([^\(\n]+)"),
    "perturbation": re.compile(r"Perturbation:\s*([^\n]+)"),
    "unexposed":    re.compile(r"\nUnexposed:\s*([^\n]+)"),
}


def condition_of(perturbation: str) -> str | None:
    if perturbation.startswith("Acute preterm lung injury"):
        return "Acute"
    if "pulmonary hypertension" in perturbation:
        return "BPD-PH"
    if perturbation.startswith("Progression from acute preterm lung injury"):
        return "BPD"
    if perturbation.startswith("Disease progression to BPD with pulmonary hypertension"):
        return "BPD-PH"
    return None


def he22_reference_by_ct(groups: dict[tuple[str, str], list[dict]], model) -> dict[str, "np.ndarray"]:
    """Mean reconstructed He22 (healthy, GW22) expression per cell type.

    Only the Acute condition's records carry a real He22 cell in their own
    "Unexposed:" field (verified against the raw prompt's Age: line -- Acute's
    baseline is GW22, BPD's and BPD-PH's is GW26/Acute26, not He22). To
    compare every condition against the SAME healthy reference, pool the
    Acute group's per-record He22 baselines per cell type and take the mean;
    that mean vector substitutes for BPD/BPD-PH's own (Acute26) "Unexposed:"
    field in delta computations that must be anchored to health, not to the
    immediately preceding disease stage.
    """
    import numpy as np
    import rank_expr_model as rem

    out = {}
    for ct in CT_ORDER:
        acute_group = groups.get((ct, "Acute"), [])
        if not acute_group:
            continue
        he22_X = rem.reconstruct_batch([c["unexposed"] for c in acute_group], model, K)
        out[ct] = he22_X.mean(axis=0)
    return out


AZI_RESULTS_PATH = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/inference_azi_results.jsonl")
_AGE_RE = re.compile(r"Age:\s*([^\n]+)")


def condition_of_age(age: str) -> str | None:
    """Bucket a record by its Age: line (COND_META age_str per pipeline.common),
    since in AZI_RESULTS_PATH every record's Perturbation: line is the SAME
    AZI-treatment string -- the disease stage is only distinguishable via age."""
    if "acute preterm injury" in age:
        return "Acute"
    if "BPD with pulmonary hypertension" in age:
        return "BPD-PH"
    if "(BPD)" in age:
        return "BPD"
    return None


def load_azi_counterfactual(jsonl_path: Path = AZI_RESULTS_PATH) -> dict[tuple[str, str], list[dict]]:
    """Human AZI counterfactual: real disease-state baseline ("Unexposed:")
    paired with the model's predicted post-AZI cell sentence ("pred"). No
    ground truth exists (gt is always empty in this file) -- azithromycin was
    never actually given to these human patients; "pred" is what the model,
    trained on rat AZI response, predicts would happen if it were."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("direction") != "fwd":
                continue
            prompt = r["prompt"]
            species = _PROMPT_RE["species"].search(prompt).group(1).strip()
            if not species.startswith("Homo"):
                continue
            ct_raw = _PROMPT_RE["cell_type"].search(prompt).group(1).strip()
            ct = CT_MAP.get(ct_raw)
            if ct is None:
                continue
            age = _AGE_RE.search(prompt).group(1).strip()
            cond = condition_of_age(age)
            if cond is None:
                continue
            unexposed = _PROMPT_RE["unexposed"].search(prompt).group(1).strip().split()[:K]
            pred = dedupe_first(r["pred"].strip().split())[:K]
            groups[(ct, cond)].append({"unexposed": unexposed, "pred": pred})
    return groups


def load_cells(jsonl_path: Path) -> dict[tuple[str, str], list[dict]]:
    """Same shape/contract as common.load_cells, filtered to Homo sapiens."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("direction") != "fwd":
                continue
            prompt = r["prompt"]
            species = _PROMPT_RE["species"].search(prompt).group(1).strip()
            if not species.startswith("Homo"):
                continue
            ct_raw = _PROMPT_RE["cell_type"].search(prompt).group(1).strip()
            ct = CT_MAP.get(ct_raw)
            if ct is None:
                continue
            pert = _PROMPT_RE["perturbation"].search(prompt).group(1).strip()
            cond = condition_of(pert)
            if cond is None:
                continue
            unexposed = _PROMPT_RE["unexposed"].search(prompt).group(1).strip().split()[:K]
            gt = r["gt"].strip().split()[:K]
            pred = dedupe_first(r["pred"].strip().split())[:K]
            groups[(ct, cond)].append({"unexposed": unexposed, "gt": gt, "pred": pred})
    return groups
