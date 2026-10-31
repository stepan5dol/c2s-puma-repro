#!/usr/bin/env python3
"""
export_for_limma.py — dump the BPD7mo arms as plain matrices so limma's camera
and fry can be run on them (limma_sets.py, through rpy2).

WHY. Everything in this suite that asks a set-level question of the disease arm
so far uses either GSEA prerank or a Mann-Whitney, and both treat genes as
independent. Genes inside a pathway are correlated, so both are anticonservative
-- the p-values are too small by an amount neither method reports. limma's
`camera` estimates the inter-gene correlation and inflates the variance by
1 + (m-1)*rho; `fry` is a self-contained rotation test whose PValue.Mixed asks
"is this set moved at all, in either direction", which is the one standard
statistic that does not cancel on a set holding both directions.

TWO ARMS ARE EXPORTED, per cell type, matching the existing table exactly:

    disease   columns = real Term0d+Term20d control, then real BPD7mo
              contrast = BPD7mo vs control. Both sides real; this is the arm
              that describes the patients.
    azi       columns = real BPD7mo, then the model's AZI-counterfactual
              predictions. contrast = prediction vs disease, i.e. "does the
              model move this pathway back". No human patient received
              azithromycin, so this arm is model behaviour throughout.

    matrix   genes x cells, log1p CPM10k, top-800-truncated -- the same values
             human_rescue_pipeline_genomewide.py computes avg_log2FC from. The
             top-800 budget is kept even on the all-real disease arm ON PURPOSE:
             the AZI arm cannot leave it (the model emits at most 800 genes per
             cell), so an untruncated disease arm would not be comparable to the
             thing it is being compared against.
    rows     the genes of human_rescue_genomewide.csv that pass min.pct on BOTH
             arms, i.e. the same universe every other figure in the deck uses
    stat     avg_log2FC_disease_top800 for those genes, in row order

THE REPLICATION UNIT DOES NOT CHANGE. Columns are cells from 2 BPD7mo and 2
control patients, so every p-value out of this is a statement about these
groups of cells, exactly as the rest of the suite's per-cell tests are.

ONE ASYMMETRY TO KEEP IN MIND ON THE AZI ARM. Its two sides are not produced
the same way: the BPD7mo columns are measured cells truncated to top-800, the
AZI columns are reconstructed from the model's rank list through the
rank->expr curve. Reconstruction flattens cell-to-cell variance, and camera
compares distributions, not just means -- so some of any difference it reports
on that arm is the reconstruction, not the drug prompt. The disease arm has no
such asymmetry.

    python export_for_limma.py   -> reports/limma_input/{ct}.{bin,json}
                                    reports/limma_input/{ct}_azi.{bin,json}
                                    reports/limma_input/sets.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

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
SUITE = HERE.parent
OUT = HERE / "reports" / "limma_input"
CONDITION = "BPD7mo"
COND_MODEL = "BPD"

COND_MODEL_OF = {"BPD7mo": "BPD", "BPDPH7mo": "BPD-PH", "Acute26": "Acute"}
OUT_OF = {"BPD7mo": OUT}
CONTROL_OF = {"BPD7mo": "term", "BPDPH7mo": "term", "Acute26": "he22"}
VARIANT = "azi"
RESULTS_DIR = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results")
MIN_AZI_CELLS = 3
LIBRARIES = ["GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022",
             "WikiPathways_2024_Human", "MSigDB_Hallmark_2020"]


def main():
    global CONDITION, COND_MODEL, OUT, VARIANT
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default=CONDITION, choices=sorted(COND_MODEL_OF),
                    help="cohort of human_rescue_genomewide.csv to export")
    ap.add_argument("--variant", nargs="+", default=["azi"],
                    help="prompt variants of the ablation run; one CSV pass serves all")
    args = ap.parse_args()
    CONDITION, VARIANTS = args.condition, args.variant
    COND_MODEL = COND_MODEL_OF[CONDITION]
    OUT_OF_VARIANT = {v: HERE / "reports" / f"ablation_limma_input_{CONDITION.lower()}_{v}"
                      for v in VARIANTS}
    print(f"[cohort] {CONDITION} (model calls it {COND_MODEL}) -> {OUT.name}")

    for d in OUT_OF_VARIANT.values():
        d.mkdir(parents=True, exist_ok=True)

    print("[calibration] fitting rank->expr model on GSE275938 compiled counts...")
    model = rem.fit_from_csv(CH.COUNTS_CSV)

    control = CONTROL_OF[CONDITION]
    if control == "term":
        print("[data] real Term0d+Term20d control...")
        term_mats = RC.real_term_matrices(model)
    else:
        print("[data] real He22 fetal GW22 control...")
        term_mats = RC.real_he22_matrices(model)
    print(f"[data] real {CONDITION}...")
    disease_mats = RC.real_disease_matrices(CONDITION, model)
    azi_groups_of = {}
    for v in VARIANTS:
        print(f"[data] model AZI-counterfactual predictions, variant {v}...")
        azi_groups_of[v] = CH.load_azi_counterfactual(RESULTS_DIR / f"inference_azi_results_{v}.jsonl")

    gw = pd.read_csv(HERE / "reports" / "human_rescue_genomewide.csv")
    gw = gw[(gw.condition == CONDITION) & gw.passes_min_pct_disease & gw.passes_min_pct_azi]

    gene_index = {g: i for i, g in enumerate(model.gene_names)}

    for ct in CH.CT_ORDER:
        if ct not in disease_mats or ct not in term_mats:
            continue
        g = gw[gw.cell_type == ct]
        keep = np.array([gene_index[x] for x in g.gene])

        ctrl = CR.truncate_topk(term_mats[ct])[:, keep]
        dis = CR.truncate_topk(disease_mats[ct])[:, keep]

        azi_of = {}
        for v in VARIANTS:
            arecs = azi_groups_of[v].get((ct, COND_MODEL), [])
            if len(arecs) < MIN_AZI_CELLS:
                print(f"  {ct}/{v}: no AZI arm, only {len(arecs)} predictions")
                continue
            azi_of[v] = rem.reconstruct_batch([r["pred"] for r in arecs], model, CR.K)[:, keep]

        arms = {("disease", v): (ctrl, dis,
                                 "term-born" if control == "term" else "fetal GW22", CONDITION)
                for v in VARIANTS}
        arms.update({("azi", v): (dis, azi_of[v], CONDITION, "AZI-pred") for v in azi_of})

        for (arm, variant), (A, B, la, lb) in arms.items():
            OUT = OUT_OF_VARIANT[variant]
            X = np.vstack([A, B]).astype(np.float32)
            group = np.array([0] * A.shape[0] + [1] * B.shape[0])
            stem = ct if arm == "disease" else f"{ct}_azi"
            X.T.tofile(OUT / f"{stem}.bin")
            (OUT / f"{stem}.json").write_text(json.dumps({
                "cell_type": ct, "arm": arm,
                "n_genes": int(X.shape[1]), "n_cells": int(X.shape[0]),
                "n_ref": int(A.shape[0]), "n_test": int(B.shape[0]),
                "label_ref": la, "label_test": lb,
                "genes": list(g.gene), "group": group.tolist(),
                "stat": g.avg_log2FC_disease_top800.tolist(),
            }))
            print(f"  {ct} [{arm}/{variant}]: {X.shape[1]} genes x {X.shape[0]} cells "
                  f"({A.shape[0]} {la} + {B.shape[0]} {lb})")

    cache = HERE / "reports" / "ablation_limma_all_terms.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
        print(f"\n  reusing {cache.name}")
    else:
        blob = {}
        for lib in LIBRARIES:
            print(f"  [{lib}] downloading...", flush=True)
            blob[lib] = gp.get_library(name=lib)
        cache.write_text(json.dumps(blob))

    sets = {}
    for lib, terms in blob.items():
        for t, genes in terms.items():
            sets[f"{lib}__{t}"] = sorted(set(genes))
    for d in OUT_OF_VARIANT.values():
        (d / "sets.json").write_text(json.dumps(sets))
    print(f"\n  {len(sets)} library terms from {len(blob)} libraries -> sets.json")
    for d in OUT_OF_VARIANT.values():
        print(f"-> {d}")


if __name__ == "__main__":
    main()
