#!/usr/bin/env python3
"""
filter_to_rat_schema.py — turn the genome-wide table into rescue workbooks
with the SAME shape as the rat ones.

scripts/rescue_gene_pipeline.py writes only the genes that pass the gate,
with exactly seven columns, sorted by p_val_adj_HO ascending, one xlsx per
cell type:

    gene | avg_log2FC_HO | p_val_adj_HO | avg_log2FC_AZI | p_val_adj_AZI
         | recovery_ratio | recovery_type

human_rescue_pipeline_genomewide.py instead keeps every gene and marks it
with boolean columns (`disease_candidate`, `azi_candidate`, `rescued`), so
the full table can be re-gated later without re-running the sweep. That is
the source, not the deliverable. This script produces the deliverable.

Column mapping (top-800 measurements -- the ones recovery_ratio is computed
from, see the genome-wide script's docstring on the gene budget):

    avg_log2FC_disease_top800  ->  avg_log2FC_HO
    p_val_adj_disease_top800   ->  p_val_adj_HO
    avg_log2FC_azi_top800      ->  avg_log2FC_AZI
    p_val_adj_azi_top800       ->  p_val_adj_AZI

Row filter is `rescued`, which the genome-wide script already defines as
rat's gate: disease_candidate AND azi_candidate AND recovery_ratio < 0.9.

Output: one xlsx per (condition, cell type) in reports/rat_schema/, Sheet1,
same header order as rat -- so common.load_celltype_rescue_genes() and
rat_panel_variants/panel_sets.py can read them unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "reports" / "human_rescue_genomewide.csv"
OUT_DIR = HERE / "reports" / "rat_schema"

RAT_COLUMNS = ["gene", "avg_log2FC_HO", "p_val_adj_HO", "avg_log2FC_AZI",
               "p_val_adj_AZI", "recovery_ratio", "recovery_type"]

RENAME = {
    "avg_log2FC_disease_top800": "avg_log2FC_HO",
    "p_val_adj_disease_top800": "p_val_adj_HO",
    "avg_log2FC_azi_top800": "avg_log2FC_AZI",
    "p_val_adj_azi_top800": "p_val_adj_AZI",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[read] {SRC}")
    df = pd.read_csv(SRC)
    print(f"  {len(df)} rows in, {df['rescued'].sum()} rescued")

    summary = []
    for (cond, ct), sub in df[df["rescued"]].groupby(["condition", "cell_type"], sort=False):
        table = sub.rename(columns=RENAME)[RAT_COLUMNS].copy()
        table = table.sort_values("p_val_adj_HO", kind="mergesort").reset_index(drop=True)

        out_path = OUT_DIR / f"human_{cond}_{ct}_rescue_predicted_azi.xlsx"
        table.to_excel(out_path, index=False, sheet_name="Sheet1")

        n_down = int((table["recovery_type"] == "Rescued_HO_down").sum())
        n_up = int((table["recovery_type"] == "Rescued_HO_up").sum())
        summary.append({"condition": cond, "cell_type": ct, "n_genes": len(table),
                        "n_down": n_down, "n_up": n_up, "file": out_path.name})
        print(f"  {cond}/{ct}: {len(table)} genes ({n_down} down / {n_up} up) -> {out_path.name}")

    rep = pd.DataFrame(summary)
    rep_path = OUT_DIR / "index.csv"
    rep.to_csv(rep_path, index=False)
    print(f"\n-> {OUT_DIR}  ({len(rep)} workbooks)")
    print(f"-> {rep_path}")


if __name__ == "__main__":
    main()
