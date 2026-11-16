#!/usr/bin/env python3
"""
panel_sets_human.py — human counterpart of rat_panel_variants/panel_sets.py.

Reads the human rescue workbooks this folder produces
(reports/rat_schema/human_{condition}_{ct}_rescue_predicted_azi.xlsx, same
seven-column rat schema) and returns Rescued_HO_down / Rescued_HO_up as
separate sets, with the gate as a parameter.

The `_rescue_predicted_azi` suffix is load-bearing: the AZI column in these
workbooks is the MODEL's prediction, not a real arm, because no human patient
received azithromycin. Older `_rescue_gene.xlsx` files from an earlier run of
filter_to_rat_schema.py are NOT read here -- if any are still sitting in
reports/rat_schema/, they are stale.

Two variants, matching the rat side one-for-one:
  "full_list"  every row of the workbook, split by recovery_type only
  "gated"      recovery_type AND p_val_adj_HO < 0.05
               AND (p_val_adj_AZI is NaN OR >= 0.05)

Column semantics here (set by human_rescue_pipeline_genomewide.py):
  p_val_adj_HO   Term0_20 -> BPD7mo, both real          "disease changed it"
  p_val_adj_AZI  Term0_20 -> model AZI-pred             "after AZI it is back
                                                         at the control level"
Note the AZI contrast is against the CONTROL (Term0_20), not against BPD7mo
-- that is rat's construction (p_val_adj_AZI there is HO+AZI vs RA, not vs
HO), and >= 0.05 therefore means "normalized", not "AZI did nothing".

Caveat inherited from the workbook: every row already passed rat's
recovery_ratio<0.9 gate, so "full_list" is the rescued set, not the whole
transcriptome -- exactly as on the rat side, where the xlsx is also
post-gate. The genome-wide table (reports/human_rescue_genomewide.csv) is
the unfiltered source if a wider universe is ever needed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
XLSX_DIR = HERE / "reports" / "rat_schema"
CONDITIONS = ("Acute26", "BPD7mo", "BPDPH7mo")
CONDITION = "BPD7mo"
CT_ORDER = ["gCap", "aCap", "Pericyte", "VEC"]

GATES = ("full_list", "gated")
RECOVERY_TYPES = ("Rescued_HO_down", "Rescued_HO_up")

GATE_DESC = {
    "full_list": "all genes of the workbook, split by recovery_type, no p-value filter",
    "gated": "p_val_adj_HO<0.05 (Term->BPD) AND p_val_adj_AZI>=0.05 (AZI back at Term level)",
}

EXPECTED = {
    ("Rescued_HO_down", "disease"): "falls", ("Rescued_HO_down", "azi"): "rises",
    ("Rescued_HO_up", "disease"): "rises", ("Rescued_HO_up", "azi"): "falls",
}


def load_panels(gate: str = "gated", condition: str = CONDITION,
                max_p_ho: float = 0.05,
                min_p_azi: float = 0.05) -> dict[str, dict[str, set[str]]]:
    """-> {cell_type: {"Rescued_HO_down": {...}, "Rescued_HO_up": {...}}}"""
    assert gate in GATES, f"unknown gate {gate!r}, expected one of {GATES}"
    assert condition in CONDITIONS, f"unknown condition {condition!r}"
    out: dict[str, dict[str, set[str]]] = {}
    for ct in CT_ORDER:
        path = XLSX_DIR / f"human_{condition}_{ct}_rescue_predicted_azi.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"{path} -- run filter_to_rat_schema.py first")
        df = pd.read_excel(path, sheet_name="Sheet1")
        if gate == "gated":
            df = df[(df["p_val_adj_HO"] < max_p_ho)
                    & (df["p_val_adj_AZI"].isna() | (df["p_val_adj_AZI"] >= min_p_azi))]
        out[ct] = {rt: set(df.loc[df["recovery_type"] == rt, "gene"]) for rt in RECOVERY_TYPES}
    return out


def main():
    tables = {g: load_panels(g) for g in GATES}
    print(f"{'cell type':10s} {'recovery_type':18s} {'full_list':>10s} {'gated':>7s}  removed by gate")
    for ct in CT_ORDER:
        for rt in RECOVERY_TYPES:
            a = len(tables["full_list"][ct][rt])
            b = len(tables["gated"][ct][rt])
            print(f"{ct:10s} {rt:18s} {a:10d} {b:7d}  {a - b}")


if __name__ == "__main__":
    main()
