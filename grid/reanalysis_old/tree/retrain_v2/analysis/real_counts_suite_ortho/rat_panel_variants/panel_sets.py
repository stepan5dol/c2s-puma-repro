#!/usr/bin/env python3
"""
panel_sets.py — panel loader for rat_panel_variants/, replacing
common.load_celltype_rescue_genes() with a version that (a) keeps
Rescued_HO_up and Rescued_HO_down as SEPARATE sets instead of returning
only "down", and (b) makes the p-value filtering optional.

Source is unchanged: the same four full rescue-gene workbooks
common.DGE_XLSX points at (gcap / acap / pericyte / Venous), Sheet1, header
  gene, avg_log2FC_HO, p_val_adj_HO, avg_log2FC_AZI, p_val_adj_AZI,
  recovery_ratio, recovery_type

Two variants:
  "full_list"  every row of the workbook, split by recovery_type only --
               no p-value filtering at all
  "gated"      recovery_type AND p_val_adj_HO < 0.05
               AND (p_val_adj_AZI is None OR >= 0.05)

"gated" is exactly what common.load_celltype_rescue_genes() applies and what
every existing rat rescue figure uses. Its second condition selects genes by
the REAL AZI outcome ("after AZI no longer significantly different from
baseline"), so in the AZI arm the REAL violin is guaranteed to show rescue by
construction -- the genes were chosen for it. "full_list" drops that.

Caveat that applies to BOTH variants: the workbooks are themselves a
pre-selected list (357-540 rows out of a ~18-20k transcriptome, every row
already carrying a recovery_type), built upstream by rescue_gene_pipeline.py.
"full_list" is less conditioned on the AZI outcome, not unconditioned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import openpyxl

ABS_DIR = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/analysis")
sys.path.insert(0, str(ABS_DIR))
import common as C

GATES = ("full_list", "gated")
RECOVERY_TYPES = ("Rescued_HO_down", "Rescued_HO_up")

GATE_DESC = {
    "full_list": "all genes of the list, split by recovery_type, no p-value filter",
    "gated": "p_val_adj_HO<0.05 AND p_val_adj_AZI>=0.05 (the panel used by every existing figure)",
}

EXPECTED = {
    ("Rescued_HO_down", "HO"): "falls", ("Rescued_HO_down", "AZI"): "rises",
    ("Rescued_HO_up", "HO"): "rises", ("Rescued_HO_up", "AZI"): "falls",
}


def load_panels(gate: str = "gated", max_p_ho: float = 0.05,
                min_p_azi: float = 0.05) -> dict[str, dict[str, set[str]]]:
    """-> {cell_type: {"Rescued_HO_down": {...}, "Rescued_HO_up": {...}}}"""
    assert gate in GATES, f"unknown gate {gate!r}, expected one of {GATES}"
    out: dict[str, dict[str, set[str]]] = {}
    for ct, path in C.DGE_XLSX.items():
        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb["Sheet1"]
        data = list(ws.iter_rows(values_only=True))[1:]
        per_rt = {}
        for rt in RECOVERY_TYPES:
            keep = set()
            for r in data:
                if r[6] != rt:
                    continue
                if gate == "gated":
                    if r[2] is None or r[2] >= max_p_ho:
                        continue
                    if not (r[4] is None or r[4] >= min_p_azi):
                        continue
                keep.add(r[0])
            per_rt[rt] = keep
        out[ct] = per_rt
        wb.close()
    return out


def main():
    """Print panel sizes for every (cell type, recovery_type, variant)."""
    tables = {g: load_panels(g) for g in GATES}
    print(f"{'cell type':10s} {'recovery_type':18s} {'full_list':>10s} {'gated':>7s}  removed by gate")
    for ct in C.CT_ORDER:
        for rt in RECOVERY_TYPES:
            a = len(tables["full_list"][ct][rt])
            b = len(tables["gated"][ct][rt])
            print(f"{ct:10s} {rt:18s} {a:10d} {b:7d}  {a - b}")


if __name__ == "__main__":
    main()
