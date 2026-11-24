#!/usr/bin/env python3
"""Per-stratum summary of the manuscript violins from the reports violins_only.py writes.

Reads report_{human,rat}_delta_<contrast>_all_rescued.csv of the five contrasts (gate full_list; the POOLED row is
not one of the 40 strata) under <violins dir>/{human,rat}/reports/. For each contrast x cell type x gene set:
  shift        median of the gene set minus median of the background, observed and model arm separately
  significant  Mann-Whitney p < 0.05 (the report's p) and the shift in the expected direction (the report's
               expected_down / expected_up; human reports have no such column: down falls, up rises)
  match        observed and model both significant or both not (the violin match count of the run tables)
  ratio        model shift / observed shift
Prints: strata, same sign, significance matches, and over the strata with a significant observed shift the median,
interquartile range and range of the ratio; then the strata without a significant observed shift.
usage: python3 violin_shift_summary.py <violins dir> <out csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONTRASTS = [
    ("human", "fetal_gw22_to_ali_gw26", "human fetal GW22 -> acute injury GW26"),
    ("human", "term_born_to_bpd_7mo", "human term-born -> BPD 7 months"),
    ("human", "term_born_to_bpd_ph_7mo", "human term-born -> BPD-PH 7 months"),
    ("rat", "room_air_to_hyperoxia", "rat room air -> hyperoxia"),
    ("rat", "hyperoxia_to_azithromycin", "rat hyperoxia -> + azithromycin"),
]


def main() -> None:
    violins, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for sp, stem, label in CONTRASTS:
        d = pd.read_csv(violins / sp / "reports" / f"report_{sp}_delta_{stem}_all_rescued.csv")
        d = d[(d.gate == "full_list") & (d.cell_type != "POOLED")]
        for _, r in d.iterrows():
            for g in ("down", "up"):
                expected = r.get(f"expected_{g}", "falls" if g == "down" else "rises")
                sign = -1 if expected == "falls" else 1
                obs = r[f"median_real_{g}"] - r["median_real_bg"]
                mod = r[f"median_model_{g}"] - r["median_model_bg"]
                rows.append({
                    "contrast": label, "cell_type": r["cell_type"], "gene_set": g, "expected": expected,
                    "n_observed": r.get("n_real"), "n_model": r.get("n_model"),
                    "shift_observed": obs, "shift_model": mod,
                    "p_observed": r[f"p_real_{g}"], "p_model": r[f"p_model_{g}"],
                    "sig_observed": bool(r[f"p_real_{g}"] < 0.05 and np.sign(obs) == sign),
                    "sig_model": bool(r[f"p_model_{g}"] < 0.05 and np.sign(mod) == sign),
                })
    t = pd.DataFrame(rows)
    t["match"] = t.sig_observed == t.sig_model
    t["same_sign"] = np.sign(t.shift_observed) == np.sign(t.shift_model)
    t["ratio"] = t.shift_model / t.shift_observed
    t.to_csv(out_csv, index=False)

    s = t[t.sig_observed]
    print(f"strata {len(t)}; same sign {int(t.same_sign.sum())}; significance matches {int(t.match.sum())}")
    print(f"significant observed shift: {len(s)}; model also significant: {int(s.sig_model.sum())}; "
          f"ratio model/observed median {s.ratio.median():.3f}, IQR {s.ratio.quantile(.25):.3f}-{s.ratio.quantile(.75):.3f}, "
          f"range {s.ratio.min():.3f}-{s.ratio.max():.3f}")
    print("\nper contrast (matches / strata):")
    print(t.groupby("contrast", sort=False).match.agg(["sum", "count"]).to_string())
    print("\nstrata without a significant observed shift:")
    cols = ["contrast", "cell_type", "gene_set", "n_observed", "shift_observed", "p_observed", "shift_model", "p_model"]
    print(t[~t.sig_observed][cols].to_string(index=False))
    print(f"\n-> {out_csv}")


if __name__ == "__main__":
    main()
