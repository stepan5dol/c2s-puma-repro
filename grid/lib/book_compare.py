"""Violin comparisons of the note (sections 2.1, 3.3, 3.4), old run vs new run, from captured per-cell scores.

Per arm x cell type x gene group x side (real / model):
  violin  = Mann-Whitney of group vs background, uncorrected, expected side of the median (as the manuscript)
  book    = violins_book_tests.choose/run_test on d = group - background of the same cell, Bonferroni 8
  MDS     = smallest shift detectable with 80% power at the Bonferroni alpha, (Z_{a/16} + Z_0.8) * SD(d) / sqrt(n)
            (Boslaugh p. 375-378), on the model side, against the real mean shift of the same stratum
usage: book_compare.py old_dir new_dir   (each with violin_scores_{rat,human,human_pairs}.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, norm

sys.path.insert(0, str(Path(__file__).parent))
from violins_book_tests import choose, run_test  # noqa: E402

ALPHA, FAMILY = 0.05, 8
ZF = norm.ppf(1 - ALPHA / FAMILY / 2) + norm.ppf(0.8)
ARMS = [("rat|ho", "rat RA->HO"), ("rat|azi", "rat HO->AZI"), ("human|He22->Acute26", "human He22->Acute26"),
        ("human|Term->BPDPH7mo", "human Term->BPD-PH"), ("human|Term->BPD7mo", "human Term->BPD")]
CT_SHORT = {"General": "gCap", "Aerocyte": "aCap", "Pericyte": "Pericyte", "Vascular": "VEC", "Pulmonary": "VEC"}


def load(d):
    if Path(d).is_file():  # one scores.json per run
        return json.load(open(d))
    sc = {}
    for f in ("rat", "human", "human_pairs"):
        sc.update(json.load(open(Path(d) / f"violin_scores_{f}.json")))
    return sc


def evaluate(sc):
    rows = {}
    for akey, aname in ARMS:
        for key, dd in sc.items():
            if not key.startswith(akey + "|") or "pooled" in key:
                continue
            ct = CT_SHORT[key.split("|")[-1].split()[0]]
            for grp in ("down", "up"):
                exp = (1 if grp == "down" else -1) if akey == "rat|azi" else (-1 if grp == "down" else 1)
                for side in ("real", "model"):
                    bg, x = np.array(dd[f"{side}|bg"]), np.array(dd[f"{side}|{grp}"])
                    d = x - bg
                    test, *_ = choose(d)
                    p, _ = run_test(test, d)
                    med = np.sign(np.median(d))
                    rows[(aname, ct, grp, side)] = dict(
                        n=len(d), test=test, mean=d.mean(),
                        book=bool(min(1, p * FAMILY) < ALPHA and med == exp),
                        violin=bool(mannwhitneyu(bg, x).pvalue < ALPHA and med == exp),
                        mds=ZF * d.std(ddof=1) / np.sqrt(len(d)))
    return rows


R = {"old": evaluate(load(sys.argv[1])), "new": evaluate(load(sys.argv[2]))}
CTS = ["gCap", "aCap", "Pericyte", "VEC"]

print("2.1 / 3.3  cell types (of 4) with the expected shift; book (Bonferroni 8) | violin (MW, uncorrected)")
print(f"{'arm':42s} {'side':5s} {'old down':>9s} {'old up':>9s} {'new down':>9s} {'new up':>9s}")
for _, aname in ARMS:
    for side in ("real", "model"):
        cells = []
        for rn in ("old", "new"):
            for grp in ("down", "up"):
                r = [R[rn][k] for k in R[rn] if k[0] == aname and k[2] == grp and k[3] == side]
                cells.append(f"{sum(x['book'] for x in r)} | {sum(x['violin'] for x in r)}")
        print(f"{aname:42s} {side:5s} " + " ".join(f"{c:>9s}" for c in cells))

for rn in ("old", "new"):
    tests = {}
    for k, x in R[rn].items():
        if k[3] == "model":
            tests[x["test"]] = tests.get(x["test"], 0) + 1
    agree = sum(x["book"] == x["violin"] for k, x in R[rn].items() if k[3] == "model")
    nm = sum(1 for k in R[rn] if k[3] == "model")
    print(f"{rn}: tests chosen (model side) {tests}; book and violin agree in {agree}/{nm}")

print("\n3.4  model side: MDS > |real mean shift| -> 'no effect in the model' is not testable there")
for rn in ("old", "new"):
    low = [f"{a}:{ct}-{g}" for (a, ct, g, s), x in R[rn].items()
           if s == "model" and x["mds"] > abs(R[rn][(a, ct, g, "real")]["mean"])]
    print(f"  {rn}: {len(low)} strata: {low}")

print("\nmodel mean d per stratum, old -> new (real for reference)")
for _, aname in ARMS:
    for ct in CTS:
        for grp in ("down", "up"):
            k = (aname, ct, grp, "model")
            if k not in R["old"]:
                continue
            o, n, re = R["old"][k], R["new"][k], R["old"][(aname, ct, grp, "real")]
            print(f"  {aname:42s} {ct:8s} {grp:4s} n={o['n']:4d}  real {re['mean']:+.3f}  "
                  f"old {o['mean']:+.3f} ({'Y' if o['book'] else '-'})  new {n['mean']:+.3f} ({'Y' if n['book'] else '-'})")
