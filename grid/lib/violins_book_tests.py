"""Book-driven test selection for the manuscript's violin comparisons (Boslaugh, 2nd ed.).

Data: per cell, mean delta of a rescued-gene group and of the background genes of the SAME
cell -> related samples, within-subjects design (p. 470; Ch. 6 p. 164; Ch. 13 p. 316-321).
Unit = paired difference d = group - background, one per cell.

Selection rule, fixed before looking at the p-values, for each comparison:
  1. extreme outliers of d by the IQR rule (beyond 3 IQR from the quartiles, p. 96-97);
  2. normality of d by Anderson-Darling at 5% (p. 72);
  3. skewness |g1| > 1 counts as highly skewed (convention; the book only says Wilcoxon is
     not appropriate for highly skewed data, p. 317, and that skewed data need n >= 40 for the
     central limit theorem, p. 60);
  4. test:
       repeated measures t (p. 164)  if no extreme outliers and
                                      (normal, or n >= 30 and not highly skewed, or n >= 40)
       Wilcoxon signed ranks (p. 317) if not chosen above and not highly skewed
       sign test (p. 312)              otherwise
  5. two-tailed (p. 65), alpha 0.05, Bonferroni within a family (p. 444); family = one arm x
     one side (real or model): 4 cell types x 2 gene groups = 8 tests;
  6. reported with the mean difference and its 95% CI (p. 166) and the smallest p the chosen
     test can reach at this n, against the Bonferroni threshold (sample size, p. 375-377).
Pooled ('All cell types pooled') panels are left out: they repeat the same cells.
usage: violins_book_tests.py <label> scores.json [<label> scores.json ...]
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import anderson, binomtest, mannwhitneyu, skew, t as tdist, ttest_1samp, wilcoxon

ALPHA, FAMILY = 0.05, 8
ARMS = [("rat|ho", "rat RA->HO"), ("rat|azi", "rat HO->AZI"), ("human|He22->Acute26", "human He22->Acute26"),
        ("human|Term->BPDPH7mo", "human Term->BPD-PH"), ("human|Term->BPD7mo", "human Term->BPD"),
        ("human|Acute26->BPD7mo", "human Acute26->BPD (pairs, dge_filtered)")]
CT_SHORT = {"General": "gCap", "Aerocyte": "aCap", "Pericyte": "Pericyte", "Vascular": "VEC", "Pulmonary": "VEC"}


def choose(d):
    n = len(d)
    q1, q3 = np.percentile(d, [25, 75])
    iqr = q3 - q1
    outl = int(((d < q1 - 3 * iqr) | (d > q3 + 3 * iqr)).sum())
    ad = anderson(d, dist="norm")
    normal = ad.statistic < ad.critical_values[list(ad.significance_level).index(5.0)]
    g1 = float(skew(d))
    highly = abs(g1) > 1
    if outl == 0 and (normal or (n >= 30 and not highly) or n >= 40):
        test = "t"
    elif not highly:
        test = "wilcoxon"
    else:
        test = "sign"
    return test, outl, normal, g1


def run_test(test, d):
    n = len(d)
    if test == "t":
        p = ttest_1samp(d, 0).pvalue
        pmin = 0.0
    elif test == "wilcoxon":
        p = wilcoxon(d).pvalue
        pmin = 2 / 2 ** n
    else:
        k, m = int((d > 0).sum()), int((d != 0).sum())
        p = binomtest(k, m, 0.5).pvalue
        pmin = 2 * 0.5 ** m
    return p, pmin


def main():
    args = sys.argv[1:]
    out = {}
    for label, path in zip(args[::2], args[1::2]):
        sc = json.load(open(path))
        rows = []
        for akey, aname in ARMS:
            for key, dd in sc.items():
                if not key.startswith(akey) or "pooled" in key:
                    continue
                ct = CT_SHORT[key.split("|")[-1].split()[0]]
                for side in ("real", "model"):
                    for grp in ("down", "up"):
                        exp = ((1 if grp == "down" else -1) if akey == "rat|azi" else (-1 if grp == "down" else 1))
                        bg, x = np.array(dd[f"{side}|bg"]), np.array(dd[f"{side}|{grp}"])
                        d = x - bg
                        test, outl, normal, g1 = choose(d)
                        p, pmin = run_test(test, d)
                        n = len(d)
                        se = d.std(ddof=1) / np.sqrt(n)
                        h = tdist.ppf(0.975, n - 1) * se
                        mw = mannwhitneyu(bg, x).pvalue
                        rows.append(dict(arm=aname, ct=ct, side=side, grp=grp, n=n, test=test, outl=outl,
                                         normal=normal, skew=g1, p=p, p_bonf=min(1, p * FAMILY), pmin=pmin,
                                         reachable=pmin * FAMILY < ALPHA, mean=d.mean(), lo=d.mean() - h,
                                         hi=d.mean() + h, exp=exp,
                                         ok=(min(1, p * FAMILY) < ALPHA) and np.sign(np.median(d)) == exp,
                                         ok_mw=(mw < ALPHA) and np.sign(np.median(d)) == exp))
        out[label] = rows
        print(f"\n######## {label}")
        print(f"{'arm':42s} {'ct':8s} {'side':5s} {'grp':4s} {'n':>4s} {'test':8s} {'outl':>4s} {'AD-norm':>7s} "
              f"{'skew':>5s} {'mean d [95% CI]':>24s} {'p':>8s} {'p_bonf':>8s} {'reach':>5s} {'book':>5s} {'violin':>6s}")
        for r in rows:
            print(f"{r['arm']:42s} {r['ct']:8s} {r['side']:5s} {r['grp']:4s} {r['n']:4d} {r['test']:8s} {r['outl']:4d} "
                  f"{str(r['normal']):>7s} {r['skew']:+5.1f} {r['mean']:+7.3f} [{r['lo']:+6.3f},{r['hi']:+6.3f}] "
                  f"{r['p']:8.1e} {r['p_bonf']:8.1e} {'yes' if r['reachable'] else 'NO':>5s} "
                  f"{'yes' if r['ok'] else 'no':>5s} {'yes' if r['ok_mw'] else 'no':>6s}")
        tests = defaultdict(int)
        for r in rows:
            tests[r["test"]] += 1
        print("tests chosen:", dict(tests), "| strata where significance is unreachable:",
              sorted({(r['arm'], r['ct'], r['n']) for r in rows if not r['reachable']}))
        print("same verdict book vs violin (MW, uncorrected):", sum(r["ok"] == r["ok_mw"] for r in rows), "/", len(rows))
        print("\nper arm, model side, cell types with the expected shift (book / violin), down | up:")
        for _, aname in ARMS:
            for side in ("real", "model"):
                s = [r for r in rows if r["arm"] == aname and r["side"] == side]
                f = lambda g, k: sum(r[k] for r in s if r["grp"] == g)  # noqa: E731
                print(f"  {aname:42s} {side:5s}  down {f('down', 'ok')}/{f('down', 'ok_mw')}   up {f('up', 'ok')}/{f('up', 'ok_mw')}")
    json.dump({k: [{kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else (float(vv) if isinstance(vv, (np.floating, float)) else vv))
                    for kk, vv in r.items()} for r in v] for k, v in out.items()},
              open(__import__("os").environ.get("VBT_OUT", "violins_book_tests.json"), "w"))


if __name__ == "__main__":
    main()
