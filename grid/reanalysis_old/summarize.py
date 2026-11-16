"""Summary of the re-run of the old grid.

usage: summarize.py [--jobinfo FILE|-]                full: reanalysis_old/report.txt and message.txt (short, for Telegram)
       summarize.py --run NAME [--jobinfo FILE|-]     short message for one run -> stdout
--jobinfo: lines "name<TAB>gpu State|Elapsed<TAB>analysis State|Elapsed" from sacct (the slurm wrappers write them).

Measures: grid/lib/decide.py unchanged (violin_strata_ok / violin_mismatch: Mann-Whitney of the gene group against the
background as in the manuscript, p < 0.05, median shift in the expected direction), on the 40 strata (5 arms x 4 cell
types x 2 gene groups; 24 human, 16 rat), for the model and for the real cells, for
  - the 8 old runs with their own rescued tables (this re-run) and as the old grid computed them (one shared table);
  - the new grid's runs as they are (grid/runs/*/violins/scores.json);
  - the manuscript model through this path.
No new statistical test: counts and per-stratum marks only. Two checks: one table per old run (gene sets of the 12
workbooks); the manuscript model through this path against the manuscript's files.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import decide  # grid/lib, on PYTHONPATH

D = Path(os.environ.get("C2S_D", "/xdisk/vkalin/dolzhenko/c2s"))
RE = D / "reanalysis_old"
BUNDLE = Path(os.environ.get("BUNDLE", "/repo/grid/reanalysis_old"))
HPV = "retrain_v2/analysis/real_counts_suite_ortho/human_panel_variants"
HUMAN = [i for i, s in enumerate(decide.STRATA) if s[0].startswith("human")]
RAT = [i for i, s in enumerate(decide.STRATA) if s[0].startswith("rat")]
COND = {"Acute26": "Acute", "BPD7mo": "BPD", "BPDPH7mo": "BPD-PH"}

ap = argparse.ArgumentParser()
ap.add_argument("--run")
ap.add_argument("--jobinfo")
args = ap.parse_args()

jobs = [dict(zip(("name", "run", "gpu", "an", "test", "azi"), ln.split("\t")))
        for ln in (RE / "jobs.tsv").read_text().splitlines() if ln.strip()]
names = [j["name"] for j in jobs]
old = [n for n in names if n != "manuscript"]
info = {}
if args.jobinfo:
    for ln in (sys.stdin if args.jobinfo == "-" else open(args.jobinfo)):
        f = ln.rstrip("\n").split("\t")
        if len(f) == 3:
            info[f[0]] = (f[1].split("|") + ["-"])[:2], (f[2].split("|") + ["-"])[:2]


def short(n):
    return n.replace("_s42", "").replace("_ep", " ep").removeprefix("lr")


def dur(el):
    """sacct Elapsed '1-02:03:04' / '01:24:13' -> '26 h 3 min' / '1 h 24 min'"""
    if not el or el == "-":
        return "?"
    d, _, hms = el.rpartition("-")
    h, m, s = (int(x) for x in hms.split(":"))
    h += 24 * int(d or 0)
    return f"{h} h {m} min" if h else f"{m} min"


def load(path):
    return json.load(open(path)) if path.exists() else None


def counts(scores):
    rows = decide._eval_scores(scores)
    ok = lambda a, ct, g, side: bool(rows.get((a, ct, g, side), {}).get("violin", False))  # noqa: E731
    m = [(ok(a, ct, g, "real"), ok(a, ct, g, "model")) for a, ct, g in decide.STRATA]
    return dict(m=m, model=sum(x[1] for x in m), real=sum(x[0] for x in m),
                mh=sum(m[i][1] for i in HUMAN), mr=sum(m[i][1] for i in RAT),
                rh=sum(m[i][0] for i in HUMAN), rr=sum(m[i][0] for i in RAT),
                mismatch=decide.violin_mismatch(scores))


def table_sets(folder):
    """{workbook: frozenset of (gene, recovery_type)} of a rat_schema folder"""
    out = {}
    for f in sorted(Path(folder).glob("human_*_rescue_predicted_azi.xlsx")):
        df = pd.read_excel(f, sheet_name="Sheet1")
        out[f.name] = frozenset(zip(df["gene"], df["recovery_type"]))
    return out


def table_size(name):
    idx = RE / name / "violins/human/reports/rat_schema/index.csv"
    if not idx.exists():
        return None
    t = pd.read_csv(idx)
    per = t.groupby("condition")["n_genes"].sum()
    return int(t["n_genes"].sum()), ", ".join(f"{COND.get(c, c)} {int(per[c])}" for c in COND if c in per)


def arrow(a, b):
    return f"{a}→{b}"


def gold_check():
    """lines: the manuscript model through this path against the manuscript's files"""
    g = RE / "manuscript/violins"
    out = []
    a, b = g / "human/reports/human_rescue_genomewide.csv", BUNDLE / "reference/human_rescue_genomewide.csv"
    out.append("table CSV: " + ("MISSING" if not a.exists() else "identical" if a.read_bytes() == b.read_bytes() else "DIFFERS"))
    mine, ref = table_sets(g / "human/reports/rat_schema"), table_sets(BUNDLE / "tree" / HPV / "reports/rat_schema")
    bad = {k: len(mine.get(k, frozenset()) ^ v) for k, v in ref.items() if mine.get(k) != v}
    out.append("12 workbooks: " + ("MISSING" if not mine else "same genes" if not bad and len(mine) == 12 else
                                   f"{len(bad)} of 12 differ (up to {max(bad.values(), default=0)} genes)"))
    for rf in sorted((BUNDLE / "reference").glob("report_*_all_rescued.csv")):
        side = "human" if rf.name.startswith("report_human") else "rat"
        nf = g / side / "reports" / rf.name
        tag = rf.name.replace("report_", "").replace("_all_rescued.csv", "")
        if not nf.exists():
            out.append(f"{tag}: MISSING")
            continue
        x, y = pd.read_csv(rf), pd.read_csv(nf)
        if x.shape != y.shape:
            out.append(f"{tag}: DIFFERS in shape")
            continue
        num = [c for c in x.select_dtypes("number").columns if c in y.columns]
        d = {k: float(np.nanmax(np.abs(x[[c for c in num if k in c]].values - y[[c for c in num if k in c]].values)))
             for k in ("real", "model")}
        out.append(f"{tag}: max |Δ| real {d['real']:.2g}, model {d['model']:.2g}"
                   + ("  identical" if d["real"] == 0 and d["model"] == 0 else ""))
    return out


def own_tables_check():
    dig = {}
    for n in old:
        sets = table_sets(RE / n / "violins/human/reports/rat_schema")
        if sets:
            h = hashlib.sha256(repr(sorted((k, sorted(v)) for k, v in sets.items())).encode()).hexdigest()[:12]
            dig.setdefault(h, []).append(n)
    shared = [ns for ns in dig.values() if len(ns) > 1]
    return ("SHARED: " + "; ".join(", ".join(map(short, ns)) for ns in shared)) if shared else \
        f"every run has its own ({sum(len(v) for v in dig.values())} tables)"


done = [n for n in names if (RE / n / "violins/scores.json").exists()]
failed = [n for n in names if n not in done and n in info and info[n][1][0] not in ("PENDING", "RUNNING", "-")]
waiting = [n for n in names if n not in done and n not in failed]

# ---------------------------------------------------------------- one run
if args.run:
    n = args.run
    j = next(x for x in jobs if x["name"] == n)
    L = [f"C2S old grid re-run: {short(n)}", f"{len(done)}/{len(names)} done" + (f", failed: {', '.join(map(short, failed))}" if failed else "")]
    if n in info:
        (gs, ge), (as_, ae) = info[n]
        L.append("AZI answers: " + ("already there" if j["gpu"] == "0" else f"{gs.lower()}, {dur(ge)}"))
        L.append(f"table + violins: {as_.lower()}, {dur(ae)}")
    own = load(RE / n / "violins/scores.json")
    if own is None:
        L.append(f"NO RESULT: log {D}/logs/re-{n}_{j['an']}.out")
    else:
        ts = table_size(n)
        if ts:
            L.append(f"own human table: {ts[0]} genes ({ts[1]})")
        c = counts(own)
        prev = load(D / "grid_old_20261105/runs" / n / "violins/scores.json") if n != "manuscript" else None
        if prev:
            o = counts(prev)
            L += ["model violins, shared → own table:",
                  f"  all {arrow(o['model'], c['model'])}/40  human {arrow(o['mh'], c['mh'])}/24  rat {arrow(o['mr'], c['mr'])}/16",
                  f"real-cell violins: {arrow(o['real'], c['real'])}/40 (human {arrow(o['rh'], c['rh'])}, rat {arrow(o['rr'], c['rr'])})",
                  f"mismatch: {arrow(o['mismatch'], c['mismatch'])}"]
        else:
            L += [f"model violins: {c['model']}/40 (human {c['mh']}/24, rat {c['mr']}/16)",
                  f"real-cell violins: {c['real']}/40 (human {c['rh']}, rat {c['rr']}); mismatch {c['mismatch']}"]
        if n == "manuscript":
            L += ["vs the manuscript's files:"] + ["  " + s for s in gold_check()]
    if waiting:
        L.append("waiting: " + ", ".join(map(short, waiting)))
    print("\n".join(L))
    sys.exit(0)

# ---------------------------------------------------------------- full summary
t0 = (RE / "jobs.tsv").stat().st_mtime
hours = (time.time() - t0) / 3600
cur = {n: load(RE / n / "violins/scores.json") for n in names}
prev = {n: load(D / "grid_old_20261105/runs" / n / "violins/scores.json") for n in old}
new = {s.parent.parent.name: load(s) for s in sorted((D / "grid/runs").glob("*/violins/scores.json"))}
C = {k: counts(v) for k, v in {**{("own", n): cur[n] for n in names}, **{("prev", n): prev[n] for n in old},
                                **{("new", n): new[n] for n in new}}.items() if v is not None}

M = [f"C2S old grid re-run: {len(done)}/{len(names)} done, {hours:.1f} h since submission"
     + (f"\nFAILED: {', '.join(map(short, failed))}" if failed else "") + (f"\nnot finished: {', '.join(map(short, waiting))}" if waiting else ""),
     "",
     "OLD GRID (input cell in prompt)",
     "model violins /40: shared → own table",
     f"{'run':10s} {'all':>6s} {'human':>6s} {'rat':>6s} {'real':>4s} {'mism':>6s}"]
for n in old:
    if ("own", n) not in C:
        M.append(f"{short(n):10s} MISSING")
        continue
    c, o = C[("own", n)], C.get(("prev", n))
    if o:
        M.append(f"{short(n):10s} {arrow(o['model'], c['model']):>6s} {arrow(o['mh'], c['mh']):>6s} {arrow(o['mr'], c['mr']):>6s} "
                 f"{c['real']:>4d} {arrow(o['mismatch'], c['mismatch']):>6s}")
    else:
        M.append(f"{short(n):10s} {c['model']:>6d} {c['mh']:>6d} {c['mr']:>6d} {c['real']:>4d} {c['mismatch']:>6d}")
if ("own", "manuscript") in C:
    c = C[("own", "manuscript")]
    M.append(f"{'manuscript':10s} {c['model']:>6d} {c['mh']:>6d} {c['mr']:>6d} {c['real']:>4d} {c['mismatch']:>6d}")
M += ["", "NEW GRID (no input cell)", f"{'run':10s} {'all':>6s} {'human':>6s} {'rat':>6s} {'real':>4s} {'mism':>6s}"]
for n in new:
    if ("new", n) in C:
        c = C[("new", n)]
        M.append(f"{short(n):10s} {c['model']:>6d} {c['mh']:>6d} {c['mr']:>6d} {c['real']:>4d} {c['mismatch']:>6d}")
M += ["", "real = real-cell strata significant (the target)", "",
      "own table per run: " + own_tables_check(), "manuscript model vs manuscript files:"] + ["  " + s for s in gold_check()]

durs = []
for n in names:
    if n in info:
        (gs, ge), (as_, ae) = info[n]
        durs.append(f"{short(n):10s} AZI {'-' if jobs[names.index(n)]['gpu'] == '0' else dur(ge):>11s}   analysis {dur(ae)}")
if durs:
    M += ["", "time per run"] + durs

R = list(M) + ["", "== per stratum (columns = runs): RM real and model significant in the expected direction,",
               "   R. only real, .M only model, .. neither", ""]
cols = [(f"own {short(n)}", ("own", n)) for n in names] + [(f"shared {short(n)}", ("prev", n)) for n in old] + \
       [(f"new {short(n)}", ("new", n)) for n in new]
cols = [(lbl, k) for lbl, k in cols if k in C]
R += [f"{i:2d} = {lbl}" for i, (lbl, _) in enumerate(cols, 1)]
R.append(f"{'arm':22s} {'cell':8s} {'grp':4s} " + " ".join(f"{i:>2d}" for i in range(1, len(cols) + 1)))
for k, (a, ct, g) in enumerate(decide.STRATA):
    R.append(f"{a:22s} {ct:8s} {g:4s} " + " ".join(("R" if C[key]["m"][k][0] else ".") + ("M" if C[key]["m"][k][1] else ".")
                                                   for _, key in cols))
(RE / "message.txt").write_text("\n".join(M) + "\n")
(RE / "report.txt").write_text("\n".join(R) + "\n")
print("\n".join(R))
