"""Summary of the re-run of the old grid -> /xdisk/vkalin/dolzhenko/c2s/reanalysis_old/report.txt

Per run, the 40 violin strata (5 arms x 4 cell types x 2 gene groups) by the grid's own measures, grid/lib/decide.py
unchanged (violin_strata_ok / violin_mismatch: Mann-Whitney of the group against the background as in the manuscript,
p < 0.05, median shift in the expected direction), for the model and for the real cells:
  - the 8 old runs with their own rescued tables (this re-run);
  - the same 8 runs as the old grid computed them on 2026-11-02..04 (one shared table);
  - the new grid's runs as they are (grid/runs/*/violins/scores.json);
  - the manuscript model through this path.
No new statistical test: counts and per-stratum marks only.
Then two checks: does every old run have its own table (gene sets of the 12 workbooks), and does the manuscript model
through this path reproduce the manuscript's files (table CSV, the 12 workbooks, the violin reports of S2-S6).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import decide  # grid/lib, on PYTHONPATH

D = Path("/xdisk/vkalin/dolzhenko/c2s")
RE = D / "reanalysis_old"
BUNDLE = Path("/repo/grid/reanalysis_old")
HPV = "retrain_v2/analysis/real_counts_suite_ortho/human_panel_variants"
lines = []


def p(s=""):
    print(s, flush=True)
    lines.append(s)


def load(path):
    return json.load(open(path)) if path.exists() else None


def marks(scores):
    rows = decide._eval_scores(scores)
    ok = lambda a, ct, g, side: bool(rows.get((a, ct, g, side), {}).get("violin", False))  # noqa: E731
    return [(ok(a, ct, g, "real"), ok(a, ct, g, "model")) for a, ct, g in decide.STRATA]


def table_sets(folder):
    """{workbook name: frozenset of (gene, recovery_type)} of a rat_schema folder"""
    out = {}
    for f in sorted(Path(folder).glob("human_*_rescue_predicted_azi.xlsx")):
        df = pd.read_excel(f, sheet_name="Sheet1")
        out[f.name] = frozenset(zip(df["gene"], df["recovery_type"]))
    return out


jobs = [ln.split("\t") for ln in (RE / "jobs.tsv").read_text().splitlines() if ln.strip()]
old = [j[0] for j in jobs if j[0] != "manuscript"]
runs = [(f"old {n}, own table", load(RE / n / "violins/scores.json")) for n in old]
runs += [(f"old {n}, shared table (2-4 Nov)", load(D / "grid_old_20261105/runs" / n / "violins/scores.json")) for n in old]
runs += [(f"new {s.parent.parent.name}", load(s)) for s in sorted((D / "grid/runs").glob("*/violins/scores.json"))]
runs.append(("manuscript model, this path", load(RE / "manuscript/violins/scores.json")))

p(f"C2S: re-run of the old grid with each run's own human rescued table ({pd.Timestamp.now():%Y-%m-%d %H:%M})")
p()
p("== 40 violin strata per run: model / real cells significant in the expected direction (decide.py); mismatch")
M = []
for i, (label, sc) in enumerate(runs, 1):
    if sc is None:
        p(f"{i:2d} {label:42s} MISSING (no scores.json; see {RE}/jobs.tsv and the job log)")
        M.append(None)
        continue
    m = marks(sc)
    M.append(m)
    p(f"{i:2d} {label:42s} model {sum(x[1] for x in m):2d}/40   real {sum(x[0] for x in m):2d}/40   "
      f"mismatch {decide.violin_mismatch(sc):2d}")
p()
p("== per stratum (columns = run numbers above): RM real and model significant in the expected direction,")
p("   R. only real, .M only model, .. neither; ?? run missing")
p(f"{'arm':22s} {'cell':8s} {'grp':4s} " + " ".join(f"{i:>2d}" for i in range(1, len(runs) + 1)))
for k, (a, ct, g) in enumerate(decide.STRATA):
    cells = " ".join("??" if m is None else ("R" if m[k][0] else ".") + ("M" if m[k][1] else ".") for m in M)
    p(f"{a:22s} {ct:8s} {g:4s} {cells}")
p()

p("== check 1: one human rescued table per old run (gene sets of the 12 workbooks)")
dig = {}
for n in old + ["manuscript"]:
    sets = table_sets(RE / n / "violins/human/reports/rat_schema")
    h = hashlib.sha256(repr(sorted((k, sorted(v)) for k, v in sets.items())).encode()).hexdigest()[:12] if sets else "none"
    dig.setdefault(h, []).append(n)
    p(f"  {n:22s} {len(sets):2d} workbooks  {sum(len(v) for v in sets.values()):6d} genes  {h}")
shared = {h: ns for h, ns in dig.items() if len(ns) > 1 and h != "none"}
p("  SHARED TABLES: " + json.dumps(shared) if shared else "  every run has its own table")
p()

p("== check 2: the manuscript model through this path against the manuscript's own files")
g = RE / "manuscript/violins"
a, b = g / "human/reports/human_rescue_genomewide.csv", BUNDLE / "reference/human_rescue_genomewide.csv"
p(f"  human_rescue_genomewide.csv: " + ("MISSING" if not a.exists() else
                                       "identical" if a.read_bytes() == b.read_bytes() else "DIFFERS"))
mine, ref = table_sets(g / "human/reports/rat_schema"), table_sets(BUNDLE / "tree" / HPV / "reports/rat_schema")
diff = [f"{k} ({len(mine.get(k, set()) ^ v)} genes differ)" for k, v in ref.items() if mine.get(k) != v]
p(f"  12 rescued workbooks: " + ("all gene sets identical" if not diff and len(mine) == 12 else f"DIFFER: {diff}"))
for rf in sorted((BUNDLE / "reference").glob("report_*_all_rescued.csv")):
    side = "human" if rf.name.startswith("report_human") else "rat"
    nf = g / side / "reports" / rf.name
    if not nf.exists():
        p(f"  {rf.name}: MISSING")
        continue
    x, y = pd.read_csv(rf), pd.read_csv(nf)
    num = [c for c in x.select_dtypes("number").columns if c in y.columns]
    if x.shape != y.shape:
        p(f"  {rf.name}: DIFFERS in shape {x.shape} vs {y.shape}")
        continue
    dr = np.nanmax(np.abs(x[[c for c in num if "real" in c]].values - y[[c for c in num if "real" in c]].values))
    dm = np.nanmax(np.abs(x[[c for c in num if "model" in c]].values - y[[c for c in num if "model" in c]].values))
    p(f"  {rf.name}: max |difference| real columns {dr:.3g}, model columns {dm:.3g}"
      + ("  (identical)" if dr == 0 and dm == 0 else ""))

(RE / "report.txt").write_text("\n".join(lines) + "\n")
print(f"-> {RE / 'report.txt'}")
