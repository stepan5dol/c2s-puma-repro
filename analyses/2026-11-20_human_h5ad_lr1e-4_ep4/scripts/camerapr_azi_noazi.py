#!/usr/bin/env python3
"""cameraPR of the manuscript (limma::cameraPR, use.ranks=TRUE, inter.gene.cor=0.01, 8,421 Enrichr terms split by the
direction the disease moved each gene) on three arms, newgrid lr1e-4_ep4_s42, K=2000:

  azi_vs_disease    model AZI cells  vs real disease cells   = the manuscript's "azi" arm (BPD7mo: 7,137 terms FDR<0.05,
                    results/2026-11-19_ali_he22_corrected_analysis/newgrid_lr1e-4_ep4_<cond>_camerapr_corrected.csv)
  noazi_vs_disease  model NO-AZI cells (test_t10, Treatment: mechanical ventilation) vs real disease cells
  azi_vs_noazi      model AZI cells vs model NO-AZI cells (both generated and reconstructed the same way)

Inputs without the counts CSV: real cells from the h5ad files (build_human_h5ads.py, layer log1p_cpm10k_full), model
cells reconstructed from the Puma jsonl with reports/rank_expr_model_k2000.npz (the K=2000 calibration that
run_corrected_analysis.py fits from the CSV, r2 0.8665). Rescue table, top-K truncation, gene universe, statistic
(seurat_log2fc), set halves and the "toward control" rule are run_corrected_analysis.py's own functions.
The first run of this analysis (camerapr_azi_noazi_2026-11-20_1507) read the CSV through run_corrected_analysis.py;
this version must reproduce it exactly (check at the end).
The 12 cohort x cell type cameraPR runs go in parallel as separate python processes (each with its own R):
the task is pickled, the process is this script with --worker <task.pkl> (multiprocessing semaphores are not available
in the agent sandbox).
usage: python3 camerapr_azi_noazi.py --h5ad-dir <folder> --out-root <folder> --workers <n> [--compare <earlier output folder>]
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

RCA_DIR = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/2026-11-19_ali_he22_corrected_analysis")
sys.path.insert(0, str(RCA_DIR))
import run_corrected_analysis as RCA  # noqa: E402

K = 2000
NPZ = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/c2s-puma-std/grid/tree/retrain_v2/analysis/reports/rank_expr_model_k2000.npz")
NG = RCA.PUMA / "results/2026-11-16_132201_puma-a100mig-repro-ep4-fp32-lr1e-4-s42/inference_float32"
COHORTS = ("BPD7mo", "BPDPH7mo", "Acute26")


def load_model():
    z = np.load(NPZ, allow_pickle=True)
    assert int(z["k"]) == K
    names = list(z["gene_names"])
    return RCA.rem.RankExprModel(float(z["slope"]), float(z["intercept"]), float(z["r2"]),
                                 {g: i for i, g in enumerate(names)}, names)


def real_mats(h5ad_dir: Path, gene_names):
    """bpd_mats / he22_mats in run_corrected_analysis.py's layout, from the h5ad files."""
    bpd, he22 = {}, {}
    for cond in COHORTS:
        a = ad.read_h5ad(h5ad_dir / f"human_{cond}_real_and_model.h5ad")
        assert list(a.var_names) == gene_names, "h5ad gene axis differs from the calibration"
        full = a.layers["log1p_cpm10k_full"]
        for grp, store in (("DISEASE", bpd.setdefault(cond, {})),
                           ("CTRL", he22 if cond == "Acute26" else bpd.setdefault("Term", {}))):
            for ct in RCA.CT_ORDER:
                m = ((a.obs["orig.ident"] == grp) & (a.obs["cell_type"] == ct)).values
                X = full[m].toarray().astype(np.float64)
                if grp == "CTRL" and cond == "BPDPH7mo":
                    assert np.array_equal(X, store[ct]), "Term controls differ between the BPD7mo and BPDPH7mo files"
                store[ct] = X
    return bpd, he22


def camerapr_three(genes, stats: dict, sets):
    """One set index per cell type, cameraPR for each statistic."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    cv = ro.default_converter + pandas2ri.converter
    ro.globalenv["gene_names"] = ro.StrVector(genes)
    ro.globalenv["sets_r"] = ro.ListVector({nm: ro.StrVector(v) for nm, v in sets.items()})
    ro.r("idx <- ids2indices(sets_r, gene_names, remove.empty=TRUE)")
    ro.r(f"idx <- idx[sapply(idx, length) >= {RCA.MIN_SET_HALF}]")
    out = {}
    for arm, lfc in stats.items():
        ro.globalenv["stat"] = ro.FloatVector(lfc)
        ro.r("names(stat) <- gene_names")
        ro.r(f"""
            pr <- cameraPR(stat, idx, use.ranks=TRUE, inter.gene.cor={RCA.FIXED_RHO}, sort=FALSE)
            pr$term <- rownames(pr)
            rownames(pr) <- NULL
        """)
        with cv.context():
            df = ro.r("pr")
        out[arm] = df.rename(columns={"Direction": "camerapr_direction", "PValue": "camerapr_p", "FDR": "camerapr_fdr"})
    return out


def worker(task):
    cond, ct, genes, stats, sets = task
    from rpy2.robjects.packages import importr
    importr("limma")
    res = camerapr_three(genes, stats, sets)
    for arm, df in res.items():
        df.insert(0, "arm", arm)
        df.insert(0, "cell_type", ct)
        df.insert(0, "condition", cond)
    return cond, ct, res


def summarize(df):
    rows = []
    for (cond, arm), s in df.groupby(["condition", "arm"], sort=False):
        half = s["term"].str.extract(r" @@(dn|up)$")[0]
        toward = ((half == "up") & (s["camerapr_direction"] == "Down")) | ((half == "dn") & (s["camerapr_direction"] == "Up"))
        sig = s["camerapr_fdr"] < 0.05
        rec = {"condition": cond, "arm": arm, "tests": len(s), "fdr05": int(sig.sum()),
               "toward_ctrl": int((sig & toward).sum()), "away_ctrl": int((sig & ~toward).sum())}
        for ct in RCA.CT_ORDER:
            m = s["cell_type"] == ct
            rec[f"{ct} toward/away"] = f"{int((sig & toward & m).sum())}/{int((sig & ~toward & m).sum())}"
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--compare", type=Path, default=None)
    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out = args.out_root / f"camerapr_azi_noazi_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    t0 = time.time()
    model = load_model()
    print(f"[calibration] {NPZ.name}: r2={model.r2:.4f}", flush=True)
    bpd_mats, he22_mats = real_mats(args.h5ad_dir, model.gene_names)
    blob = json.loads((RCA.AZ / "supplement_materials/09_full_screening_8421_terms/enrichr_libraries_all_terms.json").read_text())
    enrichr = {f"{lib}__{t}": sorted(set(g)) for lib, terms in blob.items() for t, g in terms.items()}

    topk = {}
    rescue = RCA.compute_rescued_table(NG / "inference_azi_results.jsonl", model, bpd_mats, he22_mats, k=K,
                                       acute_control="he22", conditions=list(COHORTS), topk_cache=topk)
    saved = pd.read_csv(RCA_DIR / "newgrid_lr1e-4_ep4_human_rescue_genomewide_corrected.csv")
    print(f"[check] rescued genes {int(rescue['rescued'].sum())}, saved 19.11 {int(saved['rescued'].sum())}", flush=True)
    assert int(rescue["rescued"].sum()) == int(saved["rescued"].sum())

    azi_g = RCA.CH.load_azi_counterfactual(NG / "inference_azi_results.jsonl")
    noazi_g = RCA.CH.load_azi_counterfactual(NG / "test_t10.jsonl")
    gene_index = {g: i for i, g in enumerate(model.gene_names)}
    tasks, counts = [], []
    for cond in COHORTS:
        cm = RCA.CONDITIONS[cond]
        gw = rescue[(rescue["condition"] == cond) & rescue["passes_min_pct_disease"] & rescue["passes_min_pct_azi"]]
        for ct in RCA.CT_ORDER:
            g = gw[gw["cell_type"] == ct]
            a_rec, n_rec = azi_g.get((ct, cm), []), noazi_g.get((ct, cm), [])
            counts.append({"condition": cond, "cell_type": ct, "n_disease": int(topk[((cond, ct), K)].shape[0]),
                           "n_azi": len(a_rec), "n_noazi": len(n_rec), "n_genes": len(g)})
            if len(g) == 0 or min(len(a_rec), len(n_rec)) < RCA.MIN_AZI_CELLS:
                continue
            keep = np.array([gene_index[x] for x in g["gene"]])
            genes = list(g["gene"])
            dis_lfc = dict(zip(genes, g["avg_log2FC_disease_topk"]))
            sets = {}
            for t, members in enrichr.items():
                for half, sel in (("dn", lambda v: v < -0.25), ("up", lambda v: v > 0.25)):
                    h = [m for m in members if m in dis_lfc and sel(dis_lfc[m])]
                    if len(h) >= RCA.MIN_SET_HALF:
                        sets[f"{t} @@{half}"] = h
            dis = topk[((cond, ct), K)][:, keep]
            azi = RCA.rem.reconstruct_batch([r["pred"] for r in a_rec], model, K)[:, keep]
            noazi = RCA.rem.reconstruct_batch([r["pred"] for r in n_rec], model, K)[:, keep]
            tasks.append((cond, ct, genes, {"azi_vs_disease": RCA.seurat_log2fc(azi, dis),
                                            "noazi_vs_disease": RCA.seurat_log2fc(noazi, dis),
                                            "azi_vs_noazi": RCA.seurat_log2fc(azi, noazi)}, sets))
    print(f"[cameraPR] {len(tasks)} cohort x cell type runs on {args.workers} processes", flush=True)
    tmp = out / "tasks"
    tmp.mkdir()
    paths = []
    for i, t in enumerate(tasks):
        pth = tmp / f"task_{i:02d}.pkl"
        pickle.dump(t, open(pth, "wb"))
        paths.append(pth)
    running, done = [], []
    queue = list(paths)
    while queue or running:
        while queue and len(running) < args.workers:
            pth = queue.pop(0)
            log = open(pth.with_suffix(".log"), "w")
            running.append((pth, subprocess.Popen([sys.executable, __file__, "--worker", str(pth)], stdout=log, stderr=log)))
        for item in list(running):
            if item[1].poll() is not None:
                assert item[1].returncode == 0, f"{item[0]} failed, see {item[0].with_suffix('.log')}"
                running.remove(item)
                done.append(item[0])
        time.sleep(1)
    frames = []
    for pth in paths:
        cond, ct, res = pickle.load(open(pth.with_suffix(".out.pkl"), "rb"))
        for arm, df in res.items():
            frames.append(df)
            print(f"  {cond}/{ct}/{arm}: {len(df)} sets, FDR<0.05 = {int((df['camerapr_fdr'] < 0.05).sum())}", flush=True)
    shutil.rmtree(tmp)
    allres = pd.concat(frames, ignore_index=True)
    allres.to_csv(out / "camerapr_three_arms.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "cells_per_group.csv", index=False)
    summ = summarize(allres)
    summ.to_csv(out / "summary.csv", index=False)
    for cond in COHORTS:
        old = pd.read_csv(RCA_DIR / f"newgrid_lr1e-4_ep4_{cond.lower()}_camerapr_corrected.csv")
        old = old[old["arm"] == "azi"]
        new = allres[(allres["condition"] == cond) & (allres["arm"] == "azi_vs_disease")]
        print(f"[reproduce 19.11] {cond}: FDR<0.05 saved {int((old['camerapr_fdr'] < 0.05).sum())}, now {int((new['camerapr_fdr'] < 0.05).sum())}", flush=True)
    if args.compare is not None:
        prev = pd.read_csv(args.compare / "camerapr_three_arms.csv")
        key = ["condition", "cell_type", "arm", "term"]
        mrg = allres.merge(prev, on=key, suffixes=("", "_prev"))
        print(f"[compare {args.compare.name}] rows {len(allres)} / {len(prev)} / matched {len(mrg)}, "
              f"max |FDR diff| {float((mrg['camerapr_fdr'] - mrg['camerapr_fdr_prev']).abs().max()):.2e}, "
              f"direction differs {int((mrg['camerapr_direction'] != mrg['camerapr_direction_prev']).sum())}", flush=True)
    print(summ.to_string(index=False), flush=True)
    print(f"done -> {out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        task_path = Path(sys.argv[2])
        pickle.dump(worker(pickle.load(open(task_path, "rb"))), open(task_path.with_suffix(".out.pkl"), "wb"))
    else:
        main()
