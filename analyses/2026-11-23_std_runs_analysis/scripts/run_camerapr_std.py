#!/usr/bin/env python3
"""Reproducible CLI wrapper around untouched manuscript camerapr_azi_noazi.py
for c2s_std runs (any subset of OOD drug arms: drug_vs_disease, drug_vs_noazi,
drug_vs_own + Stage 2 cross-drug DE/markers) and c2s-rep300.

All inputs/ and outputs go under the mandatory --out-root (outside git).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, pearsonr, spearmanr

AZ = Path("/Users/stepandolzhenko/Documents/AzithroGemma")
CAM_SCRIPT = AZ / "retrain_v2/c2s-puma-repro/analyses/2026-11-20_human_h5ad_lr1e-4_ep4/scripts/camerapr_azi_noazi.py"
H5AD_DIR = AZ / "retrain_v2/results/2026-11-20_human_h5ad_newgrid_lr1e-4_ep4_s42/h5ad_calib_k2000"
RAW_STD = AZ / "retrain_prompt/results/puma_answers/xdisk/vkalin/dolzhenko/c2s_std/results"
REP300_RAW = AZ / "retrain_prompt/results/puma_answers/xdisk/vkalin/dolzhenko/c2s/results/2026-11-16_132201_puma-a100mig-repro-ep4-fp32-lr1e-4-s42/inference_bfloat16_rep300"

ALL_DRUGS = ["azithromycin", "taurine", "amiloride", "anti_hedgehog", "antibiotics", "hmgb1_peptide"]


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def ensure_split_arms(out_root: Path, run_name: str, drugs: list[str]) -> Path:
    inp = out_root / "inputs" / run_name
    inp.mkdir(parents=True, exist_ok=True)
    raw_ood = RAW_STD / run_name / "inference_bfloat16/inference_results.jsonl"
    raw_test = RAW_STD / run_name / "inference_bfloat16/test_t10.jsonl"
    if not (inp / "test_t10.jsonl").exists() and raw_test.exists():
        shutil.copy2(raw_test, inp / "test_t10.jsonl")
    needed = set(drugs) | {"own"}
    missing = [a for a in needed if not (inp / ("inference_azi_results.jsonl" if a == "azithromycin" else f"inference_{a}_results.jsonl")).exists()]
    if missing:
        assert raw_ood.exists(), f"Missing {raw_ood}"
        by_arm = {a: [] for a in missing}
        keep_keys = ("idx", "direction", "prompt", "gt", "pred", "finish_reason", "gen_tokens")
        with raw_ood.open() as f:
            for line in f:
                r = json.loads(line)
                a = r.get("arm")
                if a in by_arm:
                    by_arm[a].append({k: r[k] for k in keep_keys if k in r})
        for a, rows in by_arm.items():
            fn = "inference_azi_results.jsonl" if a == "azithromycin" else f"inference_{a}_results.jsonl"
            with (inp / fn).open("w") as w:
                for r in rows:
                    w.write(json.dumps(r) + "\n")
            print(f"[split] {a}: {len(rows)} rows -> {inp / fn}", flush=True)
    return inp


def run_workers(paths: list[Path], n_workers: int, t0: float) -> None:
    running, queue = [], list(paths)
    done = 0
    while queue or running:
        while queue and len(running) < n_workers:
            pth = queue.pop(0)
            log = open(pth.with_suffix(".log"), "w")
            running.append((pth, subprocess.Popen([sys.executable, str(CAM_SCRIPT), "--worker", str(pth)], stdout=log, stderr=log)))
        for item in list(running):
            if item[1].poll() is not None:
                assert item[1].returncode == 0, f"{item[0]} failed, see {item[0].with_suffix('.log')}"
                running.remove(item)
                done += 1
                if done % 6 == 0 or done == len(paths):
                    print(f"[cameraPR] workers done: {done}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(1)


def run_std_drugs(out_root: Path, run_name: str, drugs: list[str], out_dir: Path, n_workers: int) -> None:
    sys.path.insert(0, str(CAM_SCRIPT.parent))
    import camerapr_azi_noazi as CAM

    t0 = time.time()
    inp = ensure_split_arms(out_root, run_name, drugs)
    assert not out_dir.exists(), f"Refusing to overwrite {out_dir}"
    out_dir.mkdir(parents=True)

    model = CAM.load_model()
    bpd_mats, he22_mats = CAM.real_mats(H5AD_DIR, model.gene_names)
    blob = json.loads((CAM.RCA.AZ / "supplement_materials/09_full_screening_8421_terms/enrichr_libraries_all_terms.json").read_text())
    enrichr = {f"{lib}__{t}": sorted(set(g)) for lib, terms in blob.items() for t, g in terms.items()}
    gene_index = {g: i for i, g in enumerate(model.gene_names)}

    noazi_g = CAM.RCA.CH.load_azi_counterfactual(inp / "test_t10.jsonl")
    own_g = CAM.RCA.CH.load_azi_counterfactual(inp / "inference_own_results.jsonl")

    recon_own = {}
    for cond in CAM.COHORTS:
        cm = CAM.RCA.CONDITIONS[cond]
        for ct in CAM.RCA.CT_ORDER:
            recon_own[(cond, ct)] = CAM.RCA.rem.reconstruct_batch([r["pred"] for r in own_g.get((ct, cm), [])], model, CAM.K)

    recon_drugs = {d: {} for d in drugs}
    topk, task_meta, counts_first = {}, [], []
    tmp = out_dir / "tasks"
    tmp.mkdir()

    for di, drug in enumerate(drugs):
        jp = inp / ("inference_azi_results.jsonl" if drug == "azithromycin" else f"inference_{drug}_results.jsonl")
        rescue = CAM.RCA.compute_rescued_table(
            jp, model, bpd_mats, he22_mats, k=CAM.K,
            acute_control="he22", conditions=list(CAM.COHORTS), topk_cache=topk,
        )
        print(f"[{run_name}/{drug}] rescued genes: {int(rescue['rescued'].sum())}", flush=True)
        dg = CAM.RCA.CH.load_azi_counterfactual(jp)
        for cond in CAM.COHORTS:
            cm = CAM.RCA.CONDITIONS[cond]
            gw = rescue[(rescue["condition"] == cond) & rescue["passes_min_pct_disease"] & rescue["passes_min_pct_azi"]]
            for ct in CAM.RCA.CT_ORDER:
                d_rec, n_rec, o_rec = dg.get((ct, cm), []), noazi_g.get((ct, cm), []), own_g.get((ct, cm), [])
                recon_drugs[drug][(cond, ct)] = CAM.RCA.rem.reconstruct_batch([r["pred"] for r in d_rec], model, CAM.K)
                g = gw[gw["cell_type"] == ct]
                if di == 0:
                    counts_first.append({
                        "condition": cond, "cell_type": ct,
                        "n_disease": int(topk[((cond, ct), CAM.K)].shape[0]),
                        "n_azi": len(d_rec), "n_noazi": len(n_rec), "n_own": len(o_rec), "n_genes": len(g),
                    })
                if len(g) == 0 or min(len(d_rec), len(n_rec)) < CAM.RCA.MIN_AZI_CELLS:
                    continue
                keep = np.array([gene_index[x] for x in g["gene"]])
                genes = list(g["gene"])
                dis_lfc = dict(zip(genes, g["avg_log2FC_disease_topk"]))
                sets = {}
                for t, members in enrichr.items():
                    for half, sel in (("dn", lambda v: v < -0.25), ("up", lambda v: v > 0.25)):
                        h = [m for m in members if m in dis_lfc and sel(dis_lfc[m])]
                        if len(h) >= CAM.RCA.MIN_SET_HALF:
                            sets[f"{t} @@{half}"] = h
                dis = topk[((cond, ct), CAM.K)][:, keep]
                dm = recon_drugs[drug][(cond, ct)][:, keep]
                noazi = CAM.RCA.rem.reconstruct_batch([r["pred"] for r in n_rec], model, CAM.K)[:, keep]
                own = recon_own[(cond, ct)][:, keep]
                stats = {"azi_vs_disease": CAM.RCA.seurat_log2fc(dm, dis)}
                if di == 0:
                    stats["noazi_vs_disease"] = CAM.RCA.seurat_log2fc(noazi, dis)
                stats["azi_vs_noazi"] = CAM.RCA.seurat_log2fc(dm, noazi)
                if len(o_rec) >= CAM.RCA.MIN_AZI_CELLS:
                    stats["azi_vs_own"] = CAM.RCA.seurat_log2fc(dm, own)
                pth = tmp / f"task_{len(task_meta):03d}_{drug}_{cond}_{ct}.pkl"
                pickle.dump((cond, ct, genes, stats, sets), open(pth, "wb"))
                task_meta.append((drug, cond, ct, pth))

    run_workers([m[3] for m in task_meta], n_workers, t0)

    by_drug = {d: [] for d in drugs}
    for drug, _, _, pth in task_meta:
        _, _, res = pickle.load(open(pth.with_suffix(".out.pkl"), "rb"))
        for df in res.values():
            by_drug[drug].append(df)
    shutil.rmtree(tmp)

    if len(drugs) == 1:
        df = pd.concat(by_drug[drugs[0]], ignore_index=True)
        df.to_csv(out_dir / "camerapr_three_arms.csv", index=False)
        pd.DataFrame(counts_first).to_csv(out_dir / "cells_per_group.csv", index=False)
        s = CAM.summarize(df)
        s.to_csv(out_dir / "summary.csv", index=False)
        print(s.to_string(index=False))
    else:
        summ_rows, full_rows = [], []
        for drug in drugs:
            df = pd.concat(by_drug[drug], ignore_index=True)
            ddir = out_dir / drug
            ddir.mkdir()
            df.to_csv(ddir / "camerapr_three_arms.csv", index=False)
            s = CAM.summarize(df)
            s.to_csv(ddir / "summary.csv", index=False)
            s.insert(0, "drug", drug)
            summ_rows.append(s)
            df["drug"] = drug
            full_rows.append(df)
        prefix = f"all_{len(drugs)}drugs"
        all_summ = pd.concat(summ_rows, ignore_index=True)
        all_summ.to_csv(out_dir / f"{prefix}_camerapr_summary.csv", index=False)
        pd.concat(full_rows, ignore_index=True).to_csv(out_dir / f"{prefix}_camerapr_full.csv", index=False)
        print(all_summ[all_summ["arm"] == "azi_vs_own"].to_string(index=False))

        # Stage 2: Cross-drug correlation, omnibus KW, one-vs-rest, and pairwise DE
        corr_rows, kw_rows, ovr_rows, pair_rows = [], [], [], []
        for cond in CAM.COHORTS:
            for ct in CAM.RCA.CT_ORDER:
                own_m = recon_own[(cond, ct)]
                dmats = {d: recon_drugs[d][(cond, ct)] for d in drugs}
                pct_max = (own_m > 0).mean(0)
                for d in drugs:
                    pct_max = np.maximum(pct_max, (dmats[d] > 0).mean(0))
                keep = np.where(pct_max >= 0.10)[0]
                genes = [model.gene_names[i] for i in keep]
                lfcs = {d: CAM.RCA.seurat_log2fc(dmats[d][:, keep], own_m[:, keep]) for d in drugs}

                for i, d1 in enumerate(drugs):
                    for d2 in drugs[i + 1:]:
                        r_p, _ = pearsonr(lfcs[d1], lfcs[d2])
                        r_s, _ = spearmanr(lfcs[d1], lfcs[d2])
                        corr_rows.append({"condition": cond, "cell_type": ct, "drug1": d1, "drug2": d2,
                                          "pearson_r": r_p, "spearman_r": r_s, "n_genes": len(keep)})

                pvals = np.ones(len(keep), dtype=float)
                means = {d: dmats[d][:, keep].mean(0) for d in drugs}
                for gi in range(len(keep)):
                    cols = [dmats[d][:, keep[gi]] for d in drugs]
                    if all(c.max() == c.min() for c in cols) and len({c[0] for c in cols}) == 1:
                        continue
                    try:
                        _, pvals[gi] = kruskal(*cols)
                    except ValueError:
                        pvals[gi] = 1.0
                qvals = bh_fdr(pvals)
                mean_mat = np.vstack([means[d] for d in drugs])
                span = mean_mat.max(0) - mean_mat.min(0)
                for gi in np.where(qvals < 0.05)[0]:
                    row = {"condition": cond, "cell_type": ct, "gene": genes[gi],
                           "kw_p": float(pvals[gi]), "kw_fdr": float(qvals[gi]), "expr_span": float(span[gi]),
                           "max_drug": drugs[int(np.argmax(mean_mat[:, gi]))],
                           "min_drug": drugs[int(np.argmin(mean_mat[:, gi]))]}
                    for d in drugs:
                        row[f"lfc_vs_own_{d}"] = float(lfcs[d][gi])
                    kw_rows.append(row)

                for d in drugs:
                    a_mat = dmats[d][:, keep]
                    rest_mat = np.vstack([dmats[od][:, keep] for od in drugs if od != d])
                    lfc_ovr = CAM.RCA.seurat_log2fc(a_mat, rest_mat)
                    cand = np.where(np.abs(lfc_ovr) >= 0.10)[0]
                    ps = np.ones(len(keep), dtype=float)
                    for gi in cand:
                        _, ps[gi] = mannwhitneyu(a_mat[:, gi], rest_mat[:, gi], alternative="two-sided")
                    qs = bh_fdr(ps)
                    for gi in np.where((qs < 0.05) & (np.abs(lfc_ovr) >= 0.10))[0]:
                        ovr_rows.append({
                            "condition": cond, "cell_type": ct, "drug": d, "gene": genes[gi],
                            "lfc_vs_other5": float(lfc_ovr[gi]), "lfc_vs_own": float(lfcs[d][gi]),
                            "p": float(ps[gi]), "fdr": float(qs[gi]),
                        })

                for i, d1 in enumerate(drugs):
                    for d2 in drugs[i + 1:]:
                        m1, m2 = dmats[d1][:, keep], dmats[d2][:, keep]
                        l12 = CAM.RCA.seurat_log2fc(m1, m2)
                        cand = np.where(np.abs(l12) >= 0.25)[0]
                        ps = np.ones(len(keep), dtype=float)
                        for gi in cand:
                            _, ps[gi] = mannwhitneyu(m1[:, gi], m2[:, gi], alternative="two-sided")
                        qs = bh_fdr(ps)
                        pair_rows.append({
                            "condition": cond, "cell_type": ct, "drug1": d1, "drug2": d2,
                            "n_de_fdr05_lfc025": int(((qs < 0.05) & (np.abs(l12) >= 0.25)).sum()),
                            "n_de_fdr05": int((qs < 0.05).sum()),
                        })

        pd.DataFrame(corr_rows).to_csv(out_dir / "drug_vs_own_lfc_correlations.csv", index=False)
        pd.DataFrame(kw_rows).to_csv(out_dir / "drug_omnibus_kw_markers.csv", index=False)
        pd.DataFrame(ovr_rows).to_csv(out_dir / "drug_one_vs_rest_markers.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(out_dir / "drug_pairwise_de_counts.csv", index=False)

    print(f"DONE in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


def run_rep300(out_dir: Path, n_workers: int) -> None:
    sys.path.insert(0, str(CAM_SCRIPT.parent))
    import camerapr_azi_noazi as CAM

    t0 = time.time()
    assert not out_dir.exists(), f"Refusing to overwrite {out_dir}"
    out_dir.mkdir(parents=True)
    azi_p, noazi_p = out_dir / "rep300_azi_all.jsonl", out_dir / "rep300_noazi_all.jsonl"
    with azi_p.open("w") as wa, noazi_p.open("w") as wn:
        for c in ("Acute26", "BPD7mo", "BPDPH7mo"):
            wa.write((REP300_RAW / f"{c}_azi_results.jsonl").read_text())
            wn.write((REP300_RAW / f"{c}_noazi_results.jsonl").read_text())

    model = CAM.load_model()
    bpd_mats, he22_mats = CAM.real_mats(H5AD_DIR, model.gene_names)
    blob = json.loads((CAM.RCA.AZ / "supplement_materials/09_full_screening_8421_terms/enrichr_libraries_all_terms.json").read_text())
    enrichr = {f"{lib}__{t}": sorted(set(g)) for lib, terms in blob.items() for t, g in terms.items()}
    gene_index = {g: i for i, g in enumerate(model.gene_names)}

    topk = {}
    rescue = CAM.RCA.compute_rescued_table(
        azi_p, model, bpd_mats, he22_mats, k=CAM.K, acute_control="he22",
        conditions=list(CAM.COHORTS), topk_cache=topk,
    )
    azi_g = CAM.RCA.CH.load_azi_counterfactual(azi_p)
    noazi_g = CAM.RCA.CH.load_azi_counterfactual(noazi_p)

    tmp = out_dir / "tasks"
    tmp.mkdir()
    paths = []
    for cond in CAM.COHORTS:
        cm = CAM.RCA.CONDITIONS[cond]
        gw = rescue[(rescue["condition"] == cond) & rescue["passes_min_pct_disease"] & rescue["passes_min_pct_azi"]]
        for ct in CAM.RCA.CT_ORDER:
            g = gw[gw["cell_type"] == ct]
            keep = np.array([gene_index[x] for x in g["gene"]])
            genes = list(g["gene"])
            dis_lfc = dict(zip(genes, g["avg_log2FC_disease_topk"]))
            sets = {}
            for t, members in enrichr.items():
                for half, sel in (("dn", lambda v: v < -0.25), ("up", lambda v: v > 0.25)):
                    h = [m for m in members if m in dis_lfc and sel(dis_lfc[m])]
                    if len(h) >= CAM.RCA.MIN_SET_HALF:
                        sets[f"{t} @@{half}"] = h
            dis = topk[((cond, ct), CAM.K)][:, keep]
            azi = CAM.RCA.rem.reconstruct_batch([r["pred"] for r in azi_g[(ct, cm)]], model, CAM.K)[:, keep]
            noazi = CAM.RCA.rem.reconstruct_batch([r["pred"] for r in noazi_g[(ct, cm)]], model, CAM.K)[:, keep]
            stats = {
                "azi_vs_disease": CAM.RCA.seurat_log2fc(azi, dis),
                "noazi_vs_disease": CAM.RCA.seurat_log2fc(noazi, dis),
                "azi_vs_noazi": CAM.RCA.seurat_log2fc(azi, noazi),
            }
            pth = tmp / f"task_{len(paths):02d}_{cond}_{ct}.pkl"
            pickle.dump((cond, ct, genes, stats, sets), open(pth, "wb"))
            paths.append(pth)

    run_workers(paths, n_workers, t0)
    frames = [df for pth in paths for df in pickle.load(open(pth.with_suffix(".out.pkl"), "rb"))[2].values()]
    shutil.rmtree(tmp)
    allres = pd.concat(frames, ignore_index=True)
    allres.to_csv(out_dir / "camerapr_three_arms.csv", index=False)
    summ = CAM.summarize(allres)
    summ.to_csv(out_dir / "summary.csv", index=False)
    print(summ.to_string(index=False))
    print(f"DONE in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True, help="Results root directory (outside git)")
    ap.add_argument("--run", help="c2s_std run folder name, e.g. 2026-11-19_1543_rodent_drugs_full")
    ap.add_argument("--drugs", nargs="+", default=ALL_DRUGS, help="OOD drug arms to run")
    ap.add_argument("--rep300", action="store_true", help="Run on c2s-rep300 (300 vs 300)")
    ap.add_argument("--out", type=Path, help="Output subdirectory (must not exist; defaults under --out-root)")
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 8))
    args = ap.parse_args()
    out_root = args.out_root.resolve()
    if args.rep300:
        out = args.out or (out_root / f"camerapr_rep300_{time.strftime('%Y-%m-%d_%H%M')}")
        run_rep300(out, args.workers)
    elif args.run:
        out = args.out or (out_root / f"camerapr_{args.run}_{time.strftime('%Y-%m-%d_%H%M')}")
        run_std_drugs(out_root, args.run, args.drugs, out, args.workers)
    else:
        ap.error("Specify --run <name> or --rep300")


if __name__ == "__main__":
    main()
