#!/usr/bin/env python3
"""
Effect exploration for MODEL_AZI vs MODEL_NOAZI (azithromycin token effect) with calibrated nulls.

Why this exists: the first pass (run_standard_scrna_analysis.py) was descriptive. Its "reversal" metric
selected genes on MODEL_NOAZI-vs-CTRL and then compared AZI-vs-NOAZI, which shares NOAZI noise with
opposite sign (regression to the mean) and is biased toward "reversal" even under the null.
Here every effect statistic is compared against a label-permutation null (AZI/NOAZI labels shuffled
within cohort x cell type, group sizes preserved), so effects are judged against what pure
sampling noise produces at the actual (small) n.

Blocks:
  A. Omnibus: is there any AZI effect per cohort x cell type? (#genes Welch p<0.01 vs permutation null)
  B. Gene-level pan-cell test: Stouffer across 4 cell types, permutation-calibrated p, BH.
  C. Pan-cell consistency: same-sign counts and cross-cell-type correlation of AZI shifts vs null.
  D. Reversal re-test: biased (selection on NOAZI) vs unbiased (selection on real DISEASE vs CTRL), both vs null.
  E. Autophagy / lysosome modules and genes.
  F. FOXF1 detection/expression and putative endothelial target module.
  G. Audit of the genes named in the pan-cell pathway table.

Input:  scanpy_analysis/processed_h5ad/human_<cohort>_scanpy_processed.h5ad (.raw = log1p CPM10k, top-2000 matched)
Output: scanpy_analysis/effect_exploration/*.csv + effect_exploration_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import fisher_exact, rankdata, spearmanr, t as t_dist

SEED = 20261120
N_PERM = 1000
BASE = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/2026-11-20_human_h5ad_newgrid_lr1e-4_ep4_s42/scanpy_analysis")
H5AD_DIR = BASE / "processed_h5ad"
OUT = BASE / "effect_exploration"
OUT.mkdir(parents=True, exist_ok=True)

COHORTS = ["Acute26", "BPD7mo", "BPDPH7mo"]
CTS = ["gCap", "aCap", "Pericyte", "VEC"]
MIN_DET = 0.05          # gene must be detected in >=5% of cells of the cell type
T_SEL = 3.0             # |t| threshold for "pathology DEG" selection
PAN_THR = 0.10          # |delta log1p| threshold for pan-cell same-sign counting

GENE_SETS = {
    "autophagy_core": [
        "ULK1", "ULK2", "ATG13", "RB1CC1", "ATG101", "BECN1", "PIK3C3", "PIK3R4", "ATG14", "UVRAG",
        "WIPI1", "WIPI2", "ATG2A", "ATG2B", "ATG9A", "ATG3", "ATG4B", "ATG5", "ATG7", "ATG10", "ATG12",
        "ATG16L1", "MAP1LC3A", "MAP1LC3B", "GABARAP", "GABARAPL1", "GABARAPL2",
    ],
    "autophagy_receptors_mitophagy": [
        "SQSTM1", "NBR1", "OPTN", "CALCOCO2", "TAX1BP1", "BNIP3", "BNIP3L", "FUNDC1", "PINK1", "PRKN", "FKBP8",
    ],
    "lysosome_CLEAR_vATPase": [
        "TFEB", "TFE3", "MCOLN1", "LAMP1", "LAMP2", "CTSA", "CTSB", "CTSD", "CTSZ", "HEXA", "HEXB", "GBA1", "GLA",
        "NPC1", "NPC2", "ATP6V0A1", "ATP6V0B", "ATP6V0C", "ATP6V0D1", "ATP6V1A", "ATP6V1B2", "ATP6V1C1",
        "ATP6V1E1", "ATP6V1H", "ATP6AP1", "ATP6AP2",
    ],
    # Literature-curated putative endothelial FOXF1 targets (Kalinichenko lab: Kdr/Flk1, Flt1, Pecam1, Cdh5, S1pr1).
    # NOT a validated human regulon.
    "FOXF1_putative_EC_targets": ["KDR", "FLT1", "PECAM1", "CDH5", "S1PR1"],
}
KEY_GENES = ["FOXF1", "SQSTM1", "MAP1LC3B", "BECN1", "ATG5", "ATG7", "GABARAPL1", "BNIP3", "BNIP3L",
             "TFEB", "LAMP1", "LAMP2", "CTSB", "CTSD", "ATP6V0B", "ATP6V0C"]

USER_TABLE = {
    "BPD7mo": {
        "up": ["CCL2", "BIRC3", "SAMHD1", "KITLG", "MT2A", "S100A6", "GRB10", "HECA", "AP2A2", "RIN2", "PITPNB",
               "UMAD1", "TUBB", "RPL7", "SNHG6"],
        "down": ["PARD3", "IQGAP1", "MRTFB", "ICAM1", "HDAC7", "EPC1", "NCOA2", "SETBP1", "SPPL3", "VAPB",
                 "TRAK1", "RAB3GAP2"],
    },
    "BPDPH7mo": {
        "up": ["FAM114A1", "TSC22D1", "CSRP1", "RPS7", "RPS15A", "RPS3A", "RPL19", "RPL18", "RPL7A", "RPL37",
               "SRPRA", "TUT4", "ZFP36L1", "TMBIM4", "VPS29"],
        "down": ["REV3L", "HUWE1", "UBE2E2", "ADAR", "PTBP3", "NCL", "ATP6V0B", "FP236383.3"],
    },
    "Acute26": {
        "up": ["MT2A", "UGCG", "ZBTB38", "PCMTD1", "EIF4A1", "RPS3A", "RBM42", "KHDRBS1", "TUT4", "GSK3B",
               "PLCB4", "NDUFA5", "LMNA", "HIVEP2"],
        "down": ["CDKN1C", "NFKBIA", "PRMT2", "PRDX6", "ELK3", "CEP170", "MZT2A"],
    },
}


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.minimum(ranked, 1.0)
    out[ok] = res
    return out


def group_moments(X: np.ndarray, W: np.ndarray):
    """X: cells x genes; W: cells x P (0/1). Returns mean, unbiased var (genes x P) and n (P)."""
    n = W.sum(axis=0)
    s1 = X.T @ W
    s2 = (X ** 2).T @ W
    m = s1 / n
    v = np.maximum((s2 - n * m ** 2) / np.maximum(n - 1, 1), 0.0)
    return m, v, n


def welch_t(mA, vA, nA, mB, vB, nB):
    se = np.sqrt(vA / nA + vB / nB)
    diff = mA - mB
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 1e-12, diff / se, 0.0)
    return t, diff, se, vA, vB


def welch_p(t, vA, nA, vB, nB):
    a, b = vA / nA, vB / nB
    with np.errstate(divide="ignore", invalid="ignore"):
        df = (a + b) ** 2 / (a ** 2 / np.maximum(nA - 1, 1) + b ** 2 / np.maximum(nB - 1, 1))
    df = np.where(np.isfinite(df) & (df > 0), df, 1.0)
    return 2 * t_dist.sf(np.abs(t), df)


def dense(M):
    return M.toarray().astype(np.float64) if sp.issparse(M) else np.asarray(M, dtype=np.float64)


def make_perm_matrix(n_cells: int, n_a: int, rng: np.random.Generator) -> np.ndarray:
    """Column 0 = observed labels (first n_a rows are group A); columns 1..N_PERM = shuffles."""
    W = np.zeros((n_cells, N_PERM + 1), dtype=np.float64)
    W[:n_a, 0] = 1.0
    for j in range(1, N_PERM + 1):
        W[rng.choice(n_cells, n_a, replace=False), j] = 1.0
    return W


def emp_p(obs: float, null: np.ndarray, side: str = "greater") -> float:
    null = null[np.isfinite(null)]
    if side == "greater":
        k = np.sum(null >= obs)
    elif side == "less":
        k = np.sum(null <= obs)
    else:
        k = np.sum(np.abs(null - np.mean(null)) >= abs(obs - np.mean(null)))
    return float((1 + k) / (1 + len(null)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows_omni, rows_rev, rows_mod, rows_gene, rows_foxf1, rows_pan, rows_user = [], [], [], [], [], [], []
    stouffer_tables = {}
    summary = {"seed": SEED, "n_perm": N_PERM, "min_det": MIN_DET, "t_sel": T_SEL, "pan_thr": PAN_THR}

    for cohort in COHORTS:
        print(f"[{cohort}] loading", flush=True)
        adata = ad.read_h5ad(H5AD_DIR / f"human_{cohort}_scanpy_processed.h5ad")
        raw_names = np.asarray(adata.raw.var_names)
        Xraw = adata.raw.X.tocsr() if sp.issparse(adata.raw.X) else sp.csr_matrix(adata.raw.X)
        obs = adata.obs
        per_ct = {}

        for ct in CTS:
            m_ct = (obs["cell_type"] == ct).values
            arm = obs["orig.ident"].values
            idx = {a: np.where(m_ct & (arm == a))[0] for a in ["CTRL", "DISEASE", "MODEL_NOAZI", "MODEL_AZI"]}
            det = np.asarray((Xraw[m_ct] > 0).mean(axis=0)).ravel()
            gmask = det >= MIN_DET
            genes = raw_names[gmask]
            Xc = Xraw[:, gmask]

            X_azi = dense(Xc[idx["MODEL_AZI"]])
            X_noazi = dense(Xc[idx["MODEL_NOAZI"]])
            X_ctrl = dense(Xc[idx["CTRL"]])
            X_dis = dense(Xc[idx["DISEASE"]])
            n_a, n_n = len(X_azi), len(X_noazi)
            Xm = np.vstack([X_azi, X_noazi])
            W = make_perm_matrix(n_a + n_n, n_a, rng)

            mA, vA, nA = group_moments(Xm, W)
            mN, vN, nN = group_moments(Xm, 1.0 - W)
            t_an, d_an, _, _, _ = welch_t(mA, vA, nA, mN, vN, nN)          # genes x (P+1)
            p_an_obs = welch_p(t_an[:, 0], vA[:, 0], nA[0], vN[:, 0], nN[0])

            # --- A. Omnibus
            n_p01 = (welch_p(t_an, vA, nA, vN, nN) < 0.01).sum(axis=0)
            rows_omni.append({
                "cohort": cohort, "cell_type": ct, "n_azi": n_a, "n_noazi": n_n, "n_genes": int(gmask.sum()),
                "n_genes_p01_obs": int(n_p01[0]), "null_mean": float(n_p01[1:].mean()),
                "null_q95": float(np.quantile(n_p01[1:], 0.95)), "perm_p": emp_p(n_p01[0], n_p01[1:], "greater"),
                "n_genes_bh10_welch": int((bh(p_an_obs) < 0.10).sum()),
                "median_ngenes_azi": float(np.median((X_azi > 0).sum(1))), "median_ngenes_noazi": float(np.median((X_noazi > 0).sum(1))),
            })

            # --- D. Reversal re-test
            mC, vC, nC = X_ctrl.mean(0), X_ctrl.var(0, ddof=1), len(X_ctrl)
            mD, vD, nD = X_dis.mean(0), X_dis.var(0, ddof=1), len(X_dis)
            t_nc, _, _, _, _ = welch_t(mN, vN, nN, mC[:, None], vC[:, None], nC)   # NOAZI(perm) vs CTRL
            sel_b = np.abs(t_nc) >= T_SEL
            disagree = np.sign(d_an) != np.sign(t_nc)
            rev_b = (sel_b & disagree).sum(0) / np.maximum(sel_b.sum(0), 1)
            rho_b = np.array([spearmanr(t_nc[sel_b[:, j], j], t_an[sel_b[:, j], j])[0] if sel_b[:, j].sum() > 10 else np.nan
                              for j in range(N_PERM + 1)])
            t_dc, _, _, _, _ = welch_t(mD, vD, nD, mC, vC, nC)                    # real DISEASE vs CTRL (fixed)
            sel_u = np.abs(t_dc) >= T_SEL
            rev_u = (np.sign(d_an[sel_u]) != np.sign(t_dc[sel_u])[:, None]).mean(0)
            rk_dis = rankdata(t_dc[sel_u])
            rk_an = np.apply_along_axis(rankdata, 0, t_an[sel_u])
            rho_u = np.array([np.corrcoef(rk_dis, rk_an[:, j])[0, 1] for j in range(N_PERM + 1)])
            for label, rev, rho, nsel in [("biased_select_on_NOAZI_vs_CTRL", rev_b, rho_b, int(sel_b[:, 0].sum())),
                                          ("unbiased_select_on_real_DISEASE_vs_CTRL", rev_u, rho_u, int(sel_u.sum()))]:
                rows_rev.append({
                    "cohort": cohort, "cell_type": ct, "selection": label, "n_selected_obs": nsel,
                    "reversal_obs": float(rev[0]), "reversal_null_mean": float(np.nanmean(rev[1:])),
                    "reversal_null_q95": float(np.nanquantile(rev[1:], 0.95)), "reversal_perm_p": emp_p(rev[0], rev[1:], "greater"),
                    "rho_obs": float(rho[0]), "rho_null_mean": float(np.nanmean(rho[1:])),
                    "rho_null_q05": float(np.nanquantile(rho[1:], 0.05)), "rho_perm_p_negative": emp_p(rho[0], rho[1:], "less"),
                })

            # --- E/F. modules (mean log1p over present genes) with permutation on AZI-NOAZI
            gene_pos = {g: i for i, g in enumerate(genes)}
            for set_name, gs in GENE_SETS.items():
                present = [g for g in gs if g in gene_pos]
                if len(present) < 2:
                    rows_mod.append({"cohort": cohort, "cell_type": ct, "module": set_name, "n_genes_present": len(present)})
                    continue
                cols = [gene_pos[g] for g in present]
                d_mod = d_an[cols].mean(0)
                sc = {a: float(dense(Xc[idx[a]][:, cols]).mean()) for a in idx}
                rows_mod.append({
                    "cohort": cohort, "cell_type": ct, "module": set_name, "n_genes_present": len(present),
                    "genes_present": ";".join(present),
                    "CTRL": sc["CTRL"], "DISEASE": sc["DISEASE"], "MODEL_NOAZI": sc["MODEL_NOAZI"], "MODEL_AZI": sc["MODEL_AZI"],
                    "delta_DIS_minus_CTRL": sc["DISEASE"] - sc["CTRL"],
                    "delta_NOAZI_minus_DIS": sc["MODEL_NOAZI"] - sc["DISEASE"],
                    "delta_AZI_minus_NOAZI": float(d_mod[0]),
                    "perm_p_two_sided": emp_p(d_mod[0], d_mod[1:], "two"),
                    "null_sd": float(np.std(d_mod[1:])),
                })

            # per-gene table for key genes (incl. absent-from-active-set genes)
            for g in KEY_GENES:
                rec = {"cohort": cohort, "cell_type": ct, "gene": g}
                if g in set(raw_names):
                    j = int(np.where(raw_names == g)[0][0])
                    col = Xraw[:, j]
                    for a in idx:
                        v = dense(col[idx[a]]).ravel()
                        rec[f"det_{a}"] = float((v > 0).mean()) if len(v) else np.nan
                        rec[f"mean_{a}"] = float(v.mean()) if len(v) else np.nan
                    if g in gene_pos:
                        k = gene_pos[g]
                        rec["delta_AZI_minus_NOAZI"] = float(d_an[k, 0])
                        rec["welch_p_AZI_vs_NOAZI"] = float(p_an_obs[k])
                        rec["perm_p_AZI_vs_NOAZI"] = emp_p(abs(t_an[k, 0]), np.abs(t_an[k, 1:]), "greater")
                    # detection Fisher tests
                    def fisher(a1, a2):
                        v1 = dense(col[idx[a1]]).ravel() > 0
                        v2 = dense(col[idx[a2]]).ravel() > 0
                        if len(v1) == 0 or len(v2) == 0:
                            return np.nan
                        return float(fisher_exact([[v1.sum(), (~v1).sum()], [v2.sum(), (~v2).sum()]])[1])
                    rec["fisher_det_AZI_vs_NOAZI"] = fisher("MODEL_AZI", "MODEL_NOAZI")
                    rec["fisher_det_DIS_vs_CTRL"] = fisher("DISEASE", "CTRL")
                    rec["fisher_det_NOAZI_vs_DIS"] = fisher("MODEL_NOAZI", "DISEASE")
                rows_gene.append(rec)

            per_ct[ct] = {"genes": genes, "t": t_an, "d": d_an, "p_obs": p_an_obs, "det_azi": (X_azi > 0).mean(0),
                          "det_noazi": (X_noazi > 0).mean(0)}

        # --- B/C. Pan-cell analysis on genes active in all 4 cell types
        common = sorted(set.intersection(*[set(per_ct[c]["genes"]) for c in CTS]))
        pos = {c: {g: i for i, g in enumerate(per_ct[c]["genes"])} for c in CTS}
        T = np.stack([per_ct[c]["t"][[pos[c][g] for g in common]] for c in CTS])   # 4 x G x (P+1)
        D = np.stack([per_ct[c]["d"][[pos[c][g] for g in common]] for c in CTS])
        Z = T.sum(0) / 2.0                                                          # Stouffer (unit weights)
        p_st = (1 + (np.abs(Z[:, 1:]) >= np.abs(Z[:, [0]])).sum(1)) / (1 + N_PERM)
        q_st = bh(p_st)
        same_up = ((D > PAN_THR).all(0)).sum(0)
        same_dn = ((D < -PAN_THR).all(0)).sum(0)
        same_any = ((np.sign(D) == np.sign(D[0])).all(0) & (D[0] != 0)).sum(0)
        ranks = np.stack([np.apply_along_axis(rankdata, 0, D[k]) for k in range(4)])
        pair_rho = []
        for a in range(4):
            for b in range(a + 1, 4):
                ra, rb = ranks[a] - ranks[a].mean(0), ranks[b] - ranks[b].mean(0)
                pair_rho.append((ra * rb).sum(0) / np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0)))
        mean_rho = np.mean(pair_rho, axis=0)
        rows_pan.append({
            "cohort": cohort, "n_common_genes": len(common),
            "same_sign_all4_obs": int(same_any[0]), "same_sign_all4_null_mean": float(same_any[1:].mean()),
            "same_sign_all4_perm_p": emp_p(same_any[0], same_any[1:], "greater"),
            f"up_all4_absdelta_gt{PAN_THR}_obs": int(same_up[0]), "up_null_mean": float(same_up[1:].mean()),
            "up_null_q95": float(np.quantile(same_up[1:], 0.95)), "up_perm_p": emp_p(same_up[0], same_up[1:], "greater"),
            f"down_all4_absdelta_gt{PAN_THR}_obs": int(same_dn[0]), "down_null_mean": float(same_dn[1:].mean()),
            "down_null_q95": float(np.quantile(same_dn[1:], 0.95)), "down_perm_p": emp_p(same_dn[0], same_dn[1:], "greater"),
            "mean_pairwise_spearman_obs": float(mean_rho[0]), "mean_pairwise_spearman_null_mean": float(mean_rho[1:].mean()),
            "mean_pairwise_spearman_null_q95": float(np.quantile(mean_rho[1:], 0.95)),
            "mean_pairwise_spearman_perm_p": emp_p(mean_rho[0], mean_rho[1:], "greater"),
            "stouffer_n_q10": int((q_st < 0.10).sum()), "stouffer_n_q25": int((q_st < 0.25).sum()),
            "stouffer_min_perm_p": float(p_st.min()),
        })
        st = pd.DataFrame({"gene": common, "stouffer_z": Z[:, 0], "perm_p": p_st, "bh_q": q_st})
        for k, c in enumerate(CTS):
            st[f"delta_{c}"] = D[k, :, 0]
            st[f"t_{c}"] = T[k, :, 0]
            st[f"detAZI_{c}"] = per_ct[c]["det_azi"][[pos[c][g] for g in common]]
            st[f"detNOAZI_{c}"] = per_ct[c]["det_noazi"][[pos[c][g] for g in common]]
        st = st.sort_values("perm_p")
        st.to_csv(OUT / f"{cohort}_pan_cell_stouffer.csv", index=False)
        stouffer_tables[cohort] = st.set_index("gene")

        # --- G. Audit of user's table genes
        for direction, gl in USER_TABLE[cohort].items():
            for g in gl:
                rec = {"cohort": cohort, "claimed": direction, "gene": g}
                if g in stouffer_tables[cohort].index:
                    r = stouffer_tables[cohort].loc[g]
                    rec.update({"in_all4_active": True, "stouffer_z": r["stouffer_z"], "perm_p": r["perm_p"], "bh_q": r["bh_q"]})
                    for c in CTS:
                        rec[f"delta_{c}"] = r[f"delta_{c}"]
                        rec[f"detAZI_{c}"] = r[f"detAZI_{c}"]
                        rec[f"detNOAZI_{c}"] = r[f"detNOAZI_{c}"]
                    rec["sign_consistent_4"] = bool((np.sign([r[f"delta_{c}"] for c in CTS]) == (1 if direction == "up" else -1)).all())
                    rec["n_ct_welch_p05"] = int(sum(per_ct[c]["p_obs"][pos[c][g]] < 0.05 for c in CTS))
                    rec["min_det_any_arm"] = float(min(min(r[f"detAZI_{c}"], r[f"detNOAZI_{c}"]) for c in CTS))
                else:
                    rec["in_all4_active"] = False
                rows_user.append(rec)

    pd.DataFrame(rows_omni).to_csv(OUT / "A_omnibus_azi_effect.csv", index=False)
    pd.DataFrame(rows_pan).to_csv(OUT / "C_pan_cell_consistency.csv", index=False)
    pd.DataFrame(rows_rev).to_csv(OUT / "D_reversal_retest.csv", index=False)
    mod = pd.DataFrame(rows_mod)
    if "perm_p_two_sided" in mod:
        mod["bh_q_within_module"] = mod.groupby("module")["perm_p_two_sided"].transform(lambda s: pd.Series(bh(s.values), index=s.index))
    mod.to_csv(OUT / "E_F_modules_autophagy_foxf1.csv", index=False)
    pd.DataFrame(rows_gene).to_csv(OUT / "E_F_key_genes_autophagy_foxf1.csv", index=False)
    pd.DataFrame(rows_user).to_csv(OUT / "G_user_table_audit.csv", index=False)

    # PCA-shift p-values from first pass, BH across the 12 tests
    first = json.load(open(BASE / "scanpy_standard_analysis_summary.json"))
    pca_rows = []
    for coh in COHORTS:
        for ct, v in first[coh]["step3_dimred"]["pca_disease_to_ctrl_shift_by_ct"].items():
            pca_rows.append({"cohort": coh, "cell_type": ct, "delta_toward_ctrl": v["delta_shift_toward_ctrl"], "p": v["shift_mannwhitney_p"]})
    pca = pd.DataFrame(pca_rows)
    pca["bh_q"] = bh(pca["p"].values)
    pca.to_csv(OUT / "H_pca_shift_bh.csv", index=False)

    summary["files"] = sorted(p.name for p in OUT.glob("*.csv"))
    json.dump(summary, open(OUT / "effect_exploration_summary.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
