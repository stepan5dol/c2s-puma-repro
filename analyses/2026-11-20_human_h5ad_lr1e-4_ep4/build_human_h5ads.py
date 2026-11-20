#!/usr/bin/env python3
"""
Build per-state human .h5ad files combining REAL counts (ctrl + disease) and
reconstructed MODEL predictions (model_noazi + model_azi) from the canonical
newgrid_lr1e-4_ep4_s42 run, K=2000.

Calibration: c2s-puma-std/.../reports/rank_expr_model_k2000.npz (K=2000 fit, r2 0.8665; lib/fit_calibration_k2000.py).
Until 2026-11-20 this script loaded reports/rank_expr_model.npz, the K=800 fit (r2 0.8361).
usage: python3 build_human_h5ads.py --out-dir <new folder> [--real-from <folder with earlier h5ad files>]
  --real-from: take the real cells (CTRL, DISEASE; full log1p CPM10k layer and obs) from earlier files of this script
               instead of scanning the counts CSV; real cells do not depend on the calibration.

Outputs:
  human_Acute26_real_and_model.h5ad   (He22 ctrl + Acute26 real + Acute26 model_noazi + Acute26 model_azi)
  human_BPD7mo_real_and_model.h5ad    (Term ctrl + BPD7mo real  + BPD7mo model_noazi  + BPD7mo model_azi)
  human_BPDPH7mo_real_and_model.h5ad  (Term ctrl + BPDPH7mo real + BPDPH7mo model_noazi + BPDPH7mo model_azi)

Matrix conventions:
  adata.X                         : CSR float32, log1p(CPM10k), top-2000-matched (real truncated to top-2000, model reconstructed K=2000)
  adata.layers['log1p_cpm10k_full']: CSR float32, full untruncated log1p(CPM10k) for real cells; same reconstructed K=2000 for model cells
  adata.layers['cpm10k_top2000']  : CSR float32, expm1(adata.X) (linear CPM10k scale)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE = Path("/Users/stepandolzhenko/Documents/AzithroGemma")
OUT_DIR = BASE / "retrain_v2/results/2026-11-20_human_h5ad_newgrid_lr1e-4_ep4_s42"
RUN_DIR = BASE / "retrain_prompt/results/puma_answers/xdisk/vkalin/dolzhenko/c2s/results/2026-11-16_132201_puma-a100mig-repro-ep4-fp32-lr1e-4-s42/inference_float32"
REG_CSV = BASE / "retrain_prompt/pipeline/sft_example_registry.csv"
STD_ANA = BASE / "retrain_v2/c2s-puma-std/grid/tree/retrain_v2/analysis"
NPZ_MODEL = STD_ANA / "reports/rank_expr_model_k2000.npz"

sys.path.insert(0, str(STD_ANA / "real_counts_suite_ortho"))
sys.path.insert(0, str(STD_ANA / "pathway_module_analysis"))
sys.path.insert(0, str(STD_ANA))

import common_real as CR
import rank_expr_model as rem
import real_counts as RC

K = 2000
CR.K = K
CTS = ["gCap", "aCap", "Pericyte", "VEC"]
CT_MAP_META = {
    "gCap": "gCap", "aCap": "aCap", "Pericyte": "Pericyte",
    "Pulmonary venous EC": "VEC", "VEC": "VEC",
}
CT_MAP_PROMPT = {
    "general capillary endothelial cell": "gCap",
    "aerocyte capillary endothelial cell": "aCap",
    "pericyte": "Pericyte",
    "pulmonary venous endothelial cell": "VEC",
    "vascular endothelial cell": "VEC",
}
COND_MAP_PROMPT = {
    "acute preterm lung injury": "Acute26",
    "established bronchopulmonary dysplasia": "BPD7mo",
    "established bronchopulmonary dysplasia with pulmonary hypertension": "BPDPH7mo",
}
D2C = {
    "Term infant 1": "Term0d", "Term infant 2": "Term20d",
    "Acute preterm injury 1": "Acute26",
    "BPD 1": "BPD7mo", "BPD 2": "BPD7mo",
    "BPD+PH 1": "BPDPH7mo", "BPD+PH 2": "BPDPH7mo",
}


def load_human_model() -> rem.RankExprModel:
    z = np.load(NPZ_MODEL, allow_pickle=True)
    assert int(z["k"]) == K, f"{NPZ_MODEL} is a K={int(z['k'])} calibration, need K={K}"
    names = list(z["gene_names"])
    return rem.RankExprModel(
        float(z["slope"]), float(z["intercept"]), float(z["r2"]),
        {g: i for i, g in enumerate(names)}, names,
    )


def scan_bpd_csv(model: rem.RankExprModel):
    print("[1/4] Scanning GSE275938_compiled_counts.csv once for all 5 human conditions...", flush=True)
    meta = pd.read_csv(BASE / "BPD-PH/GSE275938_cell_metadata.csv", index_col="id")
    meta["condition"] = meta["dataset"].map(D2C)
    meta["ct"] = meta["celltype"].map(CT_MAP_META)
    meta = meta[meta["condition"].isin(["Term0d", "Term20d", "Acute26", "BPD7mo", "BPDPH7mo"]) & meta["ct"].isin(CTS)]
    needed = set(meta.index)

    chunks_X, chunks_bc, genes_raw = [], [], None
    for chunk in pd.read_csv(BASE / "BPD-PH/GSE275938_compiled_counts.csv", index_col="id", chunksize=5000):
        if genes_raw is None:  # gene names as the manuscript maps them (load_data.remap_gene_names)
            genes_raw = RC.L.remap_gene_names(chunk.columns.tolist())
        hit = [bc for bc in chunk.index if bc in needed]
        if hit:
            chunks_X.append(chunk.loc[hit].values.astype(np.float64))
            chunks_bc.extend(hit)

    X_raw = np.vstack(chunks_X)
    lib = X_raw.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    X_norm = np.log1p(X_raw / lib * 1e4).astype(np.float32)

    # Align to model.gene_names (36,601) with the manuscript's function (run_corrected_analysis.single_pass_load_bpd)
    X_full = RC._reindex_to_model_genes(X_norm, genes_raw, model.gene_names).astype(np.float32)

    sub_meta = meta.loc[chunks_bc].copy()
    print(f"  loaded {X_full.shape[0]} BPD-PH/Term cells x {X_full.shape[1]} genes", flush=True)
    return X_full, sub_meta


def load_he22(model: rem.RankExprModel):
    print("[2/4] Loading He22 fetal GW24 control matrices...", flush=True)
    mats = RC.real_he22_matrices(model)
    Xs, obs_rows = [], []
    for ct in CTS:
        if ct not in mats:
            continue
        X = mats[ct].astype(np.float32)
        Xs.append(X)
        for i in range(X.shape[0]):
            obs_rows.append({
                "cell_id": f"He22_{ct}_{i:04d}",
                "orig.ident": "CTRL",
                "condition": "He22_GW24",
                "source": "real",
                "arm": "ctrl",
                "cell_type": ct,
                "dataset": "He22_fetal_GW24",
            })
    X_all = np.vstack(Xs)
    obs = pd.DataFrame(obs_rows).set_index("cell_id")
    print(f"  loaded {X_all.shape[0]} He22 cells", flush=True)
    return X_all, obs


def load_model_predictions(model: rem.RankExprModel):
    print("[3/4] Reconstructing model NO-AZI (test_t10) and AZI (inference_azi) cells (K=2000)...", flush=True)
    reg = pd.read_csv(REG_CSV)
    test_bpd = reg[(reg["split"] == "test") & (reg["direction"] == "fwd") & (reg["source"] == "bpd")].copy()
    test_bpd["ct"] = test_bpd["cell_type"].map(CT_MAP_META)
    idx_info = {int(r.line_idx): (r.condition, r.ct, str(r.barcode)) for _, r in test_bpd.iterrows()}

    noazi_recs: dict[str, list[tuple[str, str, list[str]]]] = {c: [] for c in ("Acute26", "BPD7mo", "BPDPH7mo")}
    with open(RUN_DIR / "test_t10.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["idx"] in idx_info:
                cond, ct, bc = idx_info[r["idx"]]
                if cond in noazi_recs:
                    noazi_recs[cond].append((ct, f"model_noazi_{cond}_{ct}_idx{r['idx']}_{bc}", r["pred"].strip().split()))

    azi_recs: dict[str, list[tuple[str, str, list[str]]]] = {c: [] for c in ("Acute26", "BPD7mo", "BPDPH7mo")}
    with open(RUN_DIR / "inference_azi_results.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            p = r["prompt"]
            cond_str = [ln for ln in p.splitlines() if ln.startswith("Condition:")][0].split(": ", 1)[1].strip()
            cond = COND_MAP_PROMPT[cond_str]
            ct_raw = [ln for ln in p.splitlines() if ln.startswith("Cell type:")][0].split(": ", 1)[1].split("(")[0].strip().lower()
            ct = CT_MAP_PROMPT[ct_raw]
            azi_recs[cond].append((ct, f"model_azi_{cond}_{ct}_{i:04d}", r["pred"].strip().split()))

    reconstructed = {}
    for cond in ("Acute26", "BPD7mo", "BPDPH7mo"):
        for arm, recs, ident in (
            ("model_noazi", noazi_recs[cond], "MODEL_NOAZI"),
            ("model_azi", azi_recs[cond], "MODEL_AZI"),
        ):
            X = rem.reconstruct_batch([g for _, _, g in recs], model, K).astype(np.float32)
            obs = pd.DataFrame([
                {
                    "cell_id": cid,
                    "orig.ident": ident,
                    "condition": f"{cond}_{arm}",
                    "source": "model",
                    "arm": arm,
                    "cell_type": ct,
                    "dataset": "newgrid_lr1e-4_ep4_s42",
                }
                for ct, cid, _ in recs
            ]).set_index("cell_id")
            reconstructed[(cond, arm)] = (X, obs)
            print(f"  {cond}/{arm}: {X.shape[0]} cells", flush=True)
    return reconstructed


def build_h5ad(
    cond: str,
    X_ctrl_full: np.ndarray, obs_ctrl: pd.DataFrame,
    X_dis_full: np.ndarray, obs_dis: pd.DataFrame,
    X_noazi: np.ndarray, obs_noazi: pd.DataFrame,
    X_azi: np.ndarray, obs_azi: pd.DataFrame,
    model: rem.RankExprModel,
    out_path: Path,
) -> None:
    X_ctrl_top = CR.truncate_topk(X_ctrl_full, K).astype(np.float32)
    X_dis_top = CR.truncate_topk(X_dis_full, K).astype(np.float32)

    X_top = sp.vstack([
        sp.csr_matrix(X_ctrl_top),
        sp.csr_matrix(X_dis_top),
        sp.csr_matrix(X_noazi),
        sp.csr_matrix(X_azi),
    ], format="csr", dtype=np.float32)

    X_full = sp.vstack([
        sp.csr_matrix(X_ctrl_full),
        sp.csr_matrix(X_dis_full),
        sp.csr_matrix(X_noazi),
        sp.csr_matrix(X_azi),
    ], format="csr", dtype=np.float32)

    X_cpm = X_top.copy()
    X_cpm.data = np.expm1(X_cpm.data).astype(np.float32)

    obs = pd.concat([obs_ctrl, obs_dis, obs_noazi, obs_azi], axis=0)
    obs["n_genes_nonzero"] = np.diff(X_top.indptr).astype(np.int32)
    obs["n_genes_nonzero_full"] = np.diff(X_full.indptr).astype(np.int32)
    # Categorical columns for fast scanpy/Seurat plotting
    for col in ("orig.ident", "condition", "source", "arm", "cell_type", "dataset"):
        obs[col] = obs[col].astype("category")

    var = pd.DataFrame(index=pd.Index(model.gene_names, name="gene"))
    adata = ad.AnnData(
        X=X_top,
        obs=obs,
        var=var,
        layers={
            "log1p_cpm10k_full": X_full,
            "cpm10k_top2000": X_cpm,
        },
        uns={
            "cohort": cond,
            "model_run": "newgrid_lr1e-4_ep4_s42",
            "K": K,
            "calibration": {
                "slope": float(model.slope),
                "intercept": float(model.intercept),
                "r2": float(model.r2),
            },
            "X_description": "log1p(CPM10k), top-2000-matched across real and model cells",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    adata.write_h5ad(out_path, compression="gzip")
    mb = out_path.stat().st_size / 1e6
    counts = obs["orig.ident"].value_counts().to_dict()
    print(f"  -> {out_path.name} ({mb:.1f} MB): {adata.n_obs} cells x {adata.n_vars} genes | {counts}", flush=True)


def real_from_h5ad(src: Path):
    """CTRL and DISEASE rows of earlier files: dense full log1p CPM10k and obs, per cohort."""
    out = {}
    cols = ["orig.ident", "condition", "source", "arm", "cell_type", "dataset"]
    for cond in ("Acute26", "BPD7mo", "BPDPH7mo"):
        a = ad.read_h5ad(src / f"human_{cond}_real_and_model.h5ad")
        for grp in ("CTRL", "DISEASE"):
            m = (a.obs["orig.ident"] == grp).values
            obs = a.obs.loc[m, cols].copy()
            for c in cols:
                obs[c] = obs[c].astype(str)
            out[(cond, grp)] = (a.layers["log1p_cpm10k_full"][m].toarray().astype(np.float32), obs)
        print(f"  real cells of {cond} from {src.name}: CTRL {out[(cond, 'CTRL')][0].shape[0]}, DISEASE {out[(cond, 'DISEASE')][0].shape[0]}", flush=True)
    return out


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--real-from", type=Path, default=None)
    args = ap.parse_args()
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=False)
    model = load_human_model()
    print(f"[calibration] {NPZ_MODEL.name}: slope={model.slope:.6f} intercept={model.intercept:.6f} r2={model.r2:.4f}", flush=True)
    preds = load_model_predictions(model)
    if args.real_from is not None:
        real = real_from_h5ad(args.real_from)
        print("[4/4] Writing 3 per-state .h5ad files...", flush=True)
        for cond in ("Acute26", "BPD7mo", "BPDPH7mo"):
            X_c, obs_c = real[(cond, "CTRL")]
            X_d, obs_d = real[(cond, "DISEASE")]
            X_no, obs_no = preds[(cond, "model_noazi")]
            X_az, obs_az = preds[(cond, "model_azi")]
            build_h5ad(cond, X_c, obs_c, X_d, obs_d, X_no, obs_no, X_az, obs_az,
                       model, OUT_DIR / f"human_{cond}_real_and_model.h5ad")
        print("ALL 3 H5AD FILES DONE", flush=True)
        return
    X_bpd_full, meta_bpd = scan_bpd_csv(model)
    X_he22_full, obs_he22 = load_he22(model)

    # Term0d+Term20d real control slice
    m_term = meta_bpd["condition"].isin(["Term0d", "Term20d"]).values
    X_term_full = X_bpd_full[m_term]
    obs_term = pd.DataFrame({
        "orig.ident": "CTRL",
        "condition": meta_bpd.loc[m_term, "condition"].values,
        "source": "real",
        "arm": "ctrl",
        "cell_type": meta_bpd.loc[m_term, "ct"].values,
        "dataset": meta_bpd.loc[m_term, "dataset"].values,
    }, index=pd.Index([f"real_ctrl_{bc}" for bc in meta_bpd.index[m_term]], name="cell_id"))

    print("[4/4] Writing 3 per-state .h5ad files...", flush=True)
    for cond, (X_c, obs_c) in [
        ("Acute26", (X_he22_full, obs_he22)),
        ("BPD7mo", (X_term_full, obs_term)),
        ("BPDPH7mo", (X_term_full, obs_term)),
    ]:
        m_dis = (meta_bpd["condition"] == cond).values
        X_dis_full = X_bpd_full[m_dis]
        obs_dis = pd.DataFrame({
            "orig.ident": "DISEASE",
            "condition": cond,
            "source": "real",
            "arm": "disease",
            "cell_type": meta_bpd.loc[m_dis, "ct"].values,
            "dataset": meta_bpd.loc[m_dis, "dataset"].values,
        }, index=pd.Index([f"real_dis_{bc}" for bc in meta_bpd.index[m_dis]], name="cell_id"))

        X_no, obs_no = preds[(cond, "model_noazi")]
        X_az, obs_az = preds[(cond, "model_azi")]

        build_h5ad(
            cond, X_c, obs_c, X_dis_full, obs_dis, X_no, obs_no, X_az, obs_az,
            model, OUT_DIR / f"human_{cond}_real_and_model.h5ad",
        )
    print("ALL 3 H5AD FILES DONE", flush=True)


if __name__ == "__main__":
    main()
