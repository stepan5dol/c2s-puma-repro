#!/usr/bin/env python3
"""Figure 5 and Figures 5-S1 to 5-S8 of the manuscript for newgrid_lr1e-4_ep4_s42.

Violins (5A, 5B, 5-S2 to 5-S6) and the cameraPR tables come from scripts/run_violins.sh. Panel 5C, the screens of
5-S8 and the Venn diagram of 5-S7 are drawn by the manuscript's own scripts in retrain_v2/analysis/
real_counts_suite_ortho (build_panelC_dotplot.py, venn_rat_human_model_run4.py, build_main_figure.py), run
unchanged on a copy under <run folder>/work/draw. 5-S1 is the energy distance between generated and measured cells
of each state (grid/lib/a8_baselines.py construction).
usage: python3 build_figures.py <run folder> [model jsonl ...]
  model jsonl  generated cells of the human disease states for 5-S1 (default: the test file)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/Users/stepandolzhenko/Documents/AzithroGemma")
RUN_DIR = Path(sys.argv[1]).resolve()
MODEL_JSONL = [Path(p) for p in sys.argv[2:]]
OUT_DIR = RUN_DIR / "figures"
DRAW = RUN_DIR / "work" / "draw"
VIOLINS = RUN_DIR / "work/violins/runs/newgrid_lr1e-4_ep4_s42/violins"
TEST_JSONL = BASE / ("retrain_prompt/results/puma_answers/xdisk/vkalin/dolzhenko/c2s/results/"
                     "2026-11-16_132201_puma-a100mig-repro-ep4-fp32-lr1e-4-s42/inference_float32/test_t10.jsonl")
ANALYSIS = BASE / "retrain_v2/analysis"
TREE_ANALYSIS = BASE / "retrain_v2/c2s-puma-std/grid/tree/retrain_v2/analysis"
NPZ_MODEL = TREE_ANALYSIS / "reports/rank_expr_model_k2000.npz"
GRID_LIB = BASE / "retrain_v2/c2s-puma-repro/grid/lib"

sys.path.insert(0, str(ANALYSIS / "real_counts_suite_ortho"))
sys.path.insert(0, str(TREE_ANALYSIS))
import common_human as CH  # noqa: E402
import plot_style as PS  # noqa: E402
import rank_expr_model as rem  # noqa: E402
from deck_figures import GRID, INK, MUTED, style as deck_style  # noqa: E402

K = 2000
CH.K = K
CTS = ["gCap", "aCap", "Pericyte", "VEC"]
DRAW_SCRIPTS = ["build_panelC_dotplot.py", "venn_rat_human_model_run4.py", "build_main_figure.py",
                "deck_figures.py", "ortho.py"]
ORTHO_DATA = ["ortholog_pairs.tsv", "ortholog_pairs.provenance.json", "hgnc_complete_set.tsv", "genes_rat_rgd.tsv"]


def copy_violins() -> None:
    for src, dst in [
        ("human/figures/human_delta_fetal_gw22_to_ali_gw26_all_rescued.png", "Figure_5-S2_deltas_acute_injury.png"),
        ("human/figures/human_delta_term_born_to_bpd_7mo_all_rescued.png", "Figure_5-S3_deltas_bpd.png"),
        ("human/figures/human_delta_term_born_to_bpd_ph_7mo_all_rescued.png", "Figure_5-S4_deltas_bpd_ph.png"),
        ("rat/figures/rat_delta_room_air_to_hyperoxia_all_rescued.png", "Figure_5-S5_deltas_rat_hyperoxia.png"),
        ("rat/figures/rat_delta_hyperoxia_to_azithromycin_all_rescued.png", "Figure_5-S6_deltas_rat_azithromycin.png"),
    ]:
        shutil.copy2(VIOLINS / src, OUT_DIR / dst)
        print(f"-> {dst}")


def draw_manuscript_figures() -> None:
    suite = DRAW / "analysis" / "real_counts_suite_ortho"
    hpv = suite / "human_panel_variants"
    for d in (hpv / "autophagy_program" / "reports", hpv / "reports", hpv / "figures", suite / "rat_panel_variants" / "figures"):
        d.mkdir(parents=True)
    shutil.copy2(ANALYSIS / "plot_style.py", DRAW / "analysis" / "plot_style.py")
    for f in DRAW_SCRIPTS:
        shutil.copy2(ANALYSIS / "real_counts_suite_ortho" / f, suite / f)
    for f in ORTHO_DATA:
        (suite / f).symlink_to(ANALYSIS / "real_counts_suite_ortho" / f)
    cam = VIOLINS / "human" / "camerapr"
    for cond in ("bpd7mo", "bpdph7mo", "acute26"):
        shutil.copy2(cam / f"{cond}_limma_sets.csv", hpv / "autophagy_program" / "reports" / f"{cond}_limma_sets.csv")
    shutil.copy2(cam / "human_rescue_genomewide.csv", hpv / "reports" / "human_rescue_genomewide.csv")
    shutil.copy2(VIOLINS / "human/figures/human_delta_term_born_to_bpd_7mo_all_rescued_gcap.png", hpv / "figures")
    shutil.copy2(VIOLINS / "rat/figures/rat_delta_hyperoxia_to_azithromycin_all_rescued_gcap.png",
                 suite / "rat_panel_variants" / "figures")

    def run(*args: str) -> None:
        subprocess.run([sys.executable, *args], cwd=suite, check=True)

    for cohort in ("BPD7mo", "Acute26", "BPDPH7mo"):
        run("build_panelC_dotplot.py", "--cohort", cohort)
    run("venn_rat_human_model_run4.py", "--bare")
    run("build_main_figure.py")

    for src, dst in [
        ("figures_main/main_figure.png", "Figure_5_main.png"),
        ("figures_main/main_figure.pdf", "Figure_5_main.pdf"),
        ("figures_main/main_figure_caption.txt", "Figure_5_main_caption.txt"),
        ("deck_figures/camerapr_panelC_dotplot.csv", "Figure_5_main_panelC.csv"),
        ("deck_figures/camerapr_panelC_dotplot_acute26.png", "Figure_5-S8a_screen_acute_injury.png"),
        ("deck_figures/camerapr_panelC_dotplot_acute26.pdf", "Figure_5-S8a_screen_acute_injury.pdf"),
        ("deck_figures/camerapr_panelC_dotplot_acute26.csv", "Figure_5-S8a_screen_acute_injury.csv"),
        ("deck_figures/camerapr_panelC_dotplot_bpdph7mo.png", "Figure_5-S8b_screen_bpd_ph.png"),
        ("deck_figures/camerapr_panelC_dotplot_bpdph7mo.pdf", "Figure_5-S8b_screen_bpd_ph.pdf"),
        ("deck_figures/camerapr_panelC_dotplot_bpdph7mo.csv", "Figure_5-S8b_screen_bpd_ph.csv"),
        ("figures_venn_run4/venn_rat_bpd7mo_model_bare.png", "Figure_5-S7_venn.png"),
        ("figures_venn_run4/venn_rat_bpd7mo_model_bare.pdf", "Figure_5-S7_venn.pdf"),
        ("figures_venn_run4/venn_rat_bpd7mo_model_bare.csv", "Figure_5-S7_venn.csv"),
        ("figures_venn_run4/venn_rat_bpd7mo_model_bare_triple_genes.csv", "Figure_5-S7_venn_triple_genes.csv"),
    ]:
        shutil.copy2(suite / src, OUT_DIR / dst)
        print(f"-> {dst}")


def _energy_parts(D, g, gen, keys, sp):
    """Distances R-R', P-R and P-P' between measured cells R and generated cells P, and the energy distance
    2*mean(P-R) - mean(R-R') - mean(P-P'), pooled over the groups in keys (grid/lib/a8_baselines.py construction)."""
    slope, icpt = D.CALIB[sp]
    vocab = {}
    for k in keys:
        for genes in [c["gt"] for c in g[k]] + gen[k]:
            for gene in genes:
                vocab.setdefault(gene, len(vocab))
    rr, cross, pp = [], [], []
    for k in keys:
        R = D.recon([c["gt"] for c in g[k]], vocab, slope, icpt)
        P = D.recon(gen[k], vocab, slope, icpt)
        dom = D.domain(R, P)
        R, P = R[:, dom], P[:, dom]
        rr.append(D.pdist(R)), cross.append(D.cdist(P, R).ravel()), pp.append(D.pdist(P))
    rr, cross, pp = (np.concatenate(x) for x in (rr, cross, pp))
    return rr, cross, pp, 2 * cross.mean() - rr.mean() - pp.mean()


def build_s1_energy() -> None:
    sys.path.insert(0, str(GRID_LIB))
    import decide as D
    z = np.load(NPZ_MODEL, allow_pickle=True)
    assert int(z["k"]) == K, f"{NPZ_MODEL} is not the K = {K} fit"
    rat_model = rem.fit(BASE / "rat.ho.azi.integrated.h5ad", k=K)
    D.CALIB = {"Rattus": (rat_model.slope, rat_model.intercept), "Homo": (float(z["slope"]), float(z["intercept"]))}
    g = D.energy_groups(TEST_JSONL)
    gen = {k: [c["pred"] for c in v] for k, v in g.items()}
    for path in MODEL_JSONL:
        for (ct, cond), cells in CH.load_azi_counterfactual(path).items():
            gen[("Homo", ct, cond)] = [c["pred"] for c in cells]
    states = [("Rattus", "RA", "Rat, room air"), ("Rattus", "HO", "Rat, hyperoxia"),
              ("Rattus", "AZI", "Rat, hyperoxia + azithromycin"),
              ("Homo", "He22", "Human, fetal lung GW24"), ("Homo", "Term0d", "Human, term-born day 7"),
              ("Homo", "Term20d", "Human, term-born day 20"),
              ("Homo", "Acute", "Human, acute preterm injury wk 26"),
              ("Homo", "BPD", "Human, BPD 7 months"), ("Homo", "BPD-PH", "Human, BPD-PH 7 months")]
    rows, panels = [], []
    for sp, cond, label in states:
        keys = [(sp, ct, cond) for ct in CTS if len(g.get((sp, ct, cond), [])) >= 5]
        for ct_label, ks in [(k[1], [k]) for k in keys] + [("pooled", keys)]:
            rr, cross, pp, e = _energy_parts(D, g, gen, ks, sp)
            n, n_gen = sum(len(g[k]) for k in ks), sum(len(gen[k]) for k in ks)
            rows.append({"species": sp, "condition": cond, "state": label, "cell_type": ct_label, "n_cells": n,
                         "n_generated": n_gen, "energy": e, "mean_measured_measured": rr.mean(),
                         "mean_generated_measured": cross.mean(), "mean_generated_generated": pp.mean()})
            if ct_label == "gCap":
                panels.append((label, n, n_gen, rr, cross, pp, e))
            print(f"  {sp} {label:34s} {ct_label:8s} n={n:4d} generated={n_gen:4d} energy={e:.3f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "Figure_5-S1_energy.csv", index=False)

    PS.apply()
    fig, axes = plt.subplots(3, 3, figsize=(PS.FIG_W, 6.6), layout="constrained", sharey=True)
    for ax, (label, n, n_gen, rr, cross, pp, e) in zip(axes.ravel(), panels):
        deck_style(ax)
        for arr, color, lab in ((rr, PS.GT_COLOR, "measured – measured"),
                                (cross, PS.PRED_COLOR, "generated – measured"),
                                (pp, PS.HO_COLOR, "generated – generated")):
            xs = np.sort(arr)
            ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=color, lw=1.8, label=lab)
        ax.set_title(f"{label}\n{n} measured, {n_gen} generated   ·   energy = {e:.2f}",
                     fontsize=PS.FS_TITLE - 0.5, color=INK, loc="left")
        lo = min(float(a.min()) for a in (rr, cross, pp))
        hi = max(float(a.max()) for a in (rr, cross, pp))
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(labelsize=PS.FS_TICK - 1.5, colors=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("Pairwise distance", fontsize=PS.FS_LABEL - 0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative fraction of pairs", fontsize=PS.FS_LABEL - 0.5)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=3, frameon=False, fontsize=PS.FS_LEGEND)
    fig.suptitle("General capillary endothelial cells: distances behind the energy distance", fontsize=PS.FS_SUPTITLE)
    PS.save(fig, OUT_DIR / "Figure_5-S1_energy.png")
    PS.save(fig, OUT_DIR / "Figure_5-S1_energy.pdf")
    plt.close(fig)
    print("-> Figure_5-S1_energy")


def main() -> None:
    assert OUT_DIR.is_dir() and not any(OUT_DIR.iterdir()), f"{OUT_DIR} must exist and be empty"
    assert not DRAW.exists(), f"{DRAW} exists"
    copy_violins()
    draw_manuscript_figures()
    build_s1_energy()


if __name__ == "__main__":
    main()
