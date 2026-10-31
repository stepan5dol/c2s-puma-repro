"""Assemble grid/tree: the code and small inputs the manuscript violins path reads, laid out as on the
author's Mac (root = the AzithroGemma folder), so that the job can rewrite the one path prefix and run the
code unchanged. The list comes from a traced run of the violins on 2026-10-30 (every python module loaded from
the checkout, every file opened); the two atlas .h5ad files are read through h5py and are added from the source
(pipeline_short/common.py ATLAS_CLEAN_PATH / ATLAS_RAW_PATH).
Run on the Mac:  python3 grid/tools/make_tree.py /Users/<you>/Documents/AzithroGemma
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
DST = Path(__file__).resolve().parents[1] / "tree"
R = "retrain_v2/"
A = R + "analysis/"
H = A + "real_counts_suite_ortho/human_panel_variants/"
CODE = [
    # two different pipeline_short packages exist on the Mac: the one in the root is imported first (load_data.py puts the
    # root on sys.path) and wins in sys.modules; the one in retrain_v2 supplies stage1_sft_bidir.py. Both are kept as they are.
    "pipeline/common.py", "pipeline_short/__init__.py", "pipeline_short/common.py",
    R + "pipeline_short/__init__.py", R + "pipeline_short/common.py", R + "pipeline_short/stage1_sft_bidir.py",
    A + "common.py", A + "common_human.py", A + "plot_style.py", A + "rank_expr_model.py",
    A + "pathway_module_analysis/common.py", A + "pathway_module_analysis/load_data.py", A + "pathway_module_analysis/real_counts.py",
    A + "real_counts_suite_ortho/common_real.py",
    H + "ablation_export_for_limma.py", H + "ablation_limma_sets.py", H + "ablation_rescue_pipeline.py",
    H + "build_disease_arms_trajectory.py", H + "human_updown_percell.py", H + "panel_sets_human.py",
    A + "real_counts_suite_ortho/rat_panel_variants/build_all_panel_variants.py",
    A + "real_counts_suite_ortho/rat_panel_variants/panel_sets.py", A + "real_counts_suite_ortho/rat_panel_variants/updown_core.py",
]
SMALL = [
    H + "recovered_human_test_pairs.csv", A + "real_counts_suite_ortho/rat/recovered_rat_test_pairs.csv",
    A + "reports/rank_expr_model.npz",
    "Venous_rescue gene.xlsx", "acap resuce gene.xlsx", "gcap rescue gene.xlsx", "pericyte_rescue gene.xlsx",
    "train-after-grpo-analysis/rescued_human_genes.txt", "train-after-grpo-analysis/rescued_rat_genes.txt",
] + sorted(str(p.relative_to(SRC)) for p in (SRC / (H + "reports/rat_schema")).glob("human_*_rescue_predicted_azi.xlsx"))
manifest = {}
for rel in CODE + SMALL:
    s, d = SRC / rel, DST / rel
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s, d)
    manifest[rel] = hashlib.sha256(d.read_bytes()).hexdigest()
(DST / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
print(len(CODE), "modules,", len(SMALL), "small inputs ->", DST)
