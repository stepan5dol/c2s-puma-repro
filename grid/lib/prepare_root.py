"""Build the working root for one analysis job.

The analysis code (grid/tree) was written on the author's Mac and has the path prefix
/Users/stepandolzhenko/Documents/AzithroGemma in its constants. This copies the tree to DST, replaces that one prefix
by DST in the .py files (a plain text replacement, nothing else is touched), links the big data files into the places
the code expects, and checks that no Mac prefix is left.

usage: prepare_root.py TREE DATA DST
  TREE  grid/tree of the repo
  DATA  folder with BPD-PH/GSE275938_*.csv, rat.ho.azi.integrated.h5ad, he_lung_atlas.h5ad, 2022FetalLungIntCounts.h5ad,
        cell_types_config.json   (/data inside c2s-analysis.sif)
"""
import shutil
import sys
from pathlib import Path

MAC = "/Users/stepandolzhenko/Documents/AzithroGemma"
tree, data, dst = (Path(a) for a in sys.argv[1:4])
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(tree, dst)
n = 0
for p in dst.rglob("*.py"):
    s = p.read_text()
    if MAC in s:
        p.write_text(s.replace(MAC, str(dst)))
        n += 1
left = [str(p) for p in dst.rglob("*.py") if "/Users/" in p.read_text()]
assert not left, f"Mac paths left in {left}"
# both pipeline_short/common.py files point at their own folder: the root one at <root>/..., the retrain_v2 one at <root>/retrain_v2/...
LINKS = {}
for base in ("", "retrain_v2/"):
    LINKS.update({
        base + "BPD-PH": data / "BPD-PH",
        base + "rat.ho.azi.integrated.h5ad": data / "rat.ho.azi.integrated.h5ad",
        base + "he_lung_atlas.h5ad": data / "he_lung_atlas.h5ad",
        base + "2022FetalLungIntCounts.h5ad": data / "2022FetalLungIntCounts.h5ad",
        base + "scripts/cell_types_config.json": data / "cell_types_config.json",
    })
for rel, target in LINKS.items():
    link = dst / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
# output folders the violin scripts write into
H = dst / "retrain_v2/analysis/real_counts_suite_ortho"
for d in ("human_panel_variants/figures", "human_panel_variants/reports", "rat_panel_variants/figures", "rat_panel_variants/reports"):
    (H / d).mkdir(parents=True, exist_ok=True)
print(f"root {dst}: prefix rewritten in {n} files, {len(LINKS)} data links")
