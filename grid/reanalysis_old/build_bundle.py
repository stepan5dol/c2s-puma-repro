#!/usr/bin/env python3
"""Bundle for re-running the violins of the 8 old grid runs (2026-11-02..04, grid_old_20261105) with each run's own
human rescued table, by the manuscript's code. Run on the Mac from the repository folder:

    python3 grid/reanalysis_old/build_bundle.py

It copies, byte for byte, into grid/reanalysis_old/ (nothing is edited):
  tree/        the manuscript analysis code and its small inputs from the Mac (AzithroGemma folder):
               - the 44 files of grid/tree at tag deployed-20261104-1346, the code the old grid ran (K = 800); every file
                 is checked against that tag's MANIFEST, every .py also against the tag's git blob;
               - the two manuscript scripts that build the human rescued table and are not in grid/tree:
                 human_rescue_pipeline_genomewide.py (-> reports/human_rescue_genomewide.csv) and
                 filter_to_rat_schema.py (-> reports/rat_schema/human_*_rescue_predicted_azi.xlsx)
  reference/   the manuscript's outputs, for the check run of the manuscript model: human_rescue_genomewide.csv and the
               violin reports of S2-S6 (report_*_all_rescued.csv)
  gold/        the manuscript model's answers: test set (test_t10.jsonl) and AZI prompts (inference_azi_results.jsonl)
  infer_only_20261104.slurm   hpc/infer_only.slurm at tag deployed-20261104-1346: the script that made the old grid's
               test answers (inputs from Hugging Face, only the vLLM dtype changed in the image's run_azi.py)
and writes MANIFEST.json (sha256 of every copied file). tree/, reference/ and gold/ are rebuilt on every call.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A = Path.home() / "Documents/AzithroGemma"
TAG = "deployed-20261104-1346"
H = "retrain_v2/analysis/real_counts_suite_ortho/human_panel_variants/"
RP = "retrain_v2/analysis/real_counts_suite_ortho/rat_panel_variants/"
TABLE_SCRIPTS = [H + "human_rescue_pipeline_genomewide.py", H + "filter_to_rat_schema.py"]
REFERENCE = [
    H + "reports/human_rescue_genomewide.csv",
    H + "reports/report_human_delta_fetal_gw22_to_ali_gw26_all_rescued.csv",       # S2
    H + "reports/report_human_delta_term_born_to_bpd_7mo_all_rescued.csv",         # S3
    H + "reports/report_human_delta_term_born_to_bpd_ph_7mo_all_rescued.csv",      # S4
    RP + "reports/report_rat_delta_room_air_to_hyperoxia_all_rescued.csv",         # S5
    RP + "reports/report_rat_delta_hyperoxia_to_azithromycin_all_rescued.csv",     # S6
]
GOLD = {"test_t10.jsonl": "retrain_v2/results/test_inference_results_t10.jsonl",
        "inference_azi_results.jsonl": "retrain_v2/results/inference_azi_results.jsonl"}


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(rel_dst, data, manifest):
    p = HERE / rel_dst
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    manifest[rel_dst] = sha(data)


for d in ("tree", "reference", "gold"):
    if (HERE / d).exists():
        shutil.rmtree(HERE / d)  # only this bundle's own copies
manifest = {}
tag_manifest = json.loads(git("show", f"{TAG}:grid/tree/MANIFEST.json"))
for rel, h in sorted(tag_manifest.items()):
    b = (A / rel).read_bytes()
    if sha(b) != h:
        raise SystemExit(f"{rel}: the Mac file differs from grid/tree at {TAG}")
    if rel.endswith(".py") and git("show", f"{TAG}:grid/tree/{rel}") != b:
        raise SystemExit(f"{rel}: the Mac file differs from the code in git at {TAG}")
    put("tree/" + rel, b, manifest)
for rel in TABLE_SCRIPTS:
    put("tree/" + rel, (A / rel).read_bytes(), manifest)
for rel in REFERENCE:
    put("reference/" + Path(rel).name, (A / rel).read_bytes(), manifest)
for name, rel in GOLD.items():
    put("gold/" + name, (A / rel).read_bytes(), manifest)
put("infer_only_20261104.slurm", git("show", f"{TAG}:hpc/infer_only.slurm"), manifest)
(HERE / "tree/.gitignore").write_bytes((REPO / "grid/tree/.gitignore").read_bytes())
(HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(f"{len(tag_manifest)} files of grid/tree at {TAG} (all equal to the Mac), {len(TABLE_SCRIPTS)} table scripts, "
      f"{len(REFERENCE)} reference files, {len(GOLD)} gold answer files, infer_only_20261104.slurm -> {HERE}")
