"""The manuscript violins S2-S6 for one run, with the per-cell values recorded.

usage: record_violins.py ROOT OUT
ROOT: a working root made by grid/lib/prepare_root.py from grid/reanalysis_old/tree, holding the run's test answers at
retrain_v2/results/test_inference_results_t10.jsonl and its own rescued table in
human_panel_variants/reports/rat_schema/ (analyze_one.slurm).

Runs build_disease_arms_trajectory.main() (human: He22->Acute26, Term->BPD7mo, Term->BPDPH7mo) and
build_all_panel_variants.main() (rat: RA->HO, HO->AZI) unchanged, as their __main__ does. draw_panel is wrapped only
to record the values it draws (all cell types together, gate full_list), in the format of grid/lib/violins_only.py,
which grid/lib/decide.py reads. Writes OUT/scores.json and copies the figures and reports the scripts wrote
(reports include the run's rescued table: human_rescue_genomewide.csv and rat_schema/).
"""
import json
import shutil
import sys
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
ORTHO = root / "retrain_v2/analysis/real_counts_suite_ortho"
sys.path.insert(0, str(ORTHO / "rat_panel_variants"))
sys.path.insert(0, str(ORTHO / "human_panel_variants"))
import updown_core as UC  # noqa: E402
import build_all_panel_variants as B  # noqa: E402
import human_updown_percell as DIS  # noqa: E402
import build_disease_arms_trajectory as T  # noqa: E402

CUR, rec = {}, {}


def dump(key, scores):
    rec[key] = {f"{a}|{b}": [float(x) for x in v] for (a, b), v in scores.items()}


_uc_run, _uc_draw, _dis_run, _dis_draw = UC.run, UC.draw_panel, DIS.run, DIS.draw_panel


def uc_run(arm_key, gate):
    CUR.update(arm=arm_key, gate=gate)
    return _uc_run(arm_key, gate)


def uc_draw(ax, scores, title, *a, **k):
    if UC.ONLY_CT is None and CUR["gate"] == "full_list":
        dump(f"rat|{CUR['arm']}|{title}", scores)
    return _uc_draw(ax, scores, title, *a, **k)


def dis_run(gate, *a, **k):
    CUR["gate"] = gate
    return _dis_run(gate, *a, **k)


def dis_draw(ax, scores, title, *a, **k):
    if DIS.ONLY_CT is None and CUR["gate"] == "full_list":
        dump(f"human|{DIS.SRC_LABEL}->{DIS.CONDITION}|{DIS.PAIRING}|{title}", scores)
    return _dis_draw(ax, scores, title, *a, **k)


UC.run, UC.draw_panel, DIS.run, DIS.draw_panel = uc_run, uc_draw, dis_run, dis_draw

print("######## human arms: build_disease_arms_trajectory.main()", flush=True)
T.main()
print("######## rat arms: build_all_panel_variants.main()", flush=True)
sys.argv = ["build_all_panel_variants.py"]
B.main()

arms = sorted({"|".join(k.split("|")[:2]) for k in rec})
want = ["human|He22->Acute26", "human|Term->BPD7mo", "human|Term->BPDPH7mo", "rat|azi", "rat|ho"]
if arms != want:
    raise SystemExit(f"recorded arms {arms}, expected {want}")
out.mkdir(parents=True)
for side in ("human", "rat"):
    for sub in ("figures", "reports"):
        shutil.copytree(ORTHO / f"{side}_panel_variants" / sub, out / side / sub, symlinks=False)
(out / "scores.json").write_text(json.dumps(rec))
print(f"-> {out} ({len(rec)} panels, arms {arms})", flush=True)
