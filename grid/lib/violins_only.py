"""[violins only, runs from a JSON file: usage violins_only.py runs.json]
Standard analysis of the A100 fp32 run in one process, the counts CSV read once
(copy of results/a100ep2_2026-10-22_1215/scripts/run_once.py; ms264 and a100ep2 violins already exist there).

1. scan_only.py: the one read of the counts CSV (no rescue table, no panel C).
2. The manuscript's test-set violins, scripts unchanged (rat build_all_panel_variants.py, human
   human_updown_percell.py pairs and build_disease_arms_trajectory.py), for the manuscript run (ms264)
   (here only the fp32 run). Their counts reads are served from step 1's scan (same rows in file order,
   integer counts held as float32); the atlas read is kept for the second run. Per-cell values are
   recorded from draw_panel for violins_book_tests.py. Figures and reports go to OUT/<run>/violins/,
   the manuscript's own files are restored afterwards.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

SCR = Path(os.environ["C2S_WORK"])  # writable folder for backups and per-run outputs
import os
ROOT = Path(os.environ["C2S_ROOT"])  # tree root: holds retrain_v2/, pipeline/, train-after-grpo-analysis/
R = ROOT / "retrain_v2"
ORTHO = R / "analysis/real_counts_suite_ortho"
OUT = Path(os.environ["C2S_WORK"]) / "runs"
_RAW_RUNS = json.load(open(sys.argv[1]))
DEFAULT_REG = Path(os.environ.get("EXAMPLE_REG_PATH", ROOT / "retrain_prompt/pipeline/sft_example_registry.csv"))
RUNS = {}
for _name, _val in _RAW_RUNS.items():
    if isinstance(_val, dict):
        _test = Path(_val["test"])
        _azi = Path(_val["azi"]) if _val.get("azi") else (_test.parent / "inference_azi_results.jsonl")
        _k = int(_val.get("k", 2000))
        _reg = Path(_val["registry"]) if _val.get("registry") else (Path("/nonexistent") if "registry" in _val else DEFAULT_REG)
        _model = [Path(m) for m in _val["model"]] if _val.get("model") else [_test]
    else:
        _test = Path(_val)
        _azi = _test.parent / "inference_azi_results.jsonl"
        _k = int(os.environ.get("C2S_K", "2000"))
        _reg = DEFAULT_REG
        _model = [_test]
    RUNS[_name] = {"test": _test, "azi": _azi, "k": _k, "registry": _reg, "model": _model}
_bad_k = {n: v["k"] for n, v in RUNS.items() if v["k"] != 2000}
assert not _bad_k, f"the human calibration in scan_only.py is the K = 2000 fit; runs with another K: {_bad_k}"

DIRS = [ORTHO / "rat_panel_variants", ORTHO / "human_panel_variants"]
PATTERNS = {
    "figures": ("rat_delta_*", "human_delta_*"),
    "reports": (
        "report_rat_delta_*",
        "report_human_delta_*",
        "ablation_rescue_genomewide_*.csv",
        "rat_schema/human_*_rescue_predicted_azi.xlsx",
    ),
}
_read_csv = pd.read_csv

# ---- 1. AZI arm: the sweep script unchanged (its scan is the one read of the counts CSV) ----
g = {"__name__": "sweep", "__file__": str(str(Path(__file__).with_name("scan_only.py")))}
exec(compile(open(g["__file__"]).read(), g["__file__"], "exec"), g)
L = g["L"]
ARP = g["ARP"]
CH = g["CH"]
CR = g["CR"]
s = g["_scan_once"]()
CSV = Path(L.C.BPD_COUNTS_PATH).resolve()
HEADER = pd.Index(s["raw_columns"])
BC = list(s["bc"])

in_csv = set(pd.read_csv(L.C.BPD_META_PATH, usecols=["id"])["id"])
need = set()
for _spec in RUNS.values():
    _reg_p = _spec["registry"]
    if _reg_p.exists():
        _reg = pd.read_csv(_reg_p)
        _reg_bpd = _reg[(_reg["split"] == "test") & (_reg["direction"] == "fwd") & (_reg["source"] == "bpd")]
        need |= set(_reg_bpd["barcode"]) & in_csv
    else:
        _pairs = pd.read_csv(ORTHO / "human_panel_variants/recovered_human_test_pairs.csv")
        need |= (set(_pairs.src_barcode) | set(_pairs.tgt_barcode)) & in_csv
assert need <= set(BC), f"{len(need - set(BC))} test barcodes outside the scan"


def read_csv(path, *a, **k):
    if isinstance(path, (str, os.PathLike)) and Path(path).resolve() == CSV:
        assert not a and set(k) == {"index_col", "chunksize"} and k["index_col"] == "id", k
        cs = k["chunksize"]
        return (pd.DataFrame(s["X"][i:i + cs], index=pd.Index(BC[i:i + cs], name="id"), columns=HEADER)
                for i in range(0, len(BC), cs))
    return _read_csv(path, *a, **k)


pd.read_csv = read_csv

# ---- 2. violins ----
sys.path.insert(0, str(ORTHO / "rat_panel_variants"))
sys.path.insert(0, str(ORTHO / "human_panel_variants"))
import updown_core as UC  # noqa: E402
import build_all_panel_variants as B  # noqa: E402
import human_updown_percell as DIS  # noqa: E402
import build_disease_arms_trajectory as T  # noqa: E402
import ablation_export_for_limma as EXP  # noqa: E402
import ablation_limma_sets as LIM  # noqa: E402

HPV = ORTHO / "human_panel_variants"
CAMERA_CONDITIONS = ("BPD7mo", "BPDPH7mo", "Acute26")

CUR, rec = {}, {}
DIS_DEFAULTS = {k: v for k, v in vars(DIS).items() if k.isupper()}


def dump(key, scores):
    rec[key] = {f"{a}|{b}": [float(x) for x in v] for (a, b), v in scores.items()}


_uc_run, _uc_draw = UC.run, UC.draw_panel
_dis_run, _dis_draw = DIS.run, DIS.draw_panel


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
UC.EXAMPLE_REG_PATH = DEFAULT_REG
T.EXAMPLE_REG_PATH = DEFAULT_REG
_atlas, _orig_atlas = {}, T.load_atlas_by_barcode


def load_atlas(barcodes, model):
    key = frozenset(barcodes)
    if key not in _atlas:
        _atlas[key] = _orig_atlas(barcodes, model)
    return dict(_atlas[key])


T.load_atlas_by_barcode = load_atlas


def ours(d):
    return [p for sub, pats in PATTERNS.items() for pat in pats for p in (d / sub).glob(pat)]


BACKUP = SCR / "backup_manuscript_violins"
for d in DIRS:
    for p in ours(d):
        q = BACKUP / p.relative_to(ORTHO)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
had = {p for d in DIRS for p in ours(d)}
try:
    for run, spec in RUNS.items():
        path, azi_path, k, reg_path = spec["test"], spec["azi"], spec["k"], spec["registry"]
        print(f"\n######## violins: {run} ({path.name}, K={k}, reg={'yes' if reg_path.exists() else 'recovered_csv'})", flush=True)
        t0 = time.time() - 1
        rec.clear()
        vars(DIS).update(DIS_DEFAULTS)
        CR.K = CH.K = UC.C.K = k
        UC.EXAMPLE_REG_PATH = reg_path
        T.EXAMPLE_REG_PATH = reg_path
        rat_schema_dir = ORTHO / "human_panel_variants/reports/rat_schema"
        for p in rat_schema_dir.glob("human_*_rescue_predicted_azi.xlsx"):
            p.unlink()
        assert azi_path.exists() and azi_path.stat().st_size > 0, (
            f"{run}: missing per-run AZI file {azi_path}; refusing to fall back to shared table"
        )
        print(f"[rescue] computing per-run human rescued panels from {azi_path} (K={k})", flush=True)
        ARP.VARIANTS = [run]
        ARP.AZI_JSONL_OF = {run: azi_path}
        ARP.main()
        n_xlsx = len(list(rat_schema_dir.glob("human_*_rescue_predicted_azi.xlsx")))
        assert n_xlsx == 12, f"{run}: expected 12 per-run rescued xlsx, got {n_xlsx}"
        T.TEST_INFERENCE = path
        T.MODEL_INFERENCE = spec["model"]
        T.main()
        UC.TEST_INFERENCE_PATH = path
        UC.MODEL_INFERENCE = spec["model"]
        sys.argv = ["x"]
        B.main()
        dest = OUT / run / "violins"
        for d in DIRS:
            for p in ours(d):
                if p.stat().st_mtime >= t0:
                    q = dest / ("rat" if d.name.startswith("rat") else "human") / p.relative_to(d)
                    q.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, q)
        shutil.copy2(HPV / "reports" / f"ablation_rescue_genomewide_{run}.csv",
                     HPV / "reports" / "human_rescue_genomewide.csv")
        (R / "results").mkdir(exist_ok=True)
        shutil.copy2(azi_path, R / "results" / f"inference_azi_results_{run}.jsonl")
        cam = dest / "human" / "camerapr"
        cam.mkdir(parents=True, exist_ok=True)
        for cond in CAMERA_CONDITIONS:
            sys.argv = ["x", "--condition", cond, "--variant", run]
            EXP.main()
            sys.argv = ["x", "--condition", cond, "--variant", run]
            LIM.main()
            shutil.copy2(HPV / "autophagy_program" / "reports" / f"ablation_{cond.lower()}_limma_sets_{run}.csv",
                         cam / f"{cond.lower()}_limma_sets.csv")
        shutil.copy2(HPV / "reports" / "human_rescue_genomewide.csv", cam / "human_rescue_genomewide.csv")
        (dest / "scores.json").write_text(json.dumps(rec))
        print(f"-> {dest} ({len(rec)} panels)", flush=True)
finally:
    for d in DIRS:
        for p in ours(d):
            b = BACKUP / p.relative_to(ORTHO)
            if p in had:
                shutil.copy2(b, p)
            else:
                p.unlink()
    print("manuscript violin files restored", flush=True)

# reproduction check: the manuscript run through this path vs the manuscript's own reports
bad = []
for q in [] if "ms264" not in RUNS else sorted((OUT / "ms264/violins").glob("*/reports/*.csv")):
    side = "rat_panel_variants" if q.parts[-3] == "rat" else "human_panel_variants"
    ref = BACKUP / side / "reports" / q.name
    if not ref.exists():
        bad.append((q.name, "no manuscript report"))
        continue
    a, b = pd.read_csv(q), pd.read_csv(ref)
    num = a.select_dtypes("number").columns
    if not (a.shape == b.shape and (a[num] - b[num]).abs().max().max() == 0
            and a.drop(columns=num).equals(b.drop(columns=num))):
        bad.append((q.name, "differs"))
print("reproduction of the manuscript violin reports:", "skipped (ms264 not rerun)" if "ms264" not in RUNS
      else "all identical" if not bad else bad, flush=True)
print("ALL DONE", flush=True)
