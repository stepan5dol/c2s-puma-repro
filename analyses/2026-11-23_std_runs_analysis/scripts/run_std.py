#!/usr/bin/env python3
"""
1) status  - check Puma squeue, sacct, and which c2s_std runs have finished inference
2) fetch   - download completed c2s_std runs one-by-one via rsync from uahpcfxfr
3) analyze - run the exact c2s-puma-std pipeline (prepare_root -> prepare_run -> violins_only)
             on all locally downloaded completed c2s_std runs and append to unified_all18_summary.csv

Usage:
  python3 run_std.py status
  python3 run_std.py fetch
  python3 run_std.py analyze
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HOME = Path.home()
AZ = HOME / "Documents/AzithroGemma"
REPO = AZ / "retrain_v2/c2s-puma-std"
LIB = REPO / "grid/lib"
BUILDS = AZ / "retrain_v2/retrain_prompt_std/builds"
LOCAL_RES = AZ / "retrain_prompt/results/puma_answers/xdisk/vkalin/dolzhenko/c2s_std/results"
UNIFIED_CSV = AZ / "retrain_v2/results/2026-11-20_c2s-puma-std_unified_all11/unified_all18_summary.csv"
REMOTE_RES = "/xdisk/vkalin/dolzhenko/c2s_std/results"


def cmd_status() -> None:
    script = """
D=/xdisk/vkalin/dolzhenko/c2s_std
S=$(date -d '5 days ago' +%F)
echo "=== SQUEUE ==="
squeue -u $USER -o "%.10i %.14P %.22j %.8T %.10M %.10l %.16R"
echo "=== SACCT ==="
sacct -u $USER -S $S -X -o "JobID%14,JobName%22,State%12,ExitCode%7,Elapsed%10"
echo "=== READY INFERENCE (c2s_std) ==="
for d in $D/results/2026-11-*; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  st=$(sed -n 's/^status: //p' "$d/record.md" 2>/dev/null || echo running)
  inf=$(sed -n 's/^inference: //p' "$d/record.md" 2>/dev/null | cut -d' ' -f1)
  t10=$( [ -s "$d/inference_bfloat16/test_t10.jsonl" ] && wc -l < "$d/inference_bfloat16/test_t10.jsonl" || echo 0 )
  ood=$( [ -s "$d/inference_bfloat16/inference_results.jsonl" ] && wc -l < "$d/inference_bfloat16/inference_results.jsonl" || echo 0 )
  printf "%-44s status=%-10s infer=%-10s test=%s ood=%s\n" "$n" "$st" "${inf:-none}" "$t10" "$ood"
done
"""
    subprocess.run(["ssh", "uahpclogin", "bash -s"], input=script, text=True, check=True)


def cmd_fetch() -> None:
    # List completed runs (both test_t10.jsonl and inference_results.jsonl non-empty)
    probe = f"""
for d in {REMOTE_RES}/2026-11-*; do
  [ -s "$d/inference_bfloat16/test_t10.jsonl" ] && [ -s "$d/inference_bfloat16/inference_results.jsonl" ] && basename "$d" || true
done
"""
    out = subprocess.check_output(["ssh", "uahpclogin", "bash -s"], input=probe, text=True)
    runs = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not runs:
        print("No completed c2s_std inference folders found on Puma.")
        return
    print("Completed c2s_std runs on Puma:", runs)
    for name in runs:
        dst = LOCAL_RES / name
        (dst / "inference_bfloat16").mkdir(parents=True, exist_ok=True)
        # Separate single-source rsync calls avoid macOS openrsync multi-source stalls
        for rel in ("inference_bfloat16/", "record.md", "train.log"):
            src = f"uahpcfxfr:{REMOTE_RES}/{name}/{rel}"
            tgt = str(dst / rel) if rel.endswith("/") else str(dst) + "/"
            print(f"-> syncing {name}/{rel}", flush=True)
            subprocess.run(["rsync", "-avz", "--partial", src, tgt], check=False)


def detect_build(run_dir: Path) -> str:
    rec = run_dir / "record.md"
    if rec.exists():
        m = re.search(r"^data_src:\s*\S+/data_src/([^\s/]+)", rec.read_text(), re.M)
        if m:
            return m.group(1)
    n = run_dir.name
    if "humans_only" in n:
        return "humans_only_v1"
    if "rats_upper" in n:
        return "old3_upper_v1"
    if "rodent_drugs" in n:
        return "rodent_drugs_v1"
    raise ValueError(f"Cannot infer build for {run_dir.name}")


def summarize_run(name: str, violins_dir: Path) -> dict:
    sys.path.insert(0, str(LIB))
    import decide as D

    scores = json.loads((violins_dir / "scores.json").read_text())
    rows = D._eval_scores(scores)
    match = D.violin_strata_match(scores)
    exact_40 = sum(match)
    mism = D.violin_mismatch(scores)

    # Real significant strata & per-arm exact matches (out of 8 each)
    arm_map = {
        "ALI/8": "human He22->Acute26",
        "BPD/8": "human Term->BPD",
        "BPDPH/8": "human Term->BPD-PH",
        "RA_HO/8": "rat RA->HO",
        "HO_AZI/8": "rat HO->AZI",
    }
    rsig_num, rsig_den = 0, 0
    per_arm = {}
    for col, aname in arm_map.items():
        ok = 0
        present_arm = False
        for ct in D.CTS:
            for g in ("down", "up"):
                r = rows.get((aname, ct, g, "real"))
                m = rows.get((aname, ct, g, "model"))
                if r is not None:
                    present_arm = True
                    if r["violin"]:
                        rsig_den += 1
                        if m is not None and m["violin"]:
                            rsig_num += 1
                    if m is not None and (m["violin"] == r["violin"]):
                        ok += 1
        per_arm[col] = ok if present_arm else "NA"

    # Count rescued genes from the 12 human xlsx files
    rat_schema = violins_dir / "human/reports/rat_schema"
    xlsxs = sorted(rat_schema.glob("human_*_rescue_predicted_azi.xlsx")) if rat_schema.exists() else []
    resc = {"Acute26": 0, "BPD7mo": 0, "BPDPH7mo": 0}
    for p in xlsxs:
        for cond in resc:
            if f"human_{cond}_" in p.name:
                df = pd.read_excel(p)
                resc[cond] += len(df)

    has_rat = per_arm["RA_HO/8"] != "NA"
    return {
        "run": name,
        "K": 2000,
        "xlsx": len(xlsxs),
        "exact_40": exact_40 if has_rat else f"{per_arm['ALI/8']+per_arm['BPD/8']+per_arm['BPDPH/8']}/24",
        "mism": mism if has_rat else "NA(no_rat)",
        "rsig": f"{rsig_num}/{rsig_den}",
        **per_arm,
        "resc_ALI": resc["Acute26"],
        "resc_BPD": resc["BPD7mo"],
        "resc_BPDPH": resc["BPDPH7mo"],
    }


def cmd_analyze(work_out: Path) -> None:
    WORK_OUT = work_out
    ready = []
    if LOCAL_RES.exists():
        for d in sorted(LOCAL_RES.iterdir()):
            inf = d / "inference_bfloat16"
            if (inf / "test_t10.jsonl").is_file() and (inf / "inference_results.jsonl").is_file():
                if (inf / "test_t10.jsonl").stat().st_size > 0 and (inf / "inference_results.jsonl").stat().st_size > 0:
                    ready.append(d)
    if not ready:
        print(f"No downloaded completed runs in {LOCAL_RES}. Run 'fetch' first.")
        return

    print("Preparing & analyzing runs:", [d.name for d in ready], flush=True)
    WORK_OUT.mkdir(parents=True, exist_ok=True)
    (WORK_OUT / "reference").mkdir(exist_ok=True)
    (WORK_OUT / "runs").mkdir(exist_ok=True)
    (WORK_OUT / "inputs").mkdir(exist_ok=True)

    runs_spec = {}
    rat_h5ad = AZ / "rat.ho.azi.integrated.h5ad"
    for d in ready:
        build = detect_build(d)
        reg = BUILDS / build / "sft_example_registry.csv"
        assert reg.is_file(), f"Missing registry {reg}"
        inp_dir = WORK_OUT / "inputs" / d.name
        subprocess.run(
            [sys.executable, str(LIB / "prepare_run.py"), str(d / "inference_bfloat16"), str(rat_h5ad), str(inp_dir), "--azi-arm", "azithromycin"],
            check=True,
        )
        runs_spec[f"std_{d.name}"] = {
            "test": str(inp_dir / "test_t10.jsonl"),
            "azi": str(inp_dir / "inference_azi_results.jsonl"),
            "k": 2000,
            "registry": str(reg),
        }

    runs_json = WORK_OUT / "runs.json"
    runs_json.write_text(json.dumps(runs_spec, indent=2))

    with tempfile.TemporaryDirectory(prefix="c2s_std_mac_") as td:
        td_p = Path(td)
        data_dir = td_p / "data"
        (data_dir / "BPD-PH").mkdir(parents=True)
        for f in ("GSE275938_compiled_counts.h5ad", "GSE275938_compiled_counts.csv", "GSE275938_cell_metadata.csv"):
            if (AZ / "BPD-PH" / f).exists():
                (data_dir / "BPD-PH" / f).symlink_to(AZ / "BPD-PH" / f)
        for f in ("rat.ho.azi.integrated.h5ad", "he_lung_atlas.h5ad", "2022FetalLungIntCounts.h5ad"):
            (data_dir / f).symlink_to(AZ / f)
        (data_dir / "cell_types_config.json").symlink_to(AZ / "retrain_v2/scripts/cell_types_config.json")

        root_dir = td_p / "root"
        subprocess.run([sys.executable, str(LIB / "prepare_root.py"), str(REPO / "grid/tree"), str(data_dir), str(root_dir)], check=True)

        # Copy violins_only.py to tmp with safe no-rat skip (leaves repo untouched)
        v_tmp = td_p / "violins_only_safe.py"
        v_txt = (LIB / "violins_only.py").read_text()
        old_rat = '        UC.TEST_INFERENCE_PATH = path\n        sys.argv = ["x"]\n        B.main()'
        new_rat = (
            '        _has_rat = True\n'
            '        if reg_path.exists():\n'
            '            _r = pd.read_csv(reg_path, usecols=["split", "direction", "source"])\n'
            '            _has_rat = bool(((_r["split"] == "test") & (_r["direction"] == "fwd") & (_r["source"] == "rat")).any())\n'
            '        if _has_rat:\n'
            '            UC.TEST_INFERENCE_PATH = path\n'
            '            sys.argv = ["x"]\n'
            '            B.main()\n'
            '        else:\n'
            '            print(f"[rat violins] {run}: no rat rows, skipping", flush=True)'
        )
        assert old_rat in v_txt
        v_tmp.write_text(v_txt.replace(old_rat, new_rat, 1))
        # Also copy sibling scan_only.py next to it so Path(__file__).with_name("scan_only.py") resolves
        shutil.copy2(LIB / "scan_only.py", td_p / "scan_only.py")

        env = os.environ.copy()
        env.update({
            "C2S_ROOT": str(root_dir),
            "C2S_WORK": str(WORK_OUT),
            "C2S_RESULTS": str(WORK_OUT / "reference"),
            "PYTHONPATH": str(LIB),
        })
        subprocess.run([sys.executable, "-u", str(v_tmp), str(runs_json)], env=env, check=True)

    new_rows = [summarize_run(name, WORK_OUT / "runs" / name / "violins") for name in runs_spec]
    df_new = pd.DataFrame(new_rows)
    out_new = WORK_OUT / "std_new_summary.csv"
    df_new.to_csv(out_new, index=False)

    if UNIFIED_CSV.exists():
        df_base = pd.read_csv(UNIFIED_CSV)
        df_all = pd.concat([df_base, df_new], ignore_index=True)
        out_all = WORK_OUT / "unified_all_with_std.csv"
        df_all.to_csv(out_all, index=False)
        print("\n=== COMBINED COMPARISON TABLE ===")
        print(df_all.to_string(index=False))
        print(f"\nSaved: {out_all}")
    else:
        print(df_new.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, required=True, help="Output root directory (outside git)")
    ap.add_argument("cmd", choices=("status", "fetch", "analyze"))
    args = ap.parse_args()
    if args.cmd == "analyze":
        cmd_analyze(args.out_root.resolve())
    elif args.cmd == "status":
        cmd_status()
    else:
        cmd_fetch()
