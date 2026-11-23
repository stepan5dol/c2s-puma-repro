#!/usr/bin/env python3
"""Inputs for hpc/infer_rep300.slurm: 300 generations per prompt for the human disease states and the rat states.

Human prompts (24 = 3 states x 4 cell types x 2 arms), taken verbatim from the files the grid inferred:
  NO-AZI: the forward prompt of the test set (dataset/test_segments.jsonl, segments[0].text), rows of the registry
          with split=test, direction=fwd, source=bpd; Treatment: mechanical ventilation
  AZI:    inference/inference_azi.jsonl; Treatment: mechanical ventilation; azithromycin ...
Rat prompts (12 = room air, hyperoxia, hyperoxia + azithromycin x 4 cell types): the forward prompt of the test set,
  rows of the registry with split=test, direction=fwd, source=rat.
Both source files are checked against retrain_prompt/pipeline/inputs_sha256.txt (the data_src list of the grid runs).
Each (state, cell type, arm) must have exactly one prompt.

Output: one file per human (state, arm) and per rat state = one array task, 4 cell types x N lines each, and
sha256.txt for the job's check.
usage: python3 build_rep300.py --n 300
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

P = Path.home() / "Documents/AzithroGemma/retrain_prompt/pipeline"
OUT = Path(__file__).resolve().parent
STATES = ["Acute26", "BPD7mo", "BPDPH7mo"]
CTS = ["gCap", "aCap", "Pericyte", "VEC"]
CT_REG = {"gCap": "gCap", "aCap": "aCap", "Pericyte": "Pericyte", "Pulmonary venous EC": "VEC", "VEC": "VEC"}
ARMS = ["noazi", "azi"]  # array task id = 2 * state index + arm index (hpc/infer_rep300.slurm uses the same order)
RAT_STATES = ["RA", "HO", "AZI"]  # array task ids 6, 7, 8
RAT_CT = {"gCAP": "gCap", "aCAP": "aCap", "Peri": "Pericyte", "VEC": "VEC"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="generations per prompt")
    n = ap.parse_args().n

    expected = dict(reversed(line.split("  ", 1)) for line in (P / "inputs_sha256.txt").read_text().split("\n") if line)
    for rel, path in (("dataset/test_segments.jsonl", P / "sft_dataset_bidir/test_segments.jsonl"),
                      ("inference/inference_azi.jsonl", P / "inference_azi.jsonl")):
        assert sha(path) == expected[rel], f"{path} differs from {rel} in inputs_sha256.txt"

    prompts = {}  # (state, ct, arm) -> set of prompts
    reg = pd.read_csv(P / "sft_example_registry.csv")
    t = reg[(reg.split == "test") & (reg.direction == "fwd") & (reg.source == "bpd") & reg.condition.isin(STATES)]
    want = {int(r.line_idx): (r.condition, CT_REG[r.cell_type], "noazi") for r in t.itertuples()}
    tr = reg[(reg.split == "test") & (reg.direction == "fwd") & (reg.source == "rat") & reg.condition.isin(RAT_STATES)]
    want.update({int(r.line_idx): (r.condition, RAT_CT[r.cell_type], "rat") for r in tr.itertuples()})
    with open(P / "sft_dataset_bidir/test_segments.jsonl") as f:
        for i, line in enumerate(f):
            if i in want:
                prompts.setdefault(want[i], set()).add(json.loads(line)["segments"][0]["text"])
    with open(P / "inference_azi.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] in STATES:
                prompts.setdefault((r["condition"], CT_REG[r["cell_type"]], "azi"), set()).add(r["prompt"])

    sums = []
    for state in STATES:
        for arm in ARMS:
            name = f"{state}_{arm}.jsonl"
            with open(OUT / name, "w") as out:
                for ct in CTS:
                    got = prompts.get((state, ct, arm), set())
                    assert len(got) == 1, f"{state} {ct} {arm}: {len(got)} prompts"
                    p = got.pop()
                    treat = [x for x in p.splitlines() if x.startswith("Treatment:")][0]
                    assert ("azithromycin" in treat) == (arm == "azi"), f"{state} {ct} {arm}: {treat}"
                    for k in range(n):
                        out.write(json.dumps({"prompt": p, "condition": state, "cell_type": ct, "arm": arm, "rep": k}) + "\n")
            sums.append(f"{sha(OUT / name)}  {name}")
            print(name, 4 * n, "lines")
    for state in RAT_STATES:
        name = f"Rat_{state}.jsonl"
        with open(OUT / name, "w") as out:
            for ct in CTS:
                got = prompts.get((state, ct, "rat"), set())
                assert len(got) == 1, f"rat {state} {ct}: {len(got)} prompts"
                p = got.pop()
                assert "Rattus norvegicus" in p, f"rat {state} {ct}: not a rat prompt"
                for k in range(n):
                    out.write(json.dumps({"prompt": p, "condition": state, "cell_type": ct, "rep": k}) + "\n")
        sums.append(f"{sha(OUT / name)}  {name}")
        print(name, 4 * n, "lines")
    (OUT / "sha256.txt").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
