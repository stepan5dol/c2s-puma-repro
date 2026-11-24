#!/usr/bin/env python3
"""Are the rat prompts of the 300-generation files the prompts of the test set?

For every rat (state, cell type) of the example registry (split test, direction fwd, source rat) collects the prompt
text from dataset/test_segments.jsonl and from the model's test answers (test_t10.jsonl, field prompt, matched by
idx), and the prompts of inference_bfloat16_rep300/Rat_{RA,HO,AZI}_results.jsonl. Prints per stratum the number of
distinct test prompts and whether the one prompt is the same text in all three files.
usage: python3 check_rat_prompts.py <retrain_prompt/pipeline> <test_t10.jsonl> <inference_bfloat16_rep300 dir>
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    pipe, t10_path, rep_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    reg = pd.read_csv(pipe / "sft_example_registry.csv")
    tr = reg[(reg.split == "test") & (reg.direction == "fwd") & (reg.source == "rat")]
    idx = {int(r.line_idx): (r.condition, r.cell_type) for r in tr.itertuples()}
    seg, t10, rep = collections.defaultdict(set), collections.defaultdict(set), set()
    with open(pipe / "sft_dataset_bidir/test_segments.jsonl") as f:
        for i, line in enumerate(f):
            if i in idx:
                seg[idx[i]].add(json.loads(line)["segments"][0]["text"])
    with open(t10_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("idx") in idx and r.get("direction") == "fwd":
                t10[idx[r["idx"]]].add(r["prompt"])
    for state in ("RA", "HO", "AZI"):
        with open(rep_dir / f"Rat_{state}_results.jsonl") as f:
            rep |= {json.loads(line)["prompt"] for line in f}
    print(f"distinct prompts in Rat_{{RA,HO,AZI}}_results.jsonl: {len(rep)}")
    for k in sorted(k for k in seg if k[0] in ("RA", "HO", "AZI")):
        same = len(seg[k]) == 1 and seg[k] == t10[k] and next(iter(seg[k])) in rep
        print(f"{k[0]:4s} {k[1]:5s} test_segments prompts {len(seg[k])}, test_t10 prompts {len(t10[k])}, "
              f"same text in the 300-generation files: {same}")


if __name__ == "__main__":
    main()
