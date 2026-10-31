"""One step of the grid chain, run inside c2s-analysis.sif by grid/analyze.slurm.

Reads the cells of the grid (cells.tsv), takes the finished ones, applies the rule of decide.py, writes the decision
(decisions/decision_NN.json, with the reason) and, if the rule says to train more cells, submits them with
hpc/submit_cells.sh logic (done here by calling `sbatch` through the host: this script only writes the commands to
next_cells.txt; grid/analyze.slurm submits them after it returns, outside the container).

The chain stops (and says why) when: a cell has no results, the rule says stop or finished, or MAX_CELLS cells were
trained in total. Nothing is skipped silently: every stop writes grid/STOPPED with the reason, and the Slurm mail
(--mail-type=END,FAIL) tells you the job ended.

usage: chain_step.py --grid /grid --repo /repo --max-cells 9
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decide  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--grid", type=Path, required=True)
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--max-cells", type=int, default=9)
args = ap.parse_args()
G = args.grid
DEC, RUNS = G / "decisions", G / "runs"
DEC.mkdir(exist_ok=True)
log = lambda s: print(s, flush=True)


def stop(why, code=0):
    (G / "STOPPED").write_text(why + "\n")
    log("STOP: " + why)
    sys.exit(code)


CELL = re.compile(r"lr([0-9.]+e-[0-9]+)_ep([0-9]+)_s([0-9]+)")
cells, missing = {}, []
for line in open(G / "cells.tsv"):
    name, run, tj, ij = line.rstrip("\n").split("\t")[:4]
    m = CELL.fullmatch(name)
    r = Path(run)
    inf, ts = r / "inference_float32" / "test_t10.jsonl", r / "trainer_state.json"
    if not ts.exists():  # the run of 2026-10-28: only inside the checkpoint folders of the model folder
        cks = sorted(r.glob("sft_pythia1b_short_full/checkpoint-*/trainer_state.json"), key=lambda q: int(q.parent.name.split("-")[1]))
        ts = cks[-1] if cks else ts
    scores = RUNS / name / "violins" / "scores.json"
    have = [f.name for f in (inf, ts, scores) if f.exists()]
    if len(have) < 3:
        missing.append((name, [f for f in ("test_t10.jsonl", "trainer_state.json", "scores.json") if f not in have], tj, ij))
        continue
    for f, src in (("test_t10.jsonl", inf), ("trainer_state.json", ts)):
        link = RUNS / name / f
        if not link.exists():
            link.symlink_to(src)
    cells[name] = dict(lr=float(m[1]), ep=int(m[2]), seed=int(m[3]), run=name, status="done", train_job=tj, infer_job=ij)

# This job runs only after the inference jobs of all cells of its call finished OK, and each call is analysed before
# the next one is submitted. So a cell without its outputs here is a failure, never "still running": stop and say what
# Slurm says about its jobs (sacct is only for the message; the decision does not depend on it).
if missing:
    import subprocess
    detail = []
    for name, lacking, tj, ij in missing:
        try:
            out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", f"{tj},{ij}", "-o", "JobID,JobName,State,ExitCode"],
                                 capture_output=True, text=True, timeout=60).stdout.strip().replace("\n", "; ")
        except Exception as e:  # sacct missing or hanging: still a failure
            out = f"sacct unavailable ({e})"
        detail.append(f"{name}: lacks {lacking}; Slurm: {out or 'no records'}")
    stop("a cell has no results, nothing further is submitted. " + " | ".join(detail), code=1)

ref = json.load(open(G / "reference" / "identity_scores.json"))
copy = decide.violin_mismatch(ref)
state = dict(cells=cells, runs_dir=str(RUNS), decisions_dir=str(DEC), copy_threshold=copy)
lines = []
tee = lambda s: (log(s), lines.append(s))
step = decide.next_step(state, copy, tee)
n = len(list(DEC.glob("decision_*.json")))
(DEC / f"decision_{n:02d}.json").write_text(json.dumps(dict(step=step, log=lines, state=state), ensure_ascii=False, indent=1))
log("NEXT: " + json.dumps(step, ensure_ascii=False))
(G / "next_cells.txt").unlink(missing_ok=True)
if step["action"] in ("stop", "finished", "wait"):
    stop(f"{step['action']}: {step['why']}")
elif step["action"] == "train":
    total = len(cells) + len(step["cells"])
    if total > args.max_cells + 1:  # +1: the finished cell lr1e-4_ep2_s42 counts in the grid but is not trained here
        stop(f"the cap of {args.max_cells} trainings would be exceeded ({total - 1}); the rule wants: {step['why']}")
    (G / "next_cells.txt").write_text(" ".join(f"{decide.lr_str(c['lr'])}:{c['ep']}:{c['seed']}" for c in step["cells"]) + "\n")
    log("TRAIN: " + (G / "next_cells.txt").read_text().strip())
