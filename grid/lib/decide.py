"""Grid decider: reads the finished runs and says which cell(s) to train next.

The rule is the one on the scheme (https://claude.ai/artifact/Spx6j3W35P4boV2sLZHcY3), made from the
existing scripts and checked on the three models we have (results/fp32_2026-10-30/select_rule/):

  Step 1  drop a run if, on the manuscript's violins, it does not beat the copy of the input cell.
  Step 2  among the rest take the one that names the condition right more often for human real cells
          (reverse task). Every run the leader is NOT significantly better than (McNemar, Bonferroni over the
          comparisons made) is tied with it; the tie goes to the lower eval_loss (stage A) or fewer epochs (stage B).
  Step 3  guard: the chosen run's energy distance must not be significantly worse than any other kept run's;
          if it is, the run is out and steps 2-3 repeat without it.

Stages:  A  learning rate at 2 epochs   (cells lr1e-5, lr3e-5 trained; lr1e-4 is the finished run)
         B  epochs at the winning LR    (1 and 4 trained; 2 from stage A)
         C  two more seeds at the winner
Edge:    winner at the first/last LR (or epochs) of the explored range -> one more cell beyond it, same rule.

usage:  decide.py state.json            reads the finished runs listed there, prints and writes the next step
        decide.py --selftest            reproduces fp32 > ms264 > a100ep2 on the runs we already have

state.json (written by grid.sh):
  {"cells": {"lr1e-5_ep2_s42": {"lr": 1e-5, "ep": 2, "seed": 42, "run": "<dir under grid/runs>", "status": "done"|"queued"}}}
Each finished cell has grid/runs/<run>/{test_t10.jsonl, trainer_state.json, violins/scores.json}.
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# a8_baselines.py, book_compare.py, violins_book_tests.py sit next to this file
from a8_baselines import CALIB, CT_MAP, K, RE, cond_of, dedupe, domain, recon  # noqa: E402

ALPHA = 0.05
LR_GRID = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
EP_GRID = [1, 2, 4, 8]
ARMS = [("rat|ho", "rat RA->HO"), ("rat|azi", "rat HO->AZI"), ("human|He22->Acute26", "human He22->Acute26"),
        ("human|Term->BPDPH7mo", "human Term->BPD-PH"), ("human|Term->BPD7mo", "human Term->BPD"),
        ("human|Acute26->BPD7mo", "human Acute26->BPD, pairs (dge_filtered)")]
CT_SHORT = {"General": "gCap", "Aerocyte": "aCap", "Pericyte": "Pericyte", "Vascular": "VEC", "Pulmonary": "VEC"}


# ---------------------------------------------------------------- measures (from the existing scripts)
def violin_mismatch(scores):
    """Manuscript violins, Mann-Whitney as in the manuscript: sum over 6 arms x 2 groups of
    |cell types (of 4) with the expected shift, model - real|. 0 = as the real cells, 48 = worst."""
    import io
    import contextlib
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "s.json"
    tmp.write_text(json.dumps(scores))
    argv, sys.argv = sys.argv, ["book_compare.py", str(tmp), str(tmp)]
    with contextlib.redirect_stdout(io.StringIO()):
        import book_compare as BC  # noqa: E402  (its module-level print runs once on a dummy pair)
    sys.argv = argv
    rows = BC.evaluate(scores)
    cnt = defaultdict(int)
    for (arm, ct, grp, side), x in rows.items():
        cnt[(arm, grp, side)] += int(x["violin"])
    return sum(abs(cnt[(a, g, "model")] - cnt[(a, g, "real")]) for _, a in BC.ARMS for g in ("down", "up"))


def field(text, name):
    m = re.search(rf"{name}:\s*([^\n<]+)", text)
    return m.group(1).strip() if m else ""


def reverse_right(path):
    """{idx: (species, condition named right)} for the rev records of a run_test.py-format file."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["direction"] != "rev":
            continue
        sp = "Rattus" if "Rattus" in r["prompt"].split("\n\n")[0] else "Homo"
        out[r["idx"]] = (sp, cond_of(sp, field(r["gt"], "Condition")) == cond_of(sp, field(r["pred"], "Condition")))
    return out


def energy_groups(path, use_input=False):
    g = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r["direction"] != "fwd":
            continue
        p = r["prompt"]
        sp = RE["species"].search(p).group(1).strip().split()[0]
        ct = CT_MAP.get(RE["ct"].search(p).group(1).strip())
        cond = cond_of(sp, RE["pert"].search(p).group(1).strip())
        if ct is None or cond is None:
            continue
        pred = RE["unexp"].search(p).group(1).strip().split()[:K] if use_input else dedupe(r["pred"].strip().split())[:K]
        g[(sp, ct, cond)].append(dict(gt=r["gt"].strip().split()[:K], pred=pred))
    return g


def energy_pair(ga, gb, nboot=300, seed=0):
    """Pooled energy distance to the real cells per species (a8_baselines.py construction) for A and B and the
    95% interval of E_B - E_A from a paired bootstrap over cells within groups."""
    rng = np.random.default_rng(seed)
    out = {}
    for sp in ("Rattus", "Homo"):
        slope, icpt = CALIB[sp]
        keys = sorted(k for k in ga if k[0] == sp and len(ga[k]) >= 5)
        vocab = {}
        for k in keys:
            for G in (ga, gb):
                for c in G[k]:
                    for gene in c["gt"] + c["pred"]:
                        vocab.setdefault(gene, len(vocab))
        mats = []
        for k in keys:
            gt = recon([c["gt"] for c in ga[k]], vocab, slope, icpt)
            pa = recon([c["pred"] for c in ga[k]], vocab, slope, icpt)
            pb = recon([c["pred"] for c in gb[k]], vocab, slope, icpt)
            mats.append((gt, pa, domain(gt, pa), pb, domain(gt, pb)))

        def pooled(ix_list):
            res = []
            for pi, di in ((1, 2), (3, 4)):
                rr, cross, pp = [], [], []
                for m, ix in zip(mats, ix_list):
                    gt, P = m[0][ix][:, m[di]], m[pi][ix][:, m[di]]
                    rr.append(pdist(gt)), cross.append(cdist(P, gt).ravel()), pp.append(pdist(P))
                rr, cross, pp = (np.concatenate(x) for x in (rr, cross, pp))
                res.append(2 * cross.mean() - rr.mean() - pp.mean())
            return res

        ea, eb = pooled([np.arange(len(m[0])) for m in mats])
        boot = [np.subtract(*pooled([rng.integers(0, len(m[0]), len(m[0])) for m in mats])[::-1]) for _ in range(nboot)]
        lo, hi = np.quantile(boot, [0.025, 0.975])
        out[sp] = dict(a=ea, b=eb, lo=lo, hi=hi)
    return out


def final_eval_loss(trainer_state):
    ts = json.load(open(trainer_state))
    ev = [h for h in ts["log_history"] if "eval_loss" in h]
    return ev[-1]["eval_loss"] if ev else float("nan")


# ---------------------------------------------------------------- one run = one record
def load_run(rundir):
    d = Path(rundir)
    return dict(dir=d, test=d / "test_t10.jsonl", scores=json.load(open(d / "violins/scores.json")),
                loss=final_eval_loss(d / "trainer_state.json") if (d / "trainer_state.json").exists() else float("nan"))


def mcnemar(a, b):
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, p


def pick_best(names, runs, copy_threshold, log, tiebreak):
    """Steps 1-3 among the named finished runs; tiebreak(name) is the sort key used when the reverse task cannot
    separate the leader from other runs (stage A: lower eval_loss; stage B: fewer epochs). Returns (winner, why)."""
    for n in names:
        r = runs[n]
        r["mismatch"] = violin_mismatch(r["scores"])
        r["rev"] = reverse_right(r["test"])
        r["human_ok"] = {i: ok for i, (sp, ok) in r["rev"].items() if sp == "Homo"}
    kept = [n for n in names if runs[n]["mismatch"] < copy_threshold]
    for n in names:
        log(f"  {n:22s} violins mismatch {runs[n]['mismatch']:2d} (copy of input {copy_threshold:.0f})  "
            f"eval_loss {runs[n]['loss']:.4f}  {'kept' if n in kept else 'DROPPED (not better than the copy of the input)'}")
    if not kept:
        return None, "all runs dropped by step 1"
    if len(kept) == 1:
        return kept[0], "only run left after step 1"
    ids = sorted(runs[kept[0]]["human_ok"])
    assert all(sorted(runs[n]["human_ok"]) == ids for n in kept), "the runs answered different test records"
    acc = {n: float(np.mean([runs[n]["human_ok"][i] for i in ids])) for n in kept}
    for n in kept:
        log(f"  {n:22s} reverse task, human: {100 * acc[n]:.1f}%")
    while True:
        order = sorted(kept, key=lambda n: -acc[n])
        lead = order[0]
        tied = [lead]
        for n in order[1:]:
            n10, n01, p = mcnemar([runs[n]["human_ok"][i] for i in ids], [runs[lead]["human_ok"][i] for i in ids])
            pb = min(1.0, p * (len(kept) - 1))
            log(f"  {lead} vs {n}: only {lead} right {n10}, only {n} right {n01}, McNemar p_bonf {pb:.2g}")
            if pb >= ALPHA:
                tied.append(n)
        if len(tied) == 1:
            cand, why = lead, "reverse task: significantly better than every other run"
        else:
            cand = min(tied, key=tiebreak)
            why = f"reverse task cannot separate {tied}; tie-break"
        guard = None
        for n in kept:
            if n == cand:
                continue
            e = energy_pair(energy_groups(runs[n]["test"]), energy_groups(runs[cand]["test"]))
            log(f"  energy, {cand} - {n}: " + "; ".join(
                f"{sp} {v['b'] - v['a']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]" for sp, v in e.items()))
            if any(v["lo"] > 0 for v in e.values()):
                guard = n
                break
        if guard is None:
            return cand, why + "; energy guard passed"
        log(f"  {cand} is out by the energy guard (significantly worse than {guard})")
        kept.remove(cand)
        if len(kept) == 1:
            return kept[0], f"last run left after the energy guard removed {cand}"


# ---------------------------------------------------------------- the grid
def lr_str(lr):
    """3e-05 -> '3e-5': the form used in cell names and in submit_cells.sh ({:g} would give 0.0001 for 1e-4)."""
    return re.sub(r"e-0*(\d)", r"e-\1", f"{lr:.0e}")


def cell_name(lr, ep, seed=42):
    return f"lr{lr_str(lr)}_ep{ep:g}_s{seed}"


def next_step(state, copy_threshold, log):
    cells = state["cells"]
    done = {n: c for n, c in cells.items() if c["status"] == "done"}
    queued = [n for n, c in cells.items() if c["status"] == "queued"]
    if queued:
        return dict(action="wait", why=f"{len(queued)} cell(s) still running: {queued}")
    runs = {n: load_run(state["runs_dir"] + "/" + c["run"]) for n, c in done.items()}
    loss_of = lambda n: runs[n]["loss"] if runs[n]["loss"] == runs[n]["loss"] else 1e9  # nan last
    base_ep = 2
    # stage A: LR at 2 epochs, seed 42
    stageA = [n for n, c in done.items() if c["ep"] == base_ep and c["seed"] == 42]
    lrs_done = sorted({done[n]["lr"] for n in stageA})
    need = [lr for lr in (1e-5, 3e-5) if lr not in lrs_done]
    if need:
        return dict(action="train", cells=[dict(lr=lr, ep=base_ep, seed=42) for lr in need],
                    why="stage A: the LR grid at 2 epochs is not complete")
    log("STAGE A  learning rate at 2 epochs")
    win, why = pick_best(stageA, runs, copy_threshold, log, tiebreak=loss_of)
    if win is None:
        return dict(action="stop", why=why)
    log(f"  -> {win}: {why}")
    lr_w = done[win]["lr"]
    i = LR_GRID.index(lr_w)
    nxt = None
    if lr_w == lrs_done[0] and i > 0 and LR_GRID[i - 1] not in lrs_done:
        nxt = LR_GRID[i - 1]
    elif lr_w == lrs_done[-1] and i < len(LR_GRID) - 1 and LR_GRID[i + 1] not in lrs_done:
        nxt = LR_GRID[i + 1]
    if nxt is not None:
        return dict(action="train", cells=[dict(lr=nxt, ep=base_ep, seed=42)],
                    why=f"stage A+: the winner {lr_w:g} is at the edge of the explored LRs {lrs_done}; try {nxt:g}")
    # stage B: epochs at the winning LR
    ep_cells = {c["ep"]: n for n, c in done.items() if c["lr"] == lr_w and c["seed"] == 42}
    need_ep = [e for e in (1, 4) if e not in ep_cells]
    if need_ep:
        return dict(action="train", cells=[dict(lr=lr_w, ep=e, seed=42) for e in need_ep],
                    why=f"stage B: epochs 1 and 4 at the winning LR {lr_w:g}")
    log(f"STAGE B  epochs at LR {lr_w:g}")
    names = [ep_cells[e] for e in sorted(ep_cells)]
    winb, whyb = pick_best(names, runs, copy_threshold, log, tiebreak=lambda n: (done[n]["ep"], loss_of(n)))
    if winb is None:
        return dict(action="stop", why=whyb)
    log(f"  -> {winb}: {whyb}")
    ep_w = done[winb]["ep"]
    j = EP_GRID.index(ep_w)
    if ep_w == max(ep_cells) and j < len(EP_GRID) - 1 and EP_GRID[j + 1] not in ep_cells:
        return dict(action="train", cells=[dict(lr=lr_w, ep=EP_GRID[j + 1], seed=42)],
                    why=f"stage B+: the winner {ep_w:g} epochs is at the upper edge; try {EP_GRID[j + 1]:g}")
    if ep_w == EP_GRID[0]:
        log("  the winner is 1 epoch, the lower end of the grid; fewer epochs are not tried")
    # stage C: two more seeds at the winner
    seeds = {c["seed"] for c in done.values() if c["lr"] == lr_w and c["ep"] == ep_w}
    need_s = [s for s in (43, 44) if s not in seeds]
    if need_s:
        return dict(action="train", cells=[dict(lr=lr_w, ep=ep_w, seed=s) for s in need_s],
                    why=f"stage C: two more seeds at the winner (LR {lr_w:g}, {ep_w:g} epochs)")
    return dict(action="finished", winner=dict(lr=lr_w, ep=ep_w), why=f"recipe: LR {lr_w:g}, {ep_w:g} epochs, 3 seeds done")


# ---------------------------------------------------------------- self-tests
def selftest_walk():
    """The walk over the grid with scripted winners: no data needed, every branch of next_step."""
    global load_run, pick_best
    keep = load_run, pick_best
    load_run = lambda d: dict(loss=0.0)
    log = lambda s: None

    def state(cells, queued=()):
        c = {cell_name(*x): dict(lr=x[0], ep=x[1], seed=x[2], run="r", status="done") for x in cells}
        c.update({q: dict(lr=0, ep=0, seed=0, run="r", status="queued") for q in queued})
        return dict(cells=c, runs_dir="x")

    def winner(lr=None, ep=None):
        def f(names, runs, thr, lg, tiebreak):
            # stage A is called with the 2-epoch runs of all LRs, stage B with the epochs of one LR: take the scripted LR,
            # and the scripted epochs when the call offers several epoch values
            pick = [n for n in names if lr is None or n.startswith(f"lr{lr_str(lr)}_")]
            by_ep = [n for n in pick if ep is None or f"_ep{ep:g}_" in n]
            if len({n.split("_")[1] for n in pick}) > 1:
                pick = by_ep
            return (pick[0], "scripted") if pick else (None, "scripted: no such run")
        return f

    def step(cells, win=None, queued=()):
        global pick_best
        pick_best = win or (lambda *a, **k: (None, "not called"))
        r = next_step(state(cells, queued), 22, log)
        return (r["action"], sorted((lr_str(c["lr"]), c["ep"], c["seed"]) for c in r["cells"])) if r["action"] == "train" else (r["action"], r.get("winner"))

    base = [(1e-4, 2, 42)]
    A = base + [(1e-5, 2, 42), (3e-5, 2, 42)]
    n = 0
    def check(got, want):
        nonlocal n
        assert got == want, (got, want)
        n += 1
    check(step(base), ("train", [("1e-5", 2, 42), ("3e-5", 2, 42)]))                     # stage A: two cells at once
    check(step(A, winner(3e-5, 2)), ("train", [("3e-5", 1, 42), ("3e-5", 4, 42)]))       # interior LR -> stage B, two at once
    B = A + [(3e-5, 1, 42), (3e-5, 4, 42)]
    check(step(B, winner(3e-5, 2)), ("train", [("3e-5", 2, 43), ("3e-5", 2, 44)]))       # interior epochs -> two seeds
    check(step(B + [(3e-5, 2, 43), (3e-5, 2, 44)], winner(3e-5, 2)), ("finished", dict(lr=3e-5, ep=2)))
    check(step(A, winner(1e-5, 2)), ("train", [("3e-6", 2, 42)]))                        # low edge -> one cell beyond
    check(step(A + [(3e-6, 2, 42)], winner(3e-6, 2)), ("train", [("3e-6", 1, 42), ("3e-6", 4, 42)]))  # bottom of the grid -> B
    check(step(A, winner(1e-4, 2)), ("train", [("3e-4", 2, 42)]))                        # high edge; '{:g}' would print 0.0003
    check(step(A + [(3e-4, 2, 42)], winner(3e-4, 2)), ("train", [("3e-4", 1, 42), ("3e-4", 4, 42)]))
    check(step(B, winner(3e-5, 4)), ("train", [("3e-5", 8, 42)]))                        # upper edge of epochs -> 8
    check(step(B + [(3e-5, 8, 42)], winner(3e-5, 8)), ("train", [("3e-5", 8, 43), ("3e-5", 8, 44)]))  # top of the grid -> seeds
    check(step(B, winner(3e-5, 1)), ("train", [("3e-5", 1, 43), ("3e-5", 1, 44)]))       # 1 epoch: no fractional epochs, seeds
    check(step(A, queued=["lr3e-5_ep4_s42"]), ("wait", None))
    check(step(A, lambda *a, **k: (None, "all dropped")), ("stop", None))
    check(cell_name(1e-4, 2, 42), "lr1e-4_ep2_s42")
    load_run, pick_best = keep
    print(f"walk selftest: {n} checks passed")


def selftest():
    R = Path(os.environ["C2S_RESULTS"])  # a folder with the three reference runs (see grid/README)
    spec = {"ms264": (R / "a100ep2_2026-10-22_1215/ms264/violins/scores.json", R / "test_inference_results_t10.jsonl"),
            "a100ep2": (R / "a100ep2_2026-10-22_1215/a100ep2/violins/scores.json", R / "test_inference_results_t10_a100ep2.jsonl"),
            "fp32": (R / "fp32_2026-10-30/fp32/violins/scores.json", R / "test_inference_results_t10_fp32.jsonl")}
    runs = {n: dict(scores=json.load(open(s)), test=t, loss=float("nan")) for n, (s, t) in spec.items()}
    copy = violin_mismatch(json.load(open(R / "fp32_2026-10-30/identity/violins/scores.json")))
    print(f"copy of input: violins mismatch {copy}")
    win, why = pick_best(list(runs), runs, copy, print, tiebreak=lambda n: 0)
    print(f"WINNER: {win} ({why})")
    assert win == "fp32", win
    print("\ntie-break: two runs with identical answers, labelled 4 and 1 epochs -> the one with fewer epochs")
    twin = {"ep4": dict(runs["fp32"]), "ep1": dict(runs["fp32"])}
    win, why = pick_best(["ep4", "ep1"], twin, copy, print, tiebreak=lambda n: {"ep1": 1, "ep4": 4}[n])
    print(f"WINNER: {win} ({why})")
    assert win == "ep1", win


if __name__ == "__main__":
    if sys.argv[1] == "--selftest-walk":
        selftest_walk()
        sys.exit(0)
    if sys.argv[1] == "--selftest":
        selftest_walk()
        selftest()
        sys.exit(0)
    state = json.load(open(sys.argv[1]))
    lines = []
    log = lambda s: (print(s), lines.append(s))
    step = next_step(state, state["copy_threshold"], log)
    print("\nNEXT:", json.dumps(step, ensure_ascii=False))
    out = Path(state["decisions_dir"]) / f"decision_{len(list(Path(state['decisions_dir']).glob('decision_*.json'))):02d}.json"
    out.write_text(json.dumps(dict(step=step, log=lines, state=state), ensure_ascii=False, indent=1))
