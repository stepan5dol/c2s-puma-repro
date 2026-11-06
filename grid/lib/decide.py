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
FORCE_LR: a file grid/FORCE_LR with one LR (e.g. 1e-4) makes stage B run at that LR instead of the stage A
         winner; stage A is still evaluated and logged, its edge step is skipped. The LR needs a 2-epoch run.

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
        ("human|Term->BPDPH7mo", "human Term->BPD-PH"), ("human|Term->BPD7mo", "human Term->BPD")]
CT_SHORT = {"General": "gCap", "Aerocyte": "aCap", "Pericyte": "Pericyte", "Vascular": "VEC", "Pulmonary": "VEC"}
CTS = ("gCap", "aCap", "Pericyte", "VEC")
STRATA = [(aname, ct, grp) for _, aname in ARMS for ct in CTS for grp in ("down", "up")]


# ---------------------------------------------------------------- measures (from the existing scripts)
def _eval_scores(scores):
    import io
    import contextlib
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "s.json"
    tmp.write_text(json.dumps(scores))
    argv, sys.argv = sys.argv, ["book_compare.py", str(tmp), str(tmp)]
    with contextlib.redirect_stdout(io.StringIO()):
        import book_compare as BC  # noqa: E402  (its module-level print runs once on a dummy pair)
    sys.argv = argv
    return BC.evaluate(scores)


def violin_mismatch(scores):
    """Manuscript violins, Mann-Whitney as in the manuscript: sum over 5 arms x 2 groups of
    |cell types (of 4) with the expected shift, model - real|. Missing cell types (e.g. <3 rescued
    genes) count as failed model strata (+1 mismatch)."""
    rows = _eval_scores(scores)
    cnt = defaultdict(int)
    for _, a in ARMS:
        for ct in CTS:
            for g in ("down", "up"):
                if (a, ct, g, "real") in rows:
                    cnt[(a, g, "real")] += int(rows[(a, ct, g, "real")]["violin"])
                    cnt[(a, g, "model")] += int(rows[(a, ct, g, "model")]["violin"])
                else:
                    cnt[(a, g, "real")] += 1
    return sum(abs(cnt[(a, g, "model")] - cnt[(a, g, "real")]) for _, a in ARMS for g in ("down", "up"))


def violin_strata_ok(scores):
    """40-element boolean list across STRATA (5 arms x 4 cell types x 2 directions): True iff
    the model side achieves Mann-Whitney p < 0.05 in the expected direction."""
    rows = _eval_scores(scores)
    return [bool(rows.get((a, ct, g, "model"), {}).get("violin", False)) for a, ct, g in STRATA]


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
        gt_c = cond_of(sp, field(r["gt"], "Condition"))
        pred_c = cond_of(sp, field(r["pred"], "Condition"))
        out[r["idx"]] = (sp, gt_c is not None and gt_c == pred_c)
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
        if use_input:
            m_un = RE["unexp"].search(p)
            pred = m_un.group(1).strip().split()[:K] if m_un else []
        else:
            pred = dedupe(r["pred"].strip().split())[:K]
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
    """Steps 1-3 among the named finished runs:
      Step 1  drop a run if its 5-arm full_list violin mismatch does not beat the copy threshold.
      Step 2  rank kept runs by forward violin quality across the 40 full_list strata (lowest mismatch,
              highest passed strata, McNemar across the 40 strata); tiebreak(name) breaks ties
              (stage A: lower eval_loss; stage B: fewer epochs). Reverse task accuracy is logged descriptively.
      Step 3  energy distance guard against every other kept run.
    Returns (winner, why)."""
    for n in names:
        r = runs[n]
        r["mismatch"] = violin_mismatch(r["scores"])
        r["strata_ok"] = violin_strata_ok(r["scores"])
        r["n_violins"] = int(sum(r["strata_ok"]))
        r["rev"] = reverse_right(r["test"])
        r["human_ok"] = {i: ok for i, (sp, ok) in r["rev"].items() if sp == "Homo"}
    kept = [n for n in names if runs[n]["mismatch"] < copy_threshold]
    for n in names:
        log(f"  {n:22s} violins {runs[n]['n_violins']:2d}/{len(STRATA)} (mismatch {runs[n]['mismatch']:2d}, "
            f"copy {copy_threshold:.0f})  eval_loss {runs[n]['loss']:.4f}  "
            f"{'kept' if n in kept else 'DROPPED (not better than the copy of the input)'}")
    if not kept:
        return None, "all runs dropped by step 1"
    if len(kept) == 1:
        return kept[0], "only run left after step 1"
    ids = sorted(runs[kept[0]]["human_ok"])
    assert all(sorted(runs[n]["human_ok"]) == ids for n in kept), "the runs answered different test records"
    acc = {n: float(np.mean([runs[n]["human_ok"][i] for i in ids])) if ids else float("nan") for n in kept}
    for n in kept:
        log(f"  {n:22s} forward violins: {runs[n]['n_violins']}/{len(STRATA)} (mismatch {runs[n]['mismatch']})  "
            f"reverse task, human: {100 * acc[n]:.1f}%")
    while True:
        order = sorted(kept, key=lambda n: (runs[n]["mismatch"], -runs[n]["n_violins"], tiebreak(n)))
        lead = order[0]
        tied = [lead]
        for n in order[1:]:
            n10, n01, p = mcnemar(runs[n]["strata_ok"], runs[lead]["strata_ok"])
            pb = min(1.0, p * (len(kept) - 1))
            log(f"  {lead} vs {n} (40 violin strata): only {lead} ok {n10}, only {n} ok {n01}, McNemar p_bonf {pb:.2g}")
            if pb >= ALPHA and runs[n]["mismatch"] == runs[lead]["mismatch"] and runs[n]["n_violins"] == runs[lead]["n_violins"]:
                tied.append(n)
        if len(tied) == 1:
            cand = lead
            why = f"forward violins: {runs[cand]['n_violins']}/{len(STRATA)} (mismatch {runs[cand]['mismatch']})"
        else:
            cand = min(tied, key=tiebreak)
            why = f"forward violins tied on {tied} ({runs[cand]['n_violins']}/{len(STRATA)}, mismatch {runs[cand]['mismatch']}); tie-break"
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
    # grid/FORCE_LR (chain_step.py puts it in state["force_lr"]): the author sets the LR of stage B instead of the
    # stage A winner. Stage A is still evaluated and logged, so the decision shows what the rule would have chosen;
    # the edge step of stage A is skipped.
    force = state.get("force_lr")
    if win is None and force is None:
        return dict(action="stop", why=why)
    log(f"  -> {win}: {why}")
    if force is not None:
        match = [lr for lr in lrs_done if lr_str(lr) == lr_str(force)]
        if not match:
            return dict(action="stop", why=f"FORCE_LR {lr_str(force)} has no 2-epoch run (seed 42); "
                                          f"LRs trained at 2 epochs: {[lr_str(x) for x in lrs_done]}")
        lr_w = match[0]
        log(f"  LR set by the author (grid/FORCE_LR): {lr_str(lr_w)}; the rule chose "
            f"{lr_str(done[win]['lr']) if win else 'nothing'}")
    else:
        lr_w = done[win]["lr"]
    nxt = None
    if force is None:
        i = LR_GRID.index(lr_w)
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

    def state(cells, queued=(), force=None):
        c = {cell_name(*x): dict(lr=x[0], ep=x[1], seed=x[2], run="r", status="done") for x in cells}
        c.update({q: dict(lr=0, ep=0, seed=0, run="r", status="queued") for q in queued})
        return dict(cells=c, runs_dir="x", force_lr=force)

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

    def two(lr_a, ep_b):
        """stage A (all names at 2 epochs) -> the run at lr_a (None: nothing kept); stage B -> the run at ep_b epochs"""
        def f(names, runs, thr, lg, tiebreak):
            if all("_ep2_" in n for n in names) and len(names) > 1 and len({n.split("_")[0] for n in names}) > 1:
                pick = [n for n in names if lr_a is not None and n.startswith(f"lr{lr_str(lr_a)}_")]
            else:
                pick = [n for n in names if f"_ep{ep_b:g}_" in n]
            calls.append(sorted(names))
            return (pick[0], "scripted") if pick else (None, "scripted: no such run")
        return f

    calls = []

    def step(cells, win=None, queued=(), force=None):
        global pick_best
        pick_best = win or (lambda *a, **k: (None, "not called"))
        calls.clear()
        r = next_step(state(cells, queued, force), 22, log)
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
    # FORCE_LR: the grid as on 2026-11-04 (stage A chose 3e-4, epochs 1/2/4 done there), author sets 1e-4
    G = A + [(3e-4, 2, 42), (3e-4, 1, 42), (3e-4, 4, 42)]
    check(step(G, two(3e-4, 4)), ("train", [("3e-4", 8, 42)]))                          # without FORCE_LR: the rule goes to 8
    check(step(G, two(3e-4, 4), force=1e-4), ("train", [("1e-4", 1, 42), ("1e-4", 4, 42)]))  # forced: epochs 1, 4 at 1e-4
    G2 = G + [(1e-4, 1, 42), (1e-4, 4, 42)]
    check(step(G2, two(3e-4, 4), force=1e-4), ("train", [("1e-4", 8, 42)]))             # forced, 4 wins -> 8 at 1e-4
    check(calls[-1], ["lr1e-4_ep1_s42", "lr1e-4_ep2_s42", "lr1e-4_ep4_s42"])            # stage B compared exactly these
    check(step(G2, two(3e-4, 2), force=1e-4), ("train", [("1e-4", 2, 43), ("1e-4", 2, 44)]))  # forced, 2 wins -> seeds
    check(step(G2, two(3e-4, 1), force=1e-4), ("train", [("1e-4", 1, 43), ("1e-4", 1, 44)]))  # forced, 1 wins -> seeds
    check(step(G2, two(3e-4, 4)), ("train", [("3e-4", 8, 42)]))                         # without FORCE_LR the 1e-4 cells
    check(calls[-1], ["lr3e-4_ep1_s42", "lr3e-4_ep2_s42", "lr3e-4_ep4_s42"])            # do not enter stage B
    check(step(A, two(1e-4, 2), force=1e-4), ("train", [("1e-4", 1, 42), ("1e-4", 4, 42)]))  # forced: no edge step to 3e-4
    check(step(G2, two(None, 4), force=1e-4), ("train", [("1e-4", 8, 42)]))             # stage A keeps nothing: forced goes on
    check(step(G2, two(None, 4)), ("stop", None))                                       # ... without FORCE_LR it stops
    check(step(G2, two(3e-4, 4), force=3e-6), ("stop", None))                           # forced LR without a 2-epoch run
    check(step(G2, two(3e-4, 4), force=float("1e-4")), ("train", [("1e-4", 8, 42)]))   # parsed from the file text
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
