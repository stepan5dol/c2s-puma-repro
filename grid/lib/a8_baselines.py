"""Does the A8 'fraction within the Q90 real-real envelope' separate the model from trivial
predictors? Same reconstruction (log-rank line), same gene-domain filter, same pooling as
analysis/a8_equivalence_beautiful_species.py, applied to three candidates per group:

  model     the generated cells
  identity  the prompt's own Unexposed cell (no perturbation at all)
  centroid  every prediction = the mean of the group's real target cells (one 'average cell')

Also reports the energy distance 2E|P-R| - E|R-R'| - E|P-P'|, which is 0 only when the
candidate cloud matches the real cloud in location AND spread.

The gene vocabulary is the union of genes seen in the records; genes never seen are zero in
every vector and do not change any distance.
"""
import json
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist, pdist

K = 1600
CALIB = {"Rattus": (-0.64127, 5.7583), "Homo": (-0.6121373901170273, 5.541692798614713)}
CT_MAP = {"general capillary endothelial cell": "gCap", "aerocyte capillary endothelial cell": "aCap",
          "pericyte": "Pericyte", "pulmonary venous endothelial cell": "VEC",
          "vascular endothelial cell": "VEC"}
RE = {k: re.compile(p) for k, p in {"species": r"Species:\s*([^\n]+)", "ct": r"Cell type:\s*([^\(\n]+)",
                                    "pert": r"Perturbation:\s*([^\n]+)", "unexp": r"\nUnexposed:\s*([^\n]+)"}.items()}


def cond_of(species: str, pert: str):
    if species == "Rattus":
        if pert.startswith("Hyperoxia"):
            return "HO"
        if pert.startswith("Azithromycin"):
            return "AZI"
        if "room air" in pert:
            return "RA"
        return None
    if pert.startswith("Acute preterm lung injury"):
        return "Acute"
    if "pulmonary hypertension" in pert:
        return "BPD-PH"
    if pert.startswith("Progression from acute preterm lung injury"):
        return "BPD"
    if "fetal lung development at gestational week 22" in pert:
        return "He22"
    if "postnatal day 0" in pert:
        return "Term0d"
    if "postnatal day 20" in pert:
        return "Term20d"
    return None


def dedupe(words):
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def load(path):
    groups = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("direction") != "fwd":
            continue
        p = r["prompt"]
        sp = RE["species"].search(p).group(1).strip().split()[0]
        ct = CT_MAP.get(RE["ct"].search(p).group(1).strip())
        cond = cond_of(sp, RE["pert"].search(p).group(1).strip())
        if ct is None or cond is None:
            continue
        m_un = RE["unexp"].search(p)
        groups[(sp, ct, cond)].append({
            "unexp": m_un.group(1).strip().split()[:K] if m_un else [],
            "gt": r["gt"].strip().split()[:K],
            "pred": dedupe(r["pred"].strip().split())[:K]})
    return groups


def recon(lists, vocab, slope, intercept):
    X = np.zeros((len(lists), len(vocab)), dtype=np.float32)
    vals = intercept + slope * np.log(np.arange(K) + 1.0)
    for i, genes in enumerate(lists):
        for rank, g in enumerate(genes[:K]):
            if vals[rank] > 0:
                X[i, vocab[g]] = vals[rank]
    return X


def domain(A, B, min_frac=0.2):
    return ((A != 0).mean(0) >= min_frac) | ((B != 0).mean(0) >= min_frac)


def main():
    groups = load(sys.argv[1])
    for species in ("Rattus", "Homo"):
        slope, intercept = CALIB[species]
        keys = sorted(k for k in groups if k[0] == species and len(groups[k]) >= 5)
        vocab = {}
        for k in keys:
            for c in groups[k]:
                for g in c["unexp"] + c["gt"] + c["pred"]:
                    vocab.setdefault(g, len(vocab))
        pooled = defaultdict(lambda: {"rr": [], "cross": [], "pp": []})
        dup_lines = []
        for k in keys:
            g = groups[k]
            n = len(g)
            n_gt = len({" ".join(c["gt"]) for c in g})
            n_un = len({" ".join(c["unexp"]) for c in g})
            dup_lines.append(f"{k[1]}/{k[2]}: records={n} unique_gt={n_gt} unique_prompts={n_un}")
            gt = recon([c["gt"] for c in g], vocab, slope, intercept)
            cands = {"model": recon([c["pred"] for c in g], vocab, slope, intercept)}
            if any(c["unexp"] for c in g):
                cands["identity"] = recon([c["unexp"] for c in g], vocab, slope, intercept)
            cands["centroid"] = np.repeat(gt.mean(0, keepdims=True), n, axis=0)
            for name, P in cands.items():
                dom = domain(gt, P)
                pooled[name]["rr"].append(pdist(gt[:, dom]))
                pooled[name]["cross"].append(cdist(P[:, dom], gt[:, dom]).ravel())
                pooled[name]["pp"].append(pdist(P[:, dom]))
        print(f"\n=== {species}  groups={len(keys)}")
        print("\n".join("  " + s for s in dup_lines))
        for name, d in pooled.items():
            rr, cross, pp = (np.concatenate(d[x]) for x in ("rr", "cross", "pp"))
            q90 = np.percentile(rr, 90)
            energy = 2 * cross.mean() - rr.mean() - pp.mean()
            print(f"  {name:9s} frac_within_Q90={np.mean(cross <= q90):.3f}  Q90={q90:6.2f}  "
                  f"mean cross={cross.mean():6.2f} real-real={rr.mean():6.2f} cand-cand={pp.mean():6.2f}  "
                  f"energy={energy:7.2f}")


if __name__ == "__main__":
    main()
