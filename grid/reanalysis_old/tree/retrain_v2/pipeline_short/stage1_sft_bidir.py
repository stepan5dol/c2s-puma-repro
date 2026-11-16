#!/usr/bin/env python3
"""
pipeline_short/stage1_sft_bidir.py — Leakage-free SFT dataset builder (bidirectional),
minimally-viable scope: rat gCAP/aCAP/Peri/VEC + human gCap/aCap/Pericyte/Arterial EC/
Pulmonary venous EC/Systemic venous EC/abCap/VSMC only. No mouse.

Fork of stage1_sft.py that additionally emits a reverse-direction example for
every forward transition: prompt carries 'Perturbed cell:' (the same exposed
cell instance used as the forward completion — never a separately re-sampled
pool) and asks for the exposure conditions; completion is the perturbation
string that produced it. This does NOT try to reconstruct the unexposed cell
from the exposed one (ill-posed / lossy), it infers the known perturbation
label from the expression profile. Everything else (forward pairs, common.py)
is unchanged.

Reads cell_split_registry_short.csv, loads raw counts per source, builds
(prompt, completion) pairs for all SFT transitions, writes:
  sft_dataset_bidir/{train,valid,test}.jsonl  — {"prompt":..., "completion":...}

Cell sentences: raw counts → normalize_total(1e4) → log1p → rank desc → top-K=800.
Completion has no leading space.
Gene names remapped via ALIAS_TO_CANONICAL for human.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline_short.common import (
    BASE, PIPELINE_DIR,
    RAT_PATH, ATLAS_CLEAN_PATH, ATLAS_RAW_PATH, BPD_COUNTS_PATH,
    RESCUED_RAT, RESCUED_HUMAN,
    PERTURBATION_STR, COND_META, SPECIES_STR,
    TOP_K, RANDOM_SEED,
    remap_gene_names, ALIAS_TO_CANONICAL,
    make_cell_sentences,
    build_prompt,
    cell_type_broad, cell_type_full,
)

REGISTRY_PATH = BASE / "cell_split_registry_short.csv"
SFT_DIR       = PIPELINE_DIR / "sft_dataset_bidir"
UNPAIRED_PATH = PIPELINE_DIR / "unpaired_report_bidir.csv"


def build_prompt_reverse(species: str, perturbed_cs: str) -> str:
    """Reverse-direction prompt: given ONLY the exposed/perturbed cell's expression,
    infer age, cell type, and the exposure conditions (perturbation) that produced
    it — all three go in the completion, mirroring C2S-Scale's own bidirectional
    perturbation task (paper §4.7.9: "simultaneously predicted all three labels —
    cell type, perturbation, and exposure"). Species is stated as known context, not
    predicted: it's the one thing that's NOT a meaningful inference target here,
    since human/mouse gene symbols are upper-case and rat symbols are title-case in
    our cell sentences (see load_rat_sentences) — a model could "predict" species
    just by reading letter case, which teaches nothing. This also matches C2S-Scale's
    own cell-type-annotation prompts, which always state species inline
    ("...in a Homo sapiens cell...") rather than asking for it.
    """
    return (
        f"Analyze the single cell's expression, listed in 'Perturbed cell:', from a "
        f"{species} sample, and determine its age, cell type, and the exposure "
        "conditions (perturbation) that produced it.\n\n"
        f"Perturbed cell: {perturbed_cs}\n\n"
        "Age, cell type, and conditions:"
    )


def build_completion_reverse(
    age_str: str,
    pma_weeks: int | float,
    cell_type_fine: str,
    cell_type_broad_str: str,
    rev_pert_str: str,
) -> str:
    ct_full = cell_type_full(cell_type_fine)
    return (
        f"Age: {age_str}\n"
        f"Cell type: {ct_full} ({cell_type_broad_str})\n"
        f"Condition: {rev_pert_str}"
    )


REV_CONDITION_OVERRIDE: dict[tuple, str] = {}

HUMAN_SFT = [
    ("He22", "Acute26"),
    ("Acute26", "BPD7mo"),
    ("Acute26", "BPDPH7mo"),
]

RAT_AUG_THRESHOLD = 200
RAT_AUG_SMALL = 6
RAT_AUG_LARGE = 2

HUMAN_AUG_THRESHOLD = 200
HUMAN_AUG_SMALL = 3
HUMAN_AUG_LARGE = 1

EVAL_AUG = 1

RAT_EXCLUDE_GENES = {"Hbb", "Hba-a1"}


def _src_of_human(cond: str) -> str:
    return "atlas" if cond.startswith("He") else "bpd"


def _pair_seed(split: str, pair_type: str, cell_type: str) -> int:
    key = f"{split}|{pair_type}|{cell_type}"
    return RANDOM_SEED + int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _make_batched(X, gene_names: list[str], batch_size: int = 2000) -> list[str]:
    """make_cell_sentences in batches to keep peak memory under ~300 MB/batch."""
    if sp.issparse(X):
        X = X.toarray()
    out: list[str] = []
    for i in range(0, X.shape[0], batch_size):
        out.extend(make_cell_sentences(
            X[i:i + batch_size].astype(np.float32), gene_names, seed=RANDOM_SEED))
    return out


def load_rat_sentences(barcodes: list[str]) -> dict[str, str]:
    print("  [rat] loading h5ad raw counts...")
    adata = anndata.read_h5ad(RAT_PATH)
    if adata.raw is not None:
        X, genes = adata.raw.X, adata.raw.var_names.tolist()
    elif "counts" in adata.layers:
        X, genes = adata.layers["counts"], adata.var_names.tolist()
    else:
        print("  WARN: rat using X directly (not guaranteed raw integer counts)")
        X, genes = adata.X, adata.var_names.tolist()
    bc_idx = {bc: i for i, bc in enumerate(adata.obs_names)}
    del adata
    keep_mask = [g not in RAT_EXCLUDE_GENES for g in genes]
    if sp.issparse(X):
        X = X[:, keep_mask]
    else:
        X = np.asarray(X)[:, keep_mask]
    genes = [g for g, k in zip(genes, keep_mask) if k]
    needed = [bc for bc in barcodes if bc in bc_idx]
    rows   = [bc_idx[bc] for bc in needed]
    X_sub  = (X[rows].toarray() if sp.issparse(X) else np.array(X)[rows]).astype(np.float32)
    return dict(zip(needed, _make_batched(X_sub, genes)))


def load_atlas_sentences(barcodes: list[str]) -> dict[str, str]:
    print("  [atlas] loading counts from ATLAS_CLEAN_PATH...")
    adata = anndata.read_h5ad(ATLAS_CLEAN_PATH)

    if adata.raw is not None:
        X_full    = adata.raw.X
        genes     = remap_gene_names(adata.raw.var_names.tolist())
        obs_names = adata.obs_names.tolist()
    elif "counts" in adata.layers:
        X_full    = adata.layers["counts"]
        genes     = remap_gene_names(adata.var_names.tolist())
        obs_names = adata.obs_names.tolist()
    else:
        print("  No raw in clean atlas; loading ATLAS_RAW_PATH and matching barcodes...")
        adata_raw = anndata.read_h5ad(ATLAS_RAW_PATH)
        genes     = remap_gene_names(adata_raw.var_names.tolist())
        X_raw     = adata_raw.X
        raw_obs   = adata_raw.obs_names.tolist()
        del adata_raw

        raw_bc_set  = set(raw_obs)
        raw_short   = {}
        for rbc in raw_obs:
            parts = rbc.split("-")
            raw_short["-".join(parts[:2]) if len(parts) >= 2 else rbc] = rbc

        clean_to_raw: dict[str, str] = {}
        for cbc in adata.obs_names:
            if cbc in raw_bc_set:
                clean_to_raw[cbc] = cbc
            else:
                parts = cbc.split("-")
                key   = "-".join(parts[:2]) if len(parts) >= 2 else cbc
                if key in raw_short:
                    clean_to_raw[cbc] = raw_short[key]

        n_matched = len(clean_to_raw)
        n_total   = len(adata.obs_names)
        print(f"  Barcode match: {n_matched}/{n_total}")
        if n_matched < n_total * 0.95:
            raise RuntimeError(
                f"Atlas barcode match too low ({n_matched}/{n_total}). "
                "Check ATLAS_CLEAN_PATH vs ATLAS_RAW_PATH."
            )
        raw_idx   = {bc: i for i, bc in enumerate(raw_obs)}
        obs_names = [cbc for cbc in adata.obs_names if cbc in clean_to_raw]
        rows_raw  = [raw_idx[clean_to_raw[cbc]] for cbc in obs_names]
        X_full    = X_raw[rows_raw]

    del adata

    bc_idx  = {bc: i for i, bc in enumerate(obs_names)}
    needed  = [bc for bc in barcodes if bc in bc_idx]
    rows    = [bc_idx[bc] for bc in needed]
    X_sub   = (X_full[rows].toarray() if sp.issparse(X_full)
               else np.array(X_full)[rows]).astype(np.float32)
    del X_full
    return dict(zip(needed, _make_batched(X_sub, genes)))


def load_bpd_sentences(barcodes: list[str]) -> dict[str, str]:
    """Chunked read of compiled_counts.csv → cell sentences (avoids loading full ~6 GB dense)."""
    print("  [bpd] reading compiled_counts.csv in chunks (large file, please wait)...")
    needed_set     = set(barcodes)
    bc_to_sent:    dict[str, str] = {}
    genes:         list[str] | None = None

    for chunk in pd.read_csv(BPD_COUNTS_PATH, index_col="id", chunksize=2000):
        if genes is None:
            genes = remap_gene_names(chunk.columns.tolist())
        rows_here = [bc for bc in chunk.index if bc in needed_set]
        if not rows_here:
            continue
        X_chunk = chunk.loc[rows_here].values.astype(np.float32)
        sents   = make_cell_sentences(X_chunk, genes, seed=RANDOM_SEED)
        for bc, sent in zip(rows_here, sents):
            bc_to_sent[bc] = sent

    if genes is None:
        raise RuntimeError("BPD compiled_counts.csv appears empty or unreadable")
    missing = needed_set - set(bc_to_sent)
    if missing:
        print(f"  WARN: {len(missing)} BPD barcodes not found in compiled_counts")
    print(f"  [bpd] built {len(bc_to_sent):,} sentences")
    return bc_to_sent


def _build_pairs(
    src_bcs:       list[str],
    tgt_bcs:       list[str],
    src_sent:      dict[str, str],
    tgt_sent:      dict[str, str],
    species:       str,
    age_str_src:   str,
    pma_weeks_src: int,
    cell_type:     str,
    pert_str:      str,
    pair_type:     str,
    split:         str,
    n_per_ct:      int | None,
    n_augment:     int,
    age_str_tgt:   str | None = None,
    pma_weeks_tgt: int | None = None,
    bidirectional: bool = False,
    rev_pert_str:  str | None = None,
) -> list[dict]:
    """Build forward (src→tgt) pairs; optionally also add, for every sampled
    exposed (tgt) cell, a reverse-direction example: prompt = 'Perturbed cell:'
    (the exposed cell, same instance used in the forward completion — never a
    separately re-sampled pool), completion = age + cell type + condition. This
    does NOT try to reconstruct the unexposed cell (ill-posed); it infers age/type/
    condition from the profile. rev_pert_str overrides the forward pert_str for the
    reverse completion's Condition field when the two must differ (see
    REV_CONDITION_OVERRIDE); defaults to pert_str."""
    rev_pert_str = pert_str if rev_pert_str is None else rev_pert_str
    src = [bc for bc in src_bcs if bc in src_sent]
    tgt = [bc for bc in tgt_bcs if bc in tgt_sent]
    if not src or not tgt:
        return []

    rng = np.random.default_rng(_pair_seed(split, pair_type, cell_type))
    if n_per_ct is not None:
        if len(src) > n_per_ct:
            src = list(rng.choice(src, n_per_ct, replace=False))
        if len(tgt) > n_per_ct:
            tgt = list(rng.choice(tgt, n_per_ct, replace=False))

    ct_broad = cell_type_broad(cell_type)
    pairs: list[dict] = []
    src_a, tgt_a = list(src), list(tgt)
    for _ in range(n_augment):
        rng.shuffle(src_a)
        rng.shuffle(tgt_a)
        n = min(len(src_a), len(tgt_a))
        for i in range(n):
            src_cs = src_sent[src_a[i]]
            tgt_cs = tgt_sent[tgt_a[i]]
            prompt = build_prompt(species, age_str_src, pma_weeks_src,
                                  cell_type, ct_broad, pert_str, src_cs)
            pairs.append({
                "prompt":     prompt,
                "completion": tgt_cs,
                "pair_type":  pair_type,
            })
            if bidirectional:
                rprompt = build_prompt_reverse(species, tgt_cs)
                rcompletion = build_completion_reverse(
                    age_str_tgt, pma_weeks_tgt, cell_type, ct_broad, rev_pert_str)
                pairs.append({
                    "prompt":     rprompt,
                    "completion": rcompletion,
                    "pair_type":  pair_type + "_rev",
                })
    return pairs


class SFTBuilder:

    def __init__(self):
        print("=" * 64)
        print("STAGE 1 — SFT Dataset Builder")
        print("=" * 64)
        print("Loading registry...")
        reg = pd.read_csv(REGISTRY_PATH)
        reg.index = reg["barcode"]
        self.reg = reg


    def _paired_types(self) -> set[str]:
        """All cell_types that appear in both sides of any transition."""
        paired: set[str] = set()
        rat = self.reg[self.reg["source"] == "rat"]
        for sc, tc in [("RA", "HO"), ("HO", "AZI")]:
            paired |= set(rat[rat["condition"] == sc]["cell_type"]) & \
                      set(rat[rat["condition"] == tc]["cell_type"])
        atl = self.reg[self.reg["source"] == "atlas"]
        bpd = self.reg[self.reg["source"] == "bpd"]
        for sc, tc in HUMAN_SFT:
            sr = atl if _src_of_human(sc) == "atlas" else bpd
            tr = atl if _src_of_human(tc) == "atlas" else bpd
            paired |= set(sr[sr["condition"] == sc]["cell_type"]) & \
                      set(tr[tr["condition"] == tc]["cell_type"])
        return paired

    def check_s1_1(self, allow_unknown: bool = False) -> bool:
        """[S1-1] All paired cell_types must have a known broad category."""
        print("\n[S1-1] FINE_TO_BROAD coverage for paired types:")
        paired  = self._paired_types()
        unknown = {ct for ct in paired if cell_type_broad(ct) == "unknown"}
        known   = paired - unknown
        print(f"  OK ({len(known)}) types have known broad")
        for ct in sorted(unknown):
            print(f"  UNKNOWN: {ct!r} → add to FINE_TO_BROAD in common.py")
        if unknown and not allow_unknown:
            return True
        return False

    def check_s1_2(self):
        """[S1-2] Print paired types where cell_type_full() == raw (no expansion found)."""
        print("\n[S1-2] ABBREV_TO_FULL — paired types with no full name resolved:")
        for ct in sorted(self._paired_types()):
            if cell_type_full(ct) == ct and len(ct) <= 12:
                print(f"  {ct!r} (short, add to ABBREV_TO_FULL_EXTRA if needed)")

    def _pairing_report(self) -> pd.DataFrame:
        """[S1-0] Count paired/unpaired types per transition."""
        rows: list[dict] = []

        def _row(trans, ct, n_src, n_tgt):
            rows.append({"transition": trans, "cell_type": ct,
                         "in_src": n_src > 0, "in_tgt": n_tgt > 0,
                         "n_src": n_src, "n_tgt": n_tgt})

        rat = self.reg[self.reg["source"] == "rat"]
        for sc, tc in [("RA", "HO"), ("HO", "AZI")]:
            sr, tr = rat[rat["condition"] == sc], rat[rat["condition"] == tc]
            all_ct = set(sr["cell_type"]) | set(tr["cell_type"])
            for ct in all_ct:
                _row(f"rat_{sc}_{tc}",
                     ct, sr[sr["cell_type"]==ct].shape[0], tr[tr["cell_type"]==ct].shape[0])

        atl = self.reg[self.reg["source"] == "atlas"]
        bpd = self.reg[self.reg["source"] == "bpd"]
        for sc, tc in HUMAN_SFT:
            sr = atl[atl["condition"]==sc] if _src_of_human(sc)=="atlas" else bpd[bpd["condition"]==sc]
            tr = atl[atl["condition"]==tc] if _src_of_human(tc)=="atlas" else bpd[bpd["condition"]==tc]
            for ct in set(sr["cell_type"]) | set(tr["cell_type"]):
                _row(f"human_{sc}_{tc}",
                     ct, sr[sr["cell_type"]==ct].shape[0], tr[tr["cell_type"]==ct].shape[0])

        return pd.DataFrame(rows)


    def _rat_pairs(self, split: str, sent: dict[str, str]) -> list[dict]:
        reg_s = self.reg[(self.reg["source"]=="rat") & (self.reg["split"]==split)]
        pairs: list[dict] = []
        for sc, tc in [("RA","HO"), ("HO","AZI")]:
            pert = PERTURBATION_STR[("rat", sc, tc)]
            meta_src = COND_META[("rat", sc)]
            meta_tgt = COND_META[("rat", tc)]
            src_r = reg_s[reg_s["condition"]==sc]
            tgt_r = reg_s[reg_s["condition"]==tc]
            shared = set(src_r["cell_type"]) & set(tgt_r["cell_type"])
            pt = f"rat_{sc}_{tc}"
            for ct in sorted(shared):
                src_bcs = src_r[src_r["cell_type"]==ct].index.tolist()
                tgt_bcs = tgt_r[tgt_r["cell_type"]==ct].index.tolist()
                if split != "train":
                    aug = EVAL_AUG
                else:
                    aug = (RAT_AUG_SMALL
                           if min(len(src_bcs), len(tgt_bcs)) < RAT_AUG_THRESHOLD
                           else RAT_AUG_LARGE)
                pairs.extend(_build_pairs(
                    src_bcs, tgt_bcs, sent, sent,
                    SPECIES_STR["rat"], meta_src["age_str"], meta_src["pma_weeks"],
                    ct, pert, pt, split, n_per_ct=None, n_augment=aug,
                    age_str_tgt=meta_tgt["age_str"], pma_weeks_tgt=meta_tgt["pma_weeks"],
                    bidirectional=True,
                ))
        return pairs

    def _human_pairs(self, split: str,
                     atl_s: dict[str, str], bpd_s: dict[str, str]) -> list[dict]:
        atl = self.reg[(self.reg["source"]=="atlas") & (self.reg["split"]==split)]
        bpd = self.reg[(self.reg["source"]=="bpd")   & (self.reg["split"]==split)]
        pairs: list[dict] = []
        for sc, tc in HUMAN_SFT:
            key = ("human", sc, tc)
            if key not in PERTURBATION_STR:
                continue
            pert = PERTURBATION_STR[key]
            rev_pert = REV_CONDITION_OVERRIDE.get(key, pert)
            meta_src = COND_META[("human", sc)]
            meta_tgt = COND_META[("human", tc)]
            src_reg = atl if _src_of_human(sc)=="atlas" else bpd
            tgt_reg = atl if _src_of_human(tc)=="atlas" else bpd
            ss      = atl_s if _src_of_human(sc)=="atlas" else bpd_s
            ts      = atl_s if _src_of_human(tc)=="atlas" else bpd_s
            src_r   = src_reg[src_reg["condition"]==sc]
            tgt_r   = tgt_reg[tgt_reg["condition"]==tc]
            shared  = set(src_r["cell_type"]) & set(tgt_r["cell_type"])
            pt = f"human_{sc}_{tc}"
            for ct in sorted(shared):
                src_bcs = src_r[src_r["cell_type"]==ct].index.tolist()
                tgt_bcs = tgt_r[tgt_r["cell_type"]==ct].index.tolist()
                if split != "train":
                    aug = EVAL_AUG
                else:
                    aug = (HUMAN_AUG_SMALL
                           if min(len(src_bcs), len(tgt_bcs)) < HUMAN_AUG_THRESHOLD
                           else HUMAN_AUG_LARGE)
                pairs.extend(_build_pairs(
                    src_bcs, tgt_bcs, ss, ts,
                    SPECIES_STR["human"], meta_src["age_str"], meta_src["pma_weeks"],
                    ct, pert, pt, split, n_per_ct=None, n_augment=aug,
                    age_str_tgt=meta_tgt["age_str"], pma_weeks_tgt=meta_tgt["pma_weeks"],
                    bidirectional=True, rev_pert_str=rev_pert,
                ))
        return pairs


    @staticmethod
    def _dedup_reverse(pairs: list[dict]) -> list[dict]:
        """Drop exact-duplicate (prompt,completion) reverse examples. The reverse
        prompt/completion depends only on the target cell, not which source it
        was paired with in that augmentation round, so a target cell reused
        across multiple rounds (small-pool rat/human strata) produces
        byte-identical reverse rows. Forward examples are untouched -- their
        duplicate rate is already <1% (real accidental (src,tgt) collisions)."""
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []
        for ex in pairs:
            if not ex["pair_type"].endswith("_rev"):
                out.append(ex)
                continue
            key = (ex["prompt"], ex["completion"])
            if key in seen:
                continue
            seen.add(key)
            out.append(ex)
        return out

    @staticmethod
    def _check_s1_3(examples: list[dict], label: str) -> bool:
        """[S1-3] No old gene alias in any cell-sentence completion.
        '_rev' pair_type completions are perturbation-description text, not
        cell sentences, so they're excluded from this check."""
        cell_examples = [ex for ex in examples if not ex["pair_type"].endswith("_rev")]
        alias_set = set(ALIAS_TO_CANONICAL.keys())
        bad: dict[str, int] = {}
        for ex in cell_examples:
            for tok in ex["completion"].split():
                if tok in alias_set:
                    bad[tok] = bad.get(tok, 0) + 1
        if bad:
            print(f"  [S1-3] STOP [{label}]: old aliases in completions: "
                  f"{dict(list(bad.items())[:5])}")
            return True
        print(f"  [S1-3] ✓ [{label}] 0 old aliases in {len(cell_examples):,} cell-sentence completions")
        return False

    @staticmethod
    def _check_s1_4(examples: list[dict], label: str) -> bool:
        """[S1-4] Every cell-sentence completion == TOP_K tokens.
        '_rev' pair_type completions are perturbation-description text, not
        cell sentences, so they're excluded from this check."""
        cell_examples = [ex for ex in examples if not ex["pair_type"].endswith("_rev")]
        wrong = [(i, len(ex["completion"].split()))
                 for i, ex in enumerate(cell_examples)
                 if len(ex["completion"].split()) != TOP_K]
        if wrong:
            print(f"  [S1-4] STOP [{label}]: {len(wrong)} completions ≠ {TOP_K} tokens")
            for i, n in wrong[:5]:
                print(f"    idx={i}  tokens={n}")
            return True
        print(f"  [S1-4] ✓ [{label}] All {len(cell_examples):,} cell-sentence completions = {TOP_K} tokens")
        return False

    @staticmethod
    def _check_s1_5(rat_sent: dict[str, str], human_sent: dict[str, str]):
        """[S1-5] Rescued gene coverage in sentence pools (sample check)."""
        print("\n[S1-5] Rescued gene coverage (sample from sentence pool):")
        if rat_sent:
            tokens = {t for s in list(rat_sent.values())[:2000] for t in s.split()}
            h = len(RESCUED_RAT & tokens)
            print(f"  rat  : {h}/{len(RESCUED_RAT)} rescued genes found in sampled sentences")
        if human_sent:
            tokens = {t for s in list(human_sent.values())[:2000] for t in s.split()}
            h = len(RESCUED_HUMAN & tokens)
            print(f"  human: {h}/{len(RESCUED_HUMAN)} rescued genes found in sampled sentences")


    @staticmethod
    def _save(examples: list[dict], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps({"prompt": ex["prompt"],
                                    "completion": ex["completion"]},
                                   ensure_ascii=False) + "\n")
        print(f"  Saved: {path}  ({len(examples):,} examples)")

    def run(self):
        if self.check_s1_1():
            print("\n✗ Fix FINE_TO_BROAD in common.py and re-run.")
            sys.exit(1)
        self.check_s1_2()

        print("\n[S1-0] Building pairing coverage report...")
        cov = self._pairing_report()
        cov.to_csv(UNPAIRED_PATH, index=False)
        n_paired   = cov[cov["in_src"] & cov["in_tgt"]].shape[0]
        n_unpaired = cov[~(cov["in_src"] & cov["in_tgt"])].shape[0]
        print(f"  {n_paired} paired (src∩tgt), {n_unpaired} unpaired → {UNPAIRED_PATH}")

        print("\n--- Loading counts and building cell sentences ---")
        rat_bcs = self.reg[self.reg["source"]=="rat"].index.tolist()
        atl_bcs = self.reg[self.reg["source"]=="atlas"].index.tolist()
        bpd_bcs = self.reg[self.reg["source"]=="bpd"].index.tolist()

        rat_s = load_rat_sentences(rat_bcs)   if rat_bcs else {}
        atl_s = load_atlas_sentences(atl_bcs) if atl_bcs else {}
        bpd_s = load_bpd_sentences(bpd_bcs)   if bpd_bcs else {}

        print(f"\n  Sentences: rat={len(rat_s):,}  atlas={len(atl_s):,}  "
              f"bpd={len(bpd_s):,}")

        self._check_s1_5(rat_s, {**atl_s, **bpd_s})

        print("\n--- Building pairs per split ---")
        splits: dict[str, list[dict]] = {}
        blocking = False

        for split in ["train", "valid", "test"]:
            print(f"\n  [{split.upper()}]")
            pairs: list[dict] = []
            pairs += self._rat_pairs(split, rat_s)
            pairs += self._human_pairs(split, atl_s, bpd_s)
            n_before = len(pairs)
            pairs = self._dedup_reverse(pairs)
            n_dropped = n_before - len(pairs)
            if n_dropped:
                print(f"  [dedup] dropped {n_dropped:,} exact-duplicate reverse "
                      f"(prompt,completion) pairs")
            print(f"  Total [{split}]: {len(pairs):,} pairs")
            if pairs:
                blocking |= self._check_s1_3(pairs, split)
                blocking |= self._check_s1_4(pairs, split)
            splits[split] = pairs

        if blocking:
            print("\n✗ Blocking check(s) failed. Fix and re-run.")
            sys.exit(1)

        train_bc = set(self.reg[self.reg["split"]=="train"].index)
        test_bc  = set(self.reg[self.reg["split"]=="test"].index)
        if train_bc & test_bc:
            print(f"\n✗ [S1-8] {len(train_bc & test_bc)} barcodes in both train and test!")
            sys.exit(1)
        print(f"\n[S1-8] ✓ train ∩ test barcodes = 0")

        from collections import defaultdict
        ct_counts: dict[tuple, int] = defaultdict(int)
        for ex in splits["train"]:
            for line in ex["prompt"].splitlines():
                if line.startswith("Cell type:"):
                    ct_label = line.split("Cell type:")[1].strip()
                    break
            else:
                ct_label = "?"
            ct_counts[(ex["pair_type"], ct_label)] += 1

        all_pt = sorted({ex["pair_type"] for ex in splits["train"]})
        print("\n[S1-7] PAIR COUNTS BY TRANSITION × CELL TYPE  (train split):")
        total_train = len(splits["train"])
        for pt in all_pt:
            pt_total = sum(v for (p, _), v in ct_counts.items() if p == pt)
            pct = 100 * pt_total / total_train if total_train else 0
            print(f"\n  {pt}  [{pt_total:,} pairs, {pct:.1f}% of train]")
            ct_rows = sorted(
                ((ct, n) for (p, ct), n in ct_counts.items() if p == pt),
                key=lambda x: -x[1],
            )
            for ct, n in ct_rows:
                print(f"    {ct:<55} {n:>6}")

        for spl in ["train", "valid", "test"]:
            if splits[spl]:
                ex = splits[spl][0]
                print(f"\n[S1-6] Sample [{spl}] pair_type={ex['pair_type']}")
                print(f"  prompt[:300]: {ex['prompt'][:300]}")
                print(f"  completion[:100]: {ex['completion'][:100]}...")
                break

        print("\n--- Saving JSONL ---")
        for spl, examples in splits.items():
            self._save(examples, SFT_DIR / f"{spl}.jsonl")

        total = sum(len(v) for v in splits.values())
        print(f"\n✓ Stage 1 complete. Total examples: {total:,}")


if __name__ == "__main__":
    SFTBuilder().run()
