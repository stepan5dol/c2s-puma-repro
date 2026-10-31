#!/usr/bin/env python3
"""pipeline/common.py — shared constants and utilities for all pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# ══════════════════════════════════════════════════════════════════
# ПУТИ
# ══════════════════════════════════════════════════════════════════
BASE = Path("/Users/stepandolzhenko/Documents/AzithroGemma")

RAT_PATH          = BASE / "rat.ho.azi.integrated.h5ad"
ATLAS_CLEAN_PATH  = BASE / "he_lung_atlas.h5ad"
ATLAS_RAW_PATH    = BASE / "2022FetalLungIntCounts.h5ad"
BPD_META_PATH     = BASE / "BPD-PH/GSE275938_cell_metadata.csv"
BPD_COUNTS_PATH   = BASE / "BPD-PH/GSE275938_compiled_counts.csv"
MOUSE_META_PATH   = BASE / "GSE151974_RAW/GSE151974_cell_metadata_postfilter.csv"
MOUSE_COUNTS_PATH = BASE / "GSE151974_RAW/GSE151974_raw_umi_matrix_postfilter.csv.gz"

RESCUED_RAT_PATH   = BASE / "train-after-grpo-analysis/rescued_rat_genes.txt"
RESCUED_HUMAN_PATH = BASE / "train-after-grpo-analysis/rescued_human_genes.txt"

CONFIG_PATH  = BASE / "scripts/cell_types_config.json"
PIPELINE_DIR = BASE / "pipeline_short"

TOP_K       = 800
RANDOM_SEED = 42
N_PER_CT    = 50
N_AUGMENT   = 2

# ══════════════════════════════════════════════════════════════════
# TARGET CELL TYPES — minimally-viable scope: matches GRPO_ENDO_RAT /
# GRPO_ENDO_HUMAN exactly (pipeline/stage2_grpo_pairs.py) so SFT and GRPO
# share the same cell-type scope. Canonical (post-harmonize) labels.
# ══════════════════════════════════════════════════════════════════
TARGET_RAT_CELL_TYPES: set[str] = {"gCAP", "aCAP", "Peri", "VEC"}
TARGET_HUMAN_CELL_TYPES: set[str] = {
    "gCap", "aCap", "Pulmonary venous EC", "Pericyte",
}
TARGET_RAT_CONDITIONS: set[str] = {"RA", "HO", "AZI"}
TARGET_HUMAN_CONDITIONS: set[str] = {"He22", "Acute26", "BPD7mo", "BPDPH7mo"}

# ══════════════════════════════════════════════════════════════════
# GENE ALIASES (устаревшие → канонические)
# Регистр генов НЕ меняем: крыса/мышь title-case, человек upper.
# ══════════════════════════════════════════════════════════════════
GENE_ALIASES: dict[str, list[str]] = {
    "H1-0":      ["H1F0"],
    "H1-5":      ["H1F5", "HIST1H1B", "H1B"],
    "H2AX":      ["H2AFX"],
    "H3-3A":     ["H3F3A"],
    "H3-3B":     ["H3F3B"],
    "H4C14":     ["HIST2H4", "HIST2H4A", "HIST2H4B"],
    "MT-CO2":    ["MTCO2"],
    "HIST1H2BQ": ["HIST1H2BH", "HIST1H2BF", "H2BC9", "H2BC7"],
    "CNTNAP5A":  ["CNTNAP5"],
    "GARS1":     ["GARS"],
    "DARS1":     ["DARS"],
    "SARS1":     ["SARS"],
    "TARS1":     ["TARS"],
    "EPRS1":     ["EPRS"],
    "BBLN":      ["C9orf16"],
    "CCN2":      ["CTGF"],
    "SEPTIN4":   ["SEPT4", "ARTS", "PNUTL2"],
    "MICOS10":   ["MINOS1", "C1orf151", "MIC10"],
    "MICOS13":   ["QIL1", "C19orf70", "MIC13"],
    "RPL30L2":   ["RPL30"],
    "NUPR1L1":   ["NUPR1"],
    "SNRPEL1":   ["SNRPE"],
}

# Заглушка — пополняется после Stage 2 [S2-8] аудита при необходимости
GENE_ALIASES_EXTRA: dict[str, list[str]] = {}

ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _aliases in {**GENE_ALIASES, **GENE_ALIASES_EXTRA}.items():
    for _alias in _aliases:
        ALIAS_TO_CANONICAL[_alias] = _canon


def remap_gene_names(gene_names) -> list[str]:
    return [ALIAS_TO_CANONICAL.get(g, g) for g in gene_names]


# ══════════════════════════════════════════════════════════════════
# RESCUED GENES (выверенные .txt, 159 rat / 161 human)
# ══════════════════════════════════════════════════════════════════
RESCUED_RAT   = set(RESCUED_RAT_PATH.read_text().splitlines()) - {""}
RESCUED_HUMAN = set(RESCUED_HUMAN_PATH.read_text().splitlines()) - {""}

# ══════════════════════════════════════════════════════════════════
# ABBREV_TO_FULL — аббревиатура → читаемое имя типа клетки
# Хранится с ключами в ВЕРХНЕМ регистре; lookup через cell_type_full(ct).
# .upper() применяется ТОЛЬКО к ключам словаря, никогда к генам.
# ══════════════════════════════════════════════════════════════════
_ABBREV_TO_FULL_RAW: dict[str, str] = {
    # --- Rat ---
    "AM":        "alveolar macrophage",
    "AEC":       "arterial endothelial cell",
    "aCAP":      "aerocyte capillary endothelial cell",
    "gCAP":      "general capillary endothelial cell",
    "AF1":       "adventitial fibroblast type 1",
    "AF2":       "adventitial fibroblast type 2",
    "AT1":       "alveolar type 1 cell",
    "AT2":       "alveolar type 2 cell",
    "B cell":    "B cell",
    "Ciliated":  "ciliated cell",
    "Club":      "club cell",
    "DC":        "dendritic cell",
    "IM":        "interstitial macrophage",
    "LEC":       "lymphatic endothelial cell",
    "Meso":      "mesothelial cell",
    "Mono":      "monocyte",
    "Myofib":    "myofibroblast",
    "Neu":       "neutrophil",
    "Peri":      "pericyte",
    "SMC":       "smooth muscle cell",
    "T cell":    "T cell",
    "VEC":       "vascular endothelial cell",
    "aCap":      "aerocyte capillary endothelial cell",
    "gCap":      "general capillary endothelial cell",
    # --- BPD canonical (посимвольно из GSE275938_cell_metadata.csv) ---
    "Alveolar FB":            "alveolar fibroblast",
    "Adventitial FB":         "adventitial fibroblast",
    "Alveolar MyoFB":         "alveolar myofibroblast",
    "Ductal MyoFB":           "ductal myofibroblast",
    "VSMC":                   "vascular smooth muscle cell",
    "Pericyte":               "pericyte",
    "Arterial EC":            "arterial endothelial cell",
    "Pulmonary venous EC":    "pulmonary venous endothelial cell",
    "Systemic venous EC":     "systemic venous endothelial cell",
    "Lymphatic":              "lymphatic endothelial cell",
    "abCap":                  "aberrant capillary endothelial cell",
    "Multiciliated":          "multiciliated cell",
    "Basal":                  "basal cell",
    "RASC":                   "respiratory airway secretory cell",
    "Secretory MUC5B":        "MUC5B+ secretory cell",
    "Secretory -3A1, -3A2":   "SCGB3A1/3A2+ secretory cell",
    "Alveolar Macrophage":    "alveolar macrophage",
    "Monocyte":               "monocyte",
    "cDC":                    "conventional dendritic cell",
    "pDC":                    "plasmacytoid dendritic cell",
    "Neutrophil":             "neutrophil",
    "Basophil":               "basophil",
    "Mast cell":              "mast cell",
    "T Cell":                 "T cell",
    "B Cell":                 "B cell",
    "NK Cell":                "NK cell",
    "NKT Cell":               "NKT cell",
    "Plasma cell":            "plasma cell",
    "Activated FB":           "activated fibroblast",
    # --- Atlas (He et al.) ---
    "Aerocyte":                      "aerocyte capillary endothelial cell",
    "Early cap":                     "early capillary endothelial cell",
    "Mid cap":                       "mid capillary endothelial cell",
    "Late cap":                      "late capillary endothelial cell",
    "Arterial endo":                 "arterial endothelial cell",
    "GRIA2+ arterial endo":          "GRIA2+ arterial endothelial cell",
    "Venous endo":                   "venous endothelial cell",
    "OMD+ endo":                     "OMD+ endothelial cell",
    "Lymphatic endo":                "lymphatic endothelial cell",
    "Intermediate lymphatic endo":   "intermediate lymphatic endothelial cell",
    "SCG3+ lymphatic endothelial":   "SCG3+ lymphatic endothelial cell",
    "MUC16+ ciliated":               "MUC16+ ciliated cell",
    "Deuterosomal":                  "deuterosomal cell",
    "Proximal basal":                "proximal basal cell",
    "Mid basal":                     "mid basal cell",
    "Late basal":                    "late basal cell",
    "SMG basal":                     "submucosal gland basal cell",
    "SMG":                           "submucosal gland cell",
    "Proximal secretory 1":          "proximal secretory cell type 1",
    "Proximal secretory 2":          "proximal secretory cell type 2",
    "Proximal secretory 3":          "proximal secretory cell type 3",
    "Proximal secretory progenitors":"proximal secretory progenitor cell",
    "Early tip":                     "early tip cell",
    "Mid tip":                       "mid tip cell",
    "Late tip":                      "late tip cell",
    "Early stalk":                   "early stalk cell",
    "Mid stalk":                     "mid stalk cell",
    "Late stalk":                    "late stalk cell",
    "Adventitial fibro":             "adventitial fibroblast",
    "Alveolar fibro":                "alveolar fibroblast",
    "Airway fibro":                  "airway fibroblast",
    "Early fibro":                   "early fibroblast",
    "Mid fibro":                     "mid fibroblast",
    "Interm fibro":                  "interstitial fibroblast",
    "Mesenchymal 1":                 "mesenchymal cell type 1",
    "Mesenchymal 2":                 "mesenchymal cell type 2",
    "Mesenchymal 3":                 "mesenchymal cell type 3",
    "Myofibro 1":                    "myofibroblast type 1",
    "Myofibro 2":                    "myofibroblast type 2",
    "Myofibro 3":                    "myofibroblast type 3",
    "Vascular SMC 1":                "vascular smooth muscle cell type 1",
    "Vascular SMC 2":                "vascular smooth muscle cell type 2",
    "MYL4+ SMC":                     "MYL4+ smooth muscle cell",
    "ACTC+ SMC":                     "ACTC+ smooth muscle cell",
    "Late airway SMC":               "late airway smooth muscle cell",
    "Mid airway SMC 1":              "mid airway smooth muscle cell type 1",
    "Mid airway SMC 2":              "mid airway smooth muscle cell type 2",
    "APOE+ MΦ1":                    "APOE+ macrophage type 1",
    "APOE+ MΦ2":                    "APOE+ macrophage type 2",
    "SPP1+ MΦ":                     "SPP1+ macrophage",
    "CX3CR1+ MΦ":                   "CX3CR1+ macrophage",
    "CXCL9+ MΦ":                    "CXCL9+ macrophage",
    "Non-cla. mono.":                "non-classical monocyte",
    "S100A12-hi cla. mono.":         "S100A12-hi classical monocyte",
    "S100A12-lo cla. mono.":         "S100A12-lo classical monocyte",
    "Promonocyte-like":              "promonocyte-like cell",
    "DC1":                           "dendritic cell type 1",
    "DC2":                           "dendritic cell type 2",
    "DC3":                           "dendritic cell type 3",
    "aDC 1":                         "activated dendritic cell type 1",
    "aDC 2":                         "activated dendritic cell type 2",
    "Cycling DC":                    "cycling dendritic cell",
    "pre-pDC/DC5":                   "pre-plasmacytoid dendritic cell / DC5",
    "Promyelocyte-like":             "promyelocyte-like cell",
    "Myelocyte-like":                "myelocyte-like cell",
    "Mast":                          "mast cell",
    "CD4 T":                         "CD4+ T cell",
    "CD8 T":                         "CD8+ T cell",
    "Cycling T":                     "cycling T cell",
    "Th17":                          "Th17 cell",
    "Treg":                          "regulatory T cell",
    "Tαβ_Entry":                     "Tαβ entry T cell",
    "NKT1":                          "NKT cell type 1",
    "NKT2":                          "NKT cell type 2",
    "Activated NK":                  "activated NK cell",
    "CD16+ NK":                      "CD16+ NK cell",
    "CD56bright NK":                 "CD56bright NK cell",
    "Cycling NK":                    "cycling NK cell",
    "Intermediate NK":               "intermediate NK cell",
    "CD5+ CCL22+ mature B":          "CD5+ CCL22+ mature B cell",
    "CD5+ CCL22- mature B":          "CD5+ CCL22- mature B cell",
    "CD5- Mature B":                 "CD5- mature B cell",
    "Immature B":                    "immature B cell",
    # --- Mouse ---
    "Col13a1+ fibroblast": "Col13a1+ fibroblast",
    "Col14a1+ fibroblast": "Col14a1+ fibroblast",
    "Pericyte 1":          "pericyte type 1",
    "Pericyte 2":          "pericyte type 2",
    "Neut 1":              "neutrophil type 1",
    "Neut 2":              "neutrophil type 2",
    "B cell 1":            "B cell type 1",
    "B cell 2":            "B cell type 2",
    "CD4 T cell 1":        "CD4+ T cell type 1",
    "CD4 T cell 2":        "CD4+ T cell type 2",
    "CD8 T cell 1":        "CD8+ T cell type 1",
    "CD8 T cell 2":        "CD8+ T cell type 2",
    "NK cell":             "NK cell",
    "gd T cell":           "gamma-delta T cell",
    "Mast Ba2":            "mast cell",
    "AT2 1":               "alveolar type 2 cell type 1",
    "AT2 2":               "alveolar type 2 cell type 2",
    "Mesothelial":         "mesothelial cell",
    # --- misc ---
    "Endo":      "endothelial cell",
    "Lymph":     "lymphatic endothelial cell",
    "Cap":       "capillary endothelial cell",
    "Cap-a":     "aerocyte capillary endothelial cell",
    "Art":       "arterial endothelial cell",
    "Vein":      "venous endothelial cell",
    "Alv Mf":   "alveolar macrophage",
    "Int Mf":   "interstitial macrophage",
    "ILC2":     "type 2 innate lymphoid cell",
    "ILC3":     "type 3 innate lymphoid cell",
    "ILCP":     "innate lymphoid cell progenitor",
    "CMP":      "common myeloid progenitor",
    "GMP":      "granulocyte-monocyte progenitor",
    "MEP":      "megakaryocyte-erythroid progenitor",
    "HSC":      "hematopoietic stem cell",
    "HSC/ELP":  "hematopoietic stem cell / early lymphoid progenitor",
}

ABBREV_TO_FULL: dict[str, str] = {k.upper(): v for k, v in _ABBREV_TO_FULL_RAW.items()}

# Заглушка — пополняется после [S1-2]/[S3-6] аудита
ABBREV_TO_FULL_EXTRA: dict[str, str] = {}


def cell_type_full(ct: str) -> str:
    """Resolve cell type name/abbreviation to human-readable form."""
    key = ct.upper()
    return ABBREV_TO_FULL.get(key, ABBREV_TO_FULL_EXTRA.get(key, ct))


# ══════════════════════════════════════════════════════════════════
# FINE_TO_BROAD — тип клетки → широкая категория для промпта
# Ключи в ВЕРХНЕМ регистре; lookup через cell_type_broad(ct).
# ══════════════════════════════════════════════════════════════════
_FINE_TO_BROAD_RAW: dict[str, str] = {
    # --- Rat ---
    "AT1": "epithelial", "AT2": "epithelial",
    "Club": "epithelial", "Ciliated": "epithelial",
    "AEC": "endothelial",
    "aCAP": "endothelial", "gCAP": "endothelial",
    "aCap": "endothelial", "gCap": "endothelial",
    "VEC": "endothelial", "LEC": "endothelial", "Endo": "endothelial",
    "AF1": "fibroblast", "AF2": "fibroblast",
    "Myofib": "fibroblast", "Meso": "mesothelial",
    "Peri": "mural", "SMC": "mural",
    "AM": "macrophage", "IM": "macrophage",
    "DC": "myeloid", "Mono": "myeloid", "Neu": "myeloid",
    "B cell": "lymphocyte", "T cell": "lymphocyte",
    # --- BPD canonical ---
    "aCap": "endothelial", "gCap": "endothelial",
    "abCap": "endothelial",
    "Arterial EC": "endothelial",
    "Pulmonary venous EC": "endothelial",
    "Systemic venous EC": "endothelial",
    "Lymphatic": "endothelial",
    "AT1": "epithelial", "AT2": "epithelial",
    "Multiciliated": "epithelial",
    "Basal": "epithelial",
    "RASC": "epithelial",
    "Secretory MUC5B": "epithelial",
    "Secretory -3A1, -3A2": "epithelial",
    "Alveolar FB": "fibroblast",
    "Adventitial FB": "fibroblast",
    "Alveolar MyoFB": "fibroblast",
    "Ductal MyoFB": "fibroblast",
    "VSMC": "mural",
    "Pericyte": "mural",
    "Alveolar Macrophage": "macrophage",
    "Monocyte": "myeloid",
    "cDC": "myeloid",
    "pDC": "myeloid",
    "Neutrophil": "myeloid",
    "Basophil": "myeloid",
    "Mast cell": "myeloid",
    "T Cell": "lymphocyte",
    "B Cell": "lymphocyte",
    "NK Cell": "lymphocyte",
    "NKT Cell": "lymphocyte",
    "Plasma cell": "lymphocyte",
    "Activated FB": "fibroblast",
    # --- Atlas (He et al.) ---
    "Aerocyte": "endothelial",
    "Early cap": "endothelial", "Mid cap": "endothelial", "Late cap": "endothelial",
    "Arterial endo": "endothelial", "GRIA2+ arterial endo": "endothelial",
    "Venous endo": "endothelial", "OMD+ endo": "endothelial",
    "Lymphatic endo": "endothelial", "Intermediate lymphatic endo": "endothelial",
    "SCG3+ lymphatic endothelial": "endothelial",
    "MUC16+ ciliated": "epithelial", "Deuterosomal": "epithelial",
    "Proximal basal": "epithelial", "Mid basal": "epithelial",
    "Late basal": "epithelial", "SMG basal": "epithelial",
    "SMG": "epithelial",
    "Proximal secretory 1": "epithelial", "Proximal secretory 2": "epithelial",
    "Proximal secretory 3": "epithelial", "Proximal secretory progenitors": "epithelial",
    "Early tip": "epithelial", "Mid tip": "epithelial", "Late tip": "epithelial",
    "Early stalk": "epithelial", "Mid stalk": "epithelial", "Late stalk": "epithelial",
    "Adventitial fibro": "fibroblast", "Alveolar fibro": "fibroblast",
    "Airway fibro": "fibroblast",
    "Early fibro": "fibroblast", "Mid fibro": "fibroblast", "Interm fibro": "fibroblast",
    "Mesenchymal 1": "fibroblast", "Mesenchymal 2": "fibroblast", "Mesenchymal 3": "fibroblast",
    "Myofibro 1": "fibroblast", "Myofibro 2": "fibroblast", "Myofibro 3": "fibroblast",
    "Vascular SMC 1": "mural", "Vascular SMC 2": "mural",
    "MYL4+ SMC": "mural", "ACTC+ SMC": "mural",
    "Late airway SMC": "mural", "Mid airway SMC 1": "mural", "Mid airway SMC 2": "mural",
    "APOE+ MΦ1": "macrophage", "APOE+ MΦ2": "macrophage",
    "SPP1+ MΦ": "macrophage", "CX3CR1+ MΦ": "macrophage", "CXCL9+ MΦ": "macrophage",
    "Non-cla. mono.": "myeloid", "S100A12-hi cla. mono.": "myeloid",
    "S100A12-lo cla. mono.": "myeloid", "Promonocyte-like": "myeloid",
    "DC1": "myeloid", "DC2": "myeloid", "DC3": "myeloid",
    "aDC 1": "myeloid", "aDC 2": "myeloid", "Cycling DC": "myeloid",
    "pre-pDC/DC5": "myeloid",
    "Promyelocyte-like": "myeloid", "Myelocyte-like": "myeloid",
    "Mast": "myeloid",
    "CD4 T": "lymphocyte", "CD8 T": "lymphocyte", "Cycling T": "lymphocyte",
    "Th17": "lymphocyte", "Treg": "lymphocyte", "Tαβ_Entry": "lymphocyte",
    "NKT1": "lymphocyte", "NKT2": "lymphocyte",
    "Activated NK": "lymphocyte", "CD16+ NK": "lymphocyte",
    "CD56bright NK": "lymphocyte", "Cycling NK": "lymphocyte", "Intermediate NK": "lymphocyte",
    "CD5+ CCL22+ mature B": "lymphocyte", "CD5+ CCL22- mature B": "lymphocyte",
    "CD5- Mature B": "lymphocyte", "Immature B": "lymphocyte",
    # --- Mouse ---
    "Col13a1+ fibroblast": "fibroblast", "Col14a1+ fibroblast": "fibroblast",
    "Pericyte 1": "mural", "Pericyte 2": "mural",
    "Neut 1": "myeloid", "Neut 2": "myeloid",
    "B cell 1": "lymphocyte", "B cell 2": "lymphocyte",
    "CD4 T cell 1": "lymphocyte", "CD4 T cell 2": "lymphocyte",
    "CD8 T cell 1": "lymphocyte", "CD8 T cell 2": "lymphocyte",
    "NK cell": "lymphocyte", "gd T cell": "lymphocyte",
    "Mast Ba2": "myeloid",
    "AT2 1": "epithelial", "AT2 2": "epithelial",
    "Mesothelial": "mesothelial",
    # --- misc ---
    "Art": "endothelial", "Cap": "endothelial", "Cap-a": "endothelial",
    "Lymph": "endothelial", "Vein": "endothelial",
    "Myofibroblast": "fibroblast",
    "Alv Mf": "macrophage", "Int Mf": "macrophage",
    "ILC2": "lymphocyte", "ILC3": "lymphocyte", "ILCP": "lymphocyte",
    "Chondrocyte": "other", "PNS": "other",
}

FINE_TO_BROAD: dict[str, str] = {k.upper(): v for k, v in _FINE_TO_BROAD_RAW.items()}

# Заглушка — пополняется после [S1-1]/[S3-6] аудита
FINE_TO_BROAD_EXTRA: dict[str, str] = {}


def cell_type_broad(ct: str) -> str:
    """Resolve cell type to broad category (endothelial, epithelial, ...)."""
    key = ct.upper()
    return FINE_TO_BROAD.get(key, FINE_TO_BROAD_EXTRA.get(key, "unknown"))


# ══════════════════════════════════════════════════════════════════
# CONDITION MAPPINGS — сырые метки → канонические condition-ключи
# ══════════════════════════════════════════════════════════════════
DATASET_TO_CONDITION: dict[str, str] = {
    "Term infant 1":          "Term0d",
    "Term infant 2":          "Term20d",
    "Acute preterm injury 1": "Acute26",
    "BPD 1":                  "BPD7mo",
    "BPD 2":                  "BPD7mo",
    "BPD+PH 1":               "BPDPH7mo",
    "BPD+PH 2":               "BPDPH7mo",
}

HE_STAGE_TO_CONDITION: dict[float, str] = {
    15.0: "He15",
    18.0: "He18",
    20.0: "He20",
    22.0: "He22",
}

SPECIES_STR: dict[str, str] = {
    "rat":   "Rattus norvegicus",
    "mouse": "Mus musculus",
    "human": "Homo sapiens",
}

# ══════════════════════════════════════════════════════════════════
# CONDITION METADATA — age_str и pma_weeks по canonical condition
# ══════════════════════════════════════════════════════════════════
COND_META: dict[tuple, dict] = {
    ("rat",   "RA"):      {"age_str": "postnatal day 14",                                       "pma_weeks": 52},
    ("rat",   "HO"):      {"age_str": "postnatal day 14",                                       "pma_weeks": 52},
    ("rat",   "AZI"):     {"age_str": "postnatal day 14",                                       "pma_weeks": 52},
    ("human", "He15"):    {"age_str": "gestational week 15",                                    "pma_weeks": 15},
    ("human", "He18"):    {"age_str": "gestational week 18",                                    "pma_weeks": 18},
    ("human", "He20"):    {"age_str": "gestational week 20",                                    "pma_weeks": 20},
    ("human", "He22"):    {"age_str": "gestational week 22",                                    "pma_weeks": 22},
    ("human", "Acute26"): {"age_str": "gestational week 26 (acute preterm injury)",             "pma_weeks": 26},
    ("human", "Term0d"):  {"age_str": "40 weeks, postnatal day 0",                             "pma_weeks": 40},
    ("human", "Term20d"): {"age_str": "43 weeks, postnatal day 20",                            "pma_weeks": 43},
    ("human", "BPD7mo"):  {"age_str": "7 months postnatal (BPD)",                              "pma_weeks": 70},
    ("human", "BPDPH7mo"):{"age_str": "7 months postnatal (BPD with pulmonary hypertension)",  "pma_weeks": 70},
    ("mouse", "P3"):      {"age_str": "postnatal day 3",                                        "pma_weeks": 32},
    ("mouse", "P7"):      {"age_str": "postnatal day 7",                                        "pma_weeks": 40},
    ("mouse", "P14"):     {"age_str": "postnatal day 14",                                       "pma_weeks": 52},
}

# ══════════════════════════════════════════════════════════════════
# PERTURBATION_STR — ЕДИНАЯ таблица для SFT и GRPO
# Ключ: (species, src_condition, tgt_condition) — канонические строки.
# Мышь: (species, src_oxy, src_age, tgt_oxy, tgt_age) — уже канональны.
# ══════════════════════════════════════════════════════════════════
PERTURBATION_STR: dict[tuple, str] = {
    # --- Rat ---
    ("rat", "RA",  "HO"):  "Hyperoxia exposure from birth to postnatal day 14.",
    ("rat", "HO",  "AZI"): "Azithromycin treatment (30 mg/kg IP at P7, P10, P13) during after exposure to 85% O2 for 14 days.",

    # --- Human fetal (atlas temporal) ---
    ("human", "He15", "He18"): "Three weeks of fetal lung development (GW15 → GW18).",
    ("human", "He15", "He20"): "Five weeks of fetal lung development (GW15 → GW20).",
    ("human", "He15", "He22"): "Seven weeks of fetal lung development (GW15 → GW22).",
    ("human", "He18", "He20"): "Two weeks of fetal lung development (GW18 → GW20).",
    ("human", "He18", "He22"): "Four weeks of fetal lung development (GW18 → GW22).",
    ("human", "He20", "He22"): "Two weeks of fetal lung development (GW20 → GW22).",

    # --- Human disease trajectory (cross-source) ---
    ("human", "He22", "Acute26"): (
        "Acute preterm lung injury at gestational week 26, "
        "compared to normal late canalicular fetal lung (GW22)."
    ),
    ("human", "He22", "Term0d"): (
        "Healthy term birth at 40 weeks, "
        "compared to late canalicular fetal lung (GW22)."
    ),
    ("human", "Acute26", "BPD7mo"): (
        "Progression from acute preterm lung injury (GW26) "
        "to established bronchopulmonary dysplasia (7 months postnatal)."
    ),
    ("human", "Acute26", "BPDPH7mo"): (
        "Disease progression to BPD with pulmonary hypertension."
    ),

    # --- Mouse (SFT only) ---
    ("mouse", "Normoxia",  "P3",  "Normoxia",  "P7"):  "Four days of normal postnatal lung development in mouse (P3 → P7).",
    ("mouse", "Normoxia",  "P7",  "Normoxia",  "P14"): "Seven days of normal postnatal lung development in mouse (P7 → P14).",
    ("mouse", "Normoxia",  "P3",  "Hyperoxia", "P3"):  "Hyperoxia exposure from birth to postnatal day 3 in mouse.",
    ("mouse", "Normoxia",  "P7",  "Hyperoxia", "P7"):  "Hyperoxia exposure from birth to postnatal day 7 in mouse.",
    ("mouse", "Normoxia",  "P14", "Hyperoxia", "P14"): "Hyperoxia exposure from birth to postnatal day 14 in mouse.",
    ("mouse", "Hyperoxia", "P3",  "Hyperoxia", "P7"):  "Four days of continued hyperoxia exposure in mouse (P3 → P7).",
    ("mouse", "Hyperoxia", "P7",  "Hyperoxia", "P14"): "Seven days of continued hyperoxia exposure in mouse (P7 → P14).",
}

# AZI counterfactual — inference-only Stage 4 (human clinical translation)
AZI_PERTURBATION = (
    "Oral azithromycin treatment (5 mg/kg of body weight, 3 times per week). "
    "Administered as a low maintenance dose for a duration of 10 weeks."
)

# ══════════════════════════════════════════════════════════════════
# HARMONIZE — сырые atlas-метки → канонический cell_type
# Значения посимвольно из GSE275938_cell_metadata.csv.
# BPD-метки (already canonical) — passthrough, не входят в словарь.
# Rat-метки — не трогаем.
# ══════════════════════════════════════════════════════════════════
HARMONIZE: dict[str, str] = {
    # --- Endothelial ---
    "Aerocyte":                    "aCap",
    "Early cap":                   "gCap",
    "Mid cap":                     "gCap",
    "Late cap":                    "gCap",
    "Arterial endo":               "Arterial EC",
    "GRIA2+ arterial endo":        "Arterial EC",
    "Venous endo":                 "Pulmonary venous EC",
    "Lymphatic endo":              "Lymphatic",
    "Intermediate lymphatic endo": "Lymphatic",
    "SCG3+ lymphatic endothelial": "Lymphatic",
    # "OMD+ endo": нет BPD-аналога → не маппим

    # --- Epithelial ---
    "AT1":                         "AT1",
    "AT2":                         "AT2",
    "Ciliated":                    "Multiciliated",
    "MUC16+ ciliated":             "Multiciliated",
    "Deuterosomal":                "Multiciliated",
    "Proximal basal":              "Basal",
    "Mid basal":                   "Basal",
    "Late basal":                  "Basal",
    "SMG basal":                   "Basal",
    "Club":                        "Secretory MUC5B",
    "Proximal secretory 1":        "Secretory MUC5B",
    "Proximal secretory 2":        "Secretory -3A1, -3A2",
    "Proximal secretory 3":        "Secretory -3A1, -3A2",
    "Proximal secretory progenitors": "RASC",
    "SMG":                         "Secretory MUC5B",
    "Early tip":                   "AT2",
    "Mid tip":                     "AT2",
    "Late tip":                    "AT2",
    "Early stalk":                 "Basal",
    "Mid stalk":                   "Basal",
    "Late stalk":                  "Basal",
    # без BPD-аналога: airway progenitor*, NE/neuroendocrine*, MUC5AC+ ASCL1+, Squamous

    # --- Immune ---
    "APOE+ MΦ1":             "Alveolar Macrophage",
    "APOE+ MΦ2":             "Alveolar Macrophage",
    "SPP1+ MΦ":              "Alveolar Macrophage",
    "CX3CR1+ MΦ":            "Alveolar Macrophage",
    "CXCL9+ MΦ":             "Alveolar Macrophage",
    "Non-cla. mono.":        "Monocyte",
    "S100A12-hi cla. mono.": "Monocyte",
    "S100A12-lo cla. mono.": "Monocyte",
    "Promonocyte-like":      "Monocyte",
    "DC1":                   "cDC",
    "DC2":                   "cDC",
    "DC3":                   "cDC",
    "aDC 1":                 "cDC",
    "aDC 2":                 "cDC",
    "Cycling DC":            "cDC",
    "pDC":                   "pDC",
    "pre-pDC/DC5":           "pDC",
    "Mast":                  "Mast cell",
    "Neutrophil":            "Neutrophil",
    "Promyelocyte-like":     "Neutrophil",
    "Myelocyte-like":        "Neutrophil",
    "CD4 T":                 "T Cell",
    "CD8 T":                 "T Cell",
    "Cycling T":             "T Cell",
    "Th17":                  "T Cell",
    "Treg":                  "T Cell",
    "Tαβ_Entry":             "T Cell",
    "NKT1":                  "NKT Cell",
    "NKT2":                  "NKT Cell",
    "Activated NK":          "NK Cell",
    "CD16+ NK":              "NK Cell",
    "CD56bright NK":         "NK Cell",
    "Cycling NK":            "NK Cell",
    "Intermediate NK":       "NK Cell",
    "Late pre-B":            "B Cell",
    "Large pre-B":           "B Cell",
    "λ small pre-B":         "B Cell",
    "κ small pre-B":         "B Cell",
    "Late pro-B":            "B Cell",
    "Pro-B":                 "B Cell",
    "Pro-B/Pre-B transition": "B Cell",
    "Immature B":            "B Cell",
    "CD5+ CCL22+ mature B":  "B Cell",
    "CD5+ CCL22- mature B":  "B Cell",
    "CD5- Mature B":         "B Cell",
    # без BPD-аналога: Eosinophil, HSC*, CMP/GMP/MEP, ILC2/3/ILCP,
    #   эритроид/ретикулоцит/мегакариоцит, Schwann*, мезотелий*, прогениторы дых. путей

    # --- Mesenchymal ---
    "Adventitial fibro":  "Adventitial FB",
    "Alveolar fibro":     "Alveolar FB",
    "Airway fibro":       "Adventitial FB",
    "Early fibro":        "Alveolar FB",
    "Mid fibro":          "Alveolar FB",
    "Interm fibro":       "Alveolar FB",
    "Mesenchymal 1":      "Alveolar FB",
    "Mesenchymal 2":      "Alveolar FB",
    "Mesenchymal 3":      "Alveolar FB",
    "Myofibro 1":         "Ductal MyoFB",
    "Myofibro 2":         "Ductal MyoFB",
    "Myofibro 3":         "Alveolar MyoFB",
    "Pericyte":           "Pericyte",
    "Vascular SMC 1":     "VSMC",
    "Vascular SMC 2":     "VSMC",
    "MYL4+ SMC":          "VSMC",
    "ACTC+ SMC":          "VSMC",
    "Late airway SMC":    "VSMC",
    "Mid airway SMC 1":   "VSMC",
    "Mid airway SMC 2":   "VSMC",
    # без BPD-аналога: Chondrocyte*, Mesothelial*, Schwann*, нейроны, NE, прогениторы дых. путей
}


def harmonize_cell_type(raw_ct: str) -> str:
    """Map raw atlas label to canonical BPD cell_type.
    BPD and rat labels pass through unchanged (not in HARMONIZE).
    """
    return HARMONIZE.get(raw_ct, raw_ct)


# ══════════════════════════════════════════════════════════════════
# GRPO-ЭНДО WHITELIST (из config ot_include)
# ══════════════════════════════════════════════════════════════════

def load_grpo_endo_types() -> dict[str, set[str]]:
    """Return {species: set_of_canonical_cell_types} for GRPO endothelial whitelist."""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    rat_types   = set(cfg["RAT"]["ot_include"])
    atlas_raw   = set(cfg["ATLAS_He"]["ot_include"])
    human_types = {harmonize_cell_type(t) for t in atlas_raw}
    return {"rat": rat_types, "human": human_types}


# ══════════════════════════════════════════════════════════════════
# MAKE CELL SENTENCES
# Единственная функция для всех источников (raw integer counts).
# normalize_total(1e4) → log1p → tie-break noise → argsort desc → top-K
# ══════════════════════════════════════════════════════════════════

def make_cell_sentences(
    X: np.ndarray,
    gene_names: list[str],
    top_k: int = TOP_K,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Raw counts → normalize 1e4 → log1p → rank desc → top_k gene symbols per cell."""
    adata_tmp = anndata.AnnData(X=sp.csr_matrix(X.astype(np.float32)))
    adata_tmp.var_names = pd.Index(gene_names)
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    X_norm = adata_tmp.X
    if sp.issparse(X_norm):
        X_norm = X_norm.toarray()
    X_norm = X_norm.astype(np.float32)
    local_rng = np.random.default_rng(seed)
    noise = local_rng.uniform(0, 1e-8, X_norm.shape).astype(np.float32)
    sorted_idx = np.argsort(-(X_norm + noise), axis=1)
    gene_arr = np.array(gene_names)
    return [" ".join(gene_arr[row[:top_k]]) for row in sorted_idx]


# ══════════════════════════════════════════════════════════════════
# BUILD PROMPT
# ══════════════════════════════════════════════════════════════════

def build_prompt(
    species: str,
    age_str: str,
    pma_weeks: int | float,
    cell_type_fine: str,
    cell_type_broad_str: str,
    perturbation_str: str,
    unexposed_cs: str,
) -> str:
    ct_full = cell_type_full(cell_type_fine)
    return (
        "Determine the single cell's expression changes, listed in 'Unexposed:', "
        "under the outlined conditions after exposure to the specified perturbation, "
        "and generate the cell sentence of the 'Perturbed cell'.\n\n"
        f"Species: {species}\n"
        f"Age: {age_str} (equivalent to human {int(pma_weeks)} weeks PMA)\n"
        f"Cell type: {ct_full} ({cell_type_broad_str})\n"
        f"Perturbation: {perturbation_str}\n"
        f"Unexposed: {unexposed_cs}\n\n"
        "Perturbed cell:"
    )


# ══════════════════════════════════════════════════════════════════
# GET RAW X
# ══════════════════════════════════════════════════════════════════

def get_raw_X(adata: anndata.AnnData) -> tuple[np.ndarray, list[str]]:
    if adata.raw is not None:
        X     = adata.raw.X
        genes = adata.raw.var_names.tolist()
    elif "counts" in adata.layers:
        X     = adata.layers["counts"]
        genes = adata.var_names.tolist()
    else:
        print("  WARN: raw/counts not found, using X")
        X     = adata.X
        genes = adata.var_names.tolist()
    if sp.issparse(X):
        X = X.toarray()
    return X.astype(np.float32), genes


# ══════════════════════════════════════════════════════════════════
# RESCUED COVERAGE
# ══════════════════════════════════════════════════════════════════

def rescued_coverage(
    gene_names_in_dataset: list[str],
    rescued_set: set[str],
    label: str,
) -> tuple[set[str], set[str]]:
    dataset_genes = set(gene_names_in_dataset)
    present = rescued_set & dataset_genes
    missing = rescued_set - dataset_genes
    pct = 100 * len(present) / len(rescued_set) if rescued_set else 0.0
    print(f"  [{label}] Rescued in dataset: {len(present)}/{len(rescued_set)} ({pct:.1f}%)")
    if missing:
        print(f"    Missing ({len(missing)}): {sorted(missing)}")
    else:
        print("    ✓ All rescued genes present")
    return present, missing


# ══════════════════════════════════════════════════════════════════
# GEMMA2 CHAT TEMPLATE — БЕЗ leading space перед completion
# ══════════════════════════════════════════════════════════════════

def gemma2_text(prompt: str, completion: str) -> str:
    return (
        f"<bos><start_of_turn>user\n"
        f"{prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{completion}<end_of_turn>\n"
    )


# ══════════════════════════════════════════════════════════════════
# OBS_TO_EXAMPLES — h5ad obs → список примеров для GRPO JSONL
# ══════════════════════════════════════════════════════════════════

def obs_to_examples(obs: pd.DataFrame) -> list[dict]:
    examples = []
    for _, row in obs.iterrows():
        prompt = build_prompt(
            species          = row["src_species"],
            age_str          = row["src_age_str"],
            pma_weeks        = row["src_pma_weeks"],
            cell_type_fine   = row["src_type_fine"],
            cell_type_broad_str = row["src_type_broad"],
            perturbation_str = row["perturbation_str"],
            unexposed_cs     = row["src_cell_sentence"],
        )
        examples.append({
            "text":      gemma2_text(prompt, row["tgt_cell_sentence"]),
            "pair_type": row["transition"],
        })
    return examples


# ══════════════════════════════════════════════════════════════════
# SAVE JSONL
# ══════════════════════════════════════════════════════════════════

def save_grpo_jsonl(rows: pd.DataFrame, path: Path) -> None:
    """Save GRPO format: {"text": ...}"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for _, row in rows.iterrows():
            f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
    print(f"  Saved: {path}  ({len(rows)} rows)")


def save_sft_jsonl(rows: pd.DataFrame, path: Path) -> None:
    """Save SFT format: {"prompt": ..., "completion": ...} без leading space."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for _, row in rows.iterrows():
            f.write(json.dumps({
                "prompt":     row["prompt"],
                "completion": row["completion"],
            }, ensure_ascii=False) + "\n")
    print(f"  Saved: {path}  ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════════
# DETERMINISTIC STRATUM SEED — для Stage 0 barcode-level split
# Использует MD5 вместо hash() (стабилен между процессами)
# ══════════════════════════════════════════════════════════════════

def stratum_seed(source: str, condition: str, cell_type: str) -> int:
    key = f"{source}|{condition}|{cell_type}"
    return RANDOM_SEED + int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
