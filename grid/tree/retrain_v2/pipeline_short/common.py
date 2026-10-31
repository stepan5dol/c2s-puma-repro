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

BASE = Path("/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2")

RAT_PATH          = BASE / "rat.ho.azi.integrated.h5ad"
ATLAS_CLEAN_PATH  = BASE / "he_lung_atlas.h5ad"
ATLAS_RAW_PATH    = BASE / "2022FetalLungIntCounts.h5ad"
BPD_META_PATH     = BASE / "BPD-PH/GSE275938_cell_metadata.csv"
BPD_COUNTS_PATH   = BASE / "BPD-PH/GSE275938_compiled_counts.csv"

RESCUED_RAT_PATH   = BASE / "train-after-grpo-analysis/rescued_rat_genes.txt"
RESCUED_HUMAN_PATH = BASE / "train-after-grpo-analysis/rescued_human_genes.txt"

CONFIG_PATH  = BASE / "scripts/cell_types_config.json"
PIPELINE_DIR = BASE / "pipeline_short"

TOP_K       = 800
RANDOM_SEED = 42

TARGET_RAT_CELL_TYPES: set[str] = {"gCAP", "aCAP", "Peri", "VEC"}
TARGET_HUMAN_CELL_TYPES: set[str] = {
    "gCap", "aCap", "Pulmonary venous EC", "Pericyte",
}
TARGET_RAT_CONDITIONS: set[str] = {"RA", "HO", "AZI"}
TARGET_HUMAN_CONDITIONS: set[str] = {"He22", "Acute26", "BPD7mo", "BPDPH7mo"}

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

GENE_ALIASES_EXTRA: dict[str, list[str]] = {}

ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _aliases in {**GENE_ALIASES, **GENE_ALIASES_EXTRA}.items():
    for _alias in _aliases:
        ALIAS_TO_CANONICAL[_alias] = _canon


def remap_gene_names(gene_names) -> list[str]:
    return [ALIAS_TO_CANONICAL.get(g, g) for g in gene_names]


RESCUED_RAT   = set(RESCUED_RAT_PATH.read_text().splitlines()) - {""}
RESCUED_HUMAN = set(RESCUED_HUMAN_PATH.read_text().splitlines()) - {""}

_ABBREV_TO_FULL_RAW: dict[str, str] = {
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
    "VEC":       "pulmonary venous endothelial cell",
    "aCap":      "aerocyte capillary endothelial cell",
    "gCap":      "general capillary endothelial cell",
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

ABBREV_TO_FULL_EXTRA: dict[str, str] = {}


def cell_type_full(ct: str) -> str:
    """Resolve cell type name/abbreviation to human-readable form."""
    key = ct.upper()
    return ABBREV_TO_FULL.get(key, ABBREV_TO_FULL_EXTRA.get(key, ct))


_FINE_TO_BROAD_RAW: dict[str, str] = {
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
    "Art": "endothelial", "Cap": "endothelial", "Cap-a": "endothelial",
    "Lymph": "endothelial", "Vein": "endothelial",
    "Myofibroblast": "fibroblast",
    "Alv Mf": "macrophage", "Int Mf": "macrophage",
    "ILC2": "lymphocyte", "ILC3": "lymphocyte", "ILCP": "lymphocyte",
    "Chondrocyte": "other", "PNS": "other",
}

FINE_TO_BROAD: dict[str, str] = {k.upper(): v for k, v in _FINE_TO_BROAD_RAW.items()}

FINE_TO_BROAD_EXTRA: dict[str, str] = {}


def cell_type_broad(ct: str) -> str:
    """Resolve cell type to broad category (endothelial, epithelial, ...)."""
    key = ct.upper()
    return FINE_TO_BROAD.get(key, FINE_TO_BROAD_EXTRA.get(key, "unknown"))


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

COND_META: dict[tuple, dict] = {
    ("rat",   "RA"):      {"age_str": "postnatal day 14",                                       "pma_weeks": 52},
    ("rat",   "HO"):      {"age_str": "postnatal day 14",                                       "pma_weeks": 52},
    ("rat",   "AZI"):     {"age_str": "postnatal day 14",                                       "pma_weeks": 52},
    # He et al. 2022 ages are post-conception weeks (15/18/20/22); stored here as gestational weeks (+2)
    ("human", "He15"):    {"age_str": "gestational week 17", "pma_weeks": 17},
    ("human", "He18"):    {"age_str": "gestational week 20", "pma_weeks": 20},
    ("human", "He20"):    {"age_str": "gestational week 22", "pma_weeks": 22},
    ("human", "He22"):    {"age_str": "gestational week 24", "pma_weeks": 24},
    ("human", "Acute26"): {"age_str": "born at gestational week 26, postnatal week 2 (acute preterm injury)", "pma_weeks": 28},
    ("human", "Term0d"):  {"age_str": "born at gestational week 40, postnatal day 7",          "pma_weeks": 41},
    ("human", "Term20d"): {"age_str": "born at gestational week 37, postnatal day 20",         "pma_weeks": 40},
    ("human", "BPD7mo"):  {"age_str": "born at gestational week 25-27, 50-52 weeks corrected gestational age (BPD)", "pma_weeks": 52},
    ("human", "BPDPH7mo"):{"age_str": "born at gestational week 26-27, 51-52 weeks corrected gestational age (BPD with pulmonary hypertension)", "pma_weeks": 52},
    ("mouse", "P3"):      {"age_str": "postnatal day 3",                                        "pma_weeks": 32},
    ("mouse", "P7"):      {"age_str": "postnatal day 7",                                        "pma_weeks": 40},
    ("mouse", "P14"):     {"age_str": "postnatal day 14",                                       "pma_weeks": 52},
}

PERTURBATION_STR: dict[tuple, str] = {
    ("rat", "RA",  "HO"):  "Hyperoxia exposure from birth to postnatal day 14.",
    ("rat", "HO",  "AZI"): "Azithromycin treatment (30 mg/kg IP at P7, P10, P13) during exposure to 85% O2 for 14 days.",

    ("human", "He15", "He18"): "Three weeks of fetal lung development (GW17 → GW20).",
    ("human", "He15", "He20"): "Five weeks of fetal lung development (GW17 → GW22).",
    ("human", "He15", "He22"): "Seven weeks of fetal lung development (GW17 → GW24).",
    ("human", "He18", "He20"): "Two weeks of fetal lung development (GW20 → GW22).",
    ("human", "He18", "He22"): "Four weeks of fetal lung development (GW20 → GW24).",
    ("human", "He20", "He22"): "Two weeks of fetal lung development (GW22 → GW24).",

    ("human", "He22", "Acute26"): (
        "Acute preterm lung injury (born at gestational week 26, studied at postnatal week 2), "
        "compared to normal late canalicular fetal lung (GW24)."
    ),
    ("human", "He22", "Term0d"): (
        "Healthy term birth at 40 weeks, "
        "compared to late canalicular fetal lung (GW24)."
    ),
    ("human", "Acute26", "BPD7mo"): (
        "Progression from acute preterm lung injury (28 weeks corrected gestational age) "
        "to established bronchopulmonary dysplasia (50-52 weeks corrected gestational age)."
    ),
    ("human", "Acute26", "BPDPH7mo"): (
        "Progression from acute preterm lung injury (28 weeks corrected gestational age) "
        "to established bronchopulmonary dysplasia with pulmonary hypertension (51-52 weeks corrected gestational age)."
    ),

    ("mouse", "Normoxia",  "P3",  "Normoxia",  "P7"):  "Four days of normal postnatal lung development in mouse (P3 → P7).",
    ("mouse", "Normoxia",  "P7",  "Normoxia",  "P14"): "Seven days of normal postnatal lung development in mouse (P7 → P14).",
    ("mouse", "Normoxia",  "P3",  "Hyperoxia", "P3"):  "Hyperoxia exposure from birth to postnatal day 3 in mouse.",
    ("mouse", "Normoxia",  "P7",  "Hyperoxia", "P7"):  "Hyperoxia exposure from birth to postnatal day 7 in mouse.",
    ("mouse", "Normoxia",  "P14", "Hyperoxia", "P14"): "Hyperoxia exposure from birth to postnatal day 14 in mouse.",
    ("mouse", "Hyperoxia", "P3",  "Hyperoxia", "P7"):  "Four days of continued hyperoxia exposure in mouse (P3 → P7).",
    ("mouse", "Hyperoxia", "P7",  "Hyperoxia", "P14"): "Seven days of continued hyperoxia exposure in mouse (P7 → P14).",
}

AZI_PERTURBATION = (
    "Oral azithromycin treatment (5 mg/kg of body weight, 3 times per week). "
    "Administered as a low maintenance dose for a duration of 10 weeks."
)

HARMONIZE: dict[str, str] = {
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
}


def harmonize_cell_type(raw_ct: str) -> str:
    """Map raw atlas label to canonical BPD cell_type.
    BPD and rat labels pass through unchanged (not in HARMONIZE).
    """
    return HARMONIZE.get(raw_ct, raw_ct)


def load_grpo_endo_types() -> dict[str, set[str]]:
    """Return {species: set_of_canonical_cell_types} for GRPO endothelial whitelist."""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    rat_types   = set(cfg["RAT"]["ot_include"])
    atlas_raw   = set(cfg["ATLAS_He"]["ot_include"])
    human_types = {harmonize_cell_type(t) for t in atlas_raw}
    return {"rat": rat_types, "human": human_types}


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
        f"Age: {age_str}\n"
        f"Cell type: {ct_full} ({cell_type_broad_str})\n"
        f"Perturbation: {perturbation_str}\n"
        f"Unexposed: {unexposed_cs}\n\n"
        "Perturbed cell:"
    )


def stratum_seed(source: str, condition: str, cell_type: str) -> int:
    key = f"{source}|{condition}|{cell_type}"
    return RANDOM_SEED + int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
