"""Record dry instrument metrics + SHA-256 inventory into c2s-puma-repro/lab/2_сбор/
and write a dry factual run note in results/2026-11-23_std_runs_analysis/.
Does NOT touch lab/1_ход.md, lab/3_черновики/, lab/4_эффекты.md, or lab/ГДЕ-Я.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/stepandolzhenko/.gemini/config/skills/journals")
import env_snapshot  # noqa: E402

AZ = Path("/Users/stepandolzhenko/Documents/AzithroGemma")
STD_REPO = AZ / "retrain_v2/c2s-puma-std"
REPRO_REPO = AZ / "retrain_v2/c2s-puma-repro"
ap = argparse.ArgumentParser()
ap.add_argument("--out-root", type=Path, required=True, help="Results root directory (outside git)")
RES = ap.parse_args().out_root.resolve()
LAB2 = REPRO_REPO / "lab/2_сбор"

STAMP_DT = dt.datetime.now()
STAMP = STAMP_DT.strftime("%Y-%m-%d_%H%M")
STAMP_HUMAN = STAMP_DT.strftime("%Y-%m-%d %H:%M")

LAB_MD = LAB2 / f"{STAMP}_c2s_std_and_rep300_camerapr.md"
NOTE_MD = RES / f"{STAMP}_run_note.md"
assert not LAB_MD.exists(), f"Refusing to overwrite {LAB_MD}"
assert not NOTE_MD.exists(), f"Refusing to overwrite {NOTE_MD}"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def git_head(repo: Path) -> str:
    if not (repo / ".git").exists():
        return "not a git repo"
    out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    return out + (" (dirty)" if dirty else " (clean)")


def check_manifest() -> tuple[int, int]:
    man = json.loads((STD_REPO / "grid/tree/MANIFEST.json").read_text())
    items = man["files"] if isinstance(man, dict) and "files" in man else man
    ok = 0
    for rel, expected in items.items():
        exp = expected["sha256"] if isinstance(expected, dict) else expected
        if sha256_file(STD_REPO / "grid/tree" / rel) == exp:
            ok += 1
    return ok, len(items)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# 1. Environment snapshot
env = env_snapshot.snapshot(RES, root=REPRO_REPO)
env_md = env_snapshot.as_markdown(env)

# 2. Verify grid/tree MANIFEST
man_ok, man_tot = check_manifest()

# 3. Read all exact result tables from disk
uni = pd.read_csv(RES / "unified_all_with_std.csv")
uni_std = uni[uni["run"].isin(["std_2026-11-19_1543_humans_only_full",
                               "std_2026-11-19_1543_rats_upper_full",
                               "newgrid_lr1e-4_ep4_s42"])]

cam_ho = pd.read_csv(RES / "camerapr_humans_only_full/summary.csv")
cam_ho6 = pd.read_csv(RES / "camerapr_humans_only_full_6drugs/all_6drugs_camerapr_summary.csv")
cam_ru = pd.read_csv(RES / "camerapr_rats_upper_full/summary.csv")
cam_rep = pd.read_csv(RES / "camerapr_rep300_lr1e-4_ep4/summary.csv")
rep_de = pd.read_csv(RES / "camerapr_rep300_lr1e-4_ep4/rep300_azi_vs_noazi_de.csv")
pair_de6 = pd.read_csv(RES / "camerapr_humans_only_full_6drugs/drug_pairwise_de_counts.csv")
kw6 = pd.read_csv(RES / "camerapr_humans_only_full_6drugs/drug_omnibus_kw_markers.csv")
ovr6 = pd.read_csv(RES / "camerapr_humans_only_full_6drugs/drug_one_vs_rest_markers.csv")

# Zero-ribo/mito + autophagy + interferon counts directly from full CSVs
blob = json.loads((AZ / "supplement_materials/09_full_screening_8421_terms/enrichr_libraries_all_terms.json").read_text())
enrichr = {f"{lib}__{t}": set(g) for lib, terms in blob.items() for t, g in terms.items()}
hk_pat = re.compile(r"^(RPL|RPS|MRPL|MRPS|FAU|UBA52|RACK1|MT-|NDUF|COX[0-9A-Z]|UQCR|ATP5|SDH[A-D]|CYC1|CYCS)\b|^(RPL|RPS|MRPL|MRPS)[0-9P]")
hk_map = {t: (sum(bool(hk_pat.match(g)) for g in gs) / len(gs)) if gs else 0.0 for t, gs in enrichr.items()}


def summarize_bio(csv_path: Path, arm: str, drug: str | None = None) -> dict:
    d = pd.read_csv(csv_path)
    d = d[d["arm"] == arm].copy()
    if drug is not None:
        d = d[d["drug"] == drug].copy()
    d["half"] = d["term"].str.rsplit(" @@", n=1).str[1]
    d["clean"] = d["term"].str.rsplit(" @@", n=1).str[0]
    d["name"] = d["clean"].str.split("__", n=1).str[1]
    d["toward"] = ((d["half"] == "up") & (d["camerapr_direction"] == "Down")) | ((d["half"] == "dn") & (d["camerapr_direction"] == "Up"))
    d["hk_frac"] = d["clean"].map(hk_map).fillna(0.0)
    sig = d[d["camerapr_fdr"] < 0.05]
    pure = sig[sig["hk_frac"] == 0.0]
    auto = sig[sig["name"].str.contains(r"autophag|mitophag|lysosom", case=False, regex=True)]
    ifn = sig[sig["name"].str.contains(r"interferon", case=False, regex=True)]
    return {
        "total_fdr05": len(sig),
        "tow": int(sig["toward"].sum()),
        "awy": int((~sig["toward"]).sum()),
        "pure_fdr05": len(pure),
        "pure_tow": int(pure["toward"].sum()),
        "pure_awy": int((~pure["toward"]).sum()),
        "auto_fdr05": len(auto),
        "auto_tow": int(auto["toward"].sum()),
        "ifn_fdr05": len(ifn),
        "ifn_tow": int(ifn["toward"].sum()),
    }


bio_rows = []
for drug in ("azithromycin", "anti_hedgehog", "amiloride", "taurine", "antibiotics", "hmgb1_peptide"):
    st = summarize_bio(RES / "camerapr_humans_only_full_6drugs/all_6drugs_camerapr_full.csv", "azi_vs_own", drug=drug)
    bio_rows.append({"run": "humans_only_full", "drug": drug, "arm": "drug_vs_own (63v63)", **st})
for lbl, p, arm in (
    ("humans_only_full", RES / "camerapr_humans_only_full/camerapr_three_arms.csv", "azi_vs_noazi"),
    ("rats_upper_full", RES / "camerapr_rats_upper_full/camerapr_three_arms.csv", "azi_vs_own"),
    ("rats_upper_full", RES / "camerapr_rats_upper_full/camerapr_three_arms.csv", "azi_vs_noazi"),
    ("rep300_lr1e-4_ep4", RES / "camerapr_rep300_lr1e-4_ep4/camerapr_three_arms.csv", "azi_vs_noazi"),
):
    st = summarize_bio(p, arm)
    bio_rows.append({"run": lbl, "drug": "azithromycin", "arm": arm, **st})
bio_df = pd.DataFrame(bio_rows)

# Pairwise DE summary across 6 drugs in humans_only_full
pair_sum = pair_de6.groupby(["drug1", "drug2"])["n_de_fdr05_lfc025"].sum().reset_index()
pair_sum.columns = ["drug1", "drug2", "sum_de_fdr05_lfc025_12grp"]

# 4. SHA-256 inventory
tracked = [
    ("script", STD_REPO / "grid/lib/scan_only.py"),
    ("script", STD_REPO / "grid/lib/violins_only.py"),
    ("script", STD_REPO / "hpc/train.slurm"),
    ("script", REPRO_REPO / "analyses/2026-11-20_human_h5ad_lr1e-4_ep4/scripts/camerapr_azi_noazi.py"),
    ("script", RES / "run_std.py"),
    ("data", AZ / "BPD-PH/GSE275938_compiled_counts.h5ad"),
    ("env", RES / "env.json"),
    ("output", RES / "unified_all_with_std.csv"),
    ("output", RES / "camerapr_humans_only_full/summary.csv"),
    ("output", RES / "camerapr_humans_only_full/camerapr_three_arms.csv"),
    ("output", RES / "camerapr_humans_only_full_6drugs/all_6drugs_camerapr_summary.csv"),
    ("output", RES / "camerapr_humans_only_full_6drugs/all_6drugs_camerapr_full.csv"),
    ("output", RES / "camerapr_humans_only_full_6drugs/drug_omnibus_kw_markers.csv"),
    ("output", RES / "camerapr_humans_only_full_6drugs/drug_one_vs_rest_markers.csv"),
    ("output", RES / "camerapr_humans_only_full_6drugs/drug_pairwise_de_counts.csv"),
    ("output", RES / "camerapr_rats_upper_full/summary.csv"),
    ("output", RES / "camerapr_rats_upper_full/camerapr_three_arms.csv"),
    ("output", RES / "camerapr_rep300_lr1e-4_ep4/summary.csv"),
    ("output", RES / "camerapr_rep300_lr1e-4_ep4/camerapr_three_arms.csv"),
    ("output", RES / "camerapr_rep300_lr1e-4_ep4/rep300_azi_vs_noazi_de.csv"),
]
inv_rows = []
for kind, p in tracked:
    inv_rows.append({
        "тип": kind,
        "файл": str(p.relative_to(AZ)),
        "размер": fmt_size(p.stat().st_size),
        "sha256": sha256_file(p),
    })
inv_df = pd.DataFrame(inv_rows)

# 5. Write lab/2_сбор/<STAMP>_c2s_std_and_rep300_camerapr.md (pure instrument readings, zero conclusions)
lab_text = f"""---
дата: {STAMP_HUMAN}
метка: {STAMP}
теги: [c2s_std, camerapr, rep300, humans_only_full, rats_upper_full, h5ad, 6drugs]
статус: completed
---

# Показания прогонов: `c2s_std` (`humans_only_full`, `rats_upper_full`), `c2s-rep300` и переход `scan_only`/`violins_only` на `.h5ad`

## Среда и состояние кода

{env_md}
- `c2s-puma-repro` HEAD: `{git_head(REPRO_REPO)}`
- `c2s-puma-std`: `{git_head(STD_REPO)}`; `grid/tree/MANIFEST.json`: `{man_ok}/{man_tot}` совпадений sha256
- Полный снимок среды: `retrain_v2/results/2026-11-23_std_runs_analysis/env.json`

## 1. Перевод `scan_only.py` / `violins_only.py` на `GSE275938_compiled_counts.h5ad`

- Файл: `BPD-PH/GSE275938_compiled_counts.h5ad`, матрица `(43607, 36601)`, CSR `float32`, `nnz = 86,648,604`, порядок строк и заголовков `var_names` сверен с `GSE275938_compiled_counts.csv` (`header exact match = True`).
- Верификация на `std_2026-11-19_1543_rats_upper_full` без симлинка `.csv`: длительность `52.2 s`; `scores.json` — 25 панелей, `max_abs_diff = 0.0`; 12/12 `report_*.csv` идентичны; 12/12 `rat_schema/*.xlsx` идентичны.

## 2. Стандартизированная оценка скрипок (`unified_all_with_std.csv`)

{md_table(uni_std)}

## 3. `cameraPR` (`camerapr_azi_noazi.py`, `K = 2000`)

### 3.1. `humans_only_full` (0 крыс, 0 препаратов в обучении) — 6 препаратов, рука `azi_vs_own` (63 vs 63)

{md_table(cam_ho6[cam_ho6["arm"] == "azi_vs_own"][["drug", "condition", "tests", "fdr05", "toward_ctrl", "away_ctrl", "gCap toward/away", "aCap toward/away", "Pericyte toward/away", "VEC toward/away"]])}

### 3.2. `humans_only_full` — 6 препаратов, рука `azi_vs_noazi`

{md_table(cam_ho6[cam_ho6["arm"] == "azi_vs_noazi"][["drug", "condition", "tests", "fdr05", "toward_ctrl", "away_ctrl", "gCap toward/away", "aCap toward/away", "Pericyte toward/away", "VEC toward/away"]])}

### 3.3. `rats_upper_full` (`2026-11-19_1543_rats_upper_full`, люди + крысы `azithromycin`)

{md_table(cam_ru)}

### 3.4. `c2s-rep300` (`2026-11-16_132201_puma-a100mig-repro-ep4-fp32-lr1e-4-s42`, 300 vs 300 клеток/группу)

{md_table(cam_rep)}

### 3.5. Сводка не-housekeeping путей (0% `RPL/RPS/MRPL/MRPS/MT-/OXPHOS`), аутофагии и интерферона (`BH FDR < 0.05`)

{md_table(bio_df)}

## 4. Погенные тесты и проверка воспроизводимости (`rep300` и 6 препаратов `humans_only_full`)

### 4.1. `humans_only_full`: попарные DE между 6 препаратами (Mann-Whitney `FDR < 0.05` и `|log2FC| >= 0.25`, сумма по 12 группам `cond × cell_type`)

{md_table(pair_sum)}

- Kruskal-Wallis (`FDR < 0.05`) по 6 препаратам: всего `{len(kw6)}` пар ген-группа (`HMGB1` в 12/12 группах, `min_fdr = 1.13e-12`; `HMGB2` в 7/12, `min_fdr = 9.91e-08`).
- One-vs-rest (`FDR < 0.05`, `|lfc_vs_other5| >= 0.10`): всего `{len(ovr6)}` записей (`hmgb1_peptide`: 1474, `amiloride`: 52, `azithromycin`: 23, `antibiotics`: 12, `taurine`: 12, `anti_hedgehog`: 7).

### 4.2. `c2s-rep300` (`azi vs noazi`, 300 vs 300)

- Погенный Mann-Whitney (`FDR < 0.05`) по 12 группам `cond × cell_type`: всего `{len(rep_de)}` ген (`BPDPH7mo Pericyte`, `FP236383.3`, `lfc = -0.501`, `fdr = 0.0464`).
- Непересекающийся split-half (порядок батча, 12 групп, среднее):
  - по генам: `r(63v63) = 0.053`, `r(150v150) = 0.061`;
  - по среднему `log2FC` путей (MSigDB Hallmark + KEGG + Reactome, `>= 15` генов): `r(63v63) = 0.248`, `r(150v150) = 0.228` (в т.ч. `BPD7mo VEC = 0.493`, `BPDPH7mo Pericyte = 0.466`, `Acute26 VEC = 0.374`, `BPD7mo gCap = 0.352`, `BPD7mo aCap = 0.343`; `BPDPH7mo VEC = -0.057`, `Acute26 Pericyte = -0.153`).
- Совпадение знака `signed_z` на 1 025 значимых путях `rep300`: с `orig63_noazi` — `62.0%` (`med_z = +0.51`), с `rats_upper_full (azi_vs_noazi)` — `77.1%` (`med_z = +1.11`), с `rats_upper_full (azi_vs_own)` — `45.0%` (`med_z = -0.23`), с `humans_only_full (azi_vs_own)` — `46.4%` (`med_z = -0.15`).
- Спирменовская корреляция `signed_z` по всем 112 859 общим тестам `rep300` vs `orig63_noazi` = `0.074`, vs `rats_upper_full (azi_vs_noazi)` = `0.070`, vs `rats_upper_full (azi_vs_own)` = `0.033`, vs `humans_only_full (azi_vs_own)` = `0.017`.

## 5. Реестр файлов и SHA-256

{md_table(inv_df)}

## Наблюдения автора

<!-- Заполняется только автором -->
"""

LAB_MD.write_text(lab_text, encoding="utf-8")

# 6. Write dry auto-note in results/2026-11-23_std_runs_analysis/ (outside author journals)
note_text = f"""---
дата: {STAMP_HUMAN}
метка: {STAMP}
источник_данных: {LAB_MD}
---

# Автозаметка по прогонам {STAMP_HUMAN}

## Что сделано и зачем

1. **Переход пайплайна скрипок на `.h5ad` без изменения скриптов рукописи (`grid/tree/`):**
   - Собран файл `BPD-PH/GSE275938_compiled_counts.h5ad` (`43607 × 36601`, CSR `float32`, точное совпадение заголовков `36601/36601` и порядка строк с CSV).
   - Изменены только обёртки `c2s-puma-std/grid/lib/scan_only.py` (чтение `.h5ad` с fallback на CSV) и `violins_only.py` (`HEADER` из `raw_columns`), а также добавлен симлинк `.h5ad` в `run_std.py`. Все 45/45 файлов в `grid/tree/MANIFEST.json` остались без изменений.
   - Цель: убрать ~4-минутное чтение 3.0 GB CSV при каждом запуске и проверить численную эквивалентность.

2. **Прогон стандартизированных скрипок на первых готовых моделях `c2s_std`:**
   - Скачаны и прогнаны через `run_std.py`: `2026-11-19_1543_humans_only_full` (только человеческие клетки, 0 крыс и 0 лекарств в обучении) и `2026-11-19_1543_rats_upper_full` (человек + крысы с `azithromycin`, верхний регистр генов).
   - Цель: получить сопоставимую таблицу по 40/24 скрипкам наряду с предыдущими сетками (`oldgrid`, `newgrid`).

3. **Прогон `camerapr_azi_noazi.py` на `humans_only_full` по всем 6 OOD-препаратам (`azithromycin`, `taurine`, `amiloride`, `anti_hedgehog`, `antibiotics`, `hmgb1_peptide`):**
   - Использован неизменённый скрипт `camerapr_azi_noazi.py` и его `--worker` по рукам `azi_vs_disease`, `azi_vs_noazi`, `azi_vs_own` (63 vs 63 клетки на группу), отдельно проверены не-рибосомные/не-митохондриальные пути, аутофагия, интерферон, попарные DE между препаратами и мета-анализ маркеров по 12 группам `cond × cell_type`.
   - Цель: измерить базовый сдвиг на OOD-промптах препаратов в модели, не видевшей препаратов при дообучении, и проверить, одинаков ли сдвиг между разными препаратами.

4. **Прогон `camerapr_azi_noazi.py` на `rats_upper_full` (`azithromycin`) и на 300-клеточном инференсе `c2s-rep300` (`newgrid_lr1e-4_ep4_s42`, 300 vs 300 клеток/группу):**
   - Для `rep300` дополнительно выполнен погенный Mann-Whitney DE (`azi vs noazi`) и проверка split-half воспроизводимости (непересекающиеся `63 vs 63` и `150 vs 150` по генам и по средним `log2FC` путей).
   - Цель: сравнить `humans_only_full` vs `rats_upper_full` на одном билде и оценить, как увеличение инференса с 63 до 300 клеток влияет на мощность и воспроизводимость.

## Сухие результаты

- **Эквивалентность `.h5ad` vs CSV:** полный прогон `violins_only.py` на `rats_upper_full` без симлинка CSV занял `52.2 s`; расхождение по 25 панелям `scores.json` составило `0.0`, 12/12 `report_*.csv` и 12/12 `rat_schema/*.xlsx` совпали полностью.
- **Скрипки (`unified_all_with_std.csv`):**
  - `std_2026-11-19_1543_humans_only_full`: `exact = 20/24` (`ALI 8/8`, `BPD 4/8`, `BPDPH 8/8`, крысы `NA`), спасённых генов `3959 / 4450 / 2145`.
  - `std_2026-11-19_1543_rats_upper_full`: `exact = 37/40` (`mism = 3`, `rsig = 32/35`, `ALI 8/8`, `BPD 8/8`, `BPDPH 8/8`, `RA_HO 8/8`, `HO_AZI 5/8`), спасённых генов `4027 / 4582 / 2256`.
- **`cameraPR` в `humans_only_full` (6 препаратов, `drug_vs_own`, 63 vs 63):**
  - Число путей `FDR < 0.05` (BPD7mo / BPDPH7mo / Acute26): `azithromycin` `29 (0/29) / 39 (39/0) / 0`; `anti_hedgehog` `100 (99/1) / 62 (59/3) / 0`; `amiloride` `219 (218/1) / 99 (99/0) / 663 (29/634)`; `taurine` `545 (544/1) / 88 (87/1) / 132 (132/0)`; `antibiotics` `1523 (1522/1) / 117 (88/29) / 52 (52/0)`; `hmgb1_peptide` `5611 (5596/15) / 394 (311/83) / 229 (229/0)`.
  - После исключения путей с генами `RPL/RPS/MRPL/MRPS/MT-/OXPHOS` (`zero_hk`): `azithromycin` `1`, `anti_hedgehog` `12`, `taurine` `161`, `amiloride` `284` (`246 away` в Acute26 Pericyte), `antibiotics` `555`, `hmgb1_peptide` `2879`.
  - Попарных DE-генов (`FDR < 0.05`, `|log2FC| >= 0.25`, сумма по 12 группам) между 5 препаратами (кроме `hmgb1_peptide`): от `0` до `4`; против `hmgb1_peptide`: `165–604`. В `hmgb1_peptide` ген `HMGB1` снижен в 12/12 группах (`mean_lfc = -2.84`, `FDR = 4.6e-15`), `HMGB2` повышен в 12/12 (`+1.70`).
  - Аутофагия и интерферон в `humans_only_full` для `azithromycin`: `0` тестов с `FDR < 0.05` в `azi_vs_own` и `azi_vs_noazi`.
- **`cameraPR` в `rats_upper_full` (`azithromycin`):**
  - `azi_vs_noazi`: BPD7mo `128 (128/0)`, BPDPH7mo `33 (33/0)`, Acute26 `50 (48/2)`.
  - `azi_vs_own` (63 vs 63): BPD7mo `262 (225/37)`, BPDPH7mo `0`, Acute26 `56 (56/0)`; не-ribo/mito путей `41 (41/0)`; аутофагия `2 (2/0)` (`Chaperone Mediated Autophagy` `FDR=0.012`, `Selective Autophagy` `FDR=0.023` в BPD7mo VEC); интерферон `0` (мин. `FDR = 0.056–0.059` в BPD7mo VEC).
- **`cameraPR` и воспроизводимость в `c2s-rep300` (`lr1e-4_ep4`, 300 vs 300, `azi_vs_noazi`):**
  - Всего `FDR < 0.05`: `1025` (`1022 toward / 3 away`: BPD7mo `871/1`, BPDPH7mo `37/2`, Acute26 `114/0`); не-ribo/mito `280 (280/0)`; аутофагия `8 (8/0)`; интерферон `1 (1/0)` (`Interferon Gamma Response`, BPD7mo VEC, `FDR=0.008`).
  - Погенный DE (`FDR < 0.05`, 12 групп): `1` ген (`FP236383.3`, `FDR = 0.046`).
  - Split-half (`63v63` -> `150v150`, среднее по 12 группам): по генам `r = 0.053 -> 0.061`; по путям `r = 0.248 -> 0.228` (в `BPD7mo VEC`: `0.493`, `BPDPH7mo Pericyte`: `0.466`, `Acute26 VEC`: `0.374`, `BPD7mo gCap`: `0.352`, `BPD7mo aCap`: `0.343`; в `BPDPH7mo VEC`: `-0.057`, `Acute26 Pericyte`: `-0.153`).

## Статус незавершённых задач на Puma (по данным `run_std.py status` на 08:37)

- `24724060` (`std-rodent_drugs_full`), `24756462` (`std-lora-r16`), `24756463` (`std-lora-r64`) — `RUNNING`, инференс ещё не выполнен.
- На `rats_upper_full` остальные 5 OOD-препаратов (`taurine`, `amiloride`, `anti_hedgehog`, `antibiotics`, `hmgb1_peptide`) через `cameraPR` ещё не прогонялись.
"""

NOTE_MD.write_text(note_text, encoding="utf-8")
print("WROTE LAB2:", LAB_MD)
print("WROTE NOTE:", NOTE_MD)
