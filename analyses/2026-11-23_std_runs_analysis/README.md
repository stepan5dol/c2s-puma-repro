---
дата: 2026-11-23 10:07
метка: 2026-11-23_1007
источник_данных: lab/2_сбор/2026-11-23_1007_c2s_std_and_rep300_camerapr.md
---

# Автозаметка по прогонам 2026-11-23 10:07

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

## Запуск скриптов (результаты и `inputs/` остаются в `retrain_v2/results/2026-11-23_std_runs_analysis/`)

```bash
OUT=/Users/stepandolzhenko/Documents/AzithroGemma/retrain_v2/results/2026-11-23_std_runs_analysis
python3 scripts/run_std.py --out-root "$OUT" status
python3 scripts/run_std.py --out-root "$OUT" fetch
python3 scripts/run_std.py --out-root "$OUT" analyze
python3 scripts/run_camerapr_std.py --out-root "$OUT" --run 2026-11-19_1543_humans_only_full
python3 scripts/run_camerapr_std.py --out-root "$OUT" --run 2026-11-19_1543_rats_upper_full --drugs azithromycin
python3 scripts/run_camerapr_std.py --out-root "$OUT" --rep300
python3 scripts/write_lab_record.py --out-root "$OUT"
```
