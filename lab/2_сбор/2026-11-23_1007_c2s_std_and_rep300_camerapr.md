---
дата: 2026-11-23 10:07
метка: 2026-11-23_1007
теги: [c2s_std, camerapr, rep300, humans_only_full, rats_upper_full, h5ad, 6drugs]
статус: completed
---

# Показания прогонов: `c2s_std` (`humans_only_full`, `rats_upper_full`), `c2s-rep300` и переход `scan_only`/`violins_only` на `.h5ad`

## Среда и состояние кода

- машина: `Stepans-MacBook-Pro.local`, macOS-27.2.1-arm64-arm-64bit-Mach-O
- python 3.13.12, blas accelerate
- ускоритель: mps ×1 (arm64), torch 2.12.0
- seed None, deterministic False
- зависимости: pip freeze `c0924d664c98`
- коммит `beb37128a96c`
- `c2s-puma-repro` HEAD: `beb37128a96c864d6f3a32ce3912bcc6e4d4e99c (clean)`
- `c2s-puma-std`: `not a git repo`; `grid/tree/MANIFEST.json`: `45/45` совпадений sha256
- Полный снимок среды: `retrain_v2/results/2026-11-23_std_runs_analysis/env.json`

## 1. Перевод `scan_only.py` / `violins_only.py` на `GSE275938_compiled_counts.h5ad`

- Файл: `BPD-PH/GSE275938_compiled_counts.h5ad`, матрица `(43607, 36601)`, CSR `float32`, `nnz = 86,648,604`, порядок строк и заголовков `var_names` сверен с `GSE275938_compiled_counts.csv` (`header exact match = True`).
- Верификация на `std_2026-11-19_1543_rats_upper_full` без симлинка `.csv`: длительность `52.2 s`; `scores.json` — 25 панелей, `max_abs_diff = 0.0`; 12/12 `report_*.csv` идентичны; 12/12 `rat_schema/*.xlsx` идентичны.

## 2. Стандартизированная оценка скрипок (`unified_all_with_std.csv`)

| run | K | xlsx | exact_40 | mism | rsig | ALI/8 | BPD/8 | BPDPH/8 | RA_HO/8 | HO_AZI/8 | resc_ALI | resc_BPD | resc_BPDPH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| newgrid_lr1e-4_ep4_s42 | 2000 | 12 | 38 | 2 | 33/35 | 8 | 7 | 8 | 8.0 | 7.0 | 3986 | 3823 | 2181 |
| std_2026-11-19_1543_humans_only_full | 2000 | 12 | 20/24 | NA(no_rat) | 20/24 | 8 | 4 | 8 | nan | nan | 3959 | 4450 | 2145 |
| std_2026-11-19_1543_rats_upper_full | 2000 | 12 | 37 | 3 | 32/35 | 8 | 8 | 8 | 8.0 | 5.0 | 4027 | 4582 | 2256 |

## 3. `cameraPR` (`camerapr_azi_noazi.py`, `K = 2000`)

### 3.1. `humans_only_full` (0 крыс, 0 препаратов в обучении) — 6 препаратов, рука `azi_vs_own` (63 vs 63)

| drug | condition | tests | fdr05 | toward_ctrl | away_ctrl | gCap toward/away | aCap toward/away | Pericyte toward/away | VEC toward/away |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| azithromycin | BPD7mo | 39209 | 29 | 0 | 29 | 0/0 | 0/0 | 0/29 | 0/0 |
| azithromycin | BPDPH7mo | 35253 | 39 | 39 | 0 | 0/0 | 0/0 | 0/0 | 39/0 |
| azithromycin | Acute26 | 38745 | 0 | 0 | 0 | 0/0 | 0/0 | 0/0 | 0/0 |
| taurine | BPD7mo | 39201 | 545 | 544 | 1 | 0/0 | 372/0 | 64/0 | 108/1 |
| taurine | BPDPH7mo | 35269 | 88 | 87 | 1 | 0/0 | 0/0 | 42/1 | 45/0 |
| taurine | Acute26 | 38691 | 132 | 132 | 0 | 40/0 | 35/0 | 0/0 | 57/0 |
| amiloride | BPD7mo | 39166 | 219 | 218 | 1 | 0/0 | 43/0 | 124/1 | 51/0 |
| amiloride | BPDPH7mo | 35212 | 99 | 99 | 0 | 0/0 | 22/0 | 43/0 | 34/0 |
| amiloride | Acute26 | 38759 | 663 | 29 | 634 | 0/0 | 0/0 | 29/634 | 0/0 |
| anti_hedgehog | BPD7mo | 39202 | 100 | 99 | 1 | 0/0 | 35/0 | 0/0 | 64/1 |
| anti_hedgehog | BPDPH7mo | 35259 | 62 | 59 | 3 | 0/0 | 17/0 | 42/3 | 0/0 |
| anti_hedgehog | Acute26 | 38720 | 0 | 0 | 0 | 0/0 | 0/0 | 0/0 | 0/0 |
| antibiotics | BPD7mo | 39163 | 1523 | 1522 | 1 | 77/0 | 120/0 | 477/1 | 848/0 |
| antibiotics | BPDPH7mo | 35257 | 117 | 88 | 29 | 42/29 | 0/0 | 0/0 | 46/0 |
| antibiotics | Acute26 | 38684 | 52 | 52 | 0 | 0/0 | 1/0 | 0/0 | 51/0 |
| hmgb1_peptide | BPD7mo | 39181 | 5611 | 5596 | 15 | 896/7 | 884/0 | 1584/8 | 2232/0 |
| hmgb1_peptide | BPDPH7mo | 35279 | 394 | 311 | 83 | 48/22 | 28/0 | 66/60 | 169/1 |
| hmgb1_peptide | Acute26 | 38700 | 229 | 229 | 0 | 71/0 | 59/0 | 0/0 | 99/0 |

### 3.2. `humans_only_full` — 6 препаратов, рука `azi_vs_noazi`

| drug | condition | tests | fdr05 | toward_ctrl | away_ctrl | gCap toward/away | aCap toward/away | Pericyte toward/away | VEC toward/away |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| azithromycin | BPD7mo | 39209 | 125 | 33 | 92 | 0/0 | 33/0 | 0/56 | 0/36 |
| azithromycin | BPDPH7mo | 35253 | 0 | 0 | 0 | 0/0 | 0/0 | 0/0 | 0/0 |
| azithromycin | Acute26 | 38745 | 0 | 0 | 0 | 0/0 | 0/0 | 0/0 | 0/0 |
| taurine | BPD7mo | 39201 | 70 | 70 | 0 | 0/0 | 70/0 | 0/0 | 0/0 |
| taurine | BPDPH7mo | 35269 | 80 | 61 | 19 | 0/0 | 12/0 | 49/19 | 0/0 |
| taurine | Acute26 | 38691 | 98 | 98 | 0 | 48/0 | 0/0 | 0/0 | 50/0 |
| amiloride | BPD7mo | 39166 | 34 | 34 | 0 | 0/0 | 34/0 | 0/0 | 0/0 |
| amiloride | BPDPH7mo | 35212 | 71 | 55 | 16 | 10/0 | 0/0 | 45/16 | 0/0 |
| amiloride | Acute26 | 38759 | 421 | 15 | 406 | 0/0 | 0/0 | 15/406 | 0/0 |
| anti_hedgehog | BPD7mo | 39202 | 59 | 59 | 0 | 28/0 | 31/0 | 0/0 | 0/0 |
| anti_hedgehog | BPDPH7mo | 35259 | 63 | 46 | 17 | 0/0 | 0/0 | 46/17 | 0/0 |
| anti_hedgehog | Acute26 | 38720 | 27 | 27 | 0 | 27/0 | 0/0 | 0/0 | 0/0 |
| antibiotics | BPD7mo | 39163 | 132 | 132 | 0 | 85/0 | 47/0 | 0/0 | 0/0 |
| antibiotics | BPDPH7mo | 35257 | 105 | 74 | 31 | 47/30 | 0/0 | 27/1 | 0/0 |
| antibiotics | Acute26 | 38684 | 39 | 39 | 0 | 0/0 | 0/0 | 0/0 | 39/0 |
| hmgb1_peptide | BPD7mo | 39181 | 1215 | 1214 | 1 | 860/1 | 154/0 | 0/0 | 200/0 |
| hmgb1_peptide | BPDPH7mo | 35279 | 305 | 180 | 125 | 56/26 | 27/14 | 58/85 | 39/0 |
| hmgb1_peptide | Acute26 | 38700 | 211 | 209 | 2 | 89/0 | 52/0 | 0/0 | 68/2 |

### 3.3. `rats_upper_full` (`2026-11-19_1543_rats_upper_full`, люди + крысы `azithromycin`)

| condition | arm | tests | fdr05 | toward_ctrl | away_ctrl | gCap toward/away | aCap toward/away | Pericyte toward/away | VEC toward/away |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BPD7mo | azi_vs_disease | 39223 | 13037 | 13037 | 0 | 828/0 | 4930/0 | 5094/0 | 2185/0 |
| BPD7mo | noazi_vs_disease | 39223 | 2914 | 2913 | 1 | 199/0 | 1987/0 | 727/1 | 0/0 |
| BPD7mo | azi_vs_noazi | 39223 | 128 | 128 | 0 | 0/0 | 0/0 | 33/0 | 95/0 |
| BPD7mo | azi_vs_own | 39223 | 262 | 225 | 37 | 0/0 | 45/0 | 0/37 | 180/0 |
| BPDPH7mo | azi_vs_disease | 35221 | 1929 | 1901 | 28 | 3/0 | 59/0 | 788/0 | 1051/28 |
| BPDPH7mo | noazi_vs_disease | 35221 | 929 | 854 | 75 | 119/45 | 88/0 | 402/2 | 245/28 |
| BPDPH7mo | azi_vs_noazi | 35221 | 33 | 33 | 0 | 0/0 | 0/0 | 33/0 | 0/0 |
| BPDPH7mo | azi_vs_own | 35221 | 0 | 0 | 0 | 0/0 | 0/0 | 0/0 | 0/0 |
| Acute26 | azi_vs_disease | 38765 | 441 | 426 | 15 | 46/1 | 205/1 | 99/13 | 76/0 |
| Acute26 | noazi_vs_disease | 38765 | 499 | 440 | 59 | 362/25 | 59/34 | 1/0 | 18/0 |
| Acute26 | azi_vs_noazi | 38765 | 50 | 48 | 2 | 0/0 | 48/1 | 0/0 | 0/1 |
| Acute26 | azi_vs_own | 38765 | 56 | 56 | 0 | 56/0 | 0/0 | 0/0 | 0/0 |

### 3.4. `c2s-rep300` (`2026-11-16_132201_puma-a100mig-repro-ep4-fp32-lr1e-4-s42`, 300 vs 300 клеток/группу)

| condition | arm | tests | fdr05 | toward_ctrl | away_ctrl | gCap toward/away | aCap toward/away | Pericyte toward/away | VEC toward/away |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BPD7mo | azi_vs_disease | 39154 | 18304 | 18304 | 0 | 2950/0 | 6288/0 | 6463/0 | 2603/0 |
| BPD7mo | noazi_vs_disease | 39154 | 15069 | 15040 | 29 | 1633/27 | 6187/0 | 6013/0 | 1207/2 |
| BPD7mo | azi_vs_noazi | 39154 | 872 | 871 | 1 | 67/0 | 56/0 | 607/1 | 141/0 |
| BPDPH7mo | azi_vs_disease | 35233 | 3674 | 3618 | 56 | 161/3 | 348/0 | 1440/0 | 1669/53 |
| BPDPH7mo | noazi_vs_disease | 35233 | 3964 | 3743 | 221 | 228/61 | 374/0 | 1644/36 | 1497/124 |
| BPDPH7mo | azi_vs_noazi | 35233 | 39 | 37 | 2 | 0/2 | 0/0 | 37/0 | 0/0 |
| Acute26 | azi_vs_disease | 38761 | 2913 | 2808 | 105 | 591/64 | 589/40 | 887/0 | 741/1 |
| Acute26 | noazi_vs_disease | 38761 | 1945 | 1724 | 221 | 433/150 | 345/69 | 692/2 | 254/0 |
| Acute26 | azi_vs_noazi | 38761 | 114 | 114 | 0 | 12/0 | 38/0 | 0/0 | 64/0 |

### 3.5. Сводка не-housekeeping путей (0% `RPL/RPS/MRPL/MRPS/MT-/OXPHOS`), аутофагии и интерферона (`BH FDR < 0.05`)

| run | drug | arm | total_fdr05 | tow | awy | pure_fdr05 | pure_tow | pure_awy | auto_fdr05 | auto_tow | ifn_fdr05 | ifn_tow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| humans_only_full | azithromycin | drug_vs_own (63v63) | 68 | 39 | 29 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| humans_only_full | anti_hedgehog | drug_vs_own (63v63) | 162 | 158 | 4 | 12 | 8 | 4 | 0 | 0 | 0 | 0 |
| humans_only_full | amiloride | drug_vs_own (63v63) | 981 | 346 | 635 | 284 | 37 | 247 | 6 | 0 | 0 | 0 |
| humans_only_full | taurine | drug_vs_own (63v63) | 765 | 763 | 2 | 161 | 159 | 2 | 0 | 0 | 0 | 0 |
| humans_only_full | antibiotics | drug_vs_own (63v63) | 1692 | 1662 | 30 | 555 | 554 | 1 | 8 | 8 | 3 | 3 |
| humans_only_full | hmgb1_peptide | drug_vs_own (63v63) | 6234 | 6136 | 98 | 2879 | 2858 | 21 | 68 | 68 | 39 | 39 |
| humans_only_full | azithromycin | azi_vs_noazi | 125 | 33 | 92 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| rats_upper_full | azithromycin | azi_vs_own | 318 | 281 | 37 | 41 | 41 | 0 | 2 | 2 | 0 | 0 |
| rats_upper_full | azithromycin | azi_vs_noazi | 211 | 209 | 2 | 23 | 21 | 2 | 1 | 1 | 0 | 0 |
| rep300_lr1e-4_ep4 | azithromycin | azi_vs_noazi | 1025 | 1022 | 3 | 280 | 280 | 0 | 8 | 8 | 1 | 1 |

## 4. Погенные тесты и проверка воспроизводимости (`rep300` и 6 препаратов `humans_only_full`)

### 4.1. `humans_only_full`: попарные DE между 6 препаратами (Mann-Whitney `FDR < 0.05` и `|log2FC| >= 0.25`, сумма по 12 группам `cond × cell_type`)

| drug1 | drug2 | sum_de_fdr05_lfc025_12grp |
| --- | --- | --- |
| amiloride | anti_hedgehog | 0 |
| amiloride | antibiotics | 3 |
| amiloride | hmgb1_peptide | 177 |
| anti_hedgehog | antibiotics | 0 |
| anti_hedgehog | hmgb1_peptide | 274 |
| antibiotics | hmgb1_peptide | 165 |
| azithromycin | amiloride | 0 |
| azithromycin | anti_hedgehog | 0 |
| azithromycin | antibiotics | 0 |
| azithromycin | hmgb1_peptide | 604 |
| azithromycin | taurine | 2 |
| taurine | amiloride | 0 |
| taurine | anti_hedgehog | 1 |
| taurine | antibiotics | 4 |
| taurine | hmgb1_peptide | 234 |

- Kruskal-Wallis (`FDR < 0.05`) по 6 препаратам: всего `582` пар ген-группа (`HMGB1` в 12/12 группах, `min_fdr = 1.13e-12`; `HMGB2` в 7/12, `min_fdr = 9.91e-08`).
- One-vs-rest (`FDR < 0.05`, `|lfc_vs_other5| >= 0.10`): всего `1580` записей (`hmgb1_peptide`: 1474, `amiloride`: 52, `azithromycin`: 23, `antibiotics`: 12, `taurine`: 12, `anti_hedgehog`: 7).

### 4.2. `c2s-rep300` (`azi vs noazi`, 300 vs 300)

- Погенный Mann-Whitney (`FDR < 0.05`) по 12 группам `cond × cell_type`: всего `1` ген (`BPDPH7mo Pericyte`, `FP236383.3`, `lfc = -0.501`, `fdr = 0.0464`).
- Непересекающийся split-half (порядок батча, 12 групп, среднее):
  - по генам: `r(63v63) = 0.053`, `r(150v150) = 0.061`;
  - по среднему `log2FC` путей (MSigDB Hallmark + KEGG + Reactome, `>= 15` генов): `r(63v63) = 0.248`, `r(150v150) = 0.228` (в т.ч. `BPD7mo VEC = 0.493`, `BPDPH7mo Pericyte = 0.466`, `Acute26 VEC = 0.374`, `BPD7mo gCap = 0.352`, `BPD7mo aCap = 0.343`; `BPDPH7mo VEC = -0.057`, `Acute26 Pericyte = -0.153`).
- Совпадение знака `signed_z` на 1 025 значимых путях `rep300`: с `orig63_noazi` — `62.0%` (`med_z = +0.51`), с `rats_upper_full (azi_vs_noazi)` — `77.1%` (`med_z = +1.11`), с `rats_upper_full (azi_vs_own)` — `45.0%` (`med_z = -0.23`), с `humans_only_full (azi_vs_own)` — `46.4%` (`med_z = -0.15`).
- Спирменовская корреляция `signed_z` по всем 112 859 общим тестам `rep300` vs `orig63_noazi` = `0.074`, vs `rats_upper_full (azi_vs_noazi)` = `0.070`, vs `rats_upper_full (azi_vs_own)` = `0.033`, vs `humans_only_full (azi_vs_own)` = `0.017`.

## 5. Реестр файлов и SHA-256

| тип | файл | размер | sha256 |
| --- | --- | --- | --- |
| script | retrain_v2/c2s-puma-std/grid/lib/scan_only.py | 5.1 KB | 5f8040af15a7f9e890aa84cd6ba9361333d3e653f4d43bf58528995dafae2dcb |
| script | retrain_v2/c2s-puma-std/grid/lib/violins_only.py | 8.8 KB | 469d769a53b3cebf7508627acf64c0cc44a06ec5cd7a6c2d239400af59aa472f |
| script | retrain_v2/c2s-puma-std/hpc/train.slurm | 28.0 KB | 3d769aa680228ae4a97d0a31701b834d8731b8112177573640a346507eb3194a |
| script | retrain_v2/c2s-puma-repro/analyses/2026-11-20_human_h5ad_lr1e-4_ep4/scripts/camerapr_azi_noazi.py | 11.8 KB | b185a3f0295d107aaa3431b0203443f292b7c4926fa02fc4495104642249404c |
| script | retrain_v2/results/2026-11-23_std_runs_analysis/run_std.py | 10.6 KB | f75cb6548b3824712f099944510201a967feeff52c2f061bd39275dc9380338e |
| data | BPD-PH/GSE275938_compiled_counts.h5ad | 413.2 MB | a1029d4bf0382a30a427a12ead0357b60f3edd77cb83e37116c1d8c4b07bd555 |
| env | retrain_v2/results/2026-11-23_std_runs_analysis/env.json | 868 B | 54383ec489ab88c822c21d45907ef64551aeae2c8025e8241d293c2ae622d3ab |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/unified_all_with_std.csv | 1.4 KB | 5e8fde9cbbbe496b2299c77d5448ee0d38bf713d75189b4af851b8ea7009947a |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_humans_only_full/summary.csv | 810 B | 5b05d40056d491d136d1c442265e23f32accd4235d80d3611c908b8d217a926f |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_humans_only_full/camerapr_three_arms.csv | 66.3 MB | fc585038a8bfc248148427fc51345019bcf3dcd44b24b69cc854ce65662ef084 |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_humans_only_full_6drugs/all_6drugs_camerapr_summary.csv | 4.0 KB | 13d449c0d94ad5bcdcba8c877ec64c28e891a255b697828538f86f6b2377d745 |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_humans_only_full_6drugs/all_6drugs_camerapr_full.csv | 337.2 MB | 4ce9171acd6e6ce12def4f06edf6a29e9fd1026aca4ad395b8e45fbc1f32358e |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_humans_only_full_6drugs/drug_omnibus_kw_markers.csv | 127.5 KB | 1b1a91a393fd39b605c71568473624198303db8c0b6e270a08021da6d68241ad |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_humans_only_full_6drugs/drug_one_vs_rest_markers.csv | 176.5 KB | 0d47b51d2d8ae4f7c9892a293fa09fc493769e1f0ed3bb8ebb8a916bda959504 |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_humans_only_full_6drugs/drug_pairwise_de_counts.csv | 7.4 KB | 950e62cd9c097c36bc05fd0a6223c800e4b05968dd0c45eaf1bce08ebbda2b7d |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_rats_upper_full/summary.csv | 825 B | 0d2070e1fc9d51abf61e3222f1548df97c48faa1e2dd0070739f9072c1280c97 |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_rats_upper_full/camerapr_three_arms.csv | 66.3 MB | c5e3671ab5725f729c908be19a0a8f498bf61b20632fcdd844284fa95b5e9eab |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_rep300_lr1e-4_ep4/summary.csv | 717 B | 12a5642ac5da8d5b0e21b63ec37f93aaf9f2b8a55894e4da0a44d581794acb7a |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_rep300_lr1e-4_ep4/camerapr_three_arms.csv | 50.2 MB | 55dbe83724825e997fc7d143fd0f94f817e9b1dd392c6ba84ecf40af63e94def |
| output | retrain_v2/results/2026-11-23_std_runs_analysis/camerapr_rep300_lr1e-4_ep4/rep300_azi_vs_noazi_de.csv | 126 B | f626419ce2235ea92207310019139cff5e03def1d3219a4f279d342444a46c3f |

## Наблюдения автора

<!-- Заполняется только автором -->
