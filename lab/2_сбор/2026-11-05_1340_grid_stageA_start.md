---
дата: 2026-11-05 13:40
теги: [grid, sft, stageA, puma, a100mig, slurm]
статус: queued
---

# Запуск сетки гиперпараметров (Stage A) на Puma

- **Дата и время**: 2026-11-05 13:40:12 MST
- **Кластер**: University of Arizona HPC (Puma)
- **Раздел Slurm**: `gpu_standard`, аккаунт `vkalin`
- **Ресурсы**: 3 задачи обучения по 1 MIG-срезу A100 (`nvidia_a100_80gb_pcie_3g.40gb`), 8 CPU (40 GB RAM) на задачу
- **Базовая модель**: Pythia-1B (`vandijklab/C2S-Scale-Pythia-1b-pt`)
- **Режим обучения**: `fp32` мастер-веса (`PRECISION=fp32`, `INFER_FP32=1`, `NO_EARLY_STOP=1`, `CKPTS=none`)
- **Датасет**: новый `meta -> cell`, $K=1600$, 9 условий, двунаправленный SFT (`sft_dataset_bidir/{train,valid,test}_segments.jsonl`)
- **Оценка**: 5 manuscript arms $\times$ 4 cell types $\times$ 2 directions = 40 скрипок `full_list` (без DGE-гейта), per-run генерация человеческих `rescued`-панелей по собственному `inference_azi_results.jsonl`

## Задачи в очереди

| Ячейка | Slurm Job ID | Рабочая папка на xdisk | Статус |
|---|---|---|---|
| `lr1e-4_ep2_s42` | `24542467` | `/xdisk/vkalin/dolzhenko/c2s/results/2026-11-05_134012_puma-a100mig-repro-ep2-fp32-lr1e-4-s42` | Обучение + fp32 инференс `test_t10` и `inference_azi` |
| `lr1e-5_ep2_s42` | `24542468` | `/xdisk/vkalin/dolzhenko/c2s/results/2026-11-05_134012_puma-a100mig-repro-ep2-fp32-lr1e-5-s42` | Обучение + fp32 инференс `test_t10` и `inference_azi` |
| `lr3e-5_ep2_s42` | `24542469` | `/xdisk/vkalin/dolzhenko/c2s/results/2026-11-05_134012_puma-a100mig-repro-ep2-fp32-lr3e-5-s42` | Обучение + fp32 инференс `test_t10` и `inference_azi` |
| `c2s-analyze` | `24542470` | `/xdisk/vkalin/dolzhenko/c2s/grid` | Ожидает завершения всех 3 ячеек (`afterok:24542467:24542468:24542469`) |

## Что происходит дальше

1. Каждая ячейка обучается 2 эпохи на выделенном срезе A100.
2. Сразу после обучения внутри той же Slurm-задачи генерируются `inference_float32/test_t10.jsonl` и `inference_float32/inference_azi_results.jsonl`.
3. После завершения всех трёх ячеек стартует задача `c2s-analyze` (`24542470`):
   - Вычисляет per-run human rescue workbooks и 40 скрипок `full_list` для каждой модели.
   - `decide.py` ранжирует модели по качеству прямых скрипок (число значимых из 40, парный критерий МакНемара) с тай-брейком по `eval_loss`.
   - Автоматически отправляет следующую стадию (Stage B — выбор числа эпох для победившего LR).
