# Grid: learning rate × epochs, trained and chosen automatically on Puma

Everything lives on xdisk (`/xdisk/vkalin/dolzhenko/c2s/`); nothing is uploaded to Hugging Face.

```
grid/submit_cells.sh 1e-5:2:42 3e-5:2:42     one call by you; then the chain runs by itself
  per cell:   train (fp32, no early stopping, end-of-run model)  ->  fp32 inference of the test set at T 1.0
  per call:   analysis job after all inferences:  manuscript violins -> decide.py -> next cells -> submit_cells.sh
```

`decide.py` holds the rule (the scheme at https://claude.ai/artifact/Spx6j3W35P4boV2sLZHcY3 shows it):
violins worse than the copy of the input drop a run; among the rest the reverse task for human cells
(McNemar) decides, ties go to the lower eval_loss (stage A) or fewer epochs (stage B), the energy distance is a guard.
Edge of the explored grid -> one more cell beyond it. The chain stops, and says why in `grid/STOPPED`, when the rule
says stop or finished, when a Slurm job of a cell fails, or when `MAX_CELLS` (default 9) trainings are reached.
Decisions with their reasons: `grid/decisions/decision_NN.json`; log: `grid/chain.log`.

## Once, before the first call

1. Copy the repository to xdisk (`/xdisk/vkalin/dolzhenko/c2s/repo`) and the data (7 GB) to `data_src/`:
   `GSE275938_compiled_counts.csv`, `GSE275938_cell_metadata.csv`, `rat.ho.azi.integrated.h5ad`, `he_lung_atlas.h5ad`,
   `2022FetalLungIntCounts.h5ad`, `cell_types_config.json`.
2. Build the analysis image (Python only, no R; data inside, ~1 h, an interactive session on a compute node):
   `interactive -a vkalin -t 04:00:00 -c 8` then `bash grid/build_analysis_image.sh`.
3. Check the scripts: `sbatch --test-only` on `hpc/train.slurm`, `hpc/infer_only.slurm`, `grid/analyze.slurm` checks only
   the directives; the first pair of trainings is the real test.
4. `grid/tree/` is the analysis code of the manuscript with its Mac path prefix; `grid/lib/prepare_root.py` rewrites that
   one prefix at run time. Rebuild the tree on the Mac with `python3 grid/tools/make_tree.py <AzithroGemma folder>`.
5. The grid's starting cell `lr1e-4_ep2_s42` is the finished run 2026-10-28_1152. Its fp32 inference
   (`SETS=test_t10 RUN=<that folder> sbatch hpc/infer_only.slurm`) has to be in `inference_float32/` of that folder;
   add a line `lr1e-4_ep2_s42<TAB><run folder><TAB>0<TAB>0` to `grid/cells.tsv`.
   The model of that run is checkpoint-264 (load_best_model_at_end), the other cells are the end of the run; at LR ~0
   the two differ by almost nothing (eval_loss 2.184 at both steps).
