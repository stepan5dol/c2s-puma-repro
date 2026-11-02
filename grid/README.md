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

## Commands, in order

On the Mac, normal terminal (the sandbox of Claude Code blocks git writes):

```
cd ~/Documents/AzithroGemma/retrain_v2/c2s-puma-repro
bash commit.sh          # commit and push; the push starts the GitHub build of the analysis image (Actions: build-analysis-sif, ~20 min)
bash deploy.sh          # tags the commit, one rsync (password + Duo): repo to xdisk; the 7 GB of data only if they changed
```

On Puma (`ssh dolzhenko@hpc.arizona.edu`, then `shell`), once the build-analysis-sif run is green:

```
bash /xdisk/vkalin/dolzhenko/c2s/repo/grid/start.sh
```

`start.sh` checks that the image is published and the data are in `data_src`, queues the fp32 inference of the starting
cell (run 2026-10-28_1152) unless it is already there, writes `cells.tsv`, and queues stage A (1e-5 and 3e-5, 2 epochs).
The first analysis job downloads the image from the GitHub release (tag = hash of `apptainer/c2s-analysis.def`) into
`/xdisk/vkalin/dolzhenko/c2s/images/` and checks its sha256; later jobs reuse it. Nothing is built on Puma.

Watch it:  `squeue -u $USER`,  `tail -f /xdisk/vkalin/dolzhenko/c2s/grid/chain.log`,  `ls /xdisk/vkalin/dolzhenko/c2s/grid/decisions`,
`cat /xdisk/vkalin/dolzhenko/c2s/grid/STOPPED` (exists only when the chain has stopped, with the reason).

## Notes

1. `sbatch --test-only hpc/train.slurm` (and the other scripts) checks only the directives; the first pair of trainings is the real test.
2. `grid/tree/` is the analysis code of the manuscript with its Mac path prefix; `grid/lib/prepare_root.py` rewrites that one
   prefix at run time. Rebuild the tree on the Mac with `python3 grid/tools/make_tree.py <AzithroGemma folder>`.
3. The model of the starting cell is checkpoint-264 (load_best_model_at_end); the other cells are the end of the run. At LR ~0 the two
   differ by almost nothing (eval_loss 2.184 at both steps).
