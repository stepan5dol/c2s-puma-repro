# Figure 5 and Figures 5-S1 to 5-S8, newgrid_lr1e-4_ep4_s42

Model: `/xdisk/vkalin/dolzhenko/c2s/results/2026-11-16_132201_puma-a100mig-repro-ep4-fp32-lr1e-4-s42`.

Inputs
- test set, one generated cell per test cell: `inference_float32/test_t10.jsonl`
- 300 generated cells per condition and cell type, vLLM bf16 (`hpc/infer_rep300.slurm`, job 24738461):
  `inference_bfloat16_rep300/<condition>_azi_results.jsonl` (azithromycin arm, joined into one file for the rescued
  genes) and `<condition>_noazi_results.jsonl` (model arm of the human violins and of 5-S1)

Run
```
bash scripts/run_violins.sh <out> <azi jsonl> <Acute26_noazi> <BPD7mo_noazi> <BPDPH7mo_noazi>
python3 scripts/build_figures.py <out> <Acute26_noazi> <BPD7mo_noazi> <BPDPH7mo_noazi>
python3 scripts/build_supplement_pdf.py <out>/figures <Figure.pdf> <out pdf>
```
`run_violins.sh`: `grid/lib/violins_only.py` on the grid tree — rescued genes, violins, and the cameraPR screen of
the three disease states (`ablation_export_for_limma.py`, `ablation_limma_sets.py`). `build_figures.py`: violins under
the manuscript's figure numbers, panel 5C, 5-S7 and 5-S8 by the manuscript's drawing scripts
(`analysis/real_counts_suite_ortho/build_panelC_dotplot.py`, `venn_rat_human_model_run4.py`, `build_main_figure.py`),
and 5-S1, the energy distance between generated and measured cells. `build_supplement_pdf.py`: places the figures
on pages 13-20 of the figure PDF and adds 5-S8a and 5-S8b.

Results (2026-11-23)
- violins: the real arm is unchanged; the model arm has 300 generated cells per cell type in the three human disease
  contrasts (was one per test cell, 7-150)
- panel 5C, BPD at 7 months: 26 of 32 pathway halves shift toward control at FDR < 0.05, none away
- 5-S8: acute preterm lung injury 7 of 32, BPD-PH 10 of 32, all toward control
- 5-S1, general capillary endothelial cells: energy 0.12 / 0.26 / 0.14 (rat room air / hyperoxia / + azithromycin);
  0.61 / 0.30 / 0.23 / 0.09 / 0.78 / 0.13 (human fetal GW24 / term day 7 / term day 20 / acute injury / BPD / BPD-PH)
