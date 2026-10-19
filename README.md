# c2s-puma-repro

Rerun of the training from `dolzhenkosv/c2s-pythia-1b-azithro-transfer@040f129` on a Puma
A100 MIG slice (3g.40gb), to test the stack on this hardware. The config and data are the
original ones, verified by sha256 against `env_snapshot/inputs_sha256.txt`.

- `apptainer/c2s-train.def`: image built from the pod's `env_snapshot/pip_freeze.txt`
  (nvidia/cuda 13.0.0 devel, Python 3.12.13, axolotl 1855055). Two differences from the pod:
  flash-attn 2.8.3 is a prebuilt wheel instead of a source build, and mamba-ssm (unused by
  Pythia) is not installed.
- `scripts/finalize_best.py` (in the image as `/opt/c2s/finalize_best.py`): after training,
  replaces the final model in the output dir with the weights of the best checkpoint and
  checks them by sha256. Needed because with this transformers version
  `load_best_model_at_end` does not load `embed_out.weight` into GPT-NeoX (logged as missing
  `lm_head.weight` / unexpected `embed_out.weight`), so the final model kept the output layer
  from the last step. Training itself is unchanged.
- `.github/workflows/build-sif.yml`: builds the image (manual run), compares its freeze with
  the pod's and uploads it to the public HF repo `dolzhenkosv/c2s-puma-sif`.
- `hpc/train.slurm`: the run on Puma. Needs `~/c2s/c2s.env` with `HF_TOKEN=<write token>`
  (checkpoint uploads only); optional `TG_TOKEN` and `TG_CHAT` there plus `~/c2s/tg_monitor.py`
  for Telegram updates. Output goes to `~/c2s/results/<stamp>_puma-a100mig-repro/`;
  checkpoints are mirrored to the public HF repo `dolzhenkosv/c2s-puma-runs` during training.

The run of 2026-10-17 (`runs/2026-10-17_0244_puma-a100mig-repro`) predates the fix: its
`result/sft_pythia1b_short_full/model.safetensors` is checkpoint-264 with `embed_out.weight`
from step 396. Use `checkpoint-264/`.
