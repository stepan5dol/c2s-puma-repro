# c2s-puma-repro

Rerun of the training from `dolzhenkosv/c2s-pythia-1b-azithro-transfer@040f129` on a Puma
A100 MIG slice (3g.40gb), to test the stack on this hardware, and the same training for
`vandijklab/C2S-Scale-Gemma-2-2B`. Configs and data are checked by sha256 against the
source repo's list before training.

- `apptainer/c2s-train.def`: one image with two Python environments.
  - `/opt/venv` (default): training, built from the training pod's `env_snapshot/pip_freeze.txt`
    (nvidia/cuda 13.0.0 devel, Python 3.12.13, axolotl 1855055). Differences from the pod:
    flash-attn 2.8.3 is a prebuilt wheel instead of a source build, and mamba-ssm (unused) is
    not installed.
  - `/opt/infer` (`apptainer run --app infer`): vLLM 0.30.0, built from the inference pod's
    `inference_ckpt264_2026-10-05/env_snapshot/pip_freeze.txt`. Not installed: deep_ep (local
    wheel, MoE only), python-apt, dbus-python, PyGObject (Ubuntu system packages).
- `scripts/finalize_best.py` (in the image): after training, replaces the final model in the
  output dir with the weights of the best checkpoint and checks them by sha256. Needed because
  with this transformers version `load_best_model_at_end` does not load `embed_out.weight`
  into GPT-NeoX, so the final model kept the output layer from the last step.
- `scripts/run_test.py`, `scripts/run_azi.py` (in the image): the inference pod's runners. The
  only change: prompts are tokenized with `add_special_tokens=False` and passed to vLLM as
  token ids, with `<bos>` prepended when the tokenizer adds one (Gemma), as in training.
- `scripts/build_gemma_segments.py`: makes the Gemma dataset from the Pythia one: `<bos>`
  before the prompt, `<eos>` instead of `<|endoftext|>`, nothing else. The result is in the
  private HF repo `dolzhenkosv/c2s-gemma2-2b-azithro-transfer` with `sft_gemma2_2b.yaml`
  (`configs/` here) and `inputs_sha256.txt`.
- `configs/sft_gemma2_2b.yaml`: `sft_pythia1b.yaml` with the Gemma base model and output dir,
  no `pad_token` override (Gemma has `<pad>`), and Cut Cross Entropy (fp32 accumulation of the
  output layer gradient), without which the 256k-vocabulary logits do not fit in 40 GB.
- `.github/workflows/build-sif.yml`: builds the image (manual run), compares both environments
  with the pods' freezes and uploads it to the private HF repo `dolzhenkosv/c2s-puma-sif`.
- `hpc/train.slurm`: the run on Puma. Needs `~/c2s/c2s.env` with `HF_TOKEN=<write token>`;
  optional `TG_TOKEN` and `TG_CHAT` there plus `~/c2s/tg_monitor.py` for Telegram updates.
  Output goes to `~/c2s/results/<stamp>_<slug>/`; checkpoints without `optimizer.pt` are
  mirrored to the private HF repo `dolzhenkosv/c2s-puma-runs` during training. If training
  runs out of GPU memory before the first checkpoint, it is restarted in the same job with half
  the micro batch and twice the gradient accumulation (same effective batch).

```
sbatch train.slurm                                                       # Pythia 1b
sbatch --time=3-00:00:00 --export=ALL,MODEL=gemma2b train.slurm          # Gemma 2 2B
sbatch --time=3-00:00:00 --export=ALL,MODEL=gemma2b,INFER=1 train.slurm  # + inference on test
```

`INFER=1` runs `run_test.py` on `test_segments.jsonl` with the best checkpoint at temperatures
`INFER_TEMPS` (default `0.8 1.0 1.2`); results go to `inference/` in the run dir.

The run of 2026-10-17 (`runs/2026-10-17_0244_puma-a100mig-repro`) predates the fix: its
`result/sft_pythia1b_short_full/model.safetensors` is checkpoint-264 with `embed_out.weight`
from step 396. Use `checkpoint-264/`, the only checkpoint kept.
