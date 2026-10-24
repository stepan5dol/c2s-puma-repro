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
    wheel, MoE only), python-apt, dbus-python, PyGObject, iotop (Ubuntu system packages).
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
  The image, run dirs and Slurm logs are on xdisk: `/xdisk/vkalin/dolzhenko/c2s/images/`,
  `results/<stamp>_<slug>/` and `logs/` (create `logs/` before the first `sbatch`, Slurm
  does not). `diag/` in the run dir has the job's CPUs, memory
  limit and `/tmp` at start (`start.txt`), each step with its start time (`steps.log`), job
  memory, peak, `oom_kill` count, process count and free `/tmp` every 5 s (`resources.tsv`),
  and the exit code and failed step (`end.txt`, also in `record.md` and the Telegram message).
  Checkpoints without `optimizer.pt` are
  mirrored to the private HF repo `dolzhenkosv/c2s-puma-runs` during training. If training
  runs out of GPU memory before the first checkpoint, it is restarted in the same job with half
  the micro batch and twice the gradient accumulation (same effective batch).

```
sbatch train.slurm                                                                          # Pythia 1b
sbatch --time=2-12:00:00 --cpus-per-task=16 --export=ALL,MODEL=gemma2b train.slurm          # Gemma 2 2B
sbatch --time=2-12:00:00 --cpus-per-task=16 --export=ALL,MODEL=gemma2b,INFER=1 train.slurm  # + inference
sbatch --export=ALL,EPOCHS=2,INFER=1 train.slurm                                            # Pythia, 2 epochs, + inference
sbatch --export=ALL,EPOCHS=2,INFER=1,PRECISION=fp32 train.slurm                             # same, weights in fp32
```

Gemma gets 16 CPUs for their 80 GB (5 GB per CPU on Puma). With 8 CPUs (40 GB) its preprocess
failed while saving the dataset (job 24306206, "One of the subprocesses has abruptly died", no
oom_kill); the same failure appears when preprocess memory reaches a cgroup limit.

`INFER=1` runs, with the best checkpoint, `run_test.py` on `test_segments.jsonl` at temperatures
`INFER_TEMPS` (default `1.0 0.8 1.2`) and `run_azi.py` on the three azithromycin prompt sets
at `AZI_TEMP` (default `1.0`), as in `inference_ckpt264_2026-10-05`. The prompt sets are in
`dolzhenkosv/c2s-gemma2-2b-azithro-transfer/inference/`, sha256 as in that run's snapshot.
Results go to `inference/` in the run dir.

`EPOCHS=<n>` replaces `num_epochs` in the downloaded config. The run dir keeps both the original
config and `<config>_ep<n>.yaml`, `record.md` shows the value used, and the slug gets `-ep<n>`.

`PRECISION=fp32` keeps the weights in fp32 and the computation in bf16 autocast. With `bf16: true`
axolotl 1855055 loads the model in bf16 (`resolve_dtype` in `axolotl/utils/config`), so the weights
are updated in bf16. The job writes the plugin `c2s_fp32.py` next to the config and adds it to
`plugins`: it wraps `resolve_dtype` so that `torch_dtype` is float32 while `bf16` stays on. The run
dir keeps `<config>_fp32.yaml` and `c2s_fp32.py`, `record.md` shows the `torch_dtype` from
`train.log`, and the slug gets `-fp32`. The image is unchanged.
