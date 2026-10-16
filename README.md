# c2s-puma-repro

Rerun of the training from `dolzhenkosv/c2s-pythia-1b-azithro-transfer@040f129` on a Puma
A100 MIG slice (3g.40gb), to test the stack on this hardware. The config and data are the
original ones, verified by sha256 against `env_snapshot/inputs_sha256.txt`.

- `apptainer/c2s-train.def`: image built from the pod's `env_snapshot/pip_freeze.txt`
  (nvidia/cuda 13.0.0 devel, Python 3.12.13, axolotl 1855055). Two differences from the pod:
  flash-attn 2.8.3 is a prebuilt wheel instead of a source build, and mamba-ssm (unused by
  Pythia) is not installed.
- `.github/workflows/build-sif.yml`: builds the image (manual run), compares its freeze with
  the pod's and uploads it to the public HF repo `dolzhenkosv/c2s-puma-sif`.
- `hpc/train.slurm`: the run on Puma. Output goes to `~/c2s/results/<stamp>_puma-a100mig-repro/`;
  checkpoints are mirrored to the public HF repo `dolzhenkosv/c2s-puma-runs` during training.
