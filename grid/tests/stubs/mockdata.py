"""Shared fake content for the curl and apptainer stubs of the mock grid test."""
import hashlib

CFG = """base_model: /hf/stub
plugins:
  - axolotl.integrations.liger.LigerPlugin
output_dir: /workspace/axolotl/mainwork/sft_pythia1b_short_full
micro_batch_size: 8
gradient_accumulation_steps: 8
num_epochs: 6
learning_rate: 1e-4
load_best_model_at_end: true
early_stopping_patience: 4
seed: 42
"""
SIF = b"stub c2s-train.sif\n"
KNOWN = ["sft_pythia1b.yaml", "dataset/train_segments.jsonl", "dataset/valid_segments.jsonl",
         "dataset/test_segments.jsonl", "inference/inference_azi.jsonl", "inference/inference_azi_concise.jsonl",
         "inference/inference_azi_ratmatch.jsonl"]


def content(rel):
    if rel == "sft_pythia1b.yaml":
        return CFG.encode()
    if rel == "c2s-train.sif":
        return SIF
    return f'{{"stub": "{rel}"}}\n'.encode()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def for_url(url):
    rel = url.split("/resolve/", 1)[1].split("/", 1)[1]   # drop the revision
    if rel == "c2s-train.sif.sha256":
        return f"{sha(SIF)}  c2s-train.sif\n".encode()
    if rel.endswith("sha256.txt"):
        return "".join(f"{sha(content(k))}  {k}\n" for k in KNOWN).encode()
    return content(rel)
