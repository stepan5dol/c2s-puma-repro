#!/usr/bin/env python3
import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/axolotl/mainwork/sft_pythia1b_short_full")
TEST_PATH = os.environ["TEST_PATH"]
OUT_PATH = os.environ["OUT_PATH"]

MAX_NEW_TOKENS_FWD = 4096
MAX_NEW_TOKENS_REV = 256
TEMPERATURE = float(os.environ.get("TEMPERATURE", 1.0))
MAX_MODEL_LEN = 8192

print("loading examples...", flush=True)
examples = []
with open(TEST_PATH) as f:
    for i, line in enumerate(f):
        obj = json.loads(line)
        prompt = obj["segments"][0]["text"]
        gt = obj["segments"][1]["text"]
        direction = "fwd" if "Determine the single cell's expression changes" in prompt[:64] else "rev"
        examples.append({"idx": i, "prompt": prompt, "gt": gt, "direction": direction})

rev_examples = [e for e in examples if e["direction"] == "rev"]
fwd_examples = [e for e in examples if e["direction"] == "fwd"]
print(f"loaded {len(examples)} test examples ({len(fwd_examples)} fwd / {len(rev_examples)} rev)", flush=True)

print("loading model...", flush=True)
llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.9)
tokenizer = llm.get_tokenizer()
eos_id = tokenizer.eos_token_id
bos = tokenizer.bos_token if tokenizer("x")["input_ids"][0] == tokenizer.bos_token_id else ""


def encode(prompt):
    if bos and not prompt.startswith(bos):
        prompt = bos + prompt
    return TokensPrompt(prompt_token_ids=tokenizer(prompt, add_special_tokens=False)["input_ids"])


def run_group(group, max_new, out_f, start, total_done_offset, total):
    if not group:
        return
    sp = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=max_new,
        stop_token_ids=[eos_id],
    )
    prompts = [encode(e["prompt"]) for e in group]
    outputs = llm.generate(prompts, sp)

    for e, out in zip(group, outputs):
        pred = out.outputs[0].text
        finish_reason = out.outputs[0].finish_reason
        rec = {
            "idx": e["idx"], "direction": e["direction"], "prompt": e["prompt"],
            "gt": e["gt"], "pred": pred, "finish_reason": finish_reason,
            "gen_tokens": len(out.outputs[0].token_ids),
        }
        out_f.write(json.dumps(rec) + "\n")
    out_f.flush()

    elapsed = time.time() - start
    done = total_done_offset + len(group)
    print(f"[{done}/{total}] elapsed={elapsed/60:.1f}min", flush=True)


start = time.time()
with open(OUT_PATH, "w") as out_f:
    print(f"--- rev pass: {len(rev_examples)} examples ---", flush=True)
    run_group(rev_examples, MAX_NEW_TOKENS_REV, out_f, start, 0, len(examples))

    print(f"--- fwd pass: {len(fwd_examples)} examples ---", flush=True)
    run_group(fwd_examples, MAX_NEW_TOKENS_FWD, out_f, start, len(rev_examples), len(examples))

print("done, wrote", OUT_PATH, flush=True)
