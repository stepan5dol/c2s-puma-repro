#!/usr/bin/env python3
import json
import sys
from pathlib import Path

SRC_DIR, DST_DIR = Path(sys.argv[1]), Path(sys.argv[2])
PYTHIA_EOS, BOS, EOS = "<|endoftext|>", "<bos>", "<eos>"

DST_DIR.mkdir(parents=True, exist_ok=True)
for split in ["train", "valid", "test"]:
    src = SRC_DIR / f"{split}_segments.jsonl"
    dst = DST_DIR / f"{split}_segments.jsonl"
    n = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8", newline="\n") as out:
        for line in f:
            prompt, completion = json.loads(line)["segments"]
            assert prompt["label"] is False and completion["label"] is True
            assert completion["text"].startswith(" ") and completion["text"].endswith(PYTHIA_EOS)
            body = completion["text"][: -len(PYTHIA_EOS)]
            for text in (prompt["text"], body):
                assert not any(t in text for t in (PYTHIA_EOS, BOS, EOS))
            row = {
                "segments": [
                    {"label": False, "text": BOS + prompt["text"]},
                    {"label": True, "text": body + EOS},
                ]
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"{split}: {n:,} rows -> {dst}")
