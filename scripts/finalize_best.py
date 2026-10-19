import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


out = Path(sys.argv[1])
states = sorted(out.glob("checkpoint-*/trainer_state.json"),
                key=lambda p: int(p.parent.name.split("-")[1]))
if not states:
    sys.exit(f"no checkpoints in {out}")
best_path = json.loads(states[-1].read_text()).get("best_model_checkpoint")
if not best_path:
    sys.exit(f"no best_model_checkpoint in {states[-1]}")
best = out / Path(best_path).name
weights = sorted(p.name for p in best.glob("model*.safetensors*"))
if not weights:
    sys.exit(f"no weights in {best}")

if (out / "model.safetensors").exists() or (out / "model.safetensors.index.json").exists():
    extra = sorted(p.name for p in out.glob("model*.safetensors*") if p.name not in weights)
    if extra:
        sys.exit(f"final model has files not in {best.name}: {extra}")
    for name in weights:
        shutil.copyfile(best / name, out / name)
        if sha256(best / name) != sha256(out / name):
            sys.exit(f"copy of {best.name}/{name} does not match")
        print(f"{name} <- {best.name}", file=sys.stderr)

(out / "best_checkpoint.txt").write_text(best.name + "\n")
print(best.name)
