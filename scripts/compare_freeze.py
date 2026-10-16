"""Compare the image's pip freeze with the pod's env_snapshot/pip_freeze.txt.

Exit 1 if any package is missing, extra or at a different version, except the
differences listed in ALLOWED (pod version, image version), which are printed.
"""
import re
import sys
from urllib.parse import unquote

ALLOWED = {
    # same release, prebuilt wheel instead of the pod's source build
    "flash-attn": ("2.8.3", "2.8.3+cu130torch2.13"),
    # not used by Pythia; no prebuilt wheel for torch 2.13
    "mamba-ssm": ("2.3.2.post1", None),
}


def parse(path):
    pkgs = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e "):
            # editable axolotl: checked by commit in build_info.txt
            pkgs["axolotl"] = "editable"
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:==\s*(\S+)|@\s*(\S+))$", line)
        if not m:
            pkgs[line] = "?"
            continue
        name = re.sub(r"[-_.]+", "-", m.group(1)).lower()
        ver = m.group(2) or m.group(3)
        if m.group(3) and m.group(3).startswith("file://") and m.group(3).endswith(".whl"):
            ver = unquote(m.group(3).rsplit("/", 1)[1]).split("-")[1]
        pkgs[name] = ver
    return pkgs


pod, image = parse(sys.argv[1]), parse(sys.argv[2])
bad, allowed = [], []
for name in sorted(pod.keys() | image.keys()):
    a, b = pod.get(name), image.get(name)
    if a != b:
        (allowed if ALLOWED.get(name) == (a, b) else bad).append(f"{name}: pod={a} image={b}")

print(f"pod: {len(pod)} packages, image: {len(image)} packages")
if allowed:
    print(f"{len(allowed)} allowed differences:")
    print("\n".join(allowed))
if bad:
    print(f"{len(bad)} differences:")
    print("\n".join(bad))
    sys.exit(1)
print("identical apart from the allowed differences" if allowed else "identical")
