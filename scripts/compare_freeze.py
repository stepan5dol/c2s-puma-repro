"""Compare the image's pip freeze with the pod's env_snapshot/pip_freeze.txt.

Exit 1 if any package is missing, extra or at a different version.
"""
import re
import sys


def parse(path):
    pkgs = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e ") or "file://" in line:
            # editable axolotl: checked by commit in build_info.txt
            pkgs["axolotl"] = "editable"
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:==\s*(\S+)|@\s*(\S+))$", line)
        if not m:
            pkgs[line] = "?"
            continue
        name = re.sub(r"[-_.]+", "-", m.group(1)).lower()
        pkgs[name] = m.group(2) or m.group(3)
    return pkgs


pod, image = parse(sys.argv[1]), parse(sys.argv[2])
bad = []
for name in sorted(pod.keys() | image.keys()):
    a, b = pod.get(name), image.get(name)
    if a != b:
        bad.append(f"{name}: pod={a} image={b}")

print(f"pod: {len(pod)} packages, image: {len(image)} packages")
if bad:
    print(f"{len(bad)} differences:")
    print("\n".join(bad))
    sys.exit(1)
print("identical")
