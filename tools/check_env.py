"""Report the local venv against requirements.txt (the drift check).

The pin says torch 2.6.0+cu124; the venv that produced every committed gate
result runs torch 2.5.1+cu121 with mlflow absent. This tool prints the delta
so nobody rediscovers it by an ImportError mid-training. It never installs
anything and always exits 0 (informational).
"""
from __future__ import annotations

import importlib.metadata
import re
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ = os.path.join(ROOT, "requirements.txt")


def pinned():
    out = {}
    for line in open(REQ, encoding="utf-8"):
        line = line.strip()
        if line.startswith("#") or line.startswith("--"):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)(?:\[.*\])?==([^ ;]+)", line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def main():
    print(f"python: {sys.version.split()[0]}")
    header = f"{'package':<22} {'pinned':<16} installed"
    print(header)
    for name, want in sorted(pinned().items()):
        try:
            have = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            have = "<ABSENT>"
        mark = "" if want == have else "   <-- differs"
        print(f"{name:<22} {want:<16} {have}{mark}")
    print("\nExit 0 (informational). An ABSENT mlflow breaks the trainers that")
    print("import it at module top (train.py, train_impala.py, run_pipeline.py).")


if __name__ == "__main__":
    main()
