"""Provenance block for evaluation JSONs.

A result is only reproducible if the artifact says what produced it. Every
new eval JSON gets this block: git commit and dirty flag, SHA-1 of the ROM
and savestate actually opened, package versions, Python/platform, timestamp,
seed and the measured course index. Existing (pre-2026-10-23) JSONs do not
have it; tools/audit_evals.py reports them as legacy instead of pretending
they were annotated after the fact.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys

import paths

PACKAGES = (
    "numpy", "torch", "torchvision", "py-desmume", "gymnasium",
    "stable-baselines3", "sb3-contrib", "opencv-python", "ndspy",
    "mlflow", "tensorboard", "sheeprl",
)


def sha1_file(path, chunk=1 << 20):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_state(root=None):
    root = root or paths.ROOT
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, timeout=10, check=False).stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root, capture_output=True, text=True, timeout=10,
            check=False).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def package_versions(names=PACKAGES):
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def collect(rom=None, state=None, extra=None, root=None):
    """A JSON-serializable provenance block for one eval invocation."""
    root = root or paths.ROOT
    files = {}
    for label, path in (("rom", rom), ("state", state)):
        abs_path = paths.resolve(path) if path else None
        files[label] = {
            "path": paths.display(path) if path else None,
            "sha1": sha1_file(abs_path),
            "bytes": os.path.getsize(abs_path) if abs_path and os.path.exists(abs_path) else None,
        }
    block = {
        "schema": 1,
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc)
                           .isoformat(timespec="seconds"),
        "git": git_state(root),
        "files": files,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "argv": [os.path.basename(sys.argv[0])] + sys.argv[1:],
    }
    if extra:
        block.update(extra)
    return block
