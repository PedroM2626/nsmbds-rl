"""Repository-relative paths, resolved independently of the current directory.

The project is a flat set of scripts under src/; several of them used to
resolve data/... against the caller's cwd, so running
python src/eval_pinn_realtime.py from anywhere but the repository root
failed to find the ROM. Resolve here once.

Kept deliberately small and dependency-free so tools and tests can import it
without pulling torch / DeSmuME.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

ROM_REL = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).nds"
STATE_REL = "data/0479 - New Super Mario Bros. (Europe) (En,Fr,De,Es,It).ds1"


def resolve(path, ):
    """Absolute path for a repository-relative path; absolute paths pass through."""
    if path is None:
        return None
    path = os.fspath(path)
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def display(path):
    """Repository-relative form for metrics JSON, when inside the repository."""
    if path is None:
        return None
    path = os.fspath(path)
    try:
        rel = os.path.relpath(os.path.abspath(path), ROOT)
    except ValueError:
        return path
    return path if rel.startswith("..") else rel.replace(os.sep, "/")
