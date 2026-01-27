# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_workspace_root(start: Optional[Path] = None) -> Path:
    """Find the workspace root by walking upwards.

    Heuristics:
    - Prefer a directory containing `pyproject.toml`
    - Fallback to a directory containing `src/qualivault`

    If nothing is found, returns `start` (or current working directory).
    """

    current = Path(start or Path.cwd()).resolve()

    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / "src" / "qualivault").exists():
            return parent

    return current
