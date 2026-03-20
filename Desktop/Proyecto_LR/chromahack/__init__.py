"""Workspace-level shim so `python -m chromahack...` works from Proyecto_LR."""

from __future__ import annotations

from pathlib import Path

_SHIM_DIR = Path(__file__).resolve().parent
_REAL_PACKAGE = _SHIM_DIR.parent / "PR_LR_H" / "chromahack"

if not _REAL_PACKAGE.exists():
    raise ImportError(f"Expected real chromahack package at {_REAL_PACKAGE}")

__path__ = [str(_REAL_PACKAGE)]
__file__ = str(_REAL_PACKAGE / "__init__.py")

exec(compile((_REAL_PACKAGE / "__init__.py").read_text(encoding="utf-8"), __file__, "exec"))
