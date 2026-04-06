"""Path helpers that keep Frontier artifacts rooted inside the repository."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root that contains the canonical artifacts directory."""

    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a relative path against the repository root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def resolve_input_path(path: str | Path) -> Path:
    """Resolve an input path, preferring the current working directory and then the repo root."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_path = Path.cwd() / candidate
    if cwd_path.exists():
        return cwd_path
    return project_root() / candidate
