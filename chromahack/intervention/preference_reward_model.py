"""Compatibility wrapper around the GhostMerc preference-model scaffold."""

from __future__ import annotations

from chromahack.intervention.pref_model import build_parser, main as _main

__all__ = ["build_parser", "main"]


def run(*args, **kwargs):
    raise SystemExit(
        "Use `python -m chromahack.intervention.pref_model export ...` "
        "or `python -m chromahack.intervention.pref_model train ...`."
    )


def main(argv: list[str] | None = None) -> None:
    _main(argv)


if __name__ == "__main__":
    main()
