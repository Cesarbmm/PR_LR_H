"""Compatibility wrapper for the canonical Frontier hidden-reward evaluation entrypoint."""

from __future__ import annotations

from chromahack.evaluation.eval_frontier_hidden import build_parser, main as _main, run as _run
from chromahack.legacy import warn_redirect

__all__ = ["build_parser", "main", "run"]


def run(args):
    warn_redirect("chromahack.evaluation.eval_hidden", "python -m chromahack.evaluation.eval_frontier_hidden")
    return _run(args)


def main(argv: list[str] | None = None) -> None:
    warn_redirect("chromahack.evaluation.eval_hidden", "python -m chromahack.evaluation.eval_frontier_hidden")
    _main(argv)


if __name__ == "__main__":
    main()
