"""Compatibility wrapper for the canonical Frontier PPO training entrypoint."""

from __future__ import annotations

from chromahack.legacy import warn_redirect
from chromahack.training.train_ppo_frontier import build_parser, main as _main, make_frontier_env, run as _run

make_chroma_env = make_frontier_env

__all__ = ["build_parser", "main", "make_frontier_env", "make_chroma_env", "run"]


def run(args):
    warn_redirect("chromahack.training.train_ppo", "python -m chromahack.training.train_ppo_frontier")
    return _run(args)


def main(argv: list[str] | None = None) -> None:
    warn_redirect("chromahack.training.train_ppo", "python -m chromahack.training.train_ppo_frontier")
    _main(argv)


if __name__ == "__main__":
    main()
