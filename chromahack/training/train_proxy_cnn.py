"""Deprecated proxy CNN training CLI from the superseded visual proxy pipeline."""

from __future__ import annotations

from chromahack.legacy import build_deprecated_parser, deprecated_main, deprecation_message

DEPRECATION_MESSAGE = deprecation_message("chromahack.training.train_proxy_cnn", "python -m chromahack.training.train_ppo_ghostmerc")

__all__ = ["DEPRECATION_MESSAGE", "build_parser", "main", "run"]


def build_parser():
    return build_deprecated_parser("chromahack.training.train_proxy_cnn", "python -m chromahack.training.train_ppo_ghostmerc")


def run(*args, **kwargs):
    raise SystemExit(DEPRECATION_MESSAGE)


def main(argv: list[str] | None = None) -> None:
    deprecated_main("chromahack.training.train_proxy_cnn", "python -m chromahack.training.train_ppo_ghostmerc", argv)


if __name__ == "__main__":
    main()
