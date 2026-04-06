"""Helpers for deprecation shims kept for backward compatibility."""

from __future__ import annotations

import argparse
import warnings


BRIDGE_PIVOT_MESSAGE = (
    "The older visual proxy CNN, bridge demo, and classic GhostMerc paths have been superseded by "
    "the canonical GhostMerc Frontier Territory research testbed. Use the Frontier training, "
    "evaluation, rendering, and experiment entrypoints documented in README.md."
)


def deprecation_message(old_path: str, replacement: str | None = None) -> str:
    message = f"{old_path} is deprecated. {BRIDGE_PIVOT_MESSAGE}"
    if replacement:
        message = f"{message} Replacement: {replacement}"
    return message


def warn_redirect(old_path: str, replacement: str) -> None:
    warnings.warn(deprecation_message(old_path, replacement), DeprecationWarning, stacklevel=2)


def build_deprecated_parser(old_path: str, replacement: str | None = None) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=deprecation_message(old_path, replacement))


def deprecated_main(old_path: str, replacement: str | None = None, argv: list[str] | None = None) -> None:
    parser = build_deprecated_parser(old_path, replacement)
    parser.parse_args(argv)
    raise SystemExit(deprecation_message(old_path, replacement))
