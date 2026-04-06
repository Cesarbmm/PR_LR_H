"""Deprecated dataset helpers from the superseded visual proxy pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from chromahack.legacy import deprecation_message

DEPRECATION_MESSAGE = deprecation_message("chromahack.data.dataset_io", "python -m chromahack.training.train_ppo_ghostmerc")


@dataclass(slots=True)
class DatasetSample:
    payload: dict | None = None


@dataclass(slots=True)
class DatasetStats:
    payload: dict | None = None


def load_dataset_payload(*args, **kwargs):
    raise SystemExit(DEPRECATION_MESSAGE)


def save_dataset_payload(*args, **kwargs):
    raise SystemExit(DEPRECATION_MESSAGE)
